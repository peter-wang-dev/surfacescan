#include "io.h" 
import std;
#ifdef _WIN64 
#include <windows.h> 
std::map<void*,HANDLE> g_mmap_registry;
bool save_to_mmap(const char* name, const void* data, size_t size)
{
    // Explicitly target the local session namespace
    std::string mapping_name = std::string("Local\\") + name;

    // IMPORTANT: Because this mapping is backed by the paging file (INVALID_HANDLE_VALUE),
    // it will be destroyed by the OS the moment its ref count drops to 0. 
    // We must deliberately "leak" this handle into the process scope to keep it alive.
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(size),
        mapping_name.c_str());

    if (hMapFile == NULL)
    {
        //std::println("save_to_mmap: CreateFileMappingA failed, error code={}", GetLastError());
        return false;
    }

    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, static_cast<DWORD>(size));
    if (pBuf == NULL)
    {
        CloseHandle(hMapFile);
        return false;
    }

    CopyMemory(pBuf, data, size);
    
    // Deliberately doing NOT call UnmapViewOfFile(pBuf) or CloseHandle(hMapFile) 
    // so the data stays accessible process-wide.
    //std::println("save_to_mmap: Saved {} bytes to shared memory \"{}\"", size, mapping_name);
    g_mmap_registry[pBuf] = hMapFile;
    return true;
}

void load_from_mmap(const char* name, void** data, size_t* size)
{
    std::string mapping_name = std::string("Local\\") + name;

    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_READ,
        FALSE,
        mapping_name.c_str());

    if (hMapFile == NULL)
    {
        std::println("load_from_mmap: OpenFileMappingA failed for name=\"{}\", error code={}", mapping_name, GetLastError());
        *data = nullptr;
        *size = 0;
        return;
    }

    // We must map it FIRST before we can query its size
    // Passing size=0 to MapViewOfFile maps the entire existing object.
    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
    if (pBuf == NULL)
    {
        std::println("load_from_mmap: MapViewOfFile failed, error code={}", GetLastError());
        CloseHandle(hMapFile);
        *data = nullptr;
        *size = 0;
        return;
    }

    // BUG FIX: VirtualQuery requires a memory address, NOT an OS handle!
    MEMORY_BASIC_INFORMATION mbi;
    VirtualQuery(pBuf, &mbi, sizeof(mbi));
    size_t mappedSize = mbi.RegionSize;

    *data = pBuf;
    *size = mappedSize;

    // We CAN close the handle here securely because MapViewOfFile succeeded 
    // and naturally bumps the object's reference counter, keeping it alive.
    CloseHandle(hMapFile);
}
void release_all_mmap()
{
     for (auto& mapping : g_mmap_registry) 
     {
         UnmapViewOfFile(mapping.first);
         CloseHandle(mapping.second);
     }
     g_mmap_registry.clear();
}
#endif

std::map<std::string, std::vector<float>> readRingParaCSV(const std::string& filename) {
    std::map<std::string, std::vector<float>> dataMap;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::vector<std::string> headers;

    // 1. Parse the header line
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        
        while (std::getline(ss, header, ',')) {
            // Strip any trailing carriage returns (e.g., from Windows CRLF files)
            if (!header.empty() && header.back() == '\r') {
                header.pop_back();
            }
            headers.push_back(header);
            dataMap[header] = std::vector<float>(); 
        }
    }
	//std::println("Parsed headers from CSV");
	//for(auto const& h:headers)
	//	std::println("Header: {}", h);

    // 2. Parse the data lines
    while (std::getline(file, line)) {
        // Skip completely empty lines
        if (line.empty() || line.find_first_not_of(" \r\n\t") == std::string::npos) {
            continue; 
        }

        std::stringstream ss(line);
        std::string valueStr;
        size_t colIndex = 0;

        while (std::getline(ss, valueStr, ',')) {
            if (colIndex < headers.size()) {
                try {
                    // Convert the string to float and add to the specific column vector
                    float value = std::stof(valueStr);
                    dataMap[headers[colIndex]].push_back(value);
                } catch (const std::invalid_argument&) {
                    // Fallback for empty strings or invalid float data
                    dataMap[headers[colIndex]].push_back(0.0f);
                }
                colIndex++;
            }
        }
    }
	//std::println("Parsed {} rows from CSV",dataMap.begin()->second.size());

    return dataMap;
}
