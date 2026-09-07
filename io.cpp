#ifndef _WIN64 
void save_to_mmap(const char* name,const void* data,size_t size){} 
void load_from_mmap(const char* name,void** data,size_t* size){}
void release_all_mmap(){}
#else 
#include <windows.h>
#include <string>
#include "io.h"
import std;

std::map<void*,HANDLE> g_mmap_registry;
void save_to_mmap(const char* name, const void* data, size_t size)
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
        std::println("save_to_mmap: CreateFileMappingA failed, error code={}", GetLastError());
        return;
    }

    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (pBuf == NULL)
    {
        CloseHandle(hMapFile);
        return;
    }

    CopyMemory(pBuf, data, size);
    
    // Deliberately doing NOT call UnmapViewOfFile(pBuf) or CloseHandle(hMapFile) 
    // so the data stays accessible process-wide.
    std::println("save_to_mmap: Saved {} bytes to shared memory \"{}\"", size, mapping_name);
    g_mmap_registry[pBuf] = hMapFile;
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
