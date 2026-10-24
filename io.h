#include <map>
#include <string>
#include <vector>
#ifndef _WIN64 
#include <cstddef>
bool save_to_mmap(const char* name,const void* data,size_t size){ return false; } 
void load_from_mmap(const char* name,void** data,size_t* size){}
void release_all_mmap() {}
#else 
bool save_to_mmap(const char* name,const void* data,size_t size);
void load_from_mmap(const char* name,void** data,size_t* size);
void release_all_mmap();
#endif
std::map<std::string, std::vector<float>> readRingParaCSV(const std::string& filename);
