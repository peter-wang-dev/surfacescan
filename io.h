void save_to_mmap(const char* name,const void* data,size_t size);
void load_from_mmap(const char* name,void** data,size_t* size);
void release_all_mmap();
