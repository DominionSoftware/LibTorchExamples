#ifndef SAFE_COPY_
#define SAFE_COPY_
#include <cstring>      // For memcpy, strcpy
#include <stdexcept>    // For std::out_of_range
#include <string>       // For std::string
#include <cstddef>      // For size_t

namespace safe 
{

    // Safe memcpy wrapper
    void memcpy(void* dest, size_t dest_size, const void* src, size_t count) 
        {
        if (!dest || !src) 
        {
            throw std::invalid_argument("Null pointer in memcpy");
        }
        if (count > dest_size) 
        {
            throw std::out_of_range("memcpy: count exceeds dest_size");
        }
        std::memcpy(dest, src, count);
    }

 
inline void strcpy(char* dest, size_t dest_size, const char* src) 
{
    if (!dest || !src) 
    {
        throw std::invalid_argument("Null pointer in strcpy");
    }
    size_t src_len = std::strlen(src);
    if (src_len + 1 > dest_size) 
    { // +1 for null terminator
        throw std::out_of_range("strcpy: source string too large");
    }
    std::strcpy(dest, src);
}

// Safe strncpy wrapper with null terminator guarantee
void strncpy(char* dest, size_t dest_size, const char* src, size_t count) 
{
    if (!dest || !src) 
    {
        throw std::invalid_argument("Null pointer in strncpy");
    }
    if (count > dest_size) 
    {
        throw std::out_of_range("strncpy: count exceeds dest_size");
    }
    std::strncpy(dest, src, count);
    if (count == dest_size) {
        dest[dest_size - 1] = '\0'; // Ensure null termination
    }
}

// Safe string to buffer copy (C++ string)
inline void copy_string(char* dest, size_t dest_size, const std::string& str) 
{
    if (str.size() + 1 > dest_size) 
    {
        throw std::out_of_range("copy_string: string too large for buffer");
    }
    std::strcpy(dest, str.c_str());
}

} // namespace safe

#endif