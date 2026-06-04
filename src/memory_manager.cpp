#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "memory_manager.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

void MemoryManager::track_allocation(size_t size) {
    size_t current = current_ram_usage_.fetch_add(size, std::memory_order_relaxed) + size;
    size_t peak = peak_ram_usage_.load(std::memory_order_relaxed);
    while (current > peak) {
        if (peak_ram_usage_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
            break;
        }
        // peak was updated by another thread; reload current peak and retry
    }
}

void MemoryManager::track_deallocation(size_t size) {
    current_ram_usage_.fetch_sub(size, std::memory_order_relaxed);
}

void* MemoryManager::allocate_locked(size_t size) {
    void* ptr = nullptr;
#ifdef _WIN32
    // Align allocations to system page boundary
    ptr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) {
        std::cerr << "[MemoryManager ERROR] VirtualAlloc failed with code: " << GetLastError() << std::endl;
        return nullptr;
    }
    if (!VirtualLock(ptr, size)) {
        std::cerr << "[MemoryManager WARNING] VirtualLock failed. Physical memory locking not guaranteed. Error: " << GetLastError() << std::endl;
    }
#else
    // Align allocations to 4096-byte page boundary
    int ret = posix_memalign(&ptr, 4096, size);
    if (ret != 0) {
        std::cerr << "[MemoryManager ERROR] posix_memalign failed with code: " << ret << std::endl;
        return nullptr;
    }

    // Attempt to lock memory to prevent paging/swapping
    if (mlock(ptr, size) != 0) {
        std::cerr << "[MemoryManager WARNING] mlock failed. Physical memory locking not guaranteed." << std::endl;
    }
#endif

    track_allocation(size);
    return ptr;
}

void MemoryManager::free_locked(void* ptr, size_t size) {
    if (ptr) {
#ifdef _WIN32
        VirtualUnlock(ptr, size);
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        munlock(ptr, size);
        free(ptr);
#endif
        track_deallocation(size);
    }
}

void* MemoryManager::map_file(const std::string& filepath, size_t& out_size, int& out_fd) {
    out_fd = -1;
#ifdef _WIN32
    // Open file using unbuffered and overlapped I/O to bypass OS page cache
    HANDLE hFile = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[MemoryManager ERROR] Failed to open expert file (Win32): " << filepath << " Error: " << GetLastError() << std::endl;
        return nullptr;
    }

    LARGE_INTEGER size_struct;
    if (!GetFileSizeEx(hFile, &size_struct)) {
        std::cerr << "[MemoryManager ERROR] GetFileSizeEx failed (Win32): " << filepath << " Error: " << GetLastError() << std::endl;
        CloseHandle(hFile);
        return nullptr;
    }
    out_size = static_cast<size_t>(size_struct.QuadPart);

    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
        std::cerr << "[MemoryManager ERROR] CreateFileMappingA failed (Win32): " << filepath << " Error: " << GetLastError() << std::endl;
        CloseHandle(hFile);
        return nullptr;
    }

    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
        std::cerr << "[MemoryManager ERROR] MapViewOfFile failed (Win32): " << filepath << " Error: " << GetLastError() << std::endl;
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return nullptr;
    }

    // View holds reference to mapping and file, so we can close handles immediately
    CloseHandle(hMapping);
    CloseHandle(hFile);

    track_allocation(out_size);
    return ptr;
#else
    // Try O_DIRECT | O_NONBLOCK first, fallback to standard if not supported
    int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT | O_NONBLOCK);
    if (fd < 0) {
        fd = open(filepath.c_str(), O_RDONLY);
    }
    if (fd < 0) {
        std::cerr << "[MemoryManager ERROR] Failed to open expert file: " << filepath << std::endl;
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        std::cerr << "[MemoryManager ERROR] fstat failed for: " << filepath << std::endl;
        close(fd);
        return nullptr;
    }

    out_size = sb.st_size;
    
    // Memory map the file (Read-only, shared across page cache)
    void* ptr = mmap(nullptr, out_size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "[MemoryManager ERROR] mmap failed for: " << filepath << std::endl;
        close(fd);
        return nullptr;
    }

    out_fd = fd;
    track_allocation(out_size);
    return ptr;
#endif
}

void MemoryManager::unmap_file(void* ptr, size_t size) {
    if (ptr) {
#ifdef _WIN32
        if (!UnmapViewOfFile(ptr)) {
            std::cerr << "[MemoryManager ERROR] UnmapViewOfFile failed: " << GetLastError() << std::endl;
        }
#else
        if (munmap(ptr, size) != 0) {
            std::cerr << "[MemoryManager ERROR] munmap failed for pointer: " << ptr << std::endl;
        }
#endif
        track_deallocation(size);
    }
}

void MemoryManager::advise_prefetch(void* ptr, size_t size, int fd) {
    if (ptr) {
#ifdef _WIN32
        // Windows DMA Prefetch using PrefetchVirtualMemory
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = ptr;
        entry.NumberOfBytes = size;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
        // Advise kernel that these pages will be accessed shortly
        madvise(ptr, size, MADV_WILLNEED);
        if (fd >= 0) {
            // Trigger asynchronous readahead via file descriptor
            readahead(fd, 0, size);
        }
#endif
    }
}

bool MemoryManager::init_weights_file(const std::string& filepath, size_t expert_size, size_t num_experts) {
    expert_size_ = expert_size;
    num_experts_ = num_experts;

#ifdef _WIN32
    // Open unified weights file once using non-buffered and sequential scan options
    HANDLE hFile = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[MemoryManager ERROR] Failed to open unified weights file: " << filepath << " Error: " << GetLastError() << std::endl;
        return false;
    }

    hWeightsFile_ = hFile;

    LARGE_INTEGER size_struct;
    if (!GetFileSizeEx(hFile, &size_struct)) {
        std::cerr << "[MemoryManager ERROR] GetFileSizeEx failed for unified weights: " << filepath << " Error: " << GetLastError() << std::endl;
        CloseHandle(hFile);
        hWeightsFile_ = (void*)-1;
        return false;
    }

    size_t expected_size = expert_size_ * num_experts_;
    if (static_cast<size_t>(size_struct.QuadPart) != expected_size) {
        std::cerr << "[MemoryManager WARNING] Unified weights file size (" << size_struct.QuadPart 
                  << ") is not equal to expected size (" << expected_size << ")" << std::endl;
    }

    hWeightsMapping_ = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hWeightsMapping_) {
        std::cerr << "[MemoryManager ERROR] CreateFileMappingA failed for unified weights: " << filepath << " Error: " << GetLastError() << std::endl;
        CloseHandle(hFile);
        hWeightsFile_ = (void*)-1;
        return false;
    }

    return true;
#else
    // Linux/POSIX implementation
    weights_fd_ = open(filepath.c_str(), O_RDONLY | O_DIRECT | O_NONBLOCK);
    if (weights_fd_ < 0) {
        weights_fd_ = open(filepath.c_str(), O_RDONLY);
    }
    if (weights_fd_ < 0) {
        std::cerr << "[MemoryManager ERROR] Failed to open unified weights file: " << filepath << std::endl;
        return false;
    }
    return true;
#endif
}

void* MemoryManager::map_expert(uint32_t expert_id, size_t& out_size) {
    out_size = expert_size_;
    size_t offset = static_cast<size_t>(expert_id) * expert_size_;

#ifdef _WIN32
    if (!hWeightsMapping_) {
        std::cerr << "[MemoryManager ERROR] map_expert called but weights mapping is null" << std::endl;
        return nullptr;
    }

    DWORD offset_high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
    DWORD offset_low = static_cast<DWORD>(offset & 0xFFFFFFFF);

    void* ptr = MapViewOfFile(
        hWeightsMapping_,
        FILE_MAP_READ,
        offset_high,
        offset_low,
        expert_size_
    );
    if (!ptr) {
        std::cerr << "[MemoryManager ERROR] MapViewOfFile failed for expert " << expert_id 
                  << " at offset " << offset << " Error: " << GetLastError() << std::endl;
        return nullptr;
    }

    track_allocation(expert_size_);
    return ptr;
#else
    if (weights_fd_ < 0) {
        std::cerr << "[MemoryManager ERROR] map_expert called but weights_fd_ is invalid" << std::endl;
        return nullptr;
    }

    void* ptr = mmap(nullptr, expert_size_, PROT_READ, MAP_SHARED, weights_fd_, offset);
    if (ptr == MAP_FAILED) {
        std::cerr << "[MemoryManager ERROR] mmap failed for expert " << expert_id 
                  << " at offset " << offset << std::endl;
        return nullptr;
    }

    track_allocation(expert_size_);
    return ptr;
#endif
}

void MemoryManager::unmap_expert(void* ptr, size_t size) {
    if (ptr) {
#ifdef _WIN32
        if (!UnmapViewOfFile(ptr)) {
            std::cerr << "[MemoryManager ERROR] UnmapViewOfFile failed: " << GetLastError() << std::endl;
        }
#else
        if (munmap(ptr, size) != 0) {
            std::cerr << "[MemoryManager ERROR] munmap failed for pointer: " << ptr << std::endl;
        }
#endif
        track_deallocation(size);
    }
}

void MemoryManager::close_weights_file() {
#ifdef _WIN32
    if (hWeightsMapping_) {
        CloseHandle(hWeightsMapping_);
        hWeightsMapping_ = nullptr;
    }
    if (hWeightsFile_ != (void*)-1) {
        CloseHandle(hWeightsFile_);
        hWeightsFile_ = (void*)-1;
    }
#else
    if (weights_fd_ >= 0) {
        close(weights_fd_);
        weights_fd_ = -1;
    }
#endif
    expert_size_ = 0;
    num_experts_ = 0;
}
