#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <string>
#include <atomic>
#include <cstddef>

#include "moe_api_common.h"

class MOE_API MemoryManager {
public:
    static MemoryManager& getInstance() {
        static MemoryManager instance;
        return instance;
    }

    // Disable copy/move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    /**
     * @brief Allocates page-aligned, locked physical memory. Useful for KV cache.
     */
    void* allocate_locked(size_t size);

    /**
     * @brief Unlocks and deallocates memory allocated with allocate_locked.
     */
    void free_locked(void* ptr, size_t size);

    /**
     * @brief Maps a file into virtual memory.
     * @param filepath Path to the file.
     * @param out_size Stores the size of the mapped file in bytes.
     * @param out_fd Stores the file descriptor (used for prefetching).
     * @return Pointer to the mapped virtual address, or nullptr on failure.
     */
    void* map_file(const std::string& filepath, size_t& out_size, int& out_fd);

    /**
     * @brief Unmaps a memory-mapped file.
     */
    void unmap_file(void* ptr, size_t size);

    void advise_prefetch(void* ptr, size_t size, int fd);

    /**
     * @brief Pre-allocates or opens the single contiguous weights file and creates a mapping.
     */
    bool init_weights_file(const std::string& filepath, size_t expert_size, size_t num_experts);

    /**
     * @brief Maps a single expert's memory region directly from the global mapping.
     */
    void* map_expert(uint32_t expert_id, size_t& out_size);

    /**
     * @brief Unmaps an expert's mapped view.
     */
    void unmap_expert(void* ptr, size_t size);

    /**
     * @brief Closes the unified weights file and mapping handles.
     */
    void close_weights_file();

    // Getters for telemetry
    size_t getCurrentRamUsage() const { return current_ram_usage_.load(std::memory_order_relaxed); }
    size_t getPeakRamUsage() const { return peak_ram_usage_.load(std::memory_order_relaxed); }
    size_t getCacheOccupancy() const { return cache_occupancy_.load(std::memory_order_relaxed); }

    void addCacheOccupancy(size_t size) {
        cache_occupancy_.fetch_add(size, std::memory_order_relaxed);
    }
    void subCacheOccupancy(size_t size) {
        cache_occupancy_.fetch_sub(size, std::memory_order_relaxed);
    }

    void resetPeakTracker() {
        peak_ram_usage_.store(current_ram_usage_.load(), std::memory_order_relaxed);
    }

private:
    MemoryManager() : current_ram_usage_(0), peak_ram_usage_(0), cache_occupancy_(0)
#ifdef _WIN32
        , hWeightsFile_((void*)-1), hWeightsMapping_(nullptr)
#else
        , weights_fd_(-1)
#endif
        , expert_size_(0), num_experts_(0) {}
    ~MemoryManager() { close_weights_file(); }

    void track_allocation(size_t size);
    void track_deallocation(size_t size);

    std::atomic<size_t> current_ram_usage_;
    std::atomic<size_t> peak_ram_usage_;
    std::atomic<size_t> cache_occupancy_;

#ifdef _WIN32
    void* hWeightsFile_;
    void* hWeightsMapping_;
#else
    int weights_fd_;
#endif
    size_t expert_size_;
    size_t num_experts_;
};

#endif // MEMORY_MANAGER_H
