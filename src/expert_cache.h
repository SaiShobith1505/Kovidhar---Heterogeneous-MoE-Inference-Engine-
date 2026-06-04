#ifndef EXPERT_CACHE_H
#define EXPERT_CACHE_H

#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <atomic>
#include <condition_variable>

struct ExpertEntry {
    uint32_t expert_id;
    std::string filepath;
    void* mapped_ptr = nullptr;
    size_t mapped_size = 0;
    int fd = -1;
    std::atomic<int> pin_count{0};
    std::atomic<int> wait_count{0}; // Tracks threads waiting on cache_cv_
    std::atomic<bool> is_loaded{false};

    ExpertEntry(uint32_t id, std::string path)
        : expert_id(id), filepath(std::move(path)) {}
};

#include "moe_api_common.h"

class MOE_API ExpertCache {
public:
    /**
     * @param max_bytes Cache capacity in bytes (e.g., 8 GB = 8589934592)
     */
    ExpertCache(size_t max_bytes);
    ~ExpertCache();

    // Disable copy/move
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    /**
     * @brief Gets an expert's weights and pins it.
     *        If not loaded, it will load it synchronously (Cache Miss).
     *        If loading is in progress, it will block until completed.
     */
    ExpertEntry* get_and_pin(uint32_t expert_id, const std::string& filepath);

    /**
     * @brief Unpins an expert, allowing it to be evicted.
     */
    void unpin(uint32_t expert_id);

    /**
     * @brief Asynchronously prefetches an expert. Called by the Prefetch Thread.
     * @return true if mapping succeeded or was already mapped.
     */
    bool prefetch_expert(uint32_t expert_id, const std::string& filepath);

    // Eviction routine
    void evict_if_over_budget();

    // Metrics getters
    uint64_t getHitCount() const { return hit_count_.load(std::memory_order_relaxed); }
    uint64_t getMissCount() const { return miss_count_.load(std::memory_order_relaxed); }
    uint64_t getEvictionCount() const { return eviction_count_.load(std::memory_order_relaxed); }
    
    void resetMetrics() {
        hit_count_.store(0, std::memory_order_relaxed);
        miss_count_.store(0, std::memory_order_relaxed);
        eviction_count_.store(0, std::memory_order_relaxed);
    }

private:
    void touch_lru(uint32_t expert_id);
    void evict_entry_internal(ExpertEntry* entry);

    size_t max_bytes_;
    
    std::mutex cache_mutex_;
    std::condition_variable cache_cv_; // To wait for async loading completion
    
    std::unordered_map<uint32_t, ExpertEntry*> cache_map_;
    std::list<uint32_t> lru_list_; // Tracks expert_ids, head is MRU, tail is LRU

    std::atomic<uint64_t> hit_count_;
    std::atomic<uint64_t> miss_count_;
    std::atomic<uint64_t> eviction_count_;
};

#endif // EXPERT_CACHE_H
