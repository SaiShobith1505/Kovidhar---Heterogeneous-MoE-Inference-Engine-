#ifndef EXPERT_CACHE_H
#define EXPERT_CACHE_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
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

struct ExpertStats {
    uint32_t access_count = 0;
    uint64_t last_access_time = 0;
    std::vector<uint64_t> access_history; // For LRU-K
};

enum class CachePolicyType {
    LRU,
    LRUK,
    ARC,
    HYBRID
};

// Global cache optimization settings
inline std::atomic<CachePolicyType> g_cache_policy{CachePolicyType::LRU};
inline std::atomic<double> g_hotset_ratio{0.0}; // 0.0 = disabled, e.g. 0.2 = 20% of cache size

// Phase 6: Prefetch Timeline Event Logging
struct PrefetchEvent {
    uint32_t token_index;
    uint32_t layer_id;
    uint32_t expert_id;
    bool is_speculative;
    uint64_t generated_ns;
    uint64_t push_ns;
    uint64_t pop_ns;
    uint64_t map_start_ns;
    uint64_t map_end_ns;
    uint64_t prefetch_start_ns;
    uint64_t prefetch_end_ns;
    uint64_t ready_ns;
    uint64_t consumed_ns;
};

inline std::vector<PrefetchEvent> g_prefetch_events;
inline std::mutex g_events_mutex;
inline std::atomic<bool> g_enable_pipeline_logging{false};

inline void log_prefetch_event(const PrefetchEvent& event) {
    if (!g_enable_pipeline_logging.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(g_events_mutex);
    if (g_prefetch_events.size() < 50000) {
        g_prefetch_events.push_back(event);
    }
}

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
     */
    ExpertEntry* get_and_pin(uint32_t expert_id, const std::string& filepath);

    /**
     * @brief Unpins an expert, allowing it to be evicted.
     */
    void unpin(uint32_t expert_id);

    /**
     * @brief Asynchronously prefetches an expert. Called by the Prefetch Thread.
     */
    bool prefetch_expert(uint32_t expert_id, const std::string& filepath,
                         uint64_t pop_ns = 0, uint64_t gen_ns = 0, uint64_t push_ns = 0,
                         uint32_t token_idx = 0, uint32_t layer_id = 0,
                         bool is_speculative = false, uint32_t spec_dist = 0);

    // Eviction routine
    void evict_if_over_budget();

    // Helper to log stats update
    void record_access_stats(uint32_t expert_id);

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

    std::unordered_map<uint32_t, ExpertStats> stats_map_; // Track access metrics for Phase 3/4

    std::atomic<uint64_t> hit_count_;
    std::atomic<uint64_t> miss_count_;
    std::atomic<uint64_t> eviction_count_;
};

#endif // EXPERT_CACHE_H
