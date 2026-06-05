#include "expert_cache.h"
#include "memory_manager.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <iostream>
#include <chrono>
#include <algorithm>

ExpertCache::ExpertCache(size_t max_bytes)
    : max_bytes_(max_bytes), hit_count_(0), miss_count_(0), eviction_count_(0) {}

ExpertCache::~ExpertCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& pair : cache_map_) {
        ExpertEntry* entry = pair.second;
        if (entry->mapped_ptr) {
            MemoryManager::getInstance().unmap_expert(entry->mapped_ptr, entry->mapped_size);
            MemoryManager::getInstance().subCacheOccupancy(entry->mapped_size);
        }
        delete entry;
    }
    cache_map_.clear();
    lru_list_.clear();
}

void ExpertCache::touch_lru(uint32_t expert_id) {
    lru_list_.remove(expert_id);
    lru_list_.push_front(expert_id);
}

void ExpertCache::record_access_stats(uint32_t expert_id) {
    auto& stats = stats_map_[expert_id];
    stats.access_count++;
    uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    stats.last_access_time = now;
    stats.access_history.push_back(now);
    if (stats.access_history.size() > 5) {
        stats.access_history.erase(stats.access_history.begin());
    }
}

ExpertEntry* ExpertCache::get_and_pin(uint32_t expert_id, const std::string& filepath) {
    std::unique_lock<std::mutex> lock(cache_mutex_);
    
    // Record access metrics
    record_access_stats(expert_id);

    auto it = cache_map_.find(expert_id);

    if (it != cache_map_.end()) {
        ExpertEntry* entry = it->second;
        if (entry->is_loaded.load(std::memory_order_relaxed)) {
            // Cache Hit
            entry->pin_count.fetch_add(1, std::memory_order_relaxed);
            touch_lru(expert_id);
            hit_count_.fetch_add(1, std::memory_order_relaxed);
            return entry;
        } else {
            // Cache Miss (mapping in progress by prefetch thread)
            miss_count_.fetch_add(1, std::memory_order_relaxed);
            entry->wait_count.fetch_add(1, std::memory_order_relaxed);
            cache_cv_.wait(lock, [entry]() {
                return entry->is_loaded.load(std::memory_order_relaxed);
            });
            entry->wait_count.fetch_sub(1, std::memory_order_relaxed);
            entry->pin_count.fetch_add(1, std::memory_order_relaxed);
            touch_lru(expert_id);
            return entry;
        }
    }

    // Cache Miss (not present in cache at all)
    miss_count_.fetch_add(1, std::memory_order_relaxed);
    
    // Insert a pending placeholder entry to prevent duplicate loads
    ExpertEntry* entry = new ExpertEntry(expert_id, filepath);
    cache_map_[expert_id] = entry;
    lru_list_.push_front(expert_id);

    // Release lock to perform I/O
    lock.unlock();

    size_t mapped_size = 0;
    void* ptr = MemoryManager::getInstance().map_expert(expert_id, mapped_size);
    int fd = -1;
    
    if (ptr) {
        MemoryManager::getInstance().advise_prefetch(ptr, mapped_size, fd);
    }

    // Re-acquire lock to finalize the entry
    lock.lock();
    entry->mapped_ptr = ptr;
    entry->mapped_size = mapped_size;
    entry->fd = fd;
    entry->is_loaded.store(true, std::memory_order_release);
    entry->pin_count.fetch_add(1, std::memory_order_relaxed);
    
    if (ptr) {
        MemoryManager::getInstance().addCacheOccupancy(mapped_size);
    }

    cache_cv_.notify_all();

    // Check budget and evict if necessary
    evict_if_over_budget();

    return entry;
}

void ExpertCache::unpin(uint32_t expert_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_map_.find(expert_id);
    if (it != cache_map_.end()) {
        it->second->pin_count.fetch_sub(1, std::memory_order_release);
    }
}

bool ExpertCache::prefetch_expert(uint32_t expert_id, const std::string& filepath,
                                  uint64_t pop_ns, uint64_t gen_ns, uint64_t push_ns,
                                  uint32_t token_idx, uint32_t layer_id,
                                  bool is_speculative, uint32_t spec_dist) {
    std::unique_lock<std::mutex> lock(cache_mutex_);
    
    record_access_stats(expert_id);

    auto it = cache_map_.find(expert_id);

    if (it != cache_map_.end()) {
        touch_lru(expert_id);
        
        // Log cache hit event in prefetch timeline (Phase 6)
        if (g_enable_pipeline_logging.load(std::memory_order_relaxed)) {
            PrefetchEvent ev;
            ev.token_index = token_idx;
            ev.layer_id = layer_id;
            ev.expert_id = expert_id;
            ev.is_speculative = is_speculative;
            ev.generated_ns = gen_ns;
            ev.push_ns = push_ns;
            ev.pop_ns = pop_ns;
            ev.map_start_ns = pop_ns;
            ev.map_end_ns = pop_ns;
            ev.prefetch_start_ns = pop_ns;
            ev.prefetch_end_ns = pop_ns;
            ev.ready_ns = pop_ns;
            ev.consumed_ns = 0;
            log_prefetch_event(ev);
        }
        return true;
    }

    // Insert pending placeholder
    ExpertEntry* entry = new ExpertEntry(expert_id, filepath);
    cache_map_[expert_id] = entry;
    lru_list_.push_front(expert_id);

    lock.unlock();

    // Phase 6: Mapping Start (Timestamp 4)
    uint64_t map_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    size_t mapped_size = 0;
    void* ptr = MemoryManager::getInstance().map_expert(expert_id, mapped_size);
    int fd = -1;

    // Phase 6: Mapping Complete (Timestamp 5)
    uint64_t map_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    // Phase 6: Prefetch Start (Timestamp 6)
    uint64_t pref_start = map_end;
    if (ptr) {
        MemoryManager::getInstance().advise_prefetch(ptr, mapped_size, fd);
    }

    // Phase 6: Prefetch Complete (Timestamp 7)
    uint64_t pref_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    lock.lock();
    entry->mapped_ptr = ptr;
    entry->mapped_size = mapped_size;
    entry->fd = fd;
    entry->is_loaded.store(true, std::memory_order_release);

    if (ptr) {
        MemoryManager::getInstance().addCacheOccupancy(mapped_size);
    }

    cache_cv_.notify_all();

    // Phase 6: Expert Ready (Timestamp 8)
    uint64_t ready_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    if (g_enable_pipeline_logging.load(std::memory_order_relaxed)) {
        PrefetchEvent ev;
        ev.token_index = token_idx;
        ev.layer_id = layer_id;
        ev.expert_id = expert_id;
        ev.is_speculative = is_speculative;
        ev.generated_ns = gen_ns;
        ev.push_ns = push_ns;
        ev.pop_ns = pop_ns;
        ev.map_start_ns = map_start;
        ev.map_end_ns = map_end;
        ev.prefetch_start_ns = pref_start;
        ev.prefetch_end_ns = pref_end;
        ev.ready_ns = ready_ns;
        ev.consumed_ns = 0;
        log_prefetch_event(ev);
    }

    evict_if_over_budget();

    return ptr != nullptr;
}

void ExpertCache::evict_if_over_budget() {
    size_t occupancy = MemoryManager::getInstance().getCacheOccupancy();
    
    while (occupancy > max_bytes_) {
        uint32_t best_evict_id = -1;
        bool found = false;

        CachePolicyType policy = g_cache_policy.load(std::memory_order_relaxed);
        double hot_ratio = g_hotset_ratio.load(std::memory_order_relaxed);
        size_t hot_limit = static_cast<size_t>(max_bytes_ * hot_ratio);
        
        // Phase 4: Identify Hotset Region experts to protect from eviction
        std::unordered_set<uint32_t> hotset;
        if (hot_ratio > 0.0) {
            std::vector<std::pair<uint32_t, uint32_t>> freq_list;
            for (const auto& pair : cache_map_) {
                uint32_t id = pair.first;
                freq_list.push_back({id, stats_map_[id].access_count});
            }
            std::sort(freq_list.begin(), freq_list.end(), [](const auto& a, const auto& b) {
                return a.second > b.second; // Descending
            });
            size_t accumulated_size = 0;
            for (const auto& p : freq_list) {
                ExpertEntry* entry = cache_map_[p.first];
                if (accumulated_size + entry->mapped_size <= hot_limit) {
                    hotset.insert(p.first);
                    accumulated_size += entry->mapped_size;
                } else {
                    break;
                }
            }
        }

        // Apply selected Cache Policy (Phase 3)
        if (policy == CachePolicyType::LRU) {
            for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
                uint32_t id = *it;
                ExpertEntry* entry = cache_map_[id];
                if (entry->is_loaded.load(std::memory_order_relaxed) &&
                    entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                    entry->wait_count.load(std::memory_order_relaxed) == 0 &&
                    hotset.find(id) == hotset.end()) {
                    best_evict_id = id;
                    found = true;
                    break;
                }
            }
        } 
        else if (policy == CachePolicyType::LRUK) {
            uint64_t max_back_dist = 0;
            uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();

            for (const auto& pair : cache_map_) {
                uint32_t id = pair.first;
                ExpertEntry* entry = pair.second;

                if (entry->is_loaded.load(std::memory_order_relaxed) &&
                    entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                    entry->wait_count.load(std::memory_order_relaxed) == 0 &&
                    hotset.find(id) == hotset.end()) {

                    const auto& history = stats_map_[id].access_history;
                    uint64_t back_dist = 0;
                    if (history.size() < 2) {
                        back_dist = (uint64_t)-1; // Infinity
                    } else {
                        back_dist = now - history[0];
                    }

                    if (!found || back_dist > max_back_dist) {
                        max_back_dist = back_dist;
                        best_evict_id = id;
                        found = true;
                    }
                }
            }
        } 
        else if (policy == CachePolicyType::ARC) {
            static double arc_p = 0.5 * max_bytes_; // Target size for T1
            size_t t1_size = 0;
            for (const auto& pair : cache_map_) {
                if (stats_map_[pair.first].access_count < 2) {
                    t1_size += pair.second->mapped_size;
                }
            }

            bool evict_from_t1 = (t1_size > arc_p);
            
            for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
                uint32_t id = *it;
                ExpertEntry* entry = cache_map_[id];
                if (entry->is_loaded.load(std::memory_order_relaxed) &&
                    entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                    entry->wait_count.load(std::memory_order_relaxed) == 0 &&
                    hotset.find(id) == hotset.end()) {
                    
                    bool is_t1 = (stats_map_[id].access_count < 2);
                    if (evict_from_t1 == is_t1) {
                        best_evict_id = id;
                        found = true;
                        break;
                    }
                }
            }

            // Fallback
            if (!found) {
                for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
                    uint32_t id = *it;
                    ExpertEntry* entry = cache_map_[id];
                    if (entry->is_loaded.load(std::memory_order_relaxed) &&
                        entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                        entry->wait_count.load(std::memory_order_relaxed) == 0 &&
                        hotset.find(id) == hotset.end()) {
                        best_evict_id = id;
                        found = true;
                        break;
                    }
                }
            }
        } 
        else if (policy == CachePolicyType::HYBRID) {
            double lowest_score = 1e20;
            uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();

            for (const auto& pair : cache_map_) {
                uint32_t id = pair.first;
                ExpertEntry* entry = pair.second;

                if (entry->is_loaded.load(std::memory_order_relaxed) &&
                    entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                    entry->wait_count.load(std::memory_order_relaxed) == 0 &&
                    hotset.find(id) == hotset.end()) {

                    double time_diff_sec = (double)(now - stats_map_[id].last_access_time) / 1e9;
                    double score = (double)stats_map_[id].access_count / (1.0 + 0.1 * time_diff_sec);

                    if (!found || score < lowest_score) {
                        lowest_score = score;
                        best_evict_id = id;
                        found = true;
                    }
                }
            }
        }

        if (found) {
            evict_entry_internal(cache_map_[best_evict_id]);
        } else {
            break; // All resident experts are pinned
        }
        
        occupancy = MemoryManager::getInstance().getCacheOccupancy();
    }
}

void ExpertCache::evict_entry_internal(ExpertEntry* entry) {
    uint32_t id = entry->expert_id;
    lru_list_.remove(id);
    cache_map_.erase(id);

    if (entry->mapped_ptr) {
        MemoryManager::getInstance().unmap_expert(entry->mapped_ptr, entry->mapped_size);
        MemoryManager::getInstance().subCacheOccupancy(entry->mapped_size);
    }

    delete entry;
    eviction_count_.fetch_add(1, std::memory_order_relaxed);
}
