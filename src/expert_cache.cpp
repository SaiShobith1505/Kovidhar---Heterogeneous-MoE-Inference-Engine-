#include "expert_cache.h"
#include "memory_manager.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <iostream>

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

ExpertEntry* ExpertCache::get_and_pin(uint32_t expert_id, const std::string& filepath) {
    std::unique_lock<std::mutex> lock(cache_mutex_);
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

bool ExpertCache::prefetch_expert(uint32_t expert_id, const std::string& filepath) {
    std::unique_lock<std::mutex> lock(cache_mutex_);
    auto it = cache_map_.find(expert_id);

    if (it != cache_map_.end()) {
        // Already mapped or mapping is in progress. Move to front of LRU.
        touch_lru(expert_id);
        return true;
    }

    // Insert pending placeholder
    ExpertEntry* entry = new ExpertEntry(expert_id, filepath);
    cache_map_[expert_id] = entry;
    lru_list_.push_front(expert_id);

    lock.unlock();

    size_t mapped_size = 0;
    void* ptr = MemoryManager::getInstance().map_expert(expert_id, mapped_size);
    int fd = -1;
    if (ptr) {
        MemoryManager::getInstance().advise_prefetch(ptr, mapped_size, fd);
    }

    lock.lock();
    entry->mapped_ptr = ptr;
    entry->mapped_size = mapped_size;
    entry->fd = fd;
    entry->is_loaded.store(true, std::memory_order_release);

    if (ptr) {
        MemoryManager::getInstance().addCacheOccupancy(mapped_size);
    }

    cache_cv_.notify_all();

    evict_if_over_budget();

    return ptr != nullptr;
}

void ExpertCache::evict_if_over_budget() {
    // This is run inside the lock (called from get_and_pin / prefetch_expert)
    size_t occupancy = MemoryManager::getInstance().getCacheOccupancy();
    
    while (occupancy > max_bytes_) {
        bool evicted_any = false;
        
        // Scan LRU list from tail (least recently used) to head
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            uint32_t id = *it;
            ExpertEntry* entry = cache_map_[id];
            
            // Check if the expert is loaded and not currently pinned or waited on
            if (entry->is_loaded.load(std::memory_order_relaxed) &&
                entry->pin_count.load(std::memory_order_relaxed) == 0 &&
                entry->wait_count.load(std::memory_order_relaxed) == 0) {
                
                evict_entry_internal(entry);
                evicted_any = true;
                break; // Break loop because iterator is invalidated by removal
            }
        }
        
        if (!evicted_any) {
            // Cannot evict anything because all loaded experts are currently pinned
            break;
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
