#include "prefetch_thread.h"
#include <chrono>
#include <cstdio>
#include <iostream>

PrefetchThread::PrefetchThread(
    LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue,
    ExpertCache& cache,
    std::string expert_dir
) : queue_(queue),
    cache_(cache),
    expert_dir_(std::move(expert_dir)),
    running_(false),
    total_prefetch_time_ns_(0),
    prefetch_count_(0) {}

PrefetchThread::~PrefetchThread() {
    stop();
}

void PrefetchThread::start() {
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&PrefetchThread::run, this);
}

void PrefetchThread::stop() {
    if (running_.load(std::memory_order_relaxed)) {
        running_.store(false, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void PrefetchThread::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

std::string PrefetchThread::get_expert_filepath(uint32_t expert_id) const {
    char buf[64];
    // Format expert filenames as expert_000.bin, expert_001.bin, etc.
    std::snprintf(buf, sizeof(buf), "/expert_%03u.bin", expert_id);
    return expert_dir_ + buf;
}

void PrefetchThread::run() {
    PredictedExpertMessage msg;

    while (running_.load(std::memory_order_relaxed)) {
        if (queue_.pop(msg)) {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Prefetch both experts predicted for the future layer
            for (int k = 0; k < 2; ++k) {
                uint32_t expert_id = msg.top_k_experts[k];
                
                // Prefetch mapping (non-blocking in cache class if loaded, maps if missing)
                cache_.prefetch_expert(expert_id, "");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            
            total_prefetch_time_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
            prefetch_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // SPSC Queue is empty. Yield or sleep briefly to avoid spinning CPU core.
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}
