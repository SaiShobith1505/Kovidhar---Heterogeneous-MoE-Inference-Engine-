#ifndef PREFETCH_THREAD_H
#define PREFETCH_THREAD_H

#include "lockfree_queue.h"
#include "router_thread.h"
#include "expert_cache.h"
#include <atomic>
#include <thread>
#include <string>

#include "moe_api_common.h"

class MOE_API PrefetchThread {
public:
    PrefetchThread(
        LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue,
        ExpertCache& cache,
        std::string expert_dir
    );
    ~PrefetchThread();

    void start();
    void stop();
    void join();

    // Telemetry getters
    uint64_t getTotalPrefetchTimeNs() const { return total_prefetch_time_ns_.load(std::memory_order_relaxed); }
    uint64_t getPrefetchCount() const { return prefetch_count_.load(std::memory_order_relaxed); }

    void resetMetrics() {
        total_prefetch_time_ns_.store(0, std::memory_order_relaxed);
        prefetch_count_.store(0, std::memory_order_relaxed);
    }

private:
    void run();
    std::string get_expert_filepath(uint32_t expert_id) const;

    LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue_;
    ExpertCache& cache_;
    std::string expert_dir_;

    std::atomic<bool> running_;
    std::thread thread_;

    // Telemetry
    std::atomic<uint64_t> total_prefetch_time_ns_;
    std::atomic<uint64_t> prefetch_count_;
};

#endif // PREFETCH_THREAD_H
