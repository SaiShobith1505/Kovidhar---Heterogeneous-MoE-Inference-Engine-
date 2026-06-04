#ifndef TELEMETRY_STATE_H
#define TELEMETRY_STATE_H

#include <atomic>
#include <mutex>
#include "moe_api_common.h"

struct MOE_API EngineTelemetry {
    std::atomic<uint32_t> active_layer;
    std::atomic<float> throughput;
    std::atomic<float> hit_rate;
    std::atomic<float> ssd_read_mb_s;
    std::atomic<size_t> ram_usage_bytes;
    
    std::atomic<uint32_t> lookahead_layers[4];
    std::atomic<uint32_t> lookahead_experts[4];
    std::atomic<float> prefetch_progress[4];
    
    std::atomic<bool> is_stalled;
    std::atomic<uint32_t> stall_layer;
    std::atomic<uint32_t> stall_progress;
    
    std::mutex status_mutex;
    char status_message[256];

    EngineTelemetry() {
        active_layer.store(0);
        throughput.store(0.0f);
        hit_rate.store(100.0f);
        ssd_read_mb_s.store(0.0f);
        ram_usage_bytes.store(0);
        for (int i = 0; i < 4; ++i) {
            lookahead_layers[i].store(0);
            lookahead_experts[i].store(0);
            prefetch_progress[i].store(0.0f);
        }
        is_stalled.store(false);
        stall_layer.store(0);
        stall_progress.store(0);
        status_message[0] = '\0';
    }
};

extern MOE_API EngineTelemetry g_telemetry;

#endif // TELEMETRY_STATE_H
