#ifndef ROUTER_THREAD_H
#define ROUTER_THREAD_H

#include "lockfree_queue.h"
#include <atomic>
#include <thread>
#include <vector>
#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#endif

enum class RoutingScenario {
    COLD_CACHE,
    WARM_CACHE,
    HIGH_REUSE,
    WORST_CASE_CHURN
};

struct PredictedExpertMessage {
    uint32_t token_index;
    uint32_t layer_id;
    uint32_t top_k_experts[2];
    float confidence_scores[2];
    uint64_t timestamp_ns;
};

#include "moe_api_common.h"

class MOE_API RouterThread {
public:
    RouterThread(
        LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue,
        uint32_t num_layers,
        uint32_t num_experts,
        uint32_t top_k,
        std::atomic<uint32_t>& current_compute_token,
        std::atomic<uint32_t>& current_compute_layer,
        uint32_t total_tokens,
        uint32_t routing_latency_us,
        RoutingScenario scenario
    );
    ~RouterThread();

    void start();
    void stop();
    void join();

    // Telemetry getters
    uint64_t getTotalPredictionTimeUs() const { return total_prediction_time_us_.load(std::memory_order_relaxed); }
    uint64_t getPredictionCount() const { return prediction_count_.load(std::memory_order_relaxed); }

private:
    void run();

    LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue_;
    uint32_t num_layers_;
    uint32_t num_experts_;
    uint32_t top_k_;
    std::atomic<uint32_t>& current_compute_token_;
    std::atomic<uint32_t>& current_compute_layer_;
    uint32_t total_tokens_;
    uint32_t routing_latency_us_;
    RoutingScenario scenario_;

    std::atomic<bool> running_;
    std::thread thread_;

    // Emulated NPU weight matrix
    std::vector<float> gating_weights_;
    std::vector<float> hidden_state_buffer_;

#ifdef HAS_ONNXRUNTIME
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
#endif

    // Telemetry
    std::atomic<uint64_t> total_prediction_time_us_;
    std::atomic<uint64_t> prediction_count_;
};

#endif // ROUTER_THREAD_H
