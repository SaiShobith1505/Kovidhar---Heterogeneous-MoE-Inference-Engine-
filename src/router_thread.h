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
    uint32_t num_candidates;          // 2, 4, or 8 (Phase 2)
    uint32_t top_k_experts[8];        // Holds up to Top-8 experts
    float confidence_scores[8];       // Holds up to 8 scores
    uint64_t timestamp_ns;            // Prediction Generated (Phase 6 timestamp 1)
    uint64_t push_timestamp_ns;       // Queue Push (Phase 6 timestamp 2)
    
    // For speculative prefetch (Phase 7)
    bool is_speculative;
    uint32_t spec_distance;
};

#include "moe_api_common.h"

// Global configuration switches for optimization experiments
inline std::atomic<uint32_t> g_num_candidates{2};
inline std::atomic<bool> g_use_adaptive_lookahead{false};
inline std::atomic<uint32_t> g_lookahead_depth{4};
inline std::atomic<uint32_t> g_speculative_window{0}; // 0 = disabled, otherwise N layers ahead

class MOE_API AdaptiveLookaheadController {
public:
    AdaptiveLookaheadController() : current_depth_(4), transitions_(0), avg_depth_sum_(0), count_(0), max_depth_(4) {}

    void update_metrics(double miss_rate) {
        uint32_t next_depth = 4;
        if (miss_rate < 0.05) {
            next_depth = 4;
        } else if (miss_rate < 0.20) {
            next_depth = 8;
        } else if (miss_rate < 0.40) {
            next_depth = 16;
        } else {
            next_depth = 32;
        }

        if (next_depth != current_depth_) {
            transitions_++;
            current_depth_ = next_depth;
            if (current_depth_ > max_depth_) {
                max_depth_ = current_depth_;
            }
        }

        avg_depth_sum_ += current_depth_;
        count_++;
    }

    uint32_t get_current_depth() const { return current_depth_; }
    uint32_t get_transitions() const { return transitions_; }
    uint32_t get_max_depth() const { return max_depth_; }
    double get_avg_depth() const { return count_ > 0 ? (double)avg_depth_sum_ / count_ : 4.0; }

    void reset() {
        current_depth_ = 4;
        transitions_ = 0;
        avg_depth_sum_ = 0;
        count_ = 0;
        max_depth_ = 4;
    }

private:
    uint32_t current_depth_;
    uint32_t transitions_;
    uint64_t avg_depth_sum_;
    uint64_t count_;
    uint32_t max_depth_;
};

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

    AdaptiveLookaheadController& getAdaptiveController() { return adaptive_controller_; }

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

    AdaptiveLookaheadController adaptive_controller_;

    // Telemetry
    std::atomic<uint64_t> total_prediction_time_us_;
    std::atomic<uint64_t> prediction_count_;
};

#endif // ROUTER_THREAD_H
