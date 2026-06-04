#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "expert_cache.h"
#include "router_thread.h"
#include "prefetch_thread.h"
#include <string>
#include <vector>

struct BenchmarkConfig {
    uint32_t num_experts = 256;
    size_t expert_size_bytes = 100ULL * 1024 * 1024; // 100 MB
    uint32_t num_layers = 12; // 12 layers of MoE
    uint32_t total_tokens = 20; // Generate 20 tokens for benchmarking
    size_t cache_size_bytes = 8ULL * 1024 * 1024 * 1024; // 8 GB RAM Cache
    uint32_t routing_latency_us = 500; // Simulated NPU gating latency
    uint32_t compute_latency_ms = 5; // Simulated execution latency per expert
    std::string expert_dir = "./experts";
};

struct ScenarioResult {
    std::string scenario_name;
    double total_execution_time_s = 0.0;
    double token_throughput = 0.0;
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;
    uint64_t total_evictions = 0;
    double cache_hit_rate = 0.0;
    size_t peak_ram_usage_bytes = 0;
    size_t final_cache_occupancy_bytes = 0;
    uint64_t avg_prediction_time_us = 0;
    uint64_t avg_prefetch_time_us = 0;
    double total_ssd_read_mb = 0.0;
    double pcie_bandwidth_mb_s = 0.0;
};

#include "moe_api_common.h"

class MOE_API Benchmark {
public:
    Benchmark(const BenchmarkConfig& config);
    ~Benchmark() = default;

    /**
     * @brief Automatically generates the simulated expert weights on disk as sparse files.
     */
    void generate_expert_assets();

    /**
     * @brief Deletes the generated expert files to clean up workspace.
     */
    void cleanup_expert_assets();

    /**
     * @brief Runs a specific routing scenario.
     */
    ScenarioResult run_scenario(RoutingScenario scenario, const std::string& name);

    /**
     * @brief Compares and prints results for all benchmark scenarios.
     */
    void run_all_scenarios(std::vector<ScenarioResult>& results);

    /**
     * @brief Generates profiling_report.txt based on results.
     */
    void generate_report(const std::vector<ScenarioResult>& results, const std::string& filepath);

private:
    BenchmarkConfig config_;
};

#ifdef __cplusplus
extern "C" {
#endif

MOE_API void* create_benchmark(
    const char* expert_dir,
    uint32_t num_experts,
    uint64_t expert_size,
    uint32_t num_layers,
    uint32_t total_tokens,
    uint64_t cache_size
);

MOE_API void destroy_benchmark(void* benchmark_ptr);

MOE_API void generate_assets(void* benchmark_ptr);

MOE_API void cleanup_assets(void* benchmark_ptr);

MOE_API void run_scenario_c(
    void* benchmark_ptr,
    int scenario_type,
    const char* name,
    double* out_exec_time,
    double* out_throughput,
    uint64_t* out_hits,
    uint64_t* out_misses,
    uint64_t* out_evictions,
    double* out_hit_rate
);

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_H
