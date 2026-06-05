#include "benchmark.h"
#include "memory_manager.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

namespace fs = std::filesystem;

// Helper to resolve the same experts as the RouterThread for validation
static void get_routed_experts(
    uint32_t token_index,
    uint32_t layer_id,
    uint32_t num_layers,
    uint32_t num_experts,
    RoutingScenario scenario,
    uint32_t out_experts[2]
) {
    int64_t router_flat = static_cast<int64_t>(token_index) * num_layers + layer_id;

    if (scenario == RoutingScenario::HIGH_REUSE) {
        out_experts[0] = (layer_id * 2) % 4;
        out_experts[1] = (layer_id * 2 + 1) % 4;
    } else if (scenario == RoutingScenario::WORST_CASE_CHURN) {
        out_experts[0] = (router_flat * 2) % num_experts;
        out_experts[1] = (router_flat * 2 + 1) % num_experts;
    } else {
        // Skewed Zipf-like distribution
        std::mt19937 gen(1337);
        // Advance generator state deterministically to match the router thread
        for (int64_t i = 0; i < router_flat; ++i) {
            (void)gen(); (void)gen(); (void)gen(); // Advance generator states
        }

        std::uniform_int_distribution<uint32_t> hot_dist(0, 15);
        std::uniform_int_distribution<uint32_t> cold_dist(16, num_experts - 1);
        std::uniform_int_distribution<uint32_t> prob(0, 99);

        out_experts[0] = (prob(gen) < 80) ? hot_dist(gen) : cold_dist(gen);
        out_experts[1] = (prob(gen) < 80) ? hot_dist(gen) : cold_dist(gen);

        if (out_experts[0] == out_experts[1]) {
            out_experts[1] = (out_experts[1] + 1) % num_experts;
        }
    }
}

Benchmark::Benchmark(const BenchmarkConfig& config) : config_(config) {}

void Benchmark::generate_expert_assets() {
    fs::create_directories(config_.expert_dir);
    std::string path = config_.expert_dir + "/weights.bin";
    size_t total_size = config_.num_experts * config_.expert_size_bytes;

    std::cout << "[Benchmark] Generating single unified weights file: " << path 
              << " (" << (total_size / (1024ULL * 1024 * 1024)) << " GB total)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Skip if already generated with the correct size
    if (fs::exists(path) && fs::file_size(path) == total_size) {
        std::cout << "[Benchmark] Unified weights file already exists with correct size." << std::endl;
        return;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Benchmark ERROR] Failed to create unified weights file: " << path << std::endl;
        return;
    }

    // Prepare a 4 MB buffer of dummy float data to speed up writing
    std::vector<char> write_buf(4 * 1024 * 1024, 0);
    float* fbuf = reinterpret_cast<float*>(write_buf.data());
    for (size_t i = 0; i < write_buf.size() / sizeof(float); ++i) {
        fbuf[i] = static_cast<float>(i) * 0.001f; // dummy floats
    }

    size_t bytes_written = 0;
    while (bytes_written < total_size) {
        size_t to_write = std::min(write_buf.size(), total_size - bytes_written);
        ofs.write(write_buf.data(), to_write);
        bytes_written += to_write;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[Benchmark] Done. Time taken: " << elapsed << " ms." << std::endl;
}

void Benchmark::cleanup_expert_assets() {
    std::cout << "[Benchmark] Cleaning up expert files..." << std::endl;
    std::string path = config_.expert_dir + "/weights.bin";
    std::error_code ec;
    fs::remove(path, ec);
    if (ec) {
        std::cerr << "[Benchmark ERROR] Cleanup of weights.bin failed: " << ec.message() << std::endl;
    } else {
        std::cout << "[Benchmark] Cleanup of weights.bin successful." << std::endl;
    }
}

ScenarioResult Benchmark::run_scenario(RoutingScenario scenario, const std::string& name) {
    std::cout << "\n=========================================" << std::endl;
    std::cout << " RUNNING BENCHMARK SCENARIO: " << name << std::endl;
    std::cout << "=========================================" << std::endl;

    // Reset MemoryManager metrics
    MemoryManager::getInstance().resetPeakTracker();

    std::string path = config_.expert_dir + "/weights.bin";
    if (!MemoryManager::getInstance().init_weights_file(path, config_.expert_size_bytes, config_.num_experts)) {
        std::cerr << "[Benchmark ERROR] init_weights_file failed for scenario: " << name << std::endl;
    }

    // Setup queue and cache
    LockFreeSPSCQueue<PredictedExpertMessage, 1024> queue;
    ExpertCache cache(config_.cache_size_bytes);

    std::atomic<uint32_t> current_compute_token{0};
    std::atomic<uint32_t> current_compute_layer{0};

    double total_io_stall_time_s = 0.0;
    uint64_t total_io_bytes = 0;

    // Instantiate Threads
    RouterThread router(
        queue, 
        config_.num_layers, 
        config_.num_experts, 
        2, 
        current_compute_token, 
        current_compute_layer, 
        config_.total_tokens, 
        config_.routing_latency_us, 
        scenario
    );

    PrefetchThread prefetch(queue, cache, config_.expert_dir);

    // Warm-up Cache logic if Scenario B (Warm Cache) is chosen
    if (scenario == RoutingScenario::WARM_CACHE) {
        std::cout << "[Benchmark] Pre-warming cache with hot experts (0 to 15)..." << std::endl;
        for (uint32_t i = 0; i < 16; ++i) {
            cache.prefetch_expert(i, "");
        }
    }

    // Start background worker pipelines
    prefetch.start();
    router.start();

    // Start compute timeline
    auto start_time = std::chrono::high_resolution_clock::now();

    float simulated_accum = 0.0f; // Accumulator to force RAM read page-in

    for (uint32_t t = 0; t < config_.total_tokens; ++t) {
        for (uint32_t l = 0; l < config_.num_layers; ++l) {
            // Update compute progress for lookahead thread
            current_compute_token.store(t, std::memory_order_relaxed);
            current_compute_layer.store(l, std::memory_order_relaxed);

#ifdef _WIN32
            // Wake up router thread waiting on WaitOnAddress (current_compute_layer)
            if (g_yield_strategy.load(std::memory_order_relaxed) == 4) {
                WakeByAddressSingle(&current_compute_layer);
            }
#endif

            // 1. Determine which experts we route to
            uint32_t targets[2];
            get_routed_experts(t, l, config_.num_layers, config_.num_experts, scenario, targets);

            // 2. Pin, load, and page fault with high-resolution timing to measure disk stall
            auto io_start = std::chrono::high_resolution_clock::now();
            uint64_t misses_before = cache.getMissCount();

            ExpertEntry* e0 = cache.get_and_pin(targets[0], "");
            ExpertEntry* e1 = cache.get_and_pin(targets[1], "");

            // Phase 6: Expert Consumed (Timestamp 9)
            uint64_t consumed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();

            if (g_enable_pipeline_logging.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> ev_lock(g_events_mutex);
                for (auto& ev : g_prefetch_events) {
                    if (ev.token_index == t && ev.layer_id == l && ev.consumed_ns == 0) {
                        if (ev.expert_id == targets[0] || ev.expert_id == targets[1]) {
                            ev.consumed_ns = consumed_ns;
                        }
                    }
                }
            }

            // 4. Force RAM read/page faulting to accurately measure SSD latency
            if (e0 && e0->mapped_ptr && e0->mapped_size > 0) {
                volatile const float* data = static_cast<volatile const float*>(e0->mapped_ptr);
                // Touch every 64th page (256 KB stride) to keep VM context switches light
                size_t num_floats = e0->mapped_size / sizeof(float);
                for (size_t i = 0; i < num_floats; i += 1024 * 64) {
                    simulated_accum += data[i];
                }
            }

            if (e1 && e1->mapped_ptr && e1->mapped_size > 0) {
                volatile const float* data = static_cast<volatile const float*>(e1->mapped_ptr);
                size_t num_floats = e1->mapped_size / sizeof(float);
                for (size_t i = 0; i < num_floats; i += 1024 * 64) {
                    simulated_accum += data[i];
                }
            }

            auto io_end = std::chrono::high_resolution_clock::now();
            double step_io_s = std::chrono::duration<double>(io_end - io_start).count();
            uint64_t misses_after = cache.getMissCount();
            uint64_t num_misses_step = misses_after - misses_before;
            if (num_misses_step > 0) {
                total_io_stall_time_s += step_io_s;
                total_io_bytes += num_misses_step * config_.expert_size_bytes;
            }

            // Simulate expert FFN matrix-math computation latency
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.compute_latency_ms));

            // 5. Unpin experts
            cache.unpin(targets[0]);
            cache.unpin(targets[1]);

            // Phase 1: Adaptive Lookahead Controller Metrics Update
            if (g_use_adaptive_lookahead.load(std::memory_order_relaxed)) {
                uint64_t hits = cache.getHitCount();
                uint64_t misses = cache.getMissCount();
                double miss_rate = (hits + misses > 0) ? (double)misses / (hits + misses) : 0.0;
                router.getAdaptiveController().update_metrics(miss_rate);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Prevent compiler optimizing out simulated_accum
    if (simulated_accum == 8888.0f) {
        std::cout << "Simulated GEMM hit anomaly" << std::endl;
    }

    // Stop and join threads
    router.stop();
    prefetch.stop();

    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    ScenarioResult res;
    res.scenario_name = name;
    res.total_execution_time_s = elapsed_s;
    res.token_throughput = config_.total_tokens / elapsed_s;
    
    res.total_hits = cache.getHitCount();
    res.total_misses = cache.getMissCount();
    res.total_evictions = cache.getEvictionCount();
    
    uint64_t total_queries = res.total_hits + res.total_misses;
    res.cache_hit_rate = (total_queries > 0) ? (static_cast<double>(res.total_hits) / total_queries) : 0.0;
    
    res.peak_ram_usage_bytes = MemoryManager::getInstance().getPeakRamUsage();
    res.final_cache_occupancy_bytes = MemoryManager::getInstance().getCacheOccupancy();

    uint64_t pred_count = router.getPredictionCount();
    res.avg_prediction_time_us = (pred_count > 0) ? (router.getTotalPredictionTimeUs() / pred_count) : 0;

    uint64_t pref_count = prefetch.getPrefetchCount();
    res.avg_prefetch_time_us = (pref_count > 0) ? (prefetch.getTotalPrefetchTimeNs() / 1000 / pref_count) : 0;

    // Calculate SSD read volume (100 MB per cache miss)
    res.total_ssd_read_mb = res.total_misses * (config_.expert_size_bytes / (1024.0 * 1024.0));

    // Calculate actual measured PCIe bandwidth
    if (total_io_stall_time_s > 0) {
        res.pcie_bandwidth_mb_s = (total_io_bytes / (1024.0 * 1024.0)) / total_io_stall_time_s;
    } else {
        res.pcie_bandwidth_mb_s = 0.0;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Scenario Result] Elapsed: " << res.total_execution_time_s << " s" << std::endl;
    std::cout << "[Scenario Result] Throughput: " << res.token_throughput << " tok/sec" << std::endl;
    std::cout << "[Scenario Result] Hit Rate: " << (res.cache_hit_rate * 100.0) << " % (Hits: " 
              << res.total_hits << ", Misses: " << res.total_misses << ")" << std::endl;
    std::cout << "[Scenario Result] Measured PCIe Bandwidth: " << res.pcie_bandwidth_mb_s << " MB/s" << std::endl;
    std::cout << "[Scenario Result] Evictions: " << res.total_evictions << std::endl;
    std::cout << "[Scenario Result] Peak RAM: " << (res.peak_ram_usage_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "[Scenario Result] SSD Read Volume: " << res.total_ssd_read_mb << " MB" << std::endl;

    MemoryManager::getInstance().close_weights_file();
    return res;
}

void Benchmark::run_all_scenarios(std::vector<ScenarioResult>& results) {
    generate_expert_assets();

    results.push_back(run_scenario(RoutingScenario::COLD_CACHE, "Scenario A: Cold Cache"));
    results.push_back(run_scenario(RoutingScenario::WARM_CACHE, "Scenario B: Warm Cache"));
    results.push_back(run_scenario(RoutingScenario::HIGH_REUSE, "Scenario C: High Expert Reuse"));
    results.push_back(run_scenario(RoutingScenario::WORST_CASE_CHURN, "Scenario D: Worst-Case Expert Churn"));

    cleanup_expert_assets();
}

void Benchmark::generate_report(const std::vector<ScenarioResult>& results, const std::string& filepath) {
    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "[Benchmark ERROR] Failed to write profiling report to: " << filepath << std::endl;
        return;
    }

    ofs << "======================================================================\n";
    ofs << "         HETEROGENEOUS MOE LOCAL INFERENCE PROFILING REPORT           \n";
    ofs << "======================================================================\n\n";

    ofs << std::fixed << std::setprecision(2);

    for (const auto& res : results) {
        ofs << "------------------------------------------------------------\n";
        ofs << " Scenario: " << res.scenario_name << "\n";
        ofs << "------------------------------------------------------------\n";
        ofs << "  Total Execution Time  : " << res.total_execution_time_s << " seconds\n";
        ofs << "  Token Throughput       : " << res.token_throughput << " tokens/sec\n";
        ofs << "  Cache Hit Rate         : " << (res.cache_hit_rate * 100.0) << "%\n";
        ofs << "  Cache Hits             : " << res.total_hits << "\n";
        ofs << "  Cache Misses           : " << res.total_misses << " (Stalls execution)\n";
        ofs << "  Cache Evictions        : " << res.total_evictions << "\n";
        ofs << "  Peak RAM Usage         : " << (res.peak_ram_usage_bytes / (1024.0 * 1024.0)) << " MB\n";
        ofs << "  Final Cache Occupancy  : " << (res.final_cache_occupancy_bytes / (1024.0 * 1024.0)) << " MB\n";
        ofs << "  Avg Router Latency     : " << res.avg_prediction_time_us << " us\n";
        ofs << "  Avg Prefetch I/O Time  : " << res.avg_prefetch_time_us << " us\n";
        ofs << "  SSD Read Volume        : " << res.total_ssd_read_mb << " MB\n";
        ofs << "  Measured PCIe Bandwidth: " << res.pcie_bandwidth_mb_s << " MB/s\n\n";
    }

    ofs << "======================================================================\n";
    ofs << "                        COMPARATIVE SUMMARY                           \n";
    ofs << "======================================================================\n";
    ofs << std::left << std::setw(30) << "Scenario" 
        << std::setw(18) << "Throughput (t/s)" 
        << std::setw(12) << "Hit Rate" 
        << std::setw(15) << "SSD Read (MB)" 
        << std::setw(15) << "PCIe BW (MB/s)"
        << "Peak RAM (MB)\n";
    ofs << "--------------------------------------------------------------------------------------\n";
    for (const auto& res : results) {
        std::stringstream hr_ss;
        hr_ss << (res.cache_hit_rate * 100.0) << "%";
        ofs << std::left << std::setw(30) << res.scenario_name.substr(0, 28)
            << std::setw(18) << res.token_throughput
            << std::setw(12) << hr_ss.str()
            << std::setw(15) << res.total_ssd_read_mb
            << std::setw(15) << res.pcie_bandwidth_mb_s
            << (res.peak_ram_usage_bytes / (1024.0 * 1024.0)) << "\n";
    }
    ofs << "======================================================================\n";

    std::cout << "[Benchmark] Profiling report saved to: " << filepath << std::endl;
}

// C API bindings for python integration
extern "C" {
    MOE_API void* create_benchmark(
        const char* expert_dir,
        uint32_t num_experts,
        uint64_t expert_size,
        uint32_t num_layers,
        uint32_t total_tokens,
        uint64_t cache_size
    ) {
        BenchmarkConfig config;
        if (expert_dir) config.expert_dir = expert_dir;
        config.num_experts = num_experts;
        config.expert_size_bytes = expert_size;
        config.num_layers = num_layers;
        config.total_tokens = total_tokens;
        config.cache_size_bytes = cache_size;
        
        return new Benchmark(config);
    }

    MOE_API void destroy_benchmark(void* benchmark_ptr) {
        if (benchmark_ptr) {
            delete static_cast<Benchmark*>(benchmark_ptr);
        }
    }

    MOE_API void generate_assets(void* benchmark_ptr) {
        if (benchmark_ptr) {
            static_cast<Benchmark*>(benchmark_ptr)->generate_expert_assets();
        }
    }

    MOE_API void cleanup_assets(void* benchmark_ptr) {
        if (benchmark_ptr) {
            static_cast<Benchmark*>(benchmark_ptr)->cleanup_expert_assets();
        }
    }

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
    ) {
        if (!benchmark_ptr) return;
        Benchmark* bench = static_cast<Benchmark*>(benchmark_ptr);
        
        ScenarioResult res = bench->run_scenario(static_cast<RoutingScenario>(scenario_type), name ? name : "Unnamed");
        
        if (out_exec_time) *out_exec_time = res.total_execution_time_s;
        if (out_throughput) *out_throughput = res.token_throughput;
        if (out_hits) *out_hits = res.total_hits;
        if (out_misses) *out_misses = res.total_misses;
        if (out_evictions) *out_evictions = res.total_evictions;
        if (out_hit_rate) *out_hit_rate = res.cache_hit_rate;
    }
}
