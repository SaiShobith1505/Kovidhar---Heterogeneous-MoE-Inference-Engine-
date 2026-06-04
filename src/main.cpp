#include "benchmark.h"
#include "lockfree_queue.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

struct LatencyMsg {
    uint64_t timestamp_ns;
};

void run_queue_latency_benchmark() {
    std::cout << "\n=========================================" << std::endl;
    std::cout << " RUNNING LOCK-FREE QUEUE LATENCY BENCHMARK" << std::endl;
    std::cout << "=========================================" << std::endl;

    LockFreeSPSCQueue<LatencyMsg, 4096> queue;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<uint64_t> count{0};

    // Consumer thread pops and measures duration
    std::thread consumer([&]() {
        LatencyMsg msg;
        while (running.load(std::memory_order_relaxed) || !queue.empty()) {
            if (queue.pop(msg)) {
                uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count();
                if (msg.timestamp_ns > 0 && now > msg.timestamp_ns) {
                    total_latency_ns.fetch_add(now - msg.timestamp_ns, std::memory_order_relaxed);
                    count.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                yield_processor();
            }
        }
    });

    const size_t num_messages = 100000;
    
    // Warmup the queue/thread context
    for (size_t i = 0; i < 1000; ++i) {
        LatencyMsg msg{0};
        queue.push(msg);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    // Main latency benchmark run
    for (size_t i = 0; i < num_messages; ++i) {
        LatencyMsg msg;
        msg.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        
        while (!queue.push(msg)) {
            yield_processor();
        }
    }

    // Wait until all messages are popped
    while (count.load() < num_messages) {
        yield_processor();
    }

    running.store(false, std::memory_order_relaxed);
    if (consumer.joinable()) {
        consumer.join();
    }

    uint64_t total_ns = total_latency_ns.load();
    uint64_t total_count = count.load();
    double avg_ns = total_count > 0 ? (double)total_ns / total_count : 0.0;

    std::cout << "[Queue Benchmark] Total messages: " << total_count << std::endl;
    std::cout << "[Queue Benchmark] Average SPSC Queue Latency: " << avg_ns << " ns" << std::endl;
    std::cout << "=========================================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================================================\n";
    std::cout << "          HETEROGENEOUS MOE LOCAL INFERENCE ENGINE BENCHMARK          \n";
    std::cout << "======================================================================\n";

    // Set configuration for a 70B MoE LLM Model (4-bit quantized)
    BenchmarkConfig config;
    config.num_experts = 64;
    config.expert_size_bytes = 546ULL * 1024 * 1024; // 546 MB per expert (represents 35 GB total)
    config.num_layers = 32;                          // 32-layer MoE
    config.total_tokens = 10;
    config.cache_size_bytes = 12ULL * 1024 * 1024 * 1024; // 12 GB RAM Cache budget
    config.routing_latency_us = 40;                  // Real physical NPU batch routing latency
    config.expert_dir = "./experts";

    if (argc > 1) {
        config.expert_dir = argv[1];
    }

    // Run SPSC Queue Latency micro-benchmark
    run_queue_latency_benchmark();

    // Run MoE Engine Benchmark
    Benchmark benchmark(config);
    std::vector<ScenarioResult> results;
    
    try {
        benchmark.run_all_scenarios(results);
        benchmark.generate_report(results, "profiling_report.txt");
    } catch (const std::exception& e) {
        std::cerr << "[Main Exception] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n[Main] Benchmarks completed successfully.\n";
    return 0;
}
