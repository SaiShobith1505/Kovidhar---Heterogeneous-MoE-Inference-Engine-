#include "router_thread.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <filesystem>

RouterThread::RouterThread(
    LockFreeSPSCQueue<PredictedExpertMessage, 1024>& queue,
    uint32_t num_layers,
    uint32_t num_experts,
    uint32_t top_k,
    std::atomic<uint32_t>& current_compute_token,
    std::atomic<uint32_t>& current_compute_layer,
    uint32_t total_tokens,
    uint32_t routing_latency_us,
    RoutingScenario scenario
) : queue_(queue),
    num_layers_(num_layers),
    num_experts_(num_experts),
    top_k_(top_k),
    current_compute_token_(current_compute_token),
    current_compute_layer_(current_compute_layer),
    total_tokens_(total_tokens),
    routing_latency_us_(routing_latency_us),
    scenario_(scenario),
    running_(false),
    total_prediction_time_us_(0),
    prediction_count_(0)
#ifdef HAS_ONNXRUNTIME
    , env_(ORT_LOGGING_LEVEL_WARNING, "MoEEngine")
#endif
{
    // Initialize mock NPU matrices (GEMV workload simulation)
    // 256 experts, model dimension 1024 (reduced from 4096 for fast CPU emulation while keeping NPU latency simulated)
    size_t dim = 1024;
    gating_weights_.resize(num_experts_ * dim, 0.05f);
    hidden_state_buffer_.resize(dim, 1.0f);

#ifdef HAS_ONNXRUNTIME
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::unordered_map<std::string, std::string> provider_options;
        provider_options["config_file"] = std::getenv("VITIS_AI_EP_CONFIG") ? std::getenv("VITIS_AI_EP_CONFIG") : "";
        provider_options["cacheDir"] = "D:/LocalMoEEngine/.vitisai_cache";
        
        try {
            options.AppendExecutionProvider_VitisAI(provider_options);
            std::cout << "[RouterThread] Successfully appended Vitis AI Execution Provider." << std::endl;
        } catch (const std::exception& e) {
            // Fail silently or print fallback warning
        }

        std::string model_path = "D:/LocalMoEEngine/gating_model.onnx";
        #ifndef _WIN32
        model_path = "/mnt/d/LocalMoEEngine/gating_model.onnx";
        #endif

        if (std::filesystem::exists(model_path)) {
#ifdef _WIN32
            std::wstring wmodel_path(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(env_, wmodel_path.c_str(), options);
#else
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
#endif
            std::cout << "[RouterThread] Loaded ONNX gating model: " << model_path << std::endl;

            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_inputs = session_->GetInputCount();
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name_allocated = session_->GetInputNameAllocated(i, allocator);
                input_node_names_.push_back(name_allocated.get());
            }
            size_t num_outputs = session_->GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name_allocated = session_->GetOutputNameAllocated(i, allocator);
                output_node_names_.push_back(name_allocated.get());
            }
        } else {
            std::cerr << "[RouterThread WARNING] ONNX model file " << model_path 
                      << " not found. Running in high-performance CPU emulation mode." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[RouterThread ERROR] ONNX Runtime initialization failed: " << e.what() 
                  << ". Falling back to CPU routing." << std::endl;
    }
#endif
}

RouterThread::~RouterThread() {
    stop();
}

void RouterThread::start() {
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&RouterThread::run, this);
}

void RouterThread::stop() {
    if (running_.load(std::memory_order_relaxed)) {
        running_.store(false, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void RouterThread::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void RouterThread::run() {
    uint32_t router_token = 0;
    uint32_t router_layer = 0;

    std::mt19937 gen(1337); // Deterministic seed

    while (running_.load(std::memory_order_relaxed)) {
        if (router_token >= total_tokens_) {
            break;
        }

        // 1. Flow Control: Ensure the router thread does not outrun the compute pipeline too far.
        // Limit lookahead to a maximum of 4 layers ahead of current execution.
        uint32_t comp_tok = current_compute_token_.load(std::memory_order_relaxed);
        uint32_t comp_lay = current_compute_layer_.load(std::memory_order_relaxed);
        
        int64_t compute_flat = static_cast<int64_t>(comp_tok) * num_layers_ + comp_lay;
        int64_t router_flat = static_cast<int64_t>(router_token) * num_layers_ + router_layer;

        if (router_flat >= compute_flat + 4) {
            // Sleep briefly to yield CPU
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        // 2. NPU Routing Workload Execution
        auto start_time = std::chrono::high_resolution_clock::now();

        bool run_onnx = false;
#ifdef HAS_ONNXRUNTIME
        if (session_) {
            run_onnx = true;
        }
#endif

        if (run_onnx) {
#ifdef HAS_ONNXRUNTIME
            try {
                std::vector<int64_t> input_shape = {1, 1024};
                static thread_local std::vector<float> input_tensor_values(1024, 1.0f);

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size()
                );

                std::vector<const char*> input_names;
                for (const auto& name : input_node_names_) {
                    input_names.push_back(name.c_str());
                }
                std::vector<const char*> output_names;
                for (const auto& name : output_node_names_) {
                    output_names.push_back(name.c_str());
                }

                auto output_tensors = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names.data(), &input_tensor, 1,
                    output_names.data(), 1
                );

                float* output_data = output_tensors[0].GetTensorMutableData<float>();
                float sum = 0.0f;
                size_t elements = std::min(static_cast<size_t>(num_experts_), output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
                for (size_t i = 0; i < elements; ++i) {
                    sum += output_data[i];
                }
                if (sum == 9999.0f) {
                    std::cout << "ONNX logits anomaly" << std::endl;
                }
            } catch (const std::exception& e) {
                static bool logged_error = false;
                if (!logged_error) {
                    std::cerr << "[RouterThread ERROR] ONNX Runtime Run execution failed: " << e.what() << std::endl;
                    logged_error = true;
                }
            }
#endif
        } else {
            // Emulate CPU routing work
            float sum = 0.0f;
            size_t dim = 1024;
            for (size_t e = 0; e < num_experts_; ++e) {
                float expert_logit = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    expert_logit += hidden_state_buffer_[d] * gating_weights_[e * dim + d];
                }
                sum += expert_logit;
            }
            if (sum == 9999.0f) {
                std::cout << "Gating anomaly detected!" << std::endl;
            }

            // Simulate additional hardware routing latency if requested
            if (routing_latency_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(routing_latency_us_));
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_prediction_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
        prediction_count_.fetch_add(1, std::memory_order_relaxed);

        // 3. Expert Routing Selection logic based on Scenarios
        PredictedExpertMessage msg;
        msg.token_index = router_token;
        msg.layer_id = router_layer;
        msg.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        if (scenario_ == RoutingScenario::HIGH_REUSE) {
            // Highly concentrated reuse: only 2 experts per layer, cycling in a tiny set of 4 hot experts
            msg.top_k_experts[0] = (router_layer * 2) % 4;
            msg.top_k_experts[1] = (router_layer * 2 + 1) % 4;
            msg.confidence_scores[0] = 0.7f;
            msg.confidence_scores[1] = 0.3f;
        } else if (scenario_ == RoutingScenario::WORST_CASE_CHURN) {
            // Maximum churn: sequential traversal across all experts, forcing constant cache thrashing
            msg.top_k_experts[0] = (router_flat * 2) % num_experts_;
            msg.top_k_experts[1] = (router_flat * 2 + 1) % num_experts_;
            msg.confidence_scores[0] = 0.6f;
            msg.confidence_scores[1] = 0.4f;
        } else {
            // Normal zipfian/skewed distribution
            // Expert 0 to 15 are hotter, others are colder
            std::uniform_int_distribution<uint32_t> hot_dist(0, 15);
            std::uniform_int_distribution<uint32_t> cold_dist(16, num_experts_ - 1);
            
            // 80% chance of hot expert, 20% chance of cold expert
            std::uniform_int_distribution<uint32_t> prob(0, 99);
            msg.top_k_experts[0] = (prob(gen) < 80) ? hot_dist(gen) : cold_dist(gen);
            msg.top_k_experts[1] = (prob(gen) < 80) ? hot_dist(gen) : cold_dist(gen);

            // Ensure distinct experts
            if (msg.top_k_experts[0] == msg.top_k_experts[1]) {
                msg.top_k_experts[1] = (msg.top_k_experts[1] + 1) % num_experts_;
            }

            msg.confidence_scores[0] = 0.75f;
            msg.confidence_scores[1] = 0.25f;
        }

        // 4. Enqueue prediction message (Retry until queue has slot)
        while (!queue_.push(msg) && running_.load(std::memory_order_relaxed)) {
            yield_processor();
        }

        // Advance routing cursor
        router_layer++;
        if (router_layer >= num_layers_) {
            router_layer = 0;
            router_token++;
        }
    }
}
