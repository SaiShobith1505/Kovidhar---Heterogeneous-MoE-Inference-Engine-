import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

class RouterModel(nn.Module):
    def __init__(self, hidden_dim=1024, num_experts=256):
        super().__init__()
        self.gating = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gating.weight, std=0.02)

    def forward(self, x):
        return self.gating(x)

def generate_router_onnx(path):
    model = RouterModel()
    model.eval()
    dummy_input = torch.randn(1, 1024)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./npu_cache", exist_ok=True)
    
    model_path = "./models/router.onnx"
    generate_router_onnx(model_path)
    
    # Configure session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    vaip_config = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"
    provider_options = [{
        "config_file": vaip_config,
        "cacheDir": "./npu_cache",
        "cacheKey": "router_gating"
    }, {}]
    
    print("Initializing ONNX Runtime Session for Router on NPU...")
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        model_path,
        opts,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=provider_options
    )
    session_creation_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Session created in {session_creation_time_ms:.2f} ms")
    
    input_name = session.get_inputs()[0].name
    
    # --- 1. Single Layer Routing Latency (Batch Size 1) ---
    x_single = np.ones((1, 1024), dtype=np.float32)
    
    # Warmup
    for _ in range(50):
        session.run(None, {input_name: x_single})
        
    single_latencies = []
    for _ in range(1000):
        t_start = time.perf_counter()
        session.run(None, {input_name: x_single})
        single_latencies.append((time.perf_counter() - t_start) * 1000.0) # ms
        
    single_latencies = np.array(single_latencies)
    avg_single = np.mean(single_latencies)
    p50_single = np.percentile(single_latencies, 50)
    p95_single = np.percentile(single_latencies, 95)
    p99_single = np.percentile(single_latencies, 99)
    throughput_single = 1000.0 / avg_single # tokens/sec for batch 1
    
    # --- 2. Batch Routing Latency (Batch Size 12 - e.g. all layers) ---
    x_batch = np.ones((12, 1024), dtype=np.float32)
    
    # Warmup
    for _ in range(50):
        session.run(None, {input_name: x_batch})
        
    batch_latencies = []
    for _ in range(1000):
        t_start = time.perf_counter()
        session.run(None, {input_name: x_batch})
        batch_latencies.append((time.perf_counter() - t_start) * 1000.0) # ms
        
    batch_latencies = np.array(batch_latencies)
    avg_batch = np.mean(batch_latencies)
    p50_batch = np.percentile(batch_latencies, 50)
    p95_batch = np.percentile(batch_latencies, 95)
    p99_batch = np.percentile(batch_latencies, 99)
    throughput_batch = (12 * 1000.0) / avg_batch # tokens/sec for batch 12
    
    # Answer critical question:
    # Can the NPU predict N+4 layers ahead faster than execution pipeline consumes them?
    # Execution consumes 1 layer in ~5 ms. 4 layers = 20 ms.
    # We predict 4 layers using batch size 4, or 4 single runs.
    # Single run latency * 4 = 4 layers latency.
    avg_4_layers_latency_ms = avg_single * 4.0
    can_predict_ahead = avg_4_layers_latency_ms < 20.0 # 20 ms budget for 4 layers
    
    results = {
        "session_creation_time_ms": session_creation_time_ms,
        "single_layer_routing": {
            "average_latency_ms": avg_single,
            "p50_latency_ms": p50_single,
            "p95_latency_ms": p95_single,
            "p99_latency_ms": p99_single,
            "tokens_per_sec": throughput_single
        },
        "batch_routing_layer_12": {
            "average_latency_ms": avg_batch,
            "p50_latency_ms": p50_batch,
            "p95_latency_ms": p95_batch,
            "p99_latency_ms": p99_batch,
            "tokens_per_sec": throughput_batch
        },
        "lookahead_analysis": {
            "avg_4_layers_prediction_time_ms": avg_4_layers_latency_ms,
            "execution_budget_4_layers_ms": 20.0,
            "can_predict_ahead_n_plus_4": bool(can_predict_ahead)
        }
    }
    
    with open("router_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nRouter benchmarking complete. Saved to router_benchmark.json.")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
