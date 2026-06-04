import os
import time
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

class MicroModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        # Initialize weights with dummy values
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

def generate_onnx_model(size, path):
    model = MicroModel(size)
    model.eval()
    dummy_input = torch.randn(1, size)
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

def run_benchmark(model_path, size):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Configure session with Vitis AI execution provider
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Path to the Vitis AI EP configuration file
    vaip_config = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"
    provider_options = [{
        "config_file": vaip_config,
        "cacheDir": "./npu_cache",
        "cacheKey": f"model_{size}"
    }, {}]
    
    t_start_creation = time.perf_counter()
    try:
        session = ort.InferenceSession(
            model_path, 
            opts, 
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"], 
            provider_options=provider_options
        )
    except Exception as e:
        print(f"Failed to create VitisAI session for size {size}: {e}. Falling back to CPU.")
        session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    
    session_creation_time_ms = (time.perf_counter() - t_start_creation) * 1000.0
    
    # Resolve input names and shapes
    input_name = session.get_inputs()[0].name
    x_in = np.ones((1, size), dtype=np.float32)
    
    # First inference (measure compilation/initial load overhead)
    t_start_first = time.perf_counter()
    session.run(None, {input_name: x_in})
    first_inference_latency_ms = (time.perf_counter() - t_start_first) * 1000.0
    
    # Warm inference runs (1000 iterations)
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        session.run(None, {input_name: x_in})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    avg_warm_latency_ms = sum(latencies) / len(latencies)
    throughput = 1000.0 / (avg_warm_latency_ms / 1000.0)
    
    mem_after = process.memory_info().rss
    memory_usage_mb = (mem_after - mem_before) / (1024.0 * 1024.0)
    
    return {
        "size": size,
        "session_creation_time_ms": session_creation_time_ms,
        "first_inference_latency_ms": first_inference_latency_ms,
        "warm_inference_latency_ms": avg_warm_latency_ms,
        "throughput_inf_sec": throughput,
        "memory_usage_mb": memory_usage_mb
    }

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./npu_cache", exist_ok=True)
    
    sizes = {
        "tiny": 128,
        "small": 512,
        "medium": 1024,
        "large": 4096
    }
    
    results = {}
    
    for name, size in sizes.items():
        print(f"Running microbenchmark for model scale: {name.upper()} (Dim: {size}x{size})...")
        model_path = f"./models/model_{name}.onnx"
        generate_onnx_model(size, model_path)
        
        bench_res = run_benchmark(model_path, size)
        results[name] = bench_res
        print(f"  Session Creation : {bench_res['session_creation_time_ms']:.2f} ms")
        print(f"  First Inference  : {bench_res['first_inference_latency_ms']:.2f} ms")
        print(f"  Warm Inference   : {bench_res['warm_inference_latency_ms']:.2f} ms")
        print(f"  Throughput       : {bench_res['throughput_inf_sec']:.2f} inf/sec")
        print(f"  Memory Overhead  : {bench_res['memory_usage_mb']:.2f} MB\n")
        
    with open("npu_microbenchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Microbenchmarks completed. Saved to npu_microbenchmark.json.")

if __name__ == "__main__":
    main()
