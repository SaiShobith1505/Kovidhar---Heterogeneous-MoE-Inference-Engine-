# Heterogeneous MoE Local Inference Engine for Consumer Hardware

This project implements a production-grade prototype of an asynchronous, memory-bounded, heterogeneous Mixture-of-Experts (MoE) Local Inference Engine written in standard C++20. It demonstrates how to run massive MoE architectures (e.g., 47B-class models with ~25 GB parameters) on memory-constrained consumer hardware ($\le 12$ GB RAM) by decoupling Compute and I/O.

---

## Quick Start (WSL / Linux)

### 1. Build C++ native components
```bash
# Configure CMake
/home/sai_shobith/.local/bin/cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile project (executables + library)
/home/sai_shobith/.local/bin/cmake --build build --config Release
```

### 2. Run the Benchmark Suite
```bash
./build/moe_benchmark
```
This runs the full MoE simulator over 4 scenarios: Cold Cache, Warm Cache, High Expert Reuse, and Worst-Case Expert Churn, generating `profiling_report.txt` containing full telemetry.

### 3. Launch the Web Visualizer Dashboard (Windows Native / Fast)
Run the Python script natively on your Windows host:
```powershell
python python/visualizer_server.py
```
Open a browser and navigate to:
```url
http://localhost:8082/
```

### 4. Launch the Web Visualizer inside WSL (Alternative C++ Target)
```bash
# Compile the visualizer server target
/home/sai_shobith/.local/bin/cmake --build build --target moe_visualizer_server --config Release

# Start the server on port 8082
./build/moe_visualizer_server
```
Open a browser on your host machine and navigate to:
```url
http://localhost:8082/
```

### 5. Run via Python Wrapper
```bash
python3 python/wrapper.py
```

---

## Architectural Details & Technical Documents

Detailed design specifications can be found under the [docs/](file:///d:/AI%20RESEARCH/docs/) folder:
- **[docs/ARCHITECTURE.md](file:///d:/AI%20RESEARCH/docs/ARCHITECTURE.md)**: High-level architectural blocks, design principles, and component responsibilities.
- **[docs/DATAFLOW.md](file:///d:/AI%20RESEARCH/docs/DATAFLOW.md)**: Detailed trace of the token inference pipeline and data flow.
- **[docs/MEMORY_MODEL.md](file:///d:/AI%20RESEARCH/docs/MEMORY_MODEL.md)**: Memory layout, KV cache locking, expert pool allocations, and alignment constraints.
- **[docs/THREAD_MODEL.md](file:///d:/AI%20RESEARCH/docs/THREAD_MODEL.md)**: Multi-threaded design, thread priorities, synchronization via SPSC lock-free queues, and thread safety.
- **[docs/CACHE_POLICY.md](file:///d:/AI%20RESEARCH/docs/CACHE_POLICY.md)**: Cache eviction strategy (LRU), pinning of active layers, and cache hit/miss tracking metrics.
- **[docs/README.md](file:///d:/AI%20RESEARCH/docs/README.md)**: Full detailed documentation.
