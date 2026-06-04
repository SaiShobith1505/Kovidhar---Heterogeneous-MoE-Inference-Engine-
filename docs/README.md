# Detailed Reference Specification - Heterogeneous MoE Local Inference Engine

This document provides the reference build instructions, component architecture, memory model, and visualizer details of the Heterogeneous MoE Local Inference Engine.

---

## 1. Physical Memory Partitioning

The 12 GB system RAM budget is divided into three fixed, non-overlapping regions:
* **OS & Router Runtime**: ~2 GB (virtual memory)
* **Key-Value (KV) Cache**: ~2 GB (locked in RAM using `mlock`)
* **Expert Cache RAM Pool**: ~8 GB (LRU pool managing memory mappings)

---

## 2. Dependencies & Build Instructions

### Dependencies
- **OS**: Linux (or Ubuntu WSL 2 sandbox)
- **Compiler**: GCC / Clang with C++20 support (tested on GCC 15.2.0)
- **Build System**: CMake 3.20 or later

### Compiling inside WSL / Linux
```bash
# Configure the CMake build directory
/home/sai_shobith/.local/bin/cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile all targets
/home/sai_shobith/.local/bin/cmake --build build --config Release
```

---

## 3. Running benchmarks

To execute the performance benchmarks, run the executable inside WSL:
```bash
./build/moe_benchmark
```
This automatically generates sparse parameters inside `./experts`, measures telemetry over Cold Cache, Warm Cache, High Expert Reuse, and Worst-Case Churn scenarios, and outputs results to `profiling_report.txt`.

---

## 4. Running the Web Visualizer

We have implemented a high-contrast web visualizer server `moe_visualizer_server` that serves a dashboard contrasting synchronous blocking execution (Case 1) against asynchronous prefetch pipeline (Case 2).

### Option A: Launching the Python Server natively on Windows (Recommended)
Run the Python script natively on your Windows host:
```powershell
python python/visualizer_server.py
```
Open a browser and navigate to:
```url
http://localhost:8082/
```

### Option B: Launching the C++ Server inside WSL
```bash
# Compile visualizer target
/home/sai_shobith/.local/bin/cmake --build build --target moe_visualizer_server --config Release

# Run server on port 8082
./build/moe_visualizer_server
```

Once the visualizer server outputs:
`MoE Visualizer Server listening on port 8082...`

Open a browser window on the host computer and go to:
```url
http://localhost:8082/
```

### Dashboard Layout & Details
* **Case 1 Column**: Animates a 12-layer stack representing synchronous processing. Displays a glowing active layer, a flashing red stall banner, a loading spinner on cache misses, and a Chart.js line plot displaying throughput dipping to `0.17 t/s`.
* **Case 2 Column**: Renders three parallel lanes displaying the NPU Lookahead predictions, SSD-to-RAM prefetch bars (0-100% progress), and a pulsing NPU compute indicator. A Chart.js plot displays stable, fast throughput.

---

## 5. Python ctypes Integration

Expose native benchmarks to Python scripts via the compiled shared library:
```bash
python3 python/wrapper.py
```
This loads `./build/libmoe_engine.so` and triggers execution through standard Python ctypes bindings.
