# Comparative Report: Real NPU vs. Dummy NPU Simulation

This report compares the performance characteristics of the **Dummy NPU Emulator** (running inside virtualized WSL) with the **Real AMD Ryzen AI NPU** (running natively on Windows 11 with the Vitis AI Execution Provider) utilizing the same 12-layer MoE engine configurations.

---

## 📊 Comparative Performance Summary

| Metric | Dummy NPU (WSL) | Real NPU (Windows 11) | Analysis / Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Gating Latency (1 Layer)** | ~150 us (Est. CPU loop) | **16 us (Physical NPU)** | **9.3x speedup** on NPU for routing calculations. |
| **Average Router Time (C++)**| ~870 us | **189 us** | Real-world scheduling and queue push overhead added. |
| **SPSC Queue Latency** | 303,165 ns | **359,197 ns** | Windows thread context switching latency is higher. |
| **SSD PCIe Bandwidth** | 297.65 MB/s | **4,014.93 MB/s** | **13.5x speedup** via native Win32 memory-mapped I/O. |
| **Warm Cache Throughput** | 0.73 tok/sec | **3.73 tok/sec** | **5.1x overall speedup** due to NVMe SSD direct access. |
| **Worst-Case Throughput** | 0.13 tok/sec | **1.53 tok/sec** | **11.7x overall speedup** under high cache churn. |
| **Scenario D Miss Rate** | 0.8% (2 misses) | **84.5% (203 misses)** | Fast Windows execution outran the prefetch thread. |
| **Avg Prefetch I/O Time** | 10.3 ms | **54.3 ms** | Sync page-ins during misses dominate under high speed. |

---

## 💡 Key Findings & Architectural Conclusions

### 1. The Prefetch-Speed Dilemma
* **The Paradox**: Under the Dummy/WSL run, the cache hit rate was **`99.17%`** (only 2 misses) because execution was bottlenecked by slow virtualized disk reads, giving the prefetch thread abundant time to load experts. Under native Windows, the hit rate dropped to **`15.42%`** (203 misses) because the CPU executed matrix math so rapidly that the prefetch thread was outpaced.
* **Conclusion**: To maintain cache hit rates at high execution speeds, we must **increase the lookahead routing depth** from 4 layers to **6 or 8 layers** when running natively on Windows.

### 2. High-Performance Win32 I/O Validation
* Native Windows virtual memory mapping and prefetching (`VirtualAlloc`, `MapViewOfFile`, `PrefetchVirtualMemory`) successfully bypassed the OS page cache, driving physical NVMe SSD reads to **`4.01 GB/s`** (compared to the virtualized 297 MB/s in WSL).

### 3. Recommendation for Next Architecture Phase
Based on these physical measurements, the MoE pipeline should be **modified slightly**:
* **Keep Gating on the NPU**: The 16 microsecond physical gating execution makes NPU gating extremely viable.
* **Optimize Flow Control**: Replace the `sleep_for` blocks in the router threads with lock-free `yield_processor()` spinning to reduce the 359 microseconds SPSC queue overhead.
* **Increase Lookahead Depth dynamically**: Implement a dynamic lookahead depth that scales with execution speed (N+6 or N+8 layers ahead) to ensure that the NVMe prefetch completes before execution requests the parameters.
