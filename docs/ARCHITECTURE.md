# Architecture Specification - Heterogeneous MoE Local Inference Engine

This document provides the high-level architectural design for the Heterogeneous Mixture-of-Experts (MoE) Local Inference Engine. The system is designed to execute massive MoE LLMs (e.g., 47B-class models) on consumer laptops with limited memory ($\le 12$ GB RAM) by decoupling Compute and I/O, allowing expert weights to be streamed asynchronously from NVMe SSD before they are executed.

---

## 1. High-Level Architecture Overview

The system consists of three concurrent execution pipelines:
1. **Lookahead Router (NPU Thread)**: Decides which experts will be needed for future tokens/layers.
2. **SSD Prefetch Engine (I/O Thread)**: Loads the required expert weights from the NVMe SSD into a page-aligned RAM cache.
3. **Execution Engine (Compute Thread)**: Computes the model layers sequentially using the expert weights, relying on a zero-copy memory mapping mechanism.

```mermaid
graph TD
    subgraph Lookahead Pipeline (NPU Emulated)
        RT[Router Thread] -->|Predicts N+1, N+2 Experts| LFQ[Lock-Free SPSC Queue]
    end

    subgraph I/O Prefetch Pipeline
        LFQ -->|Pops Expert Message| PT[Prefetch Thread]
        PT -->|Query Cache| EC[Expert Cache]
        EC -->|Cache Miss| MM[Memory Manager]
        MM -->|Async mmap + madvise| SSD[(NVMe SSD Storage)]
        SSD -->|Load Weights| EC
    end

    subgraph Compute Pipeline
        CT[Execution Engine] -->|Request Layer N Weights| EC
        EC -->|Provide Mapped Buffer| CT
        CT -->|Compute Gemm| CPU[CPU Threads]
        CT -->|Update KV Cache| KV[Locked KV Cache]
    end
    
    style RT fill:#4f46e5,stroke:#312e81,color:#fff
    style PT fill:#0891b2,stroke:#155e75,color:#fff
    style CT fill:#16a34a,stroke:#14532d,color:#fff
    style EC fill:#ea580c,stroke:#7c2d12,color:#fff
    style SSD fill:#64748b,stroke:#334155,color:#fff
```

---

## 2. Key Components

### 2.1 Model Fragmentation Strategy
Standard MoE models store all expert weights together with routing logic in a single monolithic weight file. This is impossible to handle on a 12 GB RAM laptop for 25 GB+ models. 
The fragmentation pipeline splits the model:
* **Router Subgraph**: Extracted as a separate compact model (e.g., in ONNX format), quantized to INT4/INT8. It contains the embedding layers and gating matrices for all layers. It is small enough ($\le 500$ MB) to remain permanently loaded in NPU/CPU RAM.
* **Individual Expert Blocks**: Each expert's Feed-Forward Network (FFN) weights (projection matrices) are extracted into independent binary files (`expert_xxx.bin`). These binary blobs are stored sequentially on disk to allow maximum sequential read performance.

### 2.2 Predictive Lookahead Engine
The gating computation for layer $N+1$ and $N+2$ is executed *before* layer $N$ has finished computing. 
* By running routing ahead of the main compute thread, the engine determines which expert files to load ahead of time.
* It outputs a `PredictedExpertMessage` containing:
  ```cpp
  struct PredictedExpertMessage {
      uint32_t token_index;
      uint32_t layer_id;
      uint32_t top_k_experts[2]; // e.g., top-2 routing
      float confidence_scores[2];
      uint64_t timestamp_ns;
  };
  ```

### 2.3 SSD Prefetch Engine
The Prefetch Engine reads messages from a lock-free Single-Producer Single-Consumer (SPSC) queue. For each predicted expert, the prefetch engine:
1. Queries the **Expert Cache** to check if the expert is already loaded (Cache Hit).
2. On a cache miss, maps the expert file using `mmap`.
3. Invokes `madvise(addr, size, MADV_WILLNEED)` and `readahead()` to signal the OS kernel to initiate aggressive, non-blocking asynchronous disk reads from the NVMe SSD.
4. Returns immediately to process the next message, avoiding blocking the Lookahead Router or Compute thread.

### 2.4 Expert Cache & Memory Manager
The Memory Manager manages physical allocations and prevents virtual memory swapping:
* **KV Cache Region**: A pre-allocated, page-locked (`mlock`) memory pool that stores the attention keys and values. It is never paged out.
* **Expert Cache (RAM Pool)**: A thread-safe Least Recently Used (LRU) cache with a fixed physical memory limit.
  * Active layers (currently undergoing execution) are pinned (`pinned = true`) in the cache to prevent the prefetch thread from evicting them.
  * The memory mapping handles (`mmap` pointers) are kept in a lookup table, and unmapped via `munmap` only upon LRU eviction.

---

## 3. Heterogeneous Integration & Fallbacks

* **AMD Ryzen AI NPU**: Designed to accelerate INT8/INT4 matrix operations. The router model (`router.onnx`) is targeted at the NPU using ONNX Runtime with the Vitis AI execution provider.
* **Software Emulation Fallback**: In environments without a physical NPU or Vitis AI drivers (such as our Linux sandbox/WSL), routing operations are executed via a highly optimized CPU emulation layer. The CPU emulation layer uses multi-threaded matrix-vector multiplication (GEMV) to calculate routing logits, simulating the execution latency of the NPU routing thread.
