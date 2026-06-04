# Thread Model Specification - Heterogeneous MoE Local Inference Engine

This document defines the multi-threaded architecture of the Heterogeneous MoE Local Inference Engine. To achieve maximum throughput, compute threads must never block on I/O. This is achieved by dedicating specific threads to routing, prefetching, and tensor computation.

---

## 1. Thread Hierarchy and Roles

The engine runs three distinct classes of threads:

```
+-----------------------------------------------------------------------------------------+
|                                    MAIN INFERENCE ENGINE                                |
+-----------------------------------------------------------------------------------------+
       | (Spawns)                                | (Spawns)                        | (Spawns)
       v                                         v                                 v
+------------------+                      +------------------+              +------------------+
|   Main Thread    |                      |  Router Thread   |              | Prefetch Thread  |
|  (CPU Compute)   |                      |  (NPU Emulated)  |              |  (SSD Disk I/O)  |
+------------------+                      +------------------+              +------------------+
| - Orchestrates   |                      | - Computes gating|              | - Pops SPSC queue|
|   generation loop|                      |   logits for N+1,|              | - Calls mmap &   |
| - Executes layers|                      |   and N+2 layers.|              |   madvise to load|
|   sequentially.  |                      | - Pushes target  |              |   expert weights |
| - Pulls from cache|                     |   experts to SPSC|              |   asynchronously.|
+------------------+                      +------------------+              +------------------+
```

### 1.1 Main Thread (Compute Thread)
* **Responsibility**: Simulates token generation and executes the model's layers sequentially.
* **Workload**: Executes Multi-threaded GEMM calculations for the active layers using CPU threads.
* **Block Behavior**: Should never block on I/O. It queries the Cache for the required experts at layer $N$. If the expert is still loading, it performs a brief spin-lock or condition-wait (Cache Miss penalty), but under normal lookup depth this wait is avoided.

### 1.2 Router Thread (Lookahead NPU Engine)
* **Responsibility**: Computes gating matrices ahead of execution.
* **Workload**: Performs INT8/FP32 matrix-vector multiplications on incoming token hidden states to determine the top-K expert indices.
* **Interprocess Communication**: Encapsulates predictions into `PredictedExpertMessage` objects and pushes them into the Single-Producer Single-Consumer (SPSC) Lock-Free Queue.

### 1.3 Prefetch Thread (SSD I/O Engine)
* **Responsibility**: Populates the Expert Cache by loading expert weights from the NVMe SSD before they are requested by the compute thread.
* **Workload**: Pops messages from the SPSC queue, queries the LRU Cache, and manages the virtual address mappings (calling `mmap`, `madvise`, `readahead` on cache misses).

---

## 2. Lock-Free Single-Producer Single-Consumer Queue

Communication between the Router and Prefetch threads is critical and must not introduce mutex bottlenecks. We implement a custom, cache-line aligned Single-Producer Single-Consumer (SPSC) lock-free ring buffer:

* **No Mutexes**: Only atomic head and tail pointers are used.
* **Cache Line Padding**: Atomic indexes are padded to 64-byte boundaries (`alignas(64)`) to avoid **false sharing** between the CPU cores running the Router and Prefetch threads.
* **Memory Ordering**: Uses `std::memory_order_acquire` and `std::memory_order_release` to enforce synchronization boundaries between the enqueue and dequeue CPU cores.

---

## 3. Real-Time Scheduling and Affinities

To maximize performance, thread assignments are mapped as follows:
* **Router Thread**: Assigned to a dedicated CPU core (or NPU execution environment) to maintain constant lookahead throughput.
* **Prefetch Thread**: Assigned to an I/O-heavy core. Operates at normal priority, utilizing the Linux kernel's asynchronous page cache mechanisms.
* **Compute Threads**: Utilizes a pool of worker threads pinned to physical high-performance cores, optimizing L1/L2 cache locality during matrix math operations.
