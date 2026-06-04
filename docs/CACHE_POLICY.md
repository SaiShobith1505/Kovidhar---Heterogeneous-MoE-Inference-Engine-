# Cache Policy Specification - Heterogeneous MoE Local Inference Engine

This document defines the Cache Policy, eviction strategies, and hit/miss mechanics of the Expert RAM Pool.

---

## 1. Eviction Strategy (Least Recently Used)

The Expert Cache maintains a fixed capacity (e.g., 8 GB maximum mapped weight size). When the memory size of loaded experts exceeds this threshold, the cache must evict experts to maintain the physical memory budget.

* **LRU Eviction**: Experts are tracked in a doubly-linked list.
  * When an expert is queried, it is moved to the head of the list (marking it as most recently used).
  * When eviction is triggered, the engine scans the list from the tail (least recently used) forward.
  * The first candidate that is **not pinned** is selected for eviction.
  * Eviction is performed via `munmap`, freeing virtual memory and dropping the physical resident pages.

---

## 2. Active Layer Pinning & Lock Protection

To prevent the Prefetch Thread from evicting an expert that is currently being computed by the main thread, the engine implements **Lock/Pin protection**:

1. **Pinning**: Before the Compute Thread begins executing Layer $N$, it locks/pins all required experts for that layer by incrementing an atomic pin counter `pin_count` on each expert:
   ```cpp
   expert->pin_count.fetch_add(1, std::memory_order_relaxed);
   ```
2. **Eviction Safety**: The eviction worker skips any expert where `pin_count > 0`.
3. **Unpinning**: Once Layer $N$ execution completes, the Compute Thread decrements the pin counter:
   ```cpp
   expert->pin_count.fetch_sub(1, std::memory_order_release);
   ```

---

## 3. Prefetch Priorities

When multiple prediction messages are enqueued, they represent lookahead requirements at different future steps:
* **Immediate Priority ($N+1$)**: Experts predicted for the next layer.
* **Secondary Priority ($N+2$)**: Experts predicted for two layers ahead.

The Prefetch Thread processes the queue sequentially. Because the Router pushes messages in order of layer execution ($N+1$ before $N+2$), the prefetch queue naturally sequences immediate priorities first. If the queue fills up, $N+2$ entries are replaced or dropped to prioritize $N+1$ mappings.

---

## 4. Metric Collection and Telemetry

The cache captures high-precision telemetry metrics to measure efficiency:

* **Cache Hit (`hit_rate`)**: Counted when the execution engine requires expert $X$ at layer $L$, and the expert is already fully mapped and paged into memory (`resident = true`).
* **Cache Miss (`miss_rate`)**: Counted when the execution engine requires expert $X$, but it has not been mapped yet or the paging operation is incomplete. In this case, the compute thread stalls until the prefetch thread finishes loading, recording this stall as `expert_load_latency`.
* **Eviction Count (`evictions`)**: Counted each time an expert is unmapped from the cache pool.
* **SSD Read Volume**: Tracked as the total size of files mapped and accessed.
