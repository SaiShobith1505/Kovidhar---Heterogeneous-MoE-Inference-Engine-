# Memory Model Specification - Heterogeneous MoE Local Inference Engine

This document defines the physical and virtual memory management policies of the Heterogeneous MoE Local Inference Engine. Given the tight constraints of consumer hardware (total system RAM budget $\le 12$ GB), memory must be strictly partitioned, tracked, and allocated with page-locked policies to avoid swapping.

---

## 1. Physical Memory Partitioning

The 12 GB system RAM budget is divided into three fixed, non-overlapping regions:

| Memory Region | Allocation Policy | Target Size | Linux API / Primitive |
| :--- | :--- | :--- | :--- |
| **System & Router OS Overhead** | Default virtual allocation | ~2 GB | Standard malloc/new |
| **Key-Value (KV) Cache** | Page-locked, aligned, non-pageable | ~2 GB | `posix_memalign`, `mlock` |
| **Expert Cache RAM Pool** | Fixed physical pool, LRU-managed | ~8 GB | `mmap` with size bounds and `munmap` |

```
+-----------------------------------------------------------------------------------+
|                              TOTAL RAM BUDGET (12 GB)                             |
+----------------------+-----------------------------+------------------------------+
|   OS & Router (2 GB) |       KV Cache (2 GB)       |    Expert Cache Pool (8 GB)  |
|   (Virtual Alloc)    |   (Page-Locked & Aligned)   |   (LRU Mapped File Handlers) |
+----------------------+-----------------------------+------------------------------+
                                      |                              ^
                                      v                              |
                         +-----------------------+        +----------+----------+
                         | Page table mapping    |        | mmap / madvise      |
                         | locked to RAM.        |        | (Zero-copy disk    |
                         | Never swapped to disk.|        | paging on demand)   |
                         +-----------------------+        +----------+----------+
                                                                     |
                                                                     v
                                                          +---------------------+
                                                          | NVMe SSD (25 GB)    |
                                                          | experts/expert_*.bin|
                                                          +---------------------+
```

---

## 2. KV Cache Allocation

The KV Cache holds key-value tensors for multi-head self-attention. If the KV cache is swapped to disk by the Linux kernel's Virtual Memory Manager, token throughput drops to zero.
* **Alignment**: Allocated at page boundaries (4096-byte boundaries) using `posix_memalign` to optimize vector instructions (AVX-512 / AVX2).
* **Page Locking**: Locked to physical RAM using `mlock`. This tells the Linux kernel that these pages must never be written to swap space under memory pressure.
* **API Details**:
  ```cpp
  void* kv_cache_ptr = nullptr;
  int ret = posix_memalign(&kv_cache_ptr, 4096, KV_CACHE_SIZE_BYTES);
  if (ret == 0) {
      mlock(kv_cache_ptr, KV_CACHE_SIZE_BYTES);
  }
  ```

---

## 3. Expert Cache (LRU File Mapping)

Instead of allocating memory dynamically for every expert load, the engine uses **Memory Mapping (`mmap`)** combined with kernel paging advisories.

### 3.1 Zero-Copy Expert Loading
* Each expert's weight matrix is mapped directly into the virtual address space using `mmap`.
* Memory mapping doesn't load the file content into RAM immediately; it reserves the virtual memory address range.
* By calling `madvise(addr, length, MADV_WILLNEED)` and `readahead()`, we hint to the kernel to read pages from the NVMe SSD into the page cache asynchronously.
* When the Compute Thread accesses the memory-mapped pointer, the pages are already present in RAM (Cache Hit), avoiding synchronous page-fault blocking.

### 3.2 Expert Eviction & Unmapping
When the total virtual mappings size of active experts exceeds the 8 GB expert cache budget:
1. The LRU scheduler identifies the least recently accessed, unpinned expert.
2. The mapping is unmapped using `munmap`.
3. The kernel immediately drops these pages from physical memory (since they are read-only mappings backed by files on SSD).
4. Unmapping frees physical RAM without generating any disk write I/O.
5. Pinned experts (those currently active in the current layer of the execution engine) are protected against eviction by an atomic lock counter.
