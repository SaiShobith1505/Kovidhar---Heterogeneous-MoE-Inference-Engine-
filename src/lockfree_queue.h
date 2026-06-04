#ifndef LOCKFREE_QUEUE_H
#define LOCKFREE_QUEUE_H

#include <atomic>
#include <vector>
#include <cstddef>
#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#endif

inline void yield_processor() {
#ifdef _WIN32
    YieldProcessor();
#else
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
#endif
}

template <typename T, size_t Capacity>
class LockFreeSPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2 for fast masking");
public:
    LockFreeSPSCQueue() : head_(0), tail_(0) {
        ring_buffer_.resize(Capacity);
    }

    // Disable copy/move to preserve atomic guarantees and alignment
    LockFreeSPSCQueue(const LockFreeSPSCQueue&) = delete;
    LockFreeSPSCQueue& operator=(const LockFreeSPSCQueue&) = delete;
    LockFreeSPSCQueue(LockFreeSPSCQueue&&) = delete;
    LockFreeSPSCQueue& operator=(LockFreeSPSCQueue&&) = delete;

    ~LockFreeSPSCQueue() = default;

    /**
     * @brief Pushes an item into the queue. Only the producer thread may call this.
     * @param item Item to copy.
     * @return true if successful, false if the queue is full.
     */
    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if ((current_head - current_tail) >= Capacity) {
            return false; // Queue is full
        }

        ring_buffer_[current_head & (Capacity - 1)] = item;
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pops an item from the queue. Only the consumer thread may call this.
     * @param item Reference to store the popped item.
     * @return true if successful, false if the queue is empty.
     */
    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if (current_tail == current_head) {
            return false; // Queue is empty
        }

        item = ring_buffer_[current_tail & (Capacity - 1)];
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Estimates the current size of the queue.
     */
    size_t size() const {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_relaxed);
        return (h >= t) ? (h - t) : 0;
    }

    /**
     * @brief Checks if the queue is empty.
     */
    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

private:
    // Align head and tail to 64-byte boundaries (standard L1 cache line size) to prevent false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    
    // The ring buffer vector itself
    std::vector<T> ring_buffer_;
};

#endif // LOCKFREE_QUEUE_H
