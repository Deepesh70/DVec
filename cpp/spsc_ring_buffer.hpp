#pragma once

#include<atomic>
#include<cstddef>
#include<cstdint>

template <typename T, std::size_t Capacity>

class SPSCRingBuffer {
    
    static_assert((Capacity > 0) && ((Capacity & (Capacity -1)) == 0),
                "Capacity must be a power of 2");


private:
    T buffer_[Capacity];

    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::atomic<std::size_t> head_{0};

public:
    SPSCRingBuffer() = default;

    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    bool Push(const T* items , std::size_t count) noexcept {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t current_head = head_.load(std::memory_order_acquire);

        if(Capacity - (current_tail - current_head) < count){
            return false;
        }

        for(std::size_t i =0; i< count; ++i){
            buffer_[(current_tail + i) & (Capacity -1)] = items[i];
        }

        tail_.store(current_tail + count, std::memory_order_release);
        return true;
    }

    bool Pop(T* destination, std::size_t count) noexcept{
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);

        if(current_tail - current_head < count){
            return false;
        }

        for(std::size_t i =0; i< count; ++i){
            destination[i] = buffer_[(current_head + i) & (Capacity - 1)];
        }
        
        head_.store(current_head + count, std::memory_order_release);
        return true;
    }
    std::size_t AvailableRead() const noexcept {
         const std::size_t current_tail = tail_.load(std::memory_order_acquire);
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        return current_tail - current_head;
    }
     std::size_t AvailableWrite() const noexcept {
        return Capacity - AvailableRead();
    }
};

