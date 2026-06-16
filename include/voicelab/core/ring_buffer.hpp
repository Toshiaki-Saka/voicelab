#pragma once
//
// voicelab/core/ring_buffer.hpp
//
// A simple monaural ring buffer specialized for streaming audio analysis.
//
// Design notes:
//   * Pre-allocated, fixed capacity. No allocation after construction.
//   * Single producer / single consumer is the intended pattern, but this
//     class itself is NOT thread-safe — wrap with a queue/lock if you
//     cross threads.
//   * Capacity is rounded up to a power of two so that index wrapping is
//     a bitmask, which is friendly to the audio callback's tight budget.

#include <bit>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace voicelab::core {

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t min_capacity)
        : mask_{round_up_pow2(min_capacity) - 1},
          data_(round_up_pow2(min_capacity), T{}) {}

    [[nodiscard]] std::size_t capacity() const noexcept { return data_.size(); }
    [[nodiscard]] std::size_t size()     const noexcept { return size_; }
    [[nodiscard]] bool        empty()    const noexcept { return size_ == 0; }

    // Push `in` samples. If the ring is full, the oldest samples are
    // overwritten (drop-oldest policy) — appropriate for analysis pipelines
    // where falling behind means losing the recent past, not the present.
    void push(std::span<const T> in) noexcept {
        for (auto v : in) {
            data_[write_ & mask_] = v;
            ++write_;
            if (size_ < data_.size()) {
                ++size_;
            } else {
                ++read_;  // overwrite: advance read head
            }
        }
    }

    // Pop up to `out.size()` samples into `out`. Returns the number copied.
    [[nodiscard]] std::size_t pop(std::span<T> out) noexcept {
        const std::size_t n = std::min(out.size(), size_);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = data_[(read_ + i) & mask_];
        }
        read_ += n;
        size_ -= n;
        return n;
    }

    // Peek the oldest `out.size()` samples without consuming them.
    [[nodiscard]] std::size_t peek(std::span<T> out) const noexcept {
        const std::size_t n = std::min(out.size(), size_);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = data_[(read_ + i) & mask_];
        }
        return n;
    }

    // Discard `n` oldest samples. No-op if n > size().
    void discard(std::size_t n) noexcept {
        n = std::min(n, size_);
        read_ += n;
        size_ -= n;
    }

private:
    static std::size_t round_up_pow2(std::size_t x) noexcept {
        if (x < 2) return 2;
        return std::bit_ceil(x);
    }

    std::size_t        mask_;
    std::vector<T>     data_;
    std::size_t        size_  = 0;
    std::size_t        read_  = 0;
    std::size_t        write_ = 0;
};

}  // namespace voicelab::core
