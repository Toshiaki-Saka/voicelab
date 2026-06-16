#pragma once
//
// voicelab/core/stft.hpp
//
// Streaming Short-Time Fourier Transform.
//
// Theory recap
// ------------
// Given a real signal x[n], the STFT is
//
//     X[m, k] = sum_{n=0}^{N-1} w[n] * x[m*H + n] * exp(-j*2*pi*k*n/N)
//
// where N is the frame size, H is the hop size, w[n] is the analysis
// window. We emit N/2 + 1 complex bins per frame (the non-redundant half
// of the real FFT).
//
// Streaming API
// -------------
// Audio arrives in arbitrary block sizes (driven by the OS / RtAudio).
// We accumulate samples in an internal ring; whenever at least N samples
// are available, we window, FFT, and emit one frame, then advance by H.
// Allocation happens only in the constructor.
//
// The callback receives a non-owning `std::span<const std::complex<float>>`
// of length `num_bins()`. The span points into an internal buffer and is
// invalidated after the callback returns; copy if you need to keep it.

#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "voicelab/core/window.hpp"

namespace voicelab::core {

class StreamingStft {
public:
    using sample_t  = float;
    using complex_t = std::complex<float>;

    struct Config {
        std::size_t   frame_size  = 1024;   // N (power of 2 recommended)
        std::size_t   hop_size    = 256;    // H (typically N/4)
        std::uint32_t sample_rate = 44100;
        WindowType    window_type = WindowType::Hann;
    };

    explicit StreamingStft(Config cfg);

    // Accepts an arbitrary block of new samples, invokes `cb` once per
    // complete frame produced. `cb` is `void(std::span<const complex_t>)`.
    template <std::invocable<std::span<const complex_t>> Cb>
    void process(std::span<const sample_t> input, Cb&& cb) {
        for (auto s : input) {
            ring_[write_pos_] = s;
            write_pos_ = (write_pos_ + 1) % ring_.size();
            ++samples_since_last_frame_;
            ++filled_;
            if (filled_ >= cfg_.frame_size &&
                samples_since_last_frame_ >= cfg_.hop_size) {
                emit_frame();
                cb(std::span<const complex_t>{spectrum_});
                samples_since_last_frame_ = 0;
            }
        }
    }

    // Type-erased variant — convenient for crossing translation units.
    using FrameCallback = std::function<void(std::span<const complex_t>)>;
    void process(std::span<const sample_t> input, const FrameCallback& cb) {
        process<const FrameCallback&>(input, cb);
    }

    [[nodiscard]] const Config& config()  const noexcept { return cfg_; }
    [[nodiscard]] std::size_t num_bins() const noexcept {
        return cfg_.frame_size / 2 + 1;
    }
    [[nodiscard]] float bin_to_hz(std::size_t bin) const noexcept {
        return static_cast<float>(bin)
             * static_cast<float>(cfg_.sample_rate)
             / static_cast<float>(cfg_.frame_size);
    }

private:
    void emit_frame();

    Config                 cfg_;
    std::vector<sample_t>  window_;
    std::vector<sample_t>  ring_;
    std::vector<sample_t>  frame_;     // windowed real frame (length N)
    std::vector<complex_t> spectrum_;  // length N/2 + 1
    std::size_t            write_pos_ = 0;
    std::size_t            filled_    = 0;  // saturates at frame_size
    std::size_t            samples_since_last_frame_ = 0;
};

}  // namespace voicelab::core
