#pragma once
//
// voicelab/core/istft.hpp
//
// Streaming Inverse Short-Time Fourier Transform via Overlap-Add (OLA).
//
// To reconstruct x[n] from frames X[m, k], we
//   1. inverse-FFT each frame → real time-domain frame y_m[n] of length N
//   2. apply a synthesis window w_s[n] (often same as analysis window)
//   3. add into the output stream at offset m*H
//
// Perfect reconstruction requires the constant-overlap-add (COLA)
// condition: sum_m w_a[n - mH] * w_s[n - mH] == constant for all n.
//
// With a Hann analysis window of length N, choosing the synthesis window
// also Hann and a hop H = N/4 gives COLA with constant 1.5; we divide by
// it to normalize. (For H = N/2 with sqrt-Hann analysis & synthesis you
// get a constant of 1.0 — also fine.)
//
// This class produces output in arbitrary block sizes via `pop()`,
// independent of frame boundaries.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "voicelab/core/ring_buffer.hpp"
#include "voicelab/core/window.hpp"

namespace voicelab::core {

class StreamingIstft {
public:
    using sample_t  = float;
    using complex_t = std::complex<float>;

    struct Config {
        std::size_t   frame_size      = 1024;
        std::size_t   hop_size        = 256;
        std::uint32_t sample_rate     = 44100;
        WindowType    synthesis_window = WindowType::Hann;
        // The analysis window used upstream — required to compute the
        // COLA normalization factor.
        WindowType    analysis_window  = WindowType::Hann;
    };

    explicit StreamingIstft(Config cfg);

    // Add one new spectrum frame (length = N/2 + 1).
    void push_frame(std::span<const complex_t> spectrum);

    // Pull up to `out.size()` reconstructed samples; returns count copied.
    [[nodiscard]] std::size_t pop(std::span<sample_t> out) noexcept;

    [[nodiscard]] std::size_t available() const noexcept;
    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

private:
    Config                 cfg_;
    std::vector<sample_t>  win_synth_;
    std::vector<complex_t> spec_in_;     // padded if necessary
    std::vector<sample_t>  time_frame_;  // IFFT output, length N
    // OLA accumulator. We keep one frame's worth ahead; samples older than
    // (frame_size - hop_size) are finalized and pushed to `out_ring_`.
    std::vector<sample_t>  ola_;
    std::size_t            ola_write_offset_ = 0;
    RingBuffer<sample_t>   out_ring_;
    float                  cola_gain_ = 1.0F;
};

}  // namespace voicelab::core
