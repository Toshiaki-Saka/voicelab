// voicelab/src/core/stft.cpp
//
// Streaming STFT implementation. The actual FFT call is delegated to
// pocketfft, which gives us a clean, header-only, BSD-licensed dependency.

#include "voicelab/core/stft.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#include "pocketfft_hdronly.h"

namespace voicelab::core {

StreamingStft::StreamingStft(Config cfg) : cfg_(cfg) {
    if (cfg_.frame_size == 0 || cfg_.hop_size == 0 ||
        cfg_.hop_size > cfg_.frame_size) {
        throw std::invalid_argument("StreamingStft: invalid frame/hop size");
    }
    window_.resize(cfg_.frame_size);
    make_window<float>(cfg_.window_type, window_);

    ring_.assign(cfg_.frame_size, 0.0F);
    frame_.assign(cfg_.frame_size, 0.0F);
    spectrum_.assign(num_bins(), {});
}

void StreamingStft::emit_frame() {
    const std::size_t N = cfg_.frame_size;

    // Copy from circular ring into a contiguous frame, applying window.
    // The newest sample sits at index (write_pos_ - 1) mod N; the start
    // of the current frame is at index `write_pos_` (oldest of N).
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t idx = (write_pos_ + n) % N;
        frame_[n] = ring_[idx] * window_[n];
    }

    // Real-to-complex FFT via pocketfft.
    pocketfft::shape_t  shape  {N};
    pocketfft::stride_t stride_in  {static_cast<long>(sizeof(float))};
    pocketfft::stride_t stride_out {static_cast<long>(sizeof(std::complex<float>))};
    pocketfft::shape_t  axes   {0};

    pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                   frame_.data(), spectrum_.data(), 1.0F);
}

}  // namespace voicelab::core
