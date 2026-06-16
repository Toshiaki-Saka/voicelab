// voicelab/src/core/istft.cpp

#include "voicelab/core/istft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "pocketfft_hdronly.h"

namespace voicelab::core {

namespace {
// Compute the COLA sum for an N-length analysis*synthesis window pair at
// the given hop. For the standard Hann/Hann with H = N/4 this returns 1.5;
// we divide each output sample by it to obtain unit-gain reconstruction.
float cola_constant(std::span<const float> w_a,
                    std::span<const float> w_s,
                    std::size_t            hop) {
    const std::size_t N = w_a.size();
    // The COLA sum is periodic with period hop; evaluate at n=0..hop-1
    // and take the max (it should be constant for a true COLA window).
    float max_sum = 0.0F;
    for (std::size_t n = 0; n < hop; ++n) {
        float s = 0.0F;
        for (std::size_t k = 0;; ++k) {
            const std::ptrdiff_t idx =
                static_cast<std::ptrdiff_t>(n) -
                static_cast<std::ptrdiff_t>(k * hop);
            if (idx < 0) break;
            if (static_cast<std::size_t>(idx) >= N) continue;
            s += w_a[static_cast<std::size_t>(idx)] *
                 w_s[static_cast<std::size_t>(idx)];
        }
        for (std::size_t k = 1;; ++k) {
            const std::size_t idx = n + k * hop;
            if (idx >= N) break;
            s += w_a[idx] * w_s[idx];
        }
        if (s > max_sum) max_sum = s;
    }
    return max_sum > 0.0F ? max_sum : 1.0F;
}
}  // namespace

StreamingIstft::StreamingIstft(Config cfg)
    : cfg_(cfg),
      win_synth_(cfg.frame_size),
      spec_in_(cfg.frame_size / 2 + 1),
      time_frame_(cfg.frame_size),
      // OLA buffer needs to hold at least `frame_size + hop_size`; we use 2N.
      ola_(2 * cfg.frame_size, 0.0F),
      out_ring_(4 * cfg.frame_size) {
    if (cfg_.frame_size == 0 || cfg_.hop_size == 0 ||
        cfg_.hop_size > cfg_.frame_size) {
        throw std::invalid_argument("StreamingIstft: invalid frame/hop size");
    }
    make_window<float>(cfg_.synthesis_window, win_synth_);

    auto w_a = make_window<float>(cfg_.analysis_window, cfg_.frame_size);
    cola_gain_ = cola_constant(w_a, win_synth_, cfg_.hop_size);
}

void StreamingIstft::push_frame(std::span<const std::complex<float>> spectrum) {
    if (spectrum.size() != spec_in_.size()) {
        throw std::invalid_argument(
            "StreamingIstft: spectrum length mismatch");
    }
    std::copy(spectrum.begin(), spectrum.end(), spec_in_.begin());

    const std::size_t N = cfg_.frame_size;

    pocketfft::shape_t  shape  {N};
    pocketfft::stride_t stride_in  {static_cast<long>(sizeof(std::complex<float>))};
    pocketfft::stride_t stride_out {static_cast<long>(sizeof(float))};
    pocketfft::shape_t  axes   {0};

    // c2r in pocketfft applies the inverse and we ask it to multiply by 1/N.
    pocketfft::c2r(shape, stride_in, stride_out, axes, pocketfft::BACKWARD,
                   spec_in_.data(), time_frame_.data(),
                   1.0F / static_cast<float>(N));

    // Overlap-add into OLA buffer.
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t idx = ola_write_offset_ + n;
        ola_[idx % ola_.size()] += time_frame_[n] * win_synth_[n] / cola_gain_;
    }

    // Push `hop_size` finalized samples to the output ring.
    for (std::size_t i = 0; i < cfg_.hop_size; ++i) {
        const std::size_t idx = (ola_write_offset_ + i) % ola_.size();
        const float v = ola_[idx];
        ola_[idx] = 0.0F;  // clear so next OLA accumulates from zero
        out_ring_.push(std::span<const float>{&v, 1});
    }
    ola_write_offset_ = (ola_write_offset_ + cfg_.hop_size) % ola_.size();
}

std::size_t StreamingIstft::pop(std::span<float> out) noexcept {
    return out_ring_.pop(out);
}

std::size_t StreamingIstft::available() const noexcept {
    return out_ring_.size();
}

}  // namespace voicelab::core
