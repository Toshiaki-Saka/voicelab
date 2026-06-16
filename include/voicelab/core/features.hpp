#pragma once
//
// voicelab/core/features.hpp
//
// Spectral and mel-frequency features computed from STFT frames.
//
//   * magnitude / power     : trivial element-wise transforms
//   * spectral centroid     : amplitude-weighted mean frequency
//   * spectral flatness     : geometric mean / arithmetic mean of power
//   * mel filterbank        : triangular filters on the mel scale
//   * log-mel               : log(1 + mel) — standard ASR front-end input
//   * MFCC                  : DCT-II of log-mel, keep first K coefficients
//
// All routines write into caller-provided spans so they can run from an
// audio callback without allocating.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace voicelab::core {

// Element-wise magnitude. out.size() must equal spec.size().
void magnitude(std::span<const std::complex<float>> spec,
               std::span<float>                     out) noexcept;

// Element-wise power = |X|^2.
void power(std::span<const std::complex<float>> spec,
           std::span<float>                     out) noexcept;

// Spectral centroid in Hz. magnitudes.size() must equal num_bins;
// num_bins == frame_size / 2 + 1.
[[nodiscard]] float spectral_centroid_hz(std::span<const float> magnitudes,
                                         std::uint32_t          sample_rate,
                                         std::size_t            frame_size) noexcept;

// Spectral flatness in [0, 1]. 1 means pure noise (flat spectrum); near 0
// means tonal. Skips DC bin.
[[nodiscard]] float spectral_flatness(std::span<const float> magnitudes) noexcept;

// Convert Hz <-> mel (Slaney / HTK style; we use the O'Shaughnessy form).
[[nodiscard]] inline float hz_to_mel(float hz) noexcept {
    return 2595.0F * std::log10(1.0F + hz / 700.0F);
}
[[nodiscard]] inline float mel_to_hz(float mel) noexcept {
    return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F);
}

// A precomputed mel filterbank. Owns its triangular filter weights.
class MelFilterBank {
public:
    struct Config {
        std::size_t   frame_size  = 1024;
        std::uint32_t sample_rate = 44100;
        std::size_t   num_filters = 40;
        float         f_min       = 0.0F;
        float         f_max       = 0.0F;  // 0 == sample_rate / 2
    };

    explicit MelFilterBank(Config cfg);

    [[nodiscard]] std::size_t num_filters() const noexcept { return cfg_.num_filters; }
    [[nodiscard]] std::size_t num_bins()    const noexcept { return cfg_.frame_size / 2 + 1; }
    [[nodiscard]] const Config& config()    const noexcept { return cfg_; }

    // Apply: out[m] = sum_k filter[m][k] * power_spectrum[k].
    // power_spectrum.size() must equal num_bins(); out.size() == num_filters().
    void apply(std::span<const float> power_spectrum,
               std::span<float>       out) const noexcept;

private:
    Config             cfg_;
    // Sparse representation: for each filter, (start_bin, weights).
    std::vector<std::size_t>           starts_;
    std::vector<std::vector<float>>    weights_;
};

// Compute log(eps + x) element-wise. Useful for log-mel.
void log_compress(std::span<float> in_out, float eps = 1e-10F) noexcept;

// DCT-II in-place along the input. Keeps first `out.size()` coefficients
// of the orthonormal DCT-II (matches scipy's `dct(..., norm="ortho")`).
// `scratch` must have at least `in.size()` floats.
void dct2_truncated(std::span<const float> in,
                    std::span<float>       out) noexcept;

}  // namespace voicelab::core
