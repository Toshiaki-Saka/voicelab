// voicelab/src/core/features.cpp

#include "voicelab/core/features.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace voicelab::core {

void magnitude(std::span<const std::complex<float>> spec,
               std::span<float> out) noexcept {
    const std::size_t n = std::min(spec.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = std::abs(spec[i]);
}

void power(std::span<const std::complex<float>> spec,
           std::span<float> out) noexcept {
    const std::size_t n = std::min(spec.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = spec[i];
        out[i] = c.real() * c.real() + c.imag() * c.imag();
    }
}

float spectral_centroid_hz(std::span<const float> mags,
                           std::uint32_t          sr,
                           std::size_t            frame_size) noexcept {
    if (mags.empty() || frame_size == 0) return 0.0F;
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < mags.size(); ++k) {
        const double m = mags[k];
        num += m * static_cast<double>(k);
        den += m;
    }
    if (den <= 0.0) return 0.0F;
    return static_cast<float>(num / den * sr / static_cast<double>(frame_size));
}

float spectral_flatness(std::span<const float> mags) noexcept {
    if (mags.size() < 2) return 0.0F;
    double log_sum = 0.0, lin_sum = 0.0;
    std::size_t n = 0;
    // Skip DC bin (index 0): it carries no useful tonality info.
    for (std::size_t k = 1; k < mags.size(); ++k) {
        const double p = static_cast<double>(mags[k]) * mags[k] + 1e-20;
        log_sum += std::log(p);
        lin_sum += p;
        ++n;
    }
    if (n == 0 || lin_sum <= 0.0) return 0.0F;
    const double geom = std::exp(log_sum / static_cast<double>(n));
    const double arith = lin_sum / static_cast<double>(n);
    return static_cast<float>(geom / arith);
}

// -------- mel filterbank -------------------------------------------------

MelFilterBank::MelFilterBank(Config cfg) : cfg_(cfg) {
    if (cfg_.f_max <= 0.0F) {
        cfg_.f_max = static_cast<float>(cfg_.sample_rate) / 2.0F;
    }
    if (cfg_.num_filters == 0 || cfg_.frame_size == 0) {
        throw std::invalid_argument("MelFilterBank: invalid configuration");
    }

    const std::size_t n_bins = num_bins();
    const float mel_min = hz_to_mel(cfg_.f_min);
    const float mel_max = hz_to_mel(cfg_.f_max);
    const std::size_t M = cfg_.num_filters;

    // Triangular filter centers in mel-space, then map back to Hz, then to bin.
    std::vector<float> bin_edges(M + 2);
    for (std::size_t m = 0; m < M + 2; ++m) {
        const float mel = mel_min + (mel_max - mel_min) *
                          static_cast<float>(m) / static_cast<float>(M + 1);
        const float hz = mel_to_hz(mel);
        const float bin = hz * static_cast<float>(cfg_.frame_size) /
                          static_cast<float>(cfg_.sample_rate);
        bin_edges[m] = bin;
    }

    starts_.assign(M, 0);
    weights_.assign(M, {});

    for (std::size_t m = 0; m < M; ++m) {
        const float left   = bin_edges[m];
        const float center = bin_edges[m + 1];
        const float right  = bin_edges[m + 2];

        const auto k_lo = static_cast<std::size_t>(std::ceil(left));
        const auto k_hi = static_cast<std::size_t>(std::floor(right));
        const std::size_t lo = std::min(k_lo, n_bins);
        const std::size_t hi = std::min(k_hi + 1, n_bins);
        starts_[m] = lo;

        if (hi <= lo) {
            // Edge case: filter too narrow at this resolution.
            weights_[m].clear();
            continue;
        }
        weights_[m].resize(hi - lo);
        for (std::size_t k = lo; k < hi; ++k) {
            const float kk = static_cast<float>(k);
            float w = 0.0F;
            if (kk <= center && center > left) {
                w = (kk - left) / (center - left);
            } else if (kk > center && right > center) {
                w = (right - kk) / (right - center);
            }
            weights_[m][k - lo] = std::max(0.0F, w);
        }
    }
}

void MelFilterBank::apply(std::span<const float> pwr,
                          std::span<float> out) const noexcept {
    const std::size_t M = cfg_.num_filters;
    for (std::size_t m = 0; m < M && m < out.size(); ++m) {
        float acc = 0.0F;
        const auto& w = weights_[m];
        const std::size_t s = starts_[m];
        for (std::size_t i = 0; i < w.size() && (s + i) < pwr.size(); ++i) {
            acc += w[i] * pwr[s + i];
        }
        out[m] = acc;
    }
}

void log_compress(std::span<float> x, float eps) noexcept {
    for (auto& v : x) v = std::log(eps + v);
}

void dct2_truncated(std::span<const float> in, std::span<float> out) noexcept {
    // Orthonormal DCT-II:
    //   X[k] = alpha[k] * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))
    //   alpha[0] = sqrt(1/N),  alpha[k>0] = sqrt(2/N)
    const std::size_t N = in.size();
    const std::size_t K = std::min(out.size(), N);
    if (N == 0) return;
    const double pi = std::numbers::pi;
    const double Nd = static_cast<double>(N);
    for (std::size_t k = 0; k < K; ++k) {
        double sum = 0.0;
        for (std::size_t n = 0; n < N; ++n) {
            sum += static_cast<double>(in[n]) *
                   std::cos(pi * (2.0 * static_cast<double>(n) + 1.0) *
                            static_cast<double>(k) / (2.0 * Nd));
        }
        const double alpha = (k == 0) ? std::sqrt(1.0 / Nd)
                                      : std::sqrt(2.0 / Nd);
        out[k] = static_cast<float>(alpha * sum);
    }
}

}  // namespace voicelab::core
