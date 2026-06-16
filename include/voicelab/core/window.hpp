#pragma once
//
// voicelab/core/window.hpp
//
// Window functions for short-time analysis.
//
// All windows are real-valued, symmetric, length-N sequences w[n], 0<=n<N.
// We use the *periodic* form (sometimes called the "DFT-even" form) by
// dividing by N rather than N-1. This is what you want for STFT analysis:
// it makes the windowed frame join smoothly when the frame is treated as
// one period of a periodic signal, which matters for invertibility under
// constant-overlap-add (COLA).
//
// Reference: Heinzel, Rüdiger, Schilling, "Spectrum and spectral density
// estimation by the Discrete Fourier transform (DFT) ...", Max-Planck-
// Institut für Gravitationsphysik, 2002.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace voicelab::core {

enum class WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
};

namespace detail {
constexpr double k_two_pi  = 2.0 * std::numbers::pi_v<double>;
constexpr double k_four_pi = 4.0 * std::numbers::pi_v<double>;
constexpr double k_six_pi  = 6.0 * std::numbers::pi_v<double>;
}  // namespace detail

// Fill `out` with a window of type `type`. `out.size()` is N.
template <typename Float>
void make_window(WindowType type, std::span<Float> out) noexcept {
    const auto N = out.size();
    if (N == 0) return;
    const auto Nd = static_cast<double>(N);
    for (std::size_t n = 0; n < N; ++n) {
        const double x = static_cast<double>(n) / Nd;  // periodic form
        double w = 1.0;
        switch (type) {
        case WindowType::Rectangular:
            w = 1.0;
            break;
        case WindowType::Hann:
            w = 0.5 - 0.5 * std::cos(detail::k_two_pi * x);
            break;
        case WindowType::Hamming:
            w = 0.54 - 0.46 * std::cos(detail::k_two_pi * x);
            break;
        case WindowType::Blackman:
            w = 0.42
              - 0.5  * std::cos(detail::k_two_pi  * x)
              + 0.08 * std::cos(detail::k_four_pi * x);
            break;
        case WindowType::BlackmanHarris:
            w = 0.35875
              - 0.48829 * std::cos(detail::k_two_pi  * x)
              + 0.14128 * std::cos(detail::k_four_pi * x)
              - 0.01168 * std::cos(detail::k_six_pi  * x);
            break;
        }
        out[n] = static_cast<Float>(w);
    }
}

template <typename Float = float>
[[nodiscard]] std::vector<Float> make_window(WindowType type, std::size_t N) {
    std::vector<Float> w(N);
    make_window<Float>(type, w);
    return w;
}

}  // namespace voicelab::core
