// examples/01_sine_spectrum.cpp
//
// Generate a 440 Hz sine wave, run it through the streaming STFT, and
// print the peak frequency of the first emitted frame. This is the
// "Hello, world" of the library and requires no audio hardware.

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <vector>

#include "voicelab/core/stft.hpp"

int main() {
    using namespace voicelab::core;

    constexpr std::uint32_t sr  = 44100;
    constexpr float         f0  = 440.0F;
    constexpr std::size_t   N   = 2048;

    StreamingStft stft({.frame_size = N, .hop_size = N / 4, .sample_rate = sr});

    std::vector<float> sine(N);
    for (std::size_t n = 0; n < N; ++n) {
        sine[n] = 0.5F * std::sin(2.0F * std::numbers::pi_v<float> *
                                  f0 * static_cast<float>(n) /
                                  static_cast<float>(sr));
    }

    bool printed = false;
    stft.process(sine, [&](std::span<const std::complex<float>> spec) {
        if (printed) return;
        std::size_t peak = 0;
        float       best = 0.0F;
        for (std::size_t k = 0; k < spec.size(); ++k) {
            const float m = std::abs(spec[k]);
            if (m > best) { best = m; peak = k; }
        }
        std::printf("peak bin = %zu, peak freq = %.2f Hz (expected ~%.2f)\n",
                    peak, static_cast<double>(stft.bin_to_hz(peak)),
                    static_cast<double>(f0));
        printed = true;
    });
    return 0;
}
