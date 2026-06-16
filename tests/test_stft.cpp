#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "voicelab/core/stft.hpp"

using namespace voicelab::core;

namespace {
std::vector<float> sine(float freq_hz, std::uint32_t sr, std::size_t n) {
    std::vector<float> s(n);
    for (std::size_t i = 0; i < n; ++i) {
        s[i] = std::sin(2.0F * std::numbers::pi_v<float> * freq_hz *
                        static_cast<float>(i) / static_cast<float>(sr));
    }
    return s;
}
}  // namespace

TEST_CASE("StreamingStft: peak of a sine lands on the right bin", "[stft]") {
    constexpr std::uint32_t sr = 44100;
    constexpr std::size_t   N  = 4096;
    constexpr float         f0 = 1000.0F;

    StreamingStft stft({.frame_size = N, .hop_size = N / 4,
                        .sample_rate = sr});
    auto buf = sine(f0, sr, N * 2);

    bool got = false;
    stft.process(buf, [&](std::span<const std::complex<float>> spec) {
        if (got) return;
        std::size_t peak = 0;
        float       best = 0.0F;
        for (std::size_t k = 1; k < spec.size(); ++k) {
            const float m = std::abs(spec[k]);
            if (m > best) { best = m; peak = k; }
        }
        // Bin resolution at N=4096, sr=44100 is ~10.77 Hz; allow 2 bins.
        const float peak_hz = stft.bin_to_hz(peak);
        REQUIRE(std::abs(peak_hz - f0) < 22.0F);
        got = true;
    });
    REQUIRE(got);
}

TEST_CASE("StreamingStft: hop matches frame count", "[stft]") {
    constexpr std::size_t   N  = 1024;
    constexpr std::size_t   H  = 256;
    StreamingStft stft({.frame_size = N, .hop_size = H,
                        .sample_rate = 44100});

    std::vector<float> input(N + H * 4, 0.1F);
    std::size_t frames = 0;
    stft.process(input, [&](std::span<const std::complex<float>>) {
        ++frames;
    });
    // First frame is emitted as soon as we have N samples (after H steps
    // past startup), then one frame per H samples thereafter.
    // We had N + 4H samples in the ring => expect 5 frames.
    REQUIRE(frames == 5);
}

TEST_CASE("StreamingStft: throws on invalid config", "[stft]") {
    REQUIRE_THROWS_AS(
        StreamingStft({.frame_size = 0, .hop_size = 4}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        StreamingStft({.frame_size = 8, .hop_size = 16}),
        std::invalid_argument);
}
