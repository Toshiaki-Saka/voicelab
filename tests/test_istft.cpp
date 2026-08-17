#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "voicelab/core/istft.hpp"
#include "voicelab/core/stft.hpp"

using namespace voicelab::core;

// ASCII "->" on purpose: catch_discover_tests registers this string with ctest
// and then passes it back as a filter, so a non-ASCII test name only survives if
// every hop (compiler, console, ctest) agrees on the encoding. /utf-8 makes that
// true for MSVC, but there is no reason to depend on it for a test name.
TEST_CASE("STFT -> ISTFT round-trip reconstructs sine wave (modulo "
          "boundary)", "[istft]") {
    constexpr std::uint32_t sr = 16000;
    constexpr std::size_t   N  = 1024;
    constexpr std::size_t   H  = N / 4;
    constexpr float         f0 = 440.0F;

    StreamingStft  fwd({.frame_size = N, .hop_size = H, .sample_rate = sr});
    StreamingIstft inv({.frame_size = N, .hop_size = H, .sample_rate = sr});

    const std::size_t total = N * 8;
    std::vector<float> in(total);
    for (std::size_t i = 0; i < total; ++i) {
        in[i] = std::sin(2.0F * std::numbers::pi_v<float> * f0 *
                         static_cast<float>(i) / static_cast<float>(sr));
    }

    std::vector<float> out;
    out.reserve(total);
    std::vector<float> chunk(256);

    fwd.process(in, [&](std::span<const std::complex<float>> spec) {
        inv.push_frame(spec);
        std::size_t got = 0;
        do {
            got = inv.pop(chunk);
            out.insert(out.end(), chunk.begin(),
                       chunk.begin() + static_cast<std::ptrdiff_t>(got));
        } while (got > 0);
    });

    // The first ~N samples are warm-up (OLA not yet filled). Compare
    // a middle window in input vs output.
    REQUIRE(out.size() >= 4 * N);
    double err = 0.0;
    const std::size_t start = 2 * N;
    const std::size_t end   = start + 2 * N;
    for (std::size_t i = start; i < end; ++i) {
        const double d = static_cast<double>(out[i] - in[i]);
        err += d * d;
    }
    err = std::sqrt(err / static_cast<double>(end - start));
    REQUIRE(err < 0.05);
}
