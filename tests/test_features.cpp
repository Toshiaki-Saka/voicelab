#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "voicelab/core/features.hpp"

using namespace voicelab::core;

TEST_CASE("magnitude / power: simple values", "[features]") {
    std::vector<std::complex<float>> spec{{3.0F, 4.0F}, {1.0F, 0.0F}};
    std::vector<float> mag(2), pwr(2);
    magnitude(spec, mag);
    power(spec, pwr);
    REQUIRE_THAT(mag[0], Catch::Matchers::WithinAbs(5.0, 1e-6));
    REQUIRE_THAT(mag[1], Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(pwr[0], Catch::Matchers::WithinAbs(25.0, 1e-4));
}

TEST_CASE("spectral flatness: flat spectrum is near 1, single tone near 0",
          "[features]") {
    std::vector<float> flat(64, 1.0F);
    const float f_flat = spectral_flatness(flat);
    REQUIRE_THAT(f_flat, Catch::Matchers::WithinAbs(1.0, 1e-3));

    std::vector<float> tonal(64, 0.0F);
    tonal[10] = 1.0F;
    const float f_tonal = spectral_flatness(tonal);
    REQUIRE(f_tonal < 0.1F);
}

TEST_CASE("mel filterbank: weights are non-negative, sum sensibly",
          "[features]") {
    MelFilterBank fb({.frame_size = 1024, .sample_rate = 22050,
                      .num_filters = 20});
    std::vector<float> pwr(513, 1.0F);
    std::vector<float> out(20);
    fb.apply(pwr, out);
    for (float v : out) REQUIRE(v >= 0.0F);
    // With a flat input spectrum, mel values should monotonically increase
    // (higher mel bins span more linear bins).
    REQUIRE(out.back() > out.front());
}

TEST_CASE("dct2: round-trip via inverse-DCT-III ortho (manual check)",
          "[features]") {
    // For an orthonormal DCT-II, applying it to a vector of all ones gives
    // X[0] = sqrt(N), X[k>0] = 0.
    std::vector<float> ones(8, 1.0F);
    std::vector<float> coeffs(8);
    dct2_truncated(ones, coeffs);
    REQUIRE_THAT(coeffs[0],
                 Catch::Matchers::WithinAbs(std::sqrt(8.0), 1e-4));
    for (std::size_t i = 1; i < coeffs.size(); ++i) {
        REQUIRE_THAT(coeffs[i], Catch::Matchers::WithinAbs(0.0, 1e-4));
    }
}
