#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "voicelab/core/window.hpp"

using namespace voicelab::core;

TEST_CASE("Window: rectangular is all ones", "[window]") {
    auto w = make_window<float>(WindowType::Rectangular, 8);
    REQUIRE(w.size() == 8);
    for (float v : w) REQUIRE(v == 1.0F);
}

TEST_CASE("Window: Hann starts at 0 and is symmetric-ish (periodic form)",
          "[window]") {
    auto w = make_window<double>(WindowType::Hann, 8);
    REQUIRE_THAT(w[0], Catch::Matchers::WithinAbs(0.0, 1e-9));
    // Periodic Hann: w[N/2] == 1
    REQUIRE_THAT(w[4], Catch::Matchers::WithinAbs(1.0, 1e-9));
}

TEST_CASE("Window: zero length is a no-op", "[window]") {
    std::vector<float> v;
    make_window<float>(WindowType::Hann, v);
    REQUIRE(v.empty());
}
