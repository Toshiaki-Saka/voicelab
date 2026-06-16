#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numbers>
#include <vector>

#include "voicelab/core/vad.hpp"

using namespace voicelab::core;

TEST_CASE("EnergyVad: silence stays unvoiced", "[vad]") {
    EnergyVad vad({.sample_rate = 16000, .frame_size = 1024, .hop_size = 256});
    std::vector<float> silence(1024, 0.0F);
    bool any_voiced = false;
    for (int i = 0; i < 50; ++i) {
        if (vad.process_frame(silence)) any_voiced = true;
    }
    REQUIRE_FALSE(any_voiced);
}

TEST_CASE("EnergyVad: loud sine triggers voiced", "[vad]") {
    EnergyVad vad({.sample_rate = 16000, .frame_size = 1024, .hop_size = 256,
                   .threshold_db = 6.0F});
    std::vector<float> silence(1024, 0.0F);
    // Adapt noise floor first.
    for (int i = 0; i < 100; ++i) vad.process_frame(silence);

    std::vector<float> tone(1024);
    for (std::size_t n = 0; n < tone.size(); ++n) {
        tone[n] = 0.5F * std::sin(2.0F * std::numbers::pi_v<float> *
                                  440.0F * static_cast<float>(n) / 16000.0F);
    }
    bool voiced = false;
    for (int i = 0; i < 5; ++i) {
        if (vad.process_frame(tone)) voiced = true;
    }
    REQUIRE(voiced);
}
