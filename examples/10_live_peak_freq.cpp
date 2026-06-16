// examples/10_live_peak_freq.cpp
//
// Open the default microphone, run STFT in real time, print the peak
// frequency. Requires VOICELAB_WITH_RTAUDIO=ON.

#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <thread>

#include "voicelab/core/stft.hpp"
#include "voicelab/io/audio_stream.hpp"

int main() {
    using namespace voicelab;

    if (!io::has_live_io()) {
        std::fprintf(stderr,
                     "voicelab was built without live I/O support.\n");
        return 1;
    }

    constexpr std::uint32_t sr = 44100;
    constexpr std::size_t   N  = 2048;

    core::StreamingStft stft({.frame_size = N, .hop_size = N / 4,
                              .sample_rate = sr});
    io::AudioInputStream mic({.sample_rate = sr, .block_size = 512,
                              .channels    = 1});

    std::atomic<float> peak_hz{0.0F};

    mic.on_audio([&](std::span<const float> block) {
        stft.process(block, [&](std::span<const std::complex<float>> spec) {
            std::size_t best_k = 0;
            float       best_m = 0.0F;
            for (std::size_t k = 1; k < spec.size(); ++k) {
                const float m = std::abs(spec[k]);
                if (m > best_m) { best_m = m; best_k = k; }
            }
            peak_hz.store(stft.bin_to_hz(best_k));
        });
    });
    mic.start();

    for (int s = 0; s < 60; ++s) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::printf("\rpeak: %7.1f Hz  ",
                    static_cast<double>(peak_hz.load()));
        std::fflush(stdout);
    }
    std::printf("\n");
    return 0;
}
