// examples/04_mfcc_dump.cpp
//
// Compute log-mel + MFCC features over a WAV file and print them as a
// CSV to stdout. Pipe to a file and inspect with your favourite tool.
//
// Usage:
//   04_mfcc_dump input.wav > mfcc.csv

#include <complex>
#include <cstdio>
#include <vector>

#include "voicelab/core/features.hpp"
#include "voicelab/core/stft.hpp"

#include "wav_io.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s input.wav [n_mfcc=13]\n", argv[0]);
        return 1;
    }
    const int n_mfcc = (argc >= 3) ? std::atoi(argv[2]) : 13;

    auto wav = examples_wav::load_wav_pcm16(argv[1]);
    std::vector<float> mono(wav.samples.size() / wav.channels);
    for (std::size_t i = 0; i < mono.size(); ++i) {
        mono[i] = wav.samples[i * wav.channels];
    }

    using namespace voicelab::core;
    constexpr std::size_t N = 1024;
    StreamingStft stft({.frame_size = N, .hop_size = N / 4,
                        .sample_rate = wav.sample_rate});
    MelFilterBank mel({.frame_size = N, .sample_rate = wav.sample_rate,
                       .num_filters = 40});

    std::vector<float> pwr(stft.num_bins());
    std::vector<float> log_mel(mel.num_filters());
    std::vector<float> mfcc(static_cast<std::size_t>(n_mfcc));

    // CSV header
    std::printf("frame");
    for (int i = 0; i < n_mfcc; ++i) std::printf(",mfcc%d", i);
    std::printf("\n");

    std::size_t frame_idx = 0;
    stft.process(mono, [&](std::span<const std::complex<float>> spec) {
        power(spec, pwr);
        mel.apply(pwr, log_mel);
        log_compress(log_mel);
        dct2_truncated(log_mel, mfcc);
        std::printf("%zu", frame_idx++);
        for (auto c : mfcc) std::printf(",%.4f", static_cast<double>(c));
        std::printf("\n");
    });
    return 0;
}
