// examples/05_analyze_wav.cpp
//
// One-pass analysis dump for a WAV file: the spectrogram in dB and the MFCC
// matrix, in a single self-describing text stream on stdout.
//
// 04_mfcc_dump prints MFCCs alone and 02_spectrogram_wav renders an image, so
// neither gives a plotting front-end the numbers it needs. apps/transcribe_demo.py
// calls this tool instead of recomputing the STFT and the mel filterbank in
// scipy: the DSP has exactly one implementation, here, and Python only draws it.
//
// Usage:
//   05_analyze_wav input.wav [n_mfcc=13] [n_mels=40] [f_max_hz=8000]
//
// Output (sections in this order, values comma-separated):
//   sample_rate,<Hz>
//   num_samples,<n>          samples per channel after the mix-down to mono
//   frame_size,<N>
//   hop_size,<H>
//   num_frames,<F>
//   num_bins,<B>             bins kept, i.e. those at or below f_max_hz
//   mfcc_count,<K>
//   mel_count,<M>
//   [freqs]                  one row, B values, Hz
//   [times]                  one row, F values, seconds (frame centres)
//   [mag_db]                 F rows of B values, 20*log10(|X| + 1e-8)
//   [mfcc]                   F rows of K values

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "voicelab/core/features.hpp"
#include "voicelab/core/stft.hpp"

#include "wav_io.hpp"

namespace {

constexpr std::size_t kFrameSize = 1024;
constexpr std::size_t kHopSize   = 256;

void print_row(const std::vector<float>& row) {
    for (std::size_t i = 0; i < row.size(); ++i) {
        std::printf(i == 0 ? "%.4f" : ",%.4f", static_cast<double>(row[i]));
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s input.wav [n_mfcc=13] [n_mels=40] "
                     "[f_max_hz=8000]\n", argv[0]);
        return 1;
    }
    const auto n_mfcc = static_cast<std::size_t>(
        (argc >= 3) ? std::atoi(argv[2]) : 13);
    const auto n_mels = static_cast<std::size_t>(
        (argc >= 4) ? std::atoi(argv[3]) : 40);
    const float f_max_hz = (argc >= 5)
        ? static_cast<float>(std::atof(argv[4])) : 8000.0F;

    if (n_mfcc == 0 || n_mels == 0 || n_mfcc > n_mels) {
        std::fprintf(stderr, "error: need 0 < n_mfcc <= n_mels\n");
        return 1;
    }

    auto wav = examples_wav::load_wav_pcm16(argv[1]);
    if (wav.channels == 0 || wav.samples.empty()) {
        std::fprintf(stderr, "error: %s has no audio\n", argv[1]);
        return 1;
    }

    // Mix down to mono by averaging the channels, matching 02_spectrogram_wav.
    std::vector<float> mono;
    if (wav.channels == 1) {
        mono = std::move(wav.samples);
    } else {
        mono.resize(wav.samples.size() / wav.channels);
        for (std::size_t i = 0; i < mono.size(); ++i) {
            float acc = 0.0F;
            for (std::uint16_t c = 0; c < wav.channels; ++c) {
                acc += wav.samples[i * wav.channels + c];
            }
            mono[i] = acc / static_cast<float>(wav.channels);
        }
    }

    using namespace voicelab::core;
    StreamingStft stft({.frame_size  = kFrameSize,
                        .hop_size    = kHopSize,
                        .sample_rate = wav.sample_rate});
    MelFilterBank mel({.frame_size  = kFrameSize,
                       .sample_rate = wav.sample_rate,
                       .num_filters = n_mels});

    // Bins at or below f_max_hz; always keep at least DC so the output is
    // well-formed even for an absurdly low f_max.
    std::size_t kept_bins = 0;
    for (std::size_t b = 0; b < stft.num_bins(); ++b) {
        if (stft.bin_to_hz(b) <= f_max_hz) kept_bins = b + 1;
    }
    if (kept_bins == 0) kept_bins = 1;

    std::vector<float> mags(stft.num_bins());
    std::vector<float> pwr(stft.num_bins());
    std::vector<float> log_mel(n_mels);
    std::vector<float> mfcc(n_mfcc);

    // Buffer the frames: the header has to carry num_frames, which is only
    // known once the whole signal has been consumed.
    std::vector<std::vector<float>> mag_db_rows;
    std::vector<std::vector<float>> mfcc_rows;

    stft.process(mono, [&](std::span<const std::complex<float>> spec) {
        magnitude(spec, mags);
        power(spec, pwr);

        std::vector<float> db(kept_bins);
        for (std::size_t b = 0; b < kept_bins; ++b) {
            db[b] = 20.0F * std::log10(mags[b] + 1e-8F);
        }
        mag_db_rows.push_back(std::move(db));

        mel.apply(pwr, log_mel);
        log_compress(log_mel);
        dct2_truncated(log_mel, mfcc);
        mfcc_rows.emplace_back(mfcc.begin(), mfcc.end());
    });

    const std::size_t num_frames = mag_db_rows.size();

    std::printf("sample_rate,%u\n", wav.sample_rate);
    std::printf("num_samples,%zu\n", mono.size());
    std::printf("frame_size,%zu\n", kFrameSize);
    std::printf("hop_size,%zu\n", kHopSize);
    std::printf("num_frames,%zu\n", num_frames);
    std::printf("num_bins,%zu\n", kept_bins);
    std::printf("mfcc_count,%zu\n", n_mfcc);
    std::printf("mel_count,%zu\n", n_mels);

    std::printf("[freqs]\n");
    std::vector<float> freqs(kept_bins);
    for (std::size_t b = 0; b < kept_bins; ++b) freqs[b] = stft.bin_to_hz(b);
    print_row(freqs);

    // Frame f covers samples [f*H, f*H + N); report its centre in seconds.
    std::printf("[times]\n");
    std::vector<float> times(num_frames);
    for (std::size_t f = 0; f < num_frames; ++f) {
        times[f] = (static_cast<float>(f * kHopSize) + kFrameSize * 0.5F)
                 / static_cast<float>(wav.sample_rate);
    }
    print_row(times);

    std::printf("[mag_db]\n");
    for (const auto& row : mag_db_rows) print_row(row);

    std::printf("[mfcc]\n");
    for (const auto& row : mfcc_rows) print_row(row);

    return 0;
}
