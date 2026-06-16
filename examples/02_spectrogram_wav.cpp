// examples/02_spectrogram_wav.cpp
//
// Read a WAV file (PCM16 mono or stereo), compute its STFT, and write
// a viridis-colored spectrogram to a PPM file. No audio hardware needed.
//
// Usage:
//   02_spectrogram_wav input.wav output.ppm

#include <complex>
#include <cstdio>
#include <vector>

#include "voicelab/core/features.hpp"
#include "voicelab/core/stft.hpp"
#include "voicelab/visualize/spectrogram.hpp"

#include "wav_io.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s input.wav output.ppm\n", argv[0]);
        return 1;
    }

    auto wav = examples_wav::load_wav_pcm16(argv[1]);
    std::printf("loaded %zu samples @ %u Hz, %u ch\n",
                wav.samples.size(), wav.sample_rate, wav.channels);

    // Mix to mono if needed.
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

    using namespace voicelab;
    constexpr std::size_t N = 1024;
    core::StreamingStft stft({.frame_size = N, .hop_size = N / 4,
                              .sample_rate = wav.sample_rate});
    visualize::SpectrogramRecorder rec({.num_bins = stft.num_bins(),
                                        .max_frames = 8192});

    std::vector<float> mags(stft.num_bins());
    stft.process(mono, [&](std::span<const std::complex<float>> spec) {
        core::magnitude(spec, mags);
        rec.push(mags);
    });

    rec.write_ppm(argv[2]);
    std::printf("wrote %s (%zu frames x %zu bins)\n",
                argv[2], rec.num_frames(), stft.num_bins());
    return 0;
}
