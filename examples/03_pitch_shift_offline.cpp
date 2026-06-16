// examples/03_pitch_shift_offline.cpp
//
// Pitch-shift a WAV file by N semitones using the phase-vocoder effect.
// Demonstrates the FFT → modify → IFFT round-trip.
//
// Usage:
//   03_pitch_shift_offline input.wav output.wav semitones

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "voicelab/effects/pitch_shift.hpp"

#include "wav_io.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s input.wav output.wav semitones\n", argv[0]);
        return 1;
    }
    const float semitones = std::strtof(argv[3], nullptr);

    auto wav = examples_wav::load_wav_pcm16(argv[1]);
    if (wav.channels != 1) {
        std::fprintf(stderr, "(input has %u ch, taking left channel only)\n",
                     wav.channels);
        std::vector<float> mono(wav.samples.size() / wav.channels);
        for (std::size_t i = 0; i < mono.size(); ++i) {
            mono[i] = wav.samples[i * wav.channels];
        }
        wav.samples = std::move(mono);
        wav.channels = 1;
    }

    using namespace voicelab;
    constexpr std::size_t N = 2048;
    constexpr std::size_t H = N / 4;
    effects::PitchShifter shifter({
        .stft  = {.frame_size = N, .hop_size = H, .sample_rate = wav.sample_rate},
        .istft = {.frame_size = N, .hop_size = H, .sample_rate = wav.sample_rate},
        .semitones = semitones,
    });

    std::vector<float> out;
    out.reserve(wav.samples.size());

    std::vector<float> chunk(2048);
    constexpr std::size_t kBlock = 1024;
    for (std::size_t i = 0; i < wav.samples.size(); i += kBlock) {
        const std::size_t n =
            std::min(kBlock, wav.samples.size() - i);
        shifter.push({wav.samples.data() + i, n});
        std::size_t got = 0;
        do {
            got = shifter.pop(chunk);
            out.insert(out.end(), chunk.begin(),
                       chunk.begin() + static_cast<std::ptrdiff_t>(got));
        } while (got > 0);
    }

    examples_wav::save_wav_pcm16(argv[2], out, wav.sample_rate, 1);
    std::printf("shifted %.2f semitones; wrote %zu samples to %s\n",
                static_cast<double>(semitones), out.size(), argv[2]);
    return 0;
}
