// Quick command-line test for transcribe_once()
#include <cstdio>
#include <span>
#include <vector>
#include <cmath>
#include "examples/wav_io.hpp"
#include "voicelab/recognize/whisper_backend.hpp"

int main() {
    std::fprintf(stderr, "has_whisper_support: %s\n",
        voicelab::recognize::has_whisper_support() ? "yes" : "no");

    auto wav = examples_wav::load_wav_pcm16(
        "simulation_output/my_voice.wav");
    std::fprintf(stderr, "WAV: %u Hz, %u ch, %zu samples\n",
        wav.sample_rate, wav.channels, wav.samples.size());

    // Mix to mono
    std::vector<float> mono;
    if (wav.channels == 2) {
        mono.resize(wav.samples.size() / 2);
        for (std::size_t i = 0; i < mono.size(); ++i)
            mono[i] = (wav.samples[2*i] + wav.samples[2*i+1]) * 0.5F;
    } else {
        mono = wav.samples;
    }

    // Resample to 16kHz
    std::vector<float> pcm16k;
    if (wav.sample_rate != 16000) {
        const double ratio = 16000.0 / wav.sample_rate;
        const auto n = static_cast<std::size_t>(
            std::floor(static_cast<double>(mono.size()) * ratio));
        pcm16k.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / ratio;
            const auto j = static_cast<std::size_t>(t);
            const float frac = static_cast<float>(t - static_cast<double>(j));
            pcm16k[i] = (j+1 < mono.size())
                ? mono[j]*(1.0F-frac)+mono[j+1]*frac : mono[j];
        }
    } else {
        pcm16k = mono;
    }
    std::fprintf(stderr, "16kHz samples: %zu\n", pcm16k.size());

    std::fprintf(stderr, "Loading model...\n");
    auto text = voicelab::recognize::transcribe_once(
        "models/ggml-model.bin", pcm16k, "ja");
    std::fprintf(stderr, "Done.\n");
    std::printf("Result: %s\n", text.c_str());
    return 0;
}
