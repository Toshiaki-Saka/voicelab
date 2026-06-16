// examples/20_live_transcribe.cpp
//
// End-to-end live transcription: mic → 16 kHz resample → whisper.cpp.
// Requires VOICELAB_WITH_RTAUDIO=ON and VOICELAB_WITH_WHISPER=ON, plus a
// downloaded ggml whisper model (e.g. ggml-base.en.bin) at runtime.
//
// Usage:
//   20_live_transcribe path/to/ggml-base.en.bin [language=auto]

#include <chrono>
#include <cstdio>
#include <thread>

#include "voicelab/io/audio_stream.hpp"
#include "voicelab/recognize/whisper_backend.hpp"

namespace {
// Minimal linear-interpolation resampler. For production use a polyphase
// filter; for an example the artefacts are inaudible at speech bands.
std::vector<float> resample_linear(std::span<const float> in,
                                    double in_rate, double out_rate) {
    if (in.empty()) return {};
    const double ratio = out_rate / in_rate;
    const auto n_out = static_cast<std::size_t>(
        std::floor(static_cast<double>(in.size()) * ratio));
    std::vector<float> out(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        const double t = static_cast<double>(i) / ratio;
        const auto j = static_cast<std::size_t>(t);
        const float frac = static_cast<float>(t - static_cast<double>(j));
        if (j + 1 < in.size()) {
            out[i] = in[j] * (1.0F - frac) + in[j + 1] * frac;
        } else {
            out[i] = in[j];
        }
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.bin [lang]\n", argv[0]);
        return 1;
    }
    using namespace voicelab;

    if (!io::has_live_io() || !recognize::has_whisper_support()) {
        std::fprintf(stderr, "Live transcription unavailable in this build.\n");
        return 1;
    }

    constexpr std::uint32_t mic_sr = 44100;
    constexpr std::uint32_t asr_sr = 16000;

    recognize::WhisperBackend asr({
        .model_path = argv[1],
        .language   = (argc >= 3) ? argv[2] : "auto",
        .sample_rate = asr_sr,
        .chunk_seconds = 5.0F,
        .overlap_seconds = 0.5F,
        .threads = 4,
    });
    asr.on_segment([](std::string_view s) {
        std::printf("%.*s", static_cast<int>(s.size()), s.data());
        std::fflush(stdout);
    });
    asr.start();

    io::AudioInputStream mic({.sample_rate = mic_sr, .block_size = 1024,
                              .channels = 1});
    mic.on_audio([&](std::span<const float> block) {
        auto down = resample_linear(block, mic_sr, asr_sr);
        asr.push_audio(down);
    });
    mic.start();

    std::printf("Listening — Ctrl-C to stop.\n");
    while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
}
