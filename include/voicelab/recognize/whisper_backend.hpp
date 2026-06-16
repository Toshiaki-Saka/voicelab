#pragma once
//
// voicelab/recognize/whisper_backend.hpp
//
// Bridges voicelab's audio pipeline to whisper.cpp for live transcription.
// Compiled only when VOICELAB_WITH_WHISPER=ON; the header is always safe
// to include and exposes a stable interface.
//
// Threading model
// ---------------
// You feed mono float samples (16 kHz, the rate whisper expects) from any
// thread via `push_audio()`; samples are queued in a lock-free ring. A
// worker thread owned by this object pulls chunks, runs whisper inference,
// and invokes `on_segment()` with the recognized text. The callback is
// called on the worker thread.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace voicelab::recognize {

[[nodiscard]] bool has_whisper_support() noexcept;

class WhisperBackend {
public:
    struct Config {
        std::filesystem::path model_path;
        std::string           language      = "auto";
        std::uint32_t         sample_rate   = 16000;  // whisper requires 16k
        float                 chunk_seconds = 5.0F;
        float                 overlap_seconds = 0.5F;
        int                   threads       = 4;
        bool                  translate     = false;
    };

    using SegmentCallback = std::function<void(std::string_view text)>;

    explicit WhisperBackend(Config cfg);
    ~WhisperBackend();

    WhisperBackend(const WhisperBackend&)            = delete;
    WhisperBackend& operator=(const WhisperBackend&) = delete;

    void on_segment(SegmentCallback cb);

    // Feed audio (mono float at config().sample_rate). Safe from RT thread.
    void push_audio(std::span<const float> samples);

    void start();
    void stop() noexcept;
    [[nodiscard]] bool running() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Synchronous one-shot transcription of a complete 16 kHz mono buffer.
// Blocks until whisper inference finishes. Returns empty string if
// Whisper support is not compiled in or on error.
[[nodiscard]] std::string transcribe_once(
    const std::filesystem::path& model_path,
    std::span<const float>       samples_16k,
    std::string_view             language = "auto",
    int                          threads  = 4);

}  // namespace voicelab::recognize
