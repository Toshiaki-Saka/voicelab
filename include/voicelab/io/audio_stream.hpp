#pragma once
//
// voicelab/io/audio_stream.hpp
//
// Thin façade over a live audio input device. The actual backend
// implementation is RtAudio, only compiled when VOICELAB_WITH_RTAUDIO=ON.
//
// The library degrades cleanly: without RtAudio, the headers still
// compile and offline use cases (push samples from a WAV / vector) work
// unchanged. Examples that need a live mic check `voicelab::io::has_live_io()`
// at startup.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace voicelab::io {

[[nodiscard]] bool has_live_io() noexcept;

class AudioInputStream {
public:
    struct Config {
        std::uint32_t sample_rate = 44100;
        std::size_t   block_size  = 512;
        unsigned int  channels    = 1;
        int           device_id   = -1;  // -1 => default device
    };

    // Mono-mixed callback: called from the audio thread.
    // Do not allocate, lock, or block here.
    using Callback = std::function<void(std::span<const float>)>;

    explicit AudioInputStream(Config cfg);
    ~AudioInputStream();

    AudioInputStream(const AudioInputStream&)            = delete;
    AudioInputStream& operator=(const AudioInputStream&) = delete;
    AudioInputStream(AudioInputStream&&)            noexcept;
    AudioInputStream& operator=(AudioInputStream&&) noexcept;

    void on_audio(Callback cb);
    void start();
    void stop() noexcept;
    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] const Config& config() const noexcept;

    // Last reported error, if any (e.g. device not available).
    [[nodiscard]] const std::string& last_error() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace voicelab::io
