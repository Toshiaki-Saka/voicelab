// voicelab/src/io/audio_stream.cpp
//
// Live audio input via RtAudio when VOICELAB_HAS_RTAUDIO is defined.
// Otherwise a no-op stub that reports `has_live_io() == false` so the
// library can be built and used without a working audio backend.

#include "voicelab/io/audio_stream.hpp"

#include <utility>

#if defined(VOICELAB_HAS_RTAUDIO)
#  include <RtAudio.h>
#endif

namespace voicelab::io {

#if defined(VOICELAB_HAS_RTAUDIO)
bool has_live_io() noexcept { return true; }

struct AudioInputStream::Impl {
    Config                    cfg;
    RtAudio                   audio;
    Callback                  callback;
    std::string               error;
    bool                      running = false;
    std::vector<float>        mix_buf;  // for multi-channel down-mix

    static int rt_callback(void* out, void* in, unsigned int nFrames,
                           double, RtAudioStreamStatus, void* user) {
        (void)out;
        auto* self = static_cast<Impl*>(user);
        if (!self->callback || in == nullptr) return 0;

        auto* samples = static_cast<float*>(in);
        const unsigned int ch = self->cfg.channels;
        if (ch == 1) {
            self->callback(std::span<const float>{samples, nFrames});
        } else {
            // Mix to mono into our preallocated buffer.
            self->mix_buf.resize(nFrames);
            for (unsigned int i = 0; i < nFrames; ++i) {
                float acc = 0.0F;
                for (unsigned int c = 0; c < ch; ++c) {
                    acc += samples[i * ch + c];
                }
                self->mix_buf[i] = acc / static_cast<float>(ch);
            }
            self->callback(std::span<const float>{self->mix_buf});
        }
        return 0;
    }
};

AudioInputStream::AudioInputStream(Config cfg) : p_(std::make_unique<Impl>()) {
    p_->cfg = cfg;
    p_->mix_buf.reserve(cfg.block_size);
}

AudioInputStream::~AudioInputStream() {
    if (p_) stop();
}

AudioInputStream::AudioInputStream(AudioInputStream&&) noexcept            = default;
AudioInputStream& AudioInputStream::operator=(AudioInputStream&&) noexcept = default;

void AudioInputStream::on_audio(Callback cb) {
    p_->callback = std::move(cb);
}

void AudioInputStream::start() {
    if (p_->running) return;

    RtAudio::StreamParameters params;
    params.deviceId = (p_->cfg.device_id < 0)
        ? p_->audio.getDefaultInputDevice()
        : static_cast<unsigned int>(p_->cfg.device_id);
    params.nChannels = p_->cfg.channels;
    params.firstChannel = 0;

    unsigned int frames = static_cast<unsigned int>(p_->cfg.block_size);
    try {
        p_->audio.openStream(nullptr, &params, RTAUDIO_FLOAT32,
                             p_->cfg.sample_rate, &frames,
                             &Impl::rt_callback, p_.get());
        p_->audio.startStream();
        p_->cfg.block_size = frames;  // backend may have changed it
        p_->running = true;
    } catch (const std::exception& e) {
        p_->error = e.what();
        throw;
    }
}

void AudioInputStream::stop() noexcept {
    if (!p_ || !p_->running) return;
    try {
        if (p_->audio.isStreamRunning()) p_->audio.stopStream();
        if (p_->audio.isStreamOpen())    p_->audio.closeStream();
    } catch (...) {
        // Best-effort shutdown.
    }
    p_->running = false;
}

bool AudioInputStream::running() const noexcept {
    return p_ && p_->running;
}

const AudioInputStream::Config& AudioInputStream::config() const noexcept {
    return p_->cfg;
}

const std::string& AudioInputStream::last_error() const noexcept {
    return p_->error;
}

#else  // ---- No RtAudio: provide a stub that never starts. -------------

bool has_live_io() noexcept { return false; }

struct AudioInputStream::Impl {
    Config      cfg;
    Callback    callback;
    std::string error = "voicelab built without RtAudio: live I/O unavailable";
};

AudioInputStream::AudioInputStream(Config cfg) : p_(std::make_unique<Impl>()) {
    p_->cfg = cfg;
}
AudioInputStream::~AudioInputStream() = default;
AudioInputStream::AudioInputStream(AudioInputStream&&) noexcept            = default;
AudioInputStream& AudioInputStream::operator=(AudioInputStream&&) noexcept = default;

void AudioInputStream::on_audio(Callback cb) { p_->callback = std::move(cb); }
void AudioInputStream::start() { /* no-op */ }
void AudioInputStream::stop() noexcept { /* no-op */ }
bool AudioInputStream::running() const noexcept { return false; }
const AudioInputStream::Config& AudioInputStream::config() const noexcept {
    return p_->cfg;
}
const std::string& AudioInputStream::last_error() const noexcept {
    return p_->error;
}

#endif

}  // namespace voicelab::io
