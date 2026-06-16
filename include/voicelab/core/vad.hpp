#pragma once
//
// voicelab/core/vad.hpp
//
// Energy-based voice activity detector. Not as robust as Silero/WebRTC-VAD
// for noisy environments, but dependency-free and useful as a teaching
// example and as a first-stage gate before heavier processing.
//
// Algorithm:
//   * Per frame, compute RMS energy in dBFS.
//   * Maintain an adaptive noise floor: exponential moving average of
//     the *minimum* recent RMS, updated only while we believe the frame
//     is non-speech.
//   * A frame is "voiced" if (rms_dbfs - noise_dbfs) > threshold_db.
//   * Apply hangover: once voiced, stay voiced for at least `hangover_ms`
//     after the last voiced frame. This bridges short consonant gaps.

#include <cstddef>
#include <cstdint>
#include <span>

namespace voicelab::core {

class EnergyVad {
public:
    struct Config {
        std::uint32_t sample_rate    = 44100;
        std::size_t   frame_size     = 1024;
        std::size_t   hop_size       = 256;
        float         threshold_db   = 9.0F;   // SNR over noise floor
        float         hangover_ms    = 200.0F;
        float         noise_attack   = 0.05F;  // EMA coeff when adapting up
        float         noise_release  = 0.005F; // EMA coeff when adapting down
        float         initial_noise_dbfs = -60.0F;
    };

    explicit EnergyVad(Config cfg);

    // Feed one frame's worth of time-domain samples. Returns true if the
    // current frame is judged voiced (after hangover).
    bool process_frame(std::span<const float> frame) noexcept;

    [[nodiscard]] float noise_dbfs()  const noexcept { return noise_dbfs_; }
    [[nodiscard]] float last_rms_dbfs() const noexcept { return last_rms_dbfs_; }

private:
    Config         cfg_;
    float          noise_dbfs_    = -60.0F;
    float          last_rms_dbfs_ = -60.0F;
    std::size_t    hangover_frames_remaining_ = 0;
    std::size_t    hangover_total_frames_ = 0;
};

}  // namespace voicelab::core
