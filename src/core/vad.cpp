// voicelab/src/core/vad.cpp

#include "voicelab/core/vad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace voicelab::core {

namespace {
float rms_dbfs(std::span<const float> frame) noexcept {
    if (frame.empty()) return -120.0F;
    double sum = 0.0;
    for (auto s : frame) sum += static_cast<double>(s) * s;
    const double rms = std::sqrt(sum / static_cast<double>(frame.size()));
    if (rms < 1e-10) return -120.0F;
    return static_cast<float>(20.0 * std::log10(rms));
}
}  // namespace

EnergyVad::EnergyVad(Config cfg) : cfg_(cfg) {
    if (cfg_.frame_size == 0 || cfg_.hop_size == 0 || cfg_.sample_rate == 0) {
        throw std::invalid_argument("EnergyVad: invalid configuration");
    }
    noise_dbfs_ = cfg_.initial_noise_dbfs;
    const double hop_seconds =
        static_cast<double>(cfg_.hop_size) / cfg_.sample_rate;
    hangover_total_frames_ = static_cast<std::size_t>(
        std::ceil(cfg_.hangover_ms / 1000.0 / hop_seconds));
}

bool EnergyVad::process_frame(std::span<const float> frame) noexcept {
    const float rms_db = rms_dbfs(frame);
    last_rms_dbfs_ = rms_db;

    const float snr = rms_db - noise_dbfs_;
    const bool above_threshold = snr > cfg_.threshold_db;

    bool voiced = false;
    if (above_threshold) {
        voiced = true;
        hangover_frames_remaining_ = hangover_total_frames_;
    } else if (hangover_frames_remaining_ > 0) {
        voiced = true;
        --hangover_frames_remaining_;
    }

    // Adapt noise floor only when we're confident it's NOT speech, and
    // adapt faster downward than upward (let the floor track the quiet
    // baseline rather than chasing speech peaks).
    if (!voiced) {
        const float coeff = (rms_db < noise_dbfs_) ? cfg_.noise_attack
                                                    : cfg_.noise_release;
        noise_dbfs_ = noise_dbfs_ + coeff * (rms_db - noise_dbfs_);
    }

    return voiced;
}

}  // namespace voicelab::core
