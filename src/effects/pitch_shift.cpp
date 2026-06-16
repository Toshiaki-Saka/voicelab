// voicelab/src/effects/pitch_shift.cpp

#include "voicelab/effects/pitch_shift.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace voicelab::effects {

namespace {
constexpr float k_two_pi = 2.0F * std::numbers::pi_v<float>;

// Wrap an angle to (-pi, pi]. The "principal argument" reduction is what
// distinguishes the phase advance due to the bin's center frequency from
// the residual that encodes the true frequency.
float principal_arg(float x) noexcept {
    x = std::fmod(x + std::numbers::pi_v<float>, k_two_pi);
    if (x < 0.0F) x += k_two_pi;
    return x - std::numbers::pi_v<float>;
}
}  // namespace

PitchShifter::PitchShifter(Config cfg)
    : cfg_(cfg),
      stft_(cfg.stft),
      istft_(cfg.istft) {
    const std::size_t nbins = stft_.num_bins();
    last_phase_.assign(nbins, 0.0F);
    sum_phase_.assign(nbins, 0.0F);
    shifted_.assign(nbins, {});
    shifted_mag_.assign(nbins, 0.0F);
    shifted_freq_.assign(nbins, 0.0F);
    set_semitones(cfg_.semitones);
}

void PitchShifter::set_semitones(float st) noexcept {
    cfg_.semitones = st;
    ratio_ = std::pow(2.0F, st / 12.0F);
}

void PitchShifter::push(std::span<const float> input) {
    stft_.process(input, [this](std::span<const std::complex<float>> spec) {
        on_frame(spec);
    });
}

std::size_t PitchShifter::pop(std::span<float> output) noexcept {
    return istft_.pop(output);
}

void PitchShifter::on_frame(std::span<const std::complex<float>> spec) {
    const std::size_t N    = cfg_.stft.frame_size;
    const std::size_t H    = cfg_.stft.hop_size;
    const auto        sr   = static_cast<float>(cfg_.stft.sample_rate);
    const std::size_t bins = spec.size();

    const float expected = k_two_pi * static_cast<float>(H) /
                           static_cast<float>(N);
    const float freq_per_bin = sr / static_cast<float>(N);

    // ---- Analysis: extract magnitude and "true" instantaneous frequency.
    std::fill(shifted_mag_.begin(),  shifted_mag_.end(),  0.0F);
    std::fill(shifted_freq_.begin(), shifted_freq_.end(), 0.0F);

    for (std::size_t k = 0; k < bins; ++k) {
        const float mag   = std::abs(spec[k]);
        const float phase = std::arg(spec[k]);

        const float dphase = phase - last_phase_[k];
        last_phase_[k] = phase;

        // Subtract expected phase advance for this bin, wrap, restore.
        const float deviation =
            principal_arg(dphase - expected * static_cast<float>(k));
        const float true_bin = static_cast<float>(k) +
            deviation * static_cast<float>(N) /
            (k_two_pi * static_cast<float>(H));

        const float true_hz = true_bin * freq_per_bin;

        // ---- Pitch shift: move energy to a new bin.
        const auto target_bin = static_cast<std::size_t>(
            std::round(static_cast<float>(k) * ratio_));
        if (target_bin < bins) {
            shifted_mag_[target_bin] += mag;
            shifted_freq_[target_bin] = true_hz * ratio_;
        }
    }

    // ---- Synthesis: rebuild phase from desired frequency per bin.
    for (std::size_t k = 0; k < bins; ++k) {
        const float deviation = shifted_freq_[k] / freq_per_bin -
                                static_cast<float>(k);
        const float phase_advance = expected * static_cast<float>(k) +
            deviation * k_two_pi * static_cast<float>(H) /
            static_cast<float>(N);
        sum_phase_[k] += phase_advance;
        const float phase = sum_phase_[k];
        shifted_[k] = {shifted_mag_[k] * std::cos(phase),
                       shifted_mag_[k] * std::sin(phase)};
    }

    istft_.push_frame(shifted_);
}

}  // namespace voicelab::effects
