#pragma once
//
// voicelab/effects/pitch_shift.hpp
//
// Phase-vocoder pitch shifter — the canonical "FFT → modify → IFFT" demo.
//
// Algorithm sketch (Laroche & Dolson, 1999):
//   1. STFT analysis with hop H_a.
//   2. Compute "true" instantaneous frequency per bin from phase
//      differences across consecutive frames.
//   3. Scale the bin positions by the shift factor (semitones → ratio).
//   4. Resynthesize at hop H_a using the modified phase advance, then
//      *time-stretch* by ratio via resampling H_s = H_a * ratio out — but
//      we keep it simple here: we resample the *output* signal instead,
//      so the pitch goes up while duration stays the same.
//
// For a first OSS release we expose the simpler "frequency-bin shift +
// phase accumulation" variant which gives recognizable, if somewhat
// "metallic", pitch shifting. A full PSOLA/Rubber Band-quality shifter
// is out of scope for this library.

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

#include "voicelab/core/istft.hpp"
#include "voicelab/core/stft.hpp"

namespace voicelab::effects {

class PitchShifter {
public:
    struct Config {
        core::StreamingStft::Config  stft;
        core::StreamingIstft::Config istft;
        float                        semitones = 0.0F;
    };

    explicit PitchShifter(Config cfg);

    // Push input samples, pull shifted samples.
    void  push(std::span<const float> input);
    [[nodiscard]] std::size_t pop(std::span<float> output) noexcept;

    void  set_semitones(float st) noexcept;
    [[nodiscard]] float semitones() const noexcept { return cfg_.semitones; }

private:
    void on_frame(std::span<const std::complex<float>> spec);

    Config                              cfg_;
    core::StreamingStft                 stft_;
    core::StreamingIstft                istft_;
    float                               ratio_ = 1.0F;

    // Phase-vocoder state.
    std::vector<float>                  last_phase_;       // per-bin
    std::vector<float>                  sum_phase_;        // per-bin
    std::vector<std::complex<float>>    shifted_;          // scratch
    std::vector<float>                  shifted_mag_;
    std::vector<float>                  shifted_freq_;     // true freq per bin
};

}  // namespace voicelab::effects
