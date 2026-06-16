// voicelab/src/visualize/spectrogram.cpp

#include "voicelab/visualize/spectrogram.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace voicelab::visualize {

namespace {
// 16-stop sample of the viridis colormap (R, G, B in [0, 255]).
// Source: matplotlib viridis. Adequate for visual inspection.
constexpr std::array<std::array<std::uint8_t, 3>, 16> kViridis{{
    { 68,   1,  84}, { 72,  35, 116}, { 64,  67, 135}, { 52,  94, 141},
    { 41, 120, 142}, { 32, 144, 140}, { 34, 167, 132}, { 68, 190, 112},
    {121, 209,  81}, {189, 222,  38}, {253, 231,  36}, {253, 200,  39},
    {253, 168,  46}, {251, 134,  56}, {243,  96,  60}, {221,  50,  60},
}};

std::array<std::uint8_t, 3> viridis(float t) noexcept {
    t = std::clamp(t, 0.0F, 1.0F);
    const float x = t * static_cast<float>(kViridis.size() - 1);
    const auto i = static_cast<std::size_t>(std::floor(x));
    const auto j = std::min(i + 1, kViridis.size() - 1);
    const float f = x - static_cast<float>(i);
    std::array<std::uint8_t, 3> out{};
    for (int c = 0; c < 3; ++c) {
        const float v = static_cast<float>(kViridis[i][c]) * (1.0F - f) +
                        static_cast<float>(kViridis[j][c]) * f;
        out[static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(v);
    }
    return out;
}
}  // namespace

SpectrogramRecorder::SpectrogramRecorder(Config cfg)
    : cfg_(cfg),
      data_(cfg.num_bins * cfg.max_frames, 0.0F) {
    if (cfg_.num_bins == 0 || cfg_.max_frames == 0) {
        throw std::invalid_argument("SpectrogramRecorder: invalid config");
    }
}

void SpectrogramRecorder::push(std::span<const float> mags) {
    const std::size_t n = std::min(mags.size(), cfg_.num_bins);
    float* row = data_.data() + write_pos_ * cfg_.num_bins;
    for (std::size_t k = 0; k < n; ++k) row[k] = mags[k];
    // Zero-fill rest if input is shorter than num_bins.
    for (std::size_t k = n; k < cfg_.num_bins; ++k) row[k] = 0.0F;
    write_pos_ = (write_pos_ + 1) % cfg_.max_frames;
    if (count_ < cfg_.max_frames) ++count_;
}

float SpectrogramRecorder::at(std::size_t frame, std::size_t bin) const noexcept {
    // `frame` is in display order (0 = oldest).
    const std::size_t base = (count_ < cfg_.max_frames)
        ? 0
        : write_pos_;  // ring head == oldest when full
    const std::size_t actual = (base + frame) % cfg_.max_frames;
    return data_[actual * cfg_.num_bins + bin];
}

void SpectrogramRecorder::write_pgm(const std::filesystem::path& path) const {
    const std::size_t W = count_;            // time on X
    const std::size_t H = cfg_.num_bins;     // frequency on Y (low at bottom)
    if (W == 0) throw std::runtime_error("Nothing recorded yet");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path.string());

    f << "P5\n" << W << " " << H << "\n255\n";
    const float floor = cfg_.db_floor, ceil = cfg_.db_ceiling;
    const float range = ceil - floor;

    for (std::size_t y = 0; y < H; ++y) {
        // Flip vertically: bin 0 (DC) drawn at the bottom => write row (H-1-y).
        const std::size_t bin = H - 1 - y;
        for (std::size_t x = 0; x < W; ++x) {
            const float mag = at(x, bin);
            const float db = 20.0F * std::log10(mag + 1e-10F);
            const float t = std::clamp((db - floor) / range, 0.0F, 1.0F);
            const auto v = static_cast<std::uint8_t>(t * 255.0F);
            f.put(static_cast<char>(v));
        }
    }
}

void SpectrogramRecorder::write_ppm(const std::filesystem::path& path) const {
    const std::size_t W = count_;
    const std::size_t H = cfg_.num_bins;
    if (W == 0) throw std::runtime_error("Nothing recorded yet");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path.string());

    f << "P6\n" << W << " " << H << "\n255\n";
    const float floor = cfg_.db_floor, ceil = cfg_.db_ceiling;
    const float range = ceil - floor;

    for (std::size_t y = 0; y < H; ++y) {
        const std::size_t bin = H - 1 - y;
        for (std::size_t x = 0; x < W; ++x) {
            const float mag = at(x, bin);
            const float db = 20.0F * std::log10(mag + 1e-10F);
            const float t = std::clamp((db - floor) / range, 0.0F, 1.0F);
            const auto rgb = viridis(t);
            f.put(static_cast<char>(rgb[0]));
            f.put(static_cast<char>(rgb[1]));
            f.put(static_cast<char>(rgb[2]));
        }
    }
}

}  // namespace voicelab::visualize
