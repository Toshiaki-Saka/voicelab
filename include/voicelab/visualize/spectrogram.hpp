#pragma once
//
// voicelab/visualize/spectrogram.hpp
//
// Off-line spectrogram visualization. Accumulates magnitude frames and
// renders them as a PGM (P5 grayscale) or PPM (P6 RGB) image — chosen so
// the library has zero image-format dependencies.
//
// The colormap is "viridis"-like (perceptually uniform, monotonic
// luminance) computed from a small lookup table.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace voicelab::visualize {

class SpectrogramRecorder {
public:
    struct Config {
        std::size_t num_bins         = 513;
        std::size_t max_frames       = 4096;   // ring; older frames dropped
        float       db_floor         = -80.0F;
        float       db_ceiling       = 0.0F;
    };

    explicit SpectrogramRecorder(Config cfg);

    // Push one frame's magnitudes (length == num_bins).
    void push(std::span<const float> magnitudes);

    [[nodiscard]] std::size_t num_frames() const noexcept { return count_; }
    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    // Write current contents as a grayscale PGM.
    void write_pgm(const std::filesystem::path& path) const;

    // Write current contents with a viridis colormap as PPM.
    void write_ppm(const std::filesystem::path& path) const;

private:
    [[nodiscard]] float at(std::size_t frame, std::size_t bin) const noexcept;

    Config             cfg_;
    std::vector<float> data_;       // ring; rows = frames, cols = bins
    std::size_t        write_pos_ = 0;
    std::size_t        count_     = 0;
};

}  // namespace voicelab::visualize
