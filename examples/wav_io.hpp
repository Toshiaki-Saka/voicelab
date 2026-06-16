// examples/wav_io.hpp
//
// Tiny WAV reader/writer for the example programs. Supports mono and
// stereo, 16-bit PCM, little-endian only. Not part of the library
// public API — examples only.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace examples_wav {

struct WavData {
    std::uint32_t       sample_rate = 0;
    std::uint16_t       channels    = 0;
    std::vector<float>  samples;       // interleaved if multi-channel
};

inline std::uint32_t read_u32_le(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<std::uint32_t>(b[0])
         | (static_cast<std::uint32_t>(b[1]) << 8)
         | (static_cast<std::uint32_t>(b[2]) << 16)
         | (static_cast<std::uint32_t>(b[3]) << 24);
}
inline std::uint16_t read_u16_le(std::ifstream& f) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(b[0]) |
        static_cast<std::uint16_t>(b[1] << 8));
}

inline WavData load_wav_pcm16(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open WAV: " + path);

    char riff[4]; f.read(riff, 4);
    (void)read_u32_le(f);  // chunk size
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Not a RIFF/WAVE file: " + path);
    }

    WavData out;
    std::uint16_t bits = 0;
    std::vector<std::int16_t> pcm;

    while (f && f.peek() != EOF) {
        char id[4]; f.read(id, 4);
        if (f.gcount() < 4) break;
        const std::uint32_t size = read_u32_le(f);
        if (std::memcmp(id, "fmt ", 4) == 0) {
            const auto fmt        = read_u16_le(f);
            out.channels          = read_u16_le(f);
            out.sample_rate       = read_u32_le(f);
            (void)read_u32_le(f);  // byte rate
            (void)read_u16_le(f);  // block align
            bits                  = read_u16_le(f);
            if (fmt != 1 || bits != 16) {
                throw std::runtime_error("Only PCM16 supported: " + path);
            }
            // Skip any extra fmt bytes.
            if (size > 16) f.ignore(size - 16);
        } else if (std::memcmp(id, "data", 4) == 0) {
            pcm.resize(size / 2);
            f.read(reinterpret_cast<char*>(pcm.data()), size);
            break;
        } else {
            f.ignore(size);
        }
    }

    out.samples.resize(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        out.samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
    }
    return out;
}

inline void save_wav_pcm16(const std::string& path,
                           const std::vector<float>& samples,
                           std::uint32_t sample_rate,
                           std::uint16_t channels = 1) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot create WAV: " + path);

    auto write_u32 = [&](std::uint32_t v) {
        unsigned char b[4] = {
            static_cast<unsigned char>(v & 0xFF),
            static_cast<unsigned char>((v >> 8) & 0xFF),
            static_cast<unsigned char>((v >> 16) & 0xFF),
            static_cast<unsigned char>((v >> 24) & 0xFF)
        };
        f.write(reinterpret_cast<char*>(b), 4);
    };
    auto write_u16 = [&](std::uint16_t v) {
        unsigned char b[2] = {
            static_cast<unsigned char>(v & 0xFF),
            static_cast<unsigned char>((v >> 8) & 0xFF)
        };
        f.write(reinterpret_cast<char*>(b), 2);
    };

    const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
    f.write("RIFF", 4); write_u32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4); write_u32(16);
    write_u16(1); write_u16(channels);
    write_u32(sample_rate);
    write_u32(sample_rate * channels * 2);
    write_u16(static_cast<std::uint16_t>(channels * 2)); write_u16(16);
    f.write("data", 4); write_u32(data_bytes);

    for (float s : samples) {
        const float clipped = std::max(-1.0F, std::min(1.0F, s));
        const auto i = static_cast<std::int16_t>(clipped * 32767.0F);
        write_u16(static_cast<std::uint16_t>(i));
    }
}

}  // namespace examples_wav
