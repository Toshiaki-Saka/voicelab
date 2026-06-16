// voicelab/src/recognize/whisper_stub.cpp
//
// Always-compiled translation unit that provides definitions for
// WhisperBackend when whisper.cpp is unavailable. With VOICELAB_HAS_WHISPER,
// the real implementation in whisper_backend.cpp wins (this file is then
// excluded from the build by src/CMakeLists.txt).

#include "voicelab/recognize/whisper_backend.hpp"

#if !defined(VOICELAB_HAS_WHISPER)

#include <stdexcept>

namespace voicelab::recognize {

bool has_whisper_support() noexcept { return false; }

struct WhisperBackend::Impl {
    Config          cfg;
    SegmentCallback cb;
};

WhisperBackend::WhisperBackend(Config cfg) : p_(std::make_unique<Impl>()) {
    p_->cfg = std::move(cfg);
    throw std::runtime_error(
        "voicelab built without whisper.cpp: "
        "reconfigure with -DVOICELAB_WITH_WHISPER=ON");
}

WhisperBackend::~WhisperBackend() = default;

void WhisperBackend::on_segment(SegmentCallback) {}
void WhisperBackend::push_audio(std::span<const float>) {}
void WhisperBackend::start() {}
void WhisperBackend::stop() noexcept {}
bool WhisperBackend::running() const noexcept { return false; }

std::string transcribe_once(
    const std::filesystem::path&,
    std::span<const float>,
    std::string_view,
    int)
{
    return {};
}

}  // namespace voicelab::recognize

#endif  // !VOICELAB_HAS_WHISPER
