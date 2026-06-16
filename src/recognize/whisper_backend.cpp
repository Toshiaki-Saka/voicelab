// voicelab/src/recognize/whisper_backend.cpp
//
// Compiled only when VOICELAB_WITH_WHISPER=ON (the source file is then
// added to the target via src/CMakeLists.txt). When OFF, the symbol
// `has_whisper_support()` is provided by a separate translation unit
// (recognize/whisper_stub.cpp) so that callers can probe at runtime
// regardless of build configuration.

#include "voicelab/recognize/whisper_backend.hpp"

#if !defined(VOICELAB_HAS_WHISPER)
#  error "whisper_backend.cpp compiled without VOICELAB_HAS_WHISPER"
#endif

#include <whisper.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace voicelab::recognize {

bool has_whisper_support() noexcept { return true; }

struct WhisperBackend::Impl {
    Config                 cfg;
    SegmentCallback        cb;
    whisper_context*       ctx = nullptr;

    std::mutex             mu;
    std::condition_variable cv;
    std::vector<float>     queue;        // append-only; consumer drains
    std::atomic<bool>      stop_flag{false};
    std::thread            worker;

    explicit Impl(Config c) : cfg(std::move(c)) {
        whisper_context_params cparams = whisper_context_default_params();
        ctx = whisper_init_from_file_with_params(
            cfg.model_path.string().c_str(), cparams);
        if (!ctx) {
            throw std::runtime_error(
                "Failed to load whisper model from: " + cfg.model_path.string());
        }
    }
    ~Impl() {
        if (ctx) whisper_free(ctx);
    }

    void run() {
        const std::size_t chunk_samples =
            static_cast<std::size_t>(cfg.chunk_seconds * cfg.sample_rate);
        const std::size_t overlap_samples =
            static_cast<std::size_t>(cfg.overlap_seconds * cfg.sample_rate);

        std::vector<float> buf;
        buf.reserve(chunk_samples + overlap_samples);

        while (!stop_flag.load(std::memory_order_acquire)) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait_for(lk, std::chrono::milliseconds(100), [&]{
                    return stop_flag.load(std::memory_order_acquire) ||
                           !queue.empty();
                });
                if (!queue.empty()) {
                    buf.insert(buf.end(), queue.begin(), queue.end());
                    queue.clear();
                }
            }
            if (buf.size() < chunk_samples) continue;

            // Take the first chunk_samples for inference; keep last
            // `overlap_samples` to seed the next chunk.
            whisper_full_params wparams =
                whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.language       = cfg.language.c_str();
            wparams.translate      = cfg.translate;
            wparams.n_threads      = cfg.threads;
            wparams.print_progress = false;
            wparams.print_realtime = false;
            wparams.print_timestamps = false;

            if (whisper_full(ctx, wparams, buf.data(),
                             static_cast<int>(chunk_samples)) == 0) {
                const int n = whisper_full_n_segments(ctx);
                for (int i = 0; i < n; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    if (cb && text) cb(text);
                }
            }
            // Slide window.
            if (buf.size() > chunk_samples - overlap_samples) {
                buf.erase(buf.begin(),
                          buf.begin() +
                          static_cast<std::ptrdiff_t>(chunk_samples - overlap_samples));
            } else {
                buf.clear();
            }
        }
    }
};

WhisperBackend::WhisperBackend(Config cfg)
    : p_(std::make_unique<Impl>(std::move(cfg))) {}

WhisperBackend::~WhisperBackend() { stop(); }

void WhisperBackend::on_segment(SegmentCallback cb) {
    p_->cb = std::move(cb);
}

void WhisperBackend::push_audio(std::span<const float> samples) {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->queue.insert(p_->queue.end(), samples.begin(), samples.end());
    }
    p_->cv.notify_one();
}

void WhisperBackend::start() {
    if (p_->worker.joinable()) return;
    p_->stop_flag.store(false);
    p_->worker = std::thread([this]{ p_->run(); });
}

void WhisperBackend::stop() noexcept {
    if (!p_ || !p_->worker.joinable()) return;
    p_->stop_flag.store(true, std::memory_order_release);
    p_->cv.notify_all();
    p_->worker.join();
}

bool WhisperBackend::running() const noexcept {
    return p_ && p_->worker.joinable();
}

std::string transcribe_once(
    const std::filesystem::path& model_path,
    std::span<const float>       samples_16k,
    std::string_view             language,
    int                          threads)
{
    whisper_context_params cparams = whisper_context_default_params();
    whisper_context* ctx = whisper_init_from_file_with_params(
        model_path.string().c_str(), cparams);
    if (!ctx) return {};

    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    const std::string lang_str{language};
    wparams.language         = lang_str.c_str();
    wparams.translate        = false;
    wparams.n_threads        = threads;
    wparams.print_progress   = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;

    std::string result;
    if (whisper_full(ctx, wparams,
                     samples_16k.data(),
                     static_cast<int>(samples_16k.size())) == 0) {
        const int n = whisper_full_n_segments(ctx);
        for (int i = 0; i < n; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            if (text) result += text;
        }
    }
    whisper_free(ctx);
    return result;
}

}  // namespace voicelab::recognize
