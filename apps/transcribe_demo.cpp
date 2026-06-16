// apps/transcribe_demo.cpp
//
// GUI demo: file picker → STFT spectrogram → whisper transcription → WAV playback
// Build: cmake -B build -DVOICELAB_BUILD_APPS=ON -DVOICELAB_WITH_WHISPER=ON
//        cmake --build build --target transcribe_demo

// ---- Windows API -----------------------------------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX          // prevent windows.h from defining min/max macros
#include <windows.h>
#include <commdlg.h>    // GetOpenFileNameA
#include <mmsystem.h>   // PlaySoundA

// ---- GLFW (brings in GL/gl.h on Windows) -----------------------------------
#include <GLFW/glfw3.h>

// ---- Additional GL constants not in gl.h 1.1 -------------------------------
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// ---- Dear ImGui ------------------------------------------------------------
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// ---- Standard library ------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ---- voicelab --------------------------------------------------------------
#include "voicelab/core/stft.hpp"
#include "voicelab/core/features.hpp"
#include "voicelab/recognize/whisper_backend.hpp"
#include "../examples/wav_io.hpp"

#ifndef DEMO_WAV_PATH
#define DEMO_WAV_PATH "simulation_output/my_voice.wav"
#endif

// ============================================================================
// Helpers
// ============================================================================
namespace {

// ---- Viridis colormap (5 key stops, linear interpolation) -----------------
struct RGB8 { uint8_t r, g, b; };

RGB8 viridis(float t) noexcept {
    static const float kr[] = {0.267F, 0.231F, 0.134F, 0.478F, 0.993F};
    static const float kg[] = {0.005F, 0.323F, 0.659F, 0.821F, 0.906F};
    static const float kb[] = {0.329F, 0.533F, 0.518F, 0.318F, 0.144F};
    t = std::clamp(t, 0.0F, 1.0F);
    float idx = t * 4.0F;
    int   i   = std::min(static_cast<int>(idx), 3);
    float f   = idx - static_cast<float>(i);
    auto lerp = [&](const float* a) -> uint8_t {
        return static_cast<uint8_t>((a[i] + (a[i+1]-a[i])*f) * 255.0F + 0.5F);
    };
    return {lerp(kr), lerp(kg), lerp(kb)};
}

// ---- Linear resampler -----------------------------------------------------
std::vector<float> resample_linear(std::span<const float> in,
                                    double in_rate, double out_rate) {
    if (in.empty()) return {};
    const double ratio = out_rate / in_rate;
    const auto   nout  = static_cast<std::size_t>(
        std::floor(static_cast<double>(in.size()) * ratio));
    std::vector<float> out(nout);
    for (std::size_t i = 0; i < nout; ++i) {
        const double t    = static_cast<double>(i) / ratio;
        const auto   j    = static_cast<std::size_t>(t);
        const float  frac = static_cast<float>(t - static_cast<double>(j));
        out[i] = (j + 1 < in.size())
                   ? in[j] * (1.0F - frac) + in[j + 1] * frac
                   : in[j];
    }
    return out;
}

// ---- Win32 file dialog ----------------------------------------------------
bool open_wav_dialog(char path[], std::size_t maxlen) {
    OPENFILENAMEA ofn{};
    char buf[MAX_PATH] = {};
    std::strncpy(buf, path, sizeof(buf) - 1);
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "WAV Files (*.wav)\0*.wav\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile   = buf;
    ofn.nMaxFile    = sizeof(buf);
    ofn.Flags       = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle  = "Select WAV File";
    if (GetOpenFileNameA(&ofn)) {
        std::strncpy(path, buf, maxlen - 1);
        path[maxlen - 1] = '\0';
        return true;
    }
    return false;
}

// ---- WAV playback ---------------------------------------------------------
void play_wav(const char* path) {
    PlaySoundA(path, nullptr, SND_FILENAME | SND_ASYNC);
}

// ---- Spectrogram (OpenGL texture) -----------------------------------------
struct SpecTex {
    GLuint id     = 0;
    int    width  = 0;
    int    height = 0;
    bool   ready  = false;

    void destroy() noexcept {
        if (id) { glDeleteTextures(1, &id); id = 0; }
        ready = false; width = 0; height = 0;
    }
};

SpecTex build_spectrogram(const char* wav_path) {
    SpecTex out;
    try {
        auto wav = examples_wav::load_wav_pcm16(wav_path);
        if (wav.channels == 0 || wav.samples.empty()) return out;

        // Mix to mono
        std::vector<float> mono;
        if (wav.channels == 2) {
            mono.resize(wav.samples.size() / 2);
            for (std::size_t i = 0; i < mono.size(); ++i)
                mono[i] = (wav.samples[2*i] + wav.samples[2*i+1]) * 0.5F;
        } else {
            mono = wav.samples;
        }

        // STFT
        constexpr std::size_t N   = 1024;
        constexpr std::size_t HOP = 256;
        voicelab::core::StreamingStft stft({
            .frame_size = N, .hop_size = HOP, .sample_rate = wav.sample_rate
        });
        const std::size_t num_bins = stft.num_bins();  // 513

        // Show bins up to 8 kHz (speech range)
        const std::size_t show_bins = std::min(
            num_bins,
            static_cast<std::size_t>(
                8000.0F / static_cast<float>(wav.sample_rate) * static_cast<float>(N) * 0.5F) + 1u);

        std::vector<std::vector<float>> frames;
        frames.reserve(2048);
        std::vector<float> mags(num_bins);
        stft.process(mono, [&](std::span<const std::complex<float>> spec) {
            voicelab::core::magnitude(spec, mags);
            frames.emplace_back(mags.begin(), mags.begin() + show_bins);
        });
        if (frames.empty()) return out;

        const int W = static_cast<int>(frames.size());
        const int H = static_cast<int>(show_bins);

        // Find global max for log-scale normalization
        float max_val = 1e-8F;
        for (const auto& fr : frames)
            for (float v : fr) max_val = std::max(max_val, v);

        const float log_max = std::log10(max_val + 1e-8F);
        const float log_min = log_max - 4.0F;  // 80 dB dynamic range

        // Build RGB pixels (Y=0 is top of image → freq high; H-1-bin flips it)
        std::vector<uint8_t> pixels(static_cast<std::size_t>(W * H * 3));
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                float lv  = std::log10(frames[x][y] + 1e-8F);
                float t   = std::clamp((lv - log_min) / (log_max - log_min),
                                       0.0F, 1.0F);
                auto [r, g, b] = viridis(t);
                int flip  = H - 1 - y;   // low freq at bottom
                std::size_t idx = static_cast<std::size_t>(flip * W + x) * 3;
                pixels[idx]   = r;
                pixels[idx+1] = g;
                pixels[idx+2] = b;
            }
        }

        // Upload to OpenGL
        GLuint tex = 0;
        glGenTextures(1, &tex);
        if (tex == 0) return out;   // GL context not ready
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   // RGB rows are not 4-byte aligned
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);   // restore default
        glBindTexture(GL_TEXTURE_2D, 0);

        out.id = tex; out.width = W; out.height = H; out.ready = true;
    } catch (...) {}
    return out;
}

// ---- App state ------------------------------------------------------------
enum class State { Idle, Running, Done, Error };

struct AppState {
    char wav_path  [MAX_PATH] = DEMO_WAV_PATH;
    char model_path[MAX_PATH] = "models/ggml-model.bin";
    std::string        result_text;
    std::string        error_text;
    std::atomic<State> state{State::Idle};
    std::mutex         mu;
    SpecTex            spec;
    bool               spec_dirty = true;
};

void run_transcription(AppState& app) {
    app.state.store(State::Running);
    try {
        if (!voicelab::recognize::has_whisper_support())
            throw std::runtime_error(
                "Whisper support not compiled in.\n"
                "Reconfigure with -DVOICELAB_WITH_WHISPER=ON.");

        auto wav = examples_wav::load_wav_pcm16(app.wav_path);

        std::vector<float> mono;
        if (wav.channels == 2) {
            mono.resize(wav.samples.size() / 2);
            for (std::size_t i = 0; i < mono.size(); ++i)
                mono[i] = (wav.samples[2*i] + wav.samples[2*i+1]) * 0.5F;
        } else {
            mono = wav.samples;
        }

        std::vector<float> pcm16k =
            (wav.sample_rate != 16000)
                ? resample_linear(mono, wav.sample_rate, 16000.0)
                : std::move(mono);

        auto text = voicelab::recognize::transcribe_once(
            app.model_path, pcm16k, "ja");

        std::lock_guard<std::mutex> lk(app.mu);
        app.result_text = text.empty() ? "(認識結果なし)" : text;
        app.state.store(State::Done);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(app.mu);
        app.error_text = e.what();
        app.state.store(State::Error);
    }
}

bool try_load_jp_font(ImGuiIO& io, float sz) {
    const char* fonts[] = {
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        nullptr
    };
    for (const char** p = fonts; *p; ++p) {
        if (std::filesystem::exists(*p)) {
            io.Fonts->AddFontFromFileTTF(*p, sz, nullptr,
                io.Fonts->GetGlyphRangesJapanese());
            return true;
        }
    }
    return false;
}

} // namespace

// ============================================================================
int main(int argc, char** argv) {
    bool autorun = (argc >= 2 && std::string_view(argv[1]) == "--autorun");

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window sized to fit within the primary monitor
    GLFWmonitor*       mon  = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = mon ? glfwGetVideoMode(mon) : nullptr;
    const int screen_h = mode ? mode->height : 800;
    // Client height = screen - title bar (35) - taskbar (45) to stay fully on screen
    const int win_h = std::min(screen_h - 35 - 45, 760);
    const int win_w = std::min(mode ? mode->width - 20 : 960, 1200);
    GLFWwindow* window = glfwCreateWindow(
        win_w, win_h, "Voicelab Transcribe Demo", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwSetWindowPos(window, 0, 0);   // top-left so nothing is clipped
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    if (!try_load_jp_font(io, 17.0F))
        io.Fonts->AddFontDefault();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    AppState    app;
    std::thread worker;
    bool        autorun_done = false;
    State       prev_state   = State::Idle;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const auto state = app.state.load();
        const bool busy  = (state == State::Running);

        // Build spectrogram when file changes (main thread — GL requirement)
        if (app.spec_dirty && !busy) {
            app.spec.destroy();
            app.spec = build_spectrogram(app.wav_path);
            app.spec_dirty = false;
        }

        // Auto-play WAV when transcription finishes
        if (prev_state == State::Running && state == State::Done)
            play_wav(app.wav_path);
        prev_state = state;

        // Full-screen ImGui window
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("##root", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar);

        // ---- Header -------------------------------------------------------
        ImGui::TextColored({0.4F, 0.8F, 1.0F, 1.0F}, "Voicelab Transcribe Demo");
        ImGui::Separator();
        ImGui::Spacing();

        // ---- WAV file path + picker button --------------------------------
        ImGui::TextUnformatted("音声ファイル:");
        ImGui::SameLine();
        float pick_btn_w = 90.0F;
        ImGui::SetNextItemWidth(
            ImGui::GetContentRegionAvail().x - pick_btn_w - ImGui::GetStyle().ItemSpacing.x);
        if (ImGui::InputText("##wav", app.wav_path, sizeof(app.wav_path),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            app.spec_dirty = true;
            std::lock_guard<std::mutex> lk(app.mu);
            app.result_text.clear(); app.error_text.clear();
            app.state.store(State::Idle);
        }
        ImGui::SameLine();
        if (ImGui::Button("ファイル選択", {pick_btn_w, 0})) {
            if (open_wav_dialog(app.wav_path, sizeof(app.wav_path))) {
                app.spec_dirty = true;
                std::lock_guard<std::mutex> lk(app.mu);
                app.result_text.clear(); app.error_text.clear();
                app.state.store(State::Idle);
            }
        }
        ImGui::Spacing();

        // ---- Model path ---------------------------------------------------
        ImGui::TextUnformatted("モデルパス  :");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.0F);
        ImGui::InputText("##model", app.model_path, sizeof(app.model_path));
        ImGui::TextDisabled(
            "  ggml モデルDL先: https://huggingface.co/ggerganov/whisper.cpp");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ---- Transcribe button --------------------------------------------
        if (busy) ImGui::BeginDisabled();
        const bool clicked = ImGui::Button("  文字起こし実行  ");
        if (busy) ImGui::EndDisabled();

        // --autorun: start immediately on first frame
        if (autorun && !autorun_done) {
            autorun_done = true;
            worker = std::thread(run_transcription, std::ref(app));
        }
        if (clicked && !busy) {
            if (worker.joinable()) worker.join();
            { std::lock_guard<std::mutex> lk(app.mu);
              app.result_text.clear(); app.error_text.clear(); }
            worker = std::thread(run_transcription, std::ref(app));
        }

        if (busy) {
            ImGui::SameLine();
            static float anim = 0.0F; anim += io.DeltaTime;
            const char* dots[] = {"処理中   ", "処理中.  ", "処理中.. ", "処理中..."};
            ImGui::TextUnformatted(dots[static_cast<int>(anim * 3.0F) % 4]);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ---- Available vertical space split: result top, spectrogram bottom
        //      Spectrogram gets 40% of available height (min 180px, max 240px).
        const float avail_h    = ImGui::GetContentRegionAvail().y;
        const float SPEC_PANEL_H = std::clamp(avail_h * 0.40F, 160.0F, 240.0F);
        constexpr float LABEL_H  = 22.0F;   // one text row
        constexpr float GAP_H    = 8.0F;
        const float res_h = std::max(avail_h - SPEC_PANEL_H - LABEL_H * 2.0F - GAP_H,
                                     60.0F);

        // ---- Transcription result -----------------------------------------
        ImGui::TextUnformatted("認識結果:");
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12F, 0.12F, 0.12F, 1.0F));
        ImGui::BeginChild("##result", ImVec2(-1.0F, res_h), true);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            if (state == State::Error && !app.error_text.empty()) {
                ImGui::TextColored({1.0F, 0.4F, 0.4F, 1.0F}, "エラー:");
                ImGui::TextWrapped("%s", app.error_text.c_str());
            } else if (!app.result_text.empty()) {
                ImGui::Spacing();
                ImGui::SetWindowFontScale(1.2F);
                ImGui::TextWrapped("%s", app.result_text.c_str());
                ImGui::SetWindowFontScale(1.0F);
            } else {
                ImGui::TextDisabled("(ここに認識結果が表示されます)");
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::Spacing();

        // ---- Spectrogram -------------------------------------------------
        ImGui::TextUnformatted("スペクトログラム:");
        ImGui::SameLine();
        if (app.spec.ready) {
            ImGui::TextDisabled("  %d x %d  texID=%u  (低周波↓ 高周波↑)",
                                app.spec.width, app.spec.height, app.spec.id);
        } else {
            ImGui::TextDisabled("  (ファイル読込後に表示)");
        }

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05F, 0.05F, 0.05F, 1.0F));
        if (ImGui::BeginChild("##spec", ImVec2(-1.0F, SPEC_PANEL_H), true)) {
            if (app.spec.ready) {
                const ImVec2 sz = ImGui::GetContentRegionAvail();
                ImGui::Image(
                    (ImTextureID)(intptr_t)app.spec.id,
                    sz);
            } else {
                ImGui::Spacing();
                ImGui::TextDisabled("  スペクトログラムを計算中...");
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::End();

        // ---- Render -------------------------------------------------------
        ImGui::Render();
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.08F, 0.08F, 0.08F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (worker.joinable()) worker.join();
    app.spec.destroy();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
