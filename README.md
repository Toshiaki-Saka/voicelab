# voicelab

Modern C++ (C++20) library for real-time audio signal processing.
Streaming STFT/ISTFT, mel/MFCC features, voice-activity detection, phase-vocoder
pitch shifting, spectrogram rendering, and an optional bridge to
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech-to-text.

The library is intentionally small and dependency-light. It is designed to
serve two audiences at once:

- **Learners** who want to read the code and understand how STFT, mel
  filterbanks, overlap-add and phase vocoders work, with comments and
  references inline.
- **Builders** who want a no-fuss real-time audio pipeline they can drop
  into a desktop or embedded application.

> **Scope.** voicelab is a focused, *readable* implementation of the core
> real-time audio DSP building blocks (~2.4k lines). It is **not** a replacement
> for [librosa](https://librosa.org/), [Essentia](https://essentia.upf.edu/) or
> [aubio](https://aubio.org/) and does not aim for feature completeness — its
> value is in being small, dependency-light and easy to read line by line.

## Status

Pre-1.0. Public headers may break. Bug reports and patches welcome.

## Pipeline at a glance

```
                            +-- magnitude / power ----+
                            |                         |
  mic (RtAudio) ─► STFT ────┼-- mel → log → MFCC ─────► features
                            |                         |
                            +-- modify ─► ISTFT ──────► audio out
                            |             (OLA)        (effects)
                            |
                            +-- whisper.cpp ──────────► text
```

The "modify" branch is where pitch shifting, denoising, voice changing, etc.
live; the "whisper.cpp" branch is the neural ASR escape hatch when you want
real recognition instead of writing one from scratch.

## Building

Requirements: a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+) and CMake 3.24+.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The default build is "offline" — pocketfft (BSD-3) only, no live audio, no
neural model. To enable live capture and speech recognition:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOICELAB_WITH_RTAUDIO=ON \
  -DVOICELAB_WITH_WHISPER=ON
cmake --build build --parallel
```

### Speech-to-text model

The whisper.cpp bridge needs a ggml model file at run time (it is **not**
bundled — the `tiny`/`base` models are ~75–148 MB). Download one with the
script that whisper.cpp ships and point voicelab at it:

```bash
# from a whisper.cpp checkout:
./models/download-ggml-model.sh base.en
# -> models/ggml-base.en.bin
```

> The `models/` directory is git-ignored on purpose. **Do not commit the
> `*.bin` weights** — they are large binaries and are obtained per the above.

CMake options:

| Option                          | Default | Effect                                  |
|---------------------------------|---------|-----------------------------------------|
| `VOICELAB_BUILD_EXAMPLES`       | ON      | Build the `examples/` programs          |
| `VOICELAB_BUILD_TESTS`          | ON      | Build Catch2-based unit tests           |
| `VOICELAB_BUILD_APPS`           | OFF     | Build the ImGui desktop demo (needs Dear ImGui + GLFW) |
| `VOICELAB_WITH_RTAUDIO`         | OFF     | Pull in RtAudio for live I/O            |
| `VOICELAB_WITH_WHISPER`         | OFF     | Pull in whisper.cpp for ASR             |
| `VOICELAB_WARNINGS_AS_ERRORS`   | OFF     | `-Werror` / `/WX`                       |

### One-command build + visualization (Windows)

`build_and_visualize.ps1` is a convenience wrapper for Windows/PowerShell that
configures and builds the project, generates a chirp WAV, runs the
`01_sine_spectrum` and `04_mfcc_dump` examples, and opens a matplotlib GUI
showing the waveform, spectrogram and MFCCs. It needs Python with `matplotlib`
and `numpy` (installed automatically if missing).

```powershell
.\build_and_visualize.ps1                 # Release
.\build_and_visualize.ps1 -BuildType Debug
```

### Python GUI (`apps/transcribe_demo.py`)

A PyQt6 front-end: pick a WAV, see its waveform, spectrogram and MFCCs, and
transcribe it with Whisper.

**It computes no DSP of its own.** The STFT, the mel filterbank and the DCT all
run in the C++ core through the `05_analyze_wav` example, which prints the
spectrogram and MFCC matrices as text; the GUI parses them and draws. Build the
examples first, or the analysis panel will tell you what to run:

```bash
cmake -S . -B build && cmake --build build --config Release
pip install PyQt6 numpy matplotlib scipy openai-whisper
python apps/transcribe_demo.py
```

Set `VOICELAB_ANALYZER` to point at the executable if it lives outside the usual
`build/examples/` layout. `scipy` is still needed, but only for `resample_poly`
when handing audio to Whisper at 16 kHz — none of the spectral analysis is
duplicated in Python.

`05_analyze_wav` is a normal CLI, so it is also the easiest way to get the
numbers into any other tool:

```bash
./build/examples/05_analyze_wav simulation_output/chirp.wav 13 40 8000 > analysis.txt
```

## Quick start

Take a sine wave, run streaming STFT, print the peak bin:

```cpp
#include <voicelab/core/stft.hpp>

voicelab::core::StreamingStft stft({.frame_size = 2048,
                                    .hop_size   = 512,
                                    .sample_rate = 44100});

stft.process(samples, [&](std::span<const std::complex<float>> spec) {
    // ... do something per frame ...
});
```

Live mic → peak frequency (requires `VOICELAB_WITH_RTAUDIO=ON`):

```cpp
voicelab::io::AudioInputStream mic({.sample_rate = 44100, .channels = 1});
mic.on_audio([&](std::span<const float> block) {
    stft.process(block, [&](auto spec) { /* ... */ });
});
mic.start();
```

Live mic → whisper.cpp transcription (requires both `RTAUDIO` and `WHISPER` ON):
see `examples/20_live_transcribe.cpp`.

## Layout

```
include/voicelab/   Public headers, namespaced as voicelab::{core,io,effects,recognize,visualize}
src/                Implementation
examples/           Standalone programs, each focusing on one concept
tests/              Catch2 unit tests
cmake/              Helper modules (warnings, deps via FetchContent)
docs_en/            Design notes, algorithm derivations (English, authoritative)
docs_ja/            Japanese translation of docs_en/
```

## Design notes

**Allocation discipline.** Hot paths (everything called from an audio
callback) allocate only at construction. `process()` and `push_frame()` reuse
internal buffers. The public surface uses `std::span` for borrowed views and
`std::function` is reserved for cold setup paths.

**Streaming first.** STFT is offered as an incremental processor (`process`)
rather than a batch function. The same applies to the inverse STFT: you push
spectra in and pull samples out at whatever block size your sink wants.

**Dependencies.** pocketfft (BSD-3) is the only required dependency, vendored
via `FetchContent`. RtAudio and whisper.cpp are pulled in only when their
features are turned on, and have permissive licenses (MIT) that compose
cleanly with this library's Apache-2.0 license.

**Portability.** CI builds on Linux, macOS and Windows. No POSIX-specific
calls leak into the public API.

## License

Apache-2.0. See [LICENSE](LICENSE). Third-party dependencies retain their own
licenses; review them before redistribution.
