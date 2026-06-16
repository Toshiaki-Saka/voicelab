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

CMake options:

| Option                          | Default | Effect                                  |
|---------------------------------|---------|-----------------------------------------|
| `VOICELAB_BUILD_EXAMPLES`       | ON      | Build the `examples/` programs          |
| `VOICELAB_BUILD_TESTS`          | ON      | Build Catch2-based unit tests           |
| `VOICELAB_BUILD_APPS`           | OFF     | Build the GUI demo (placeholder)        |
| `VOICELAB_WITH_RTAUDIO`         | OFF     | Pull in RtAudio for live I/O            |
| `VOICELAB_WITH_WHISPER`         | OFF     | Pull in whisper.cpp for ASR             |
| `VOICELAB_WARNINGS_AS_ERRORS`   | OFF     | `-Werror` / `/WX`                       |

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
docs/               Design notes, algorithm derivations
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
cleanly with this library's MIT license.

**Portability.** CI builds on Linux, macOS and Windows. No POSIX-specific
calls leak into the public API.

## License

MIT. See [LICENSE](LICENSE). Third-party dependencies retain their own
licenses; review them before redistribution.
