# ---- voicelab_core (no live I/O, no neural models) ---------------------
add_library(voicelab_core STATIC
    core/stft.cpp
    core/istft.cpp
    core/features.cpp
    core/vad.cpp
    visualize/spectrogram.cpp
    effects/pitch_shift.cpp
    io/audio_stream.cpp
    recognize/whisper_stub.cpp
)

target_include_directories(voicelab_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(voicelab_core
    PUBLIC  pocketfft::pocketfft
    PRIVATE voicelab_warnings
)

# Optional backends ------------------------------------------------------
if(VOICELAB_WITH_RTAUDIO)
    target_compile_definitions(voicelab_core PUBLIC VOICELAB_HAS_RTAUDIO=1)
    target_link_libraries(voicelab_core PRIVATE rtaudio)
endif()

if(VOICELAB_WITH_WHISPER)
    target_sources(voicelab_core PRIVATE recognize/whisper_backend.cpp)
    target_compile_definitions(voicelab_core PUBLIC VOICELAB_HAS_WHISPER=1)
    target_link_libraries(voicelab_core PRIVATE whisper)
endif()

# Standard alias so downstream can write `voicelab::core`.
add_library(voicelab::core ALIAS voicelab_core)
