cmake_minimum_required(VERSION 3.24)

project(voicelab
    VERSION 0.1.0
    DESCRIPTION "Modern C++ realtime audio processing library: STFT, features, effects, recognition"
    HOMEPAGE_URL "https://github.com/yourname/voicelab"
    LANGUAGES CXX
)

# ---- Standard ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code so the static lib can be linked into shared libs.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clangd / clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options -------------------------------------------------------------
option(VOICELAB_BUILD_EXAMPLES "Build example programs"          ON)
option(VOICELAB_BUILD_TESTS    "Build unit tests"                ON)
option(VOICELAB_BUILD_APPS     "Build GUI demo applications"     OFF)
option(VOICELAB_WITH_RTAUDIO   "Enable RtAudio live input/output" OFF)
option(VOICELAB_WITH_WHISPER   "Enable whisper.cpp backend"      OFF)
option(VOICELAB_WARNINGS_AS_ERRORS "Treat warnings as errors"    OFF)

# ---- Helpers -------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Dependencies)

# ---- Library -------------------------------------------------------------
add_subdirectory(src)

# ---- Subdirs -------------------------------------------------------------
if(VOICELAB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(VOICELAB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VOICELAB_BUILD_APPS)
    add_subdirectory(apps)
endif()

# ---- Summary -------------------------------------------------------------
message(STATUS "")
message(STATUS "voicelab ${PROJECT_VERSION} configuration")
message(STATUS "  C++ standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Examples              : ${VOICELAB_BUILD_EXAMPLES}")
message(STATUS "  Tests                 : ${VOICELAB_BUILD_TESTS}")
message(STATUS "  GUI apps              : ${VOICELAB_BUILD_APPS}")
message(STATUS "  RtAudio live I/O      : ${VOICELAB_WITH_RTAUDIO}")
message(STATUS "  whisper.cpp backend   : ${VOICELAB_WITH_WHISPER}")
message(STATUS "")
