include(FetchContent)

# We want consistent shared/static behavior across deps:
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# ---- pocketfft (header-only) -------------------------------------------
# BSD-3-Clause. The "cpp" branch is the C++ header-only variant.
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        cpp
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pocketfft)

# pocketfft does not ship a CMake target — wrap it ourselves.
if(NOT TARGET pocketfft::pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft SYSTEM INTERFACE
        ${pocketfft_SOURCE_DIR}
    )
    add_library(pocketfft::pocketfft ALIAS pocketfft)
endif()

# ---- RtAudio (optional, live I/O) --------------------------------------
if(VOICELAB_WITH_RTAUDIO)
    FetchContent_Declare(
        rtaudio
        GIT_REPOSITORY https://github.com/thestk/rtaudio.git
        GIT_TAG        6.0.1
        GIT_SHALLOW    TRUE
    )
    set(RTAUDIO_BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    set(RTAUDIO_BUILD_STATIC_LIBS   ON  CACHE BOOL "" FORCE)
    set(RTAUDIO_BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(rtaudio)
endif()

# ---- whisper.cpp (optional, neural ASR) --------------------------------
if(VOICELAB_WITH_WHISPER)
    FetchContent_Declare(
        whisper_cpp
        GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
        GIT_TAG        v1.5.4
        GIT_SHALLOW    TRUE
    )
    set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(whisper_cpp)
endif()

# ---- GLFW + Dear ImGui (optional, GUI apps) --------------------------------
if(VOICELAB_BUILD_APPS)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.90.9
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)

    add_library(imgui_impl STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui_impl PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui_impl PUBLIC glfw opengl32)
endif()

# ---- Catch2 (tests) -----------------------------------------------------
if(VOICELAB_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()
