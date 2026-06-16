add_executable(transcribe_cli ../test_transcribe.cpp)
target_link_libraries(transcribe_cli PRIVATE voicelab::core)
if(MSVC)
    target_compile_options(transcribe_cli PRIVATE /utf-8)
endif()

add_executable(transcribe_demo transcribe_demo.cpp)

target_link_libraries(transcribe_demo PRIVATE
    voicelab::core
    imgui_impl
    $<$<PLATFORM_ID:Windows>:winmm>
)

target_compile_definitions(transcribe_demo PRIVATE
    DEMO_WAV_PATH="${CMAKE_SOURCE_DIR}/simulation_output/my_voice.wav"
)

if(MSVC)
    target_compile_options(transcribe_demo PRIVATE /utf-8)
endif()
