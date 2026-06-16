include(Catch)

add_executable(voicelab_tests
    test_window.cpp
    test_stft.cpp
    test_istft.cpp
    test_features.cpp
    test_vad.cpp
    test_ring_buffer.cpp
)
target_link_libraries(voicelab_tests PRIVATE
    voicelab::core
    voicelab_warnings
    Catch2::Catch2WithMain
)
catch_discover_tests(voicelab_tests)
