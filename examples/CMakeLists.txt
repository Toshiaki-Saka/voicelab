function(add_voicelab_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE voicelab::core voicelab_warnings)
endfunction()

add_voicelab_example(01_sine_spectrum)
add_voicelab_example(02_spectrogram_wav)
add_voicelab_example(03_pitch_shift_offline)
add_voicelab_example(04_mfcc_dump)
# Backs apps/transcribe_demo.py — the GUI plots what this prints rather than
# recomputing the STFT and mel filterbank in scipy.
add_voicelab_example(05_analyze_wav)

if(VOICELAB_WITH_RTAUDIO)
    add_voicelab_example(10_live_peak_freq)
endif()

if(VOICELAB_WITH_RTAUDIO AND VOICELAB_WITH_WHISPER)
    add_voicelab_example(20_live_transcribe)
endif()
