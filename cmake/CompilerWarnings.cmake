# Strict warnings for our own targets. We DO NOT propagate these to
# third-party deps (FetchContent'd targets) — that would be hostile.
add_library(voicelab_warnings INTERFACE)

if(MSVC)
    target_compile_options(voicelab_warnings INTERFACE
        # /utf-8 is not a warning switch but belongs on every target that
        # compiles our sources. Several headers carry UTF-8 text (arrows, Greek
        # letters in the DSP comments and in one Catch2 test name). Without it
        # MSVC decodes them with the machine's ANSI code page, which warns
        # C4819 and — worse — re-encodes string literals, so a TEST_CASE name
        # stops matching what catch_discover_tests registered with ctest.
        /utf-8
        /W4 /permissive-
        /w14242 /w14254 /w14263 /w14265 /w14287 /we4289
        /w14296 /w14311 /w14545 /w14546 /w14547 /w14549
        /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928
    )
    if(VOICELAB_WARNINGS_AS_ERRORS)
        target_compile_options(voicelab_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(voicelab_warnings INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wconversion -Wsign-conversion -Wnull-dereference
        -Wdouble-promotion -Wformat=2
    )
    if(VOICELAB_WARNINGS_AS_ERRORS)
        target_compile_options(voicelab_warnings INTERFACE -Werror)
    endif()
endif()
