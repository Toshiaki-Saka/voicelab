#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

#include "voicelab/core/ring_buffer.hpp"

using namespace voicelab::core;

TEST_CASE("RingBuffer: push then pop preserves order", "[ring]") {
    RingBuffer<int> rb(16);
    std::array<int, 5> in{1, 2, 3, 4, 5};
    rb.push(in);
    std::array<int, 5> out{};
    REQUIRE(rb.pop(out) == 5);
    REQUIRE(out == in);
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer: capacity is rounded up to power of two",
          "[ring]") {
    RingBuffer<float> rb(7);
    REQUIRE(rb.capacity() == 8);
}

TEST_CASE("RingBuffer: overflow drops oldest", "[ring]") {
    RingBuffer<int> rb(4);  // capacity 4
    std::array<int, 6> in{1, 2, 3, 4, 5, 6};
    rb.push(in);
    REQUIRE(rb.size() == 4);
    std::array<int, 4> out{};
    REQUIRE(rb.pop(out) == 4);
    // Oldest two (1, 2) should have been dropped.
    REQUIRE(out == std::array<int, 4>{3, 4, 5, 6});
}
