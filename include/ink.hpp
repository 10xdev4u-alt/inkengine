#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include "ink_lut.hpp"

namespace ink {

// Bit-exact NVFP4 dequantization of one 16-element block.
// `codes` has 16 entries in [0,16); `scale_code` is an E4M3 byte.
inline std::array<float, 16> dequant_block(std::span<const uint8_t, 16> codes,
                                           uint8_t scale_code) {
    const float scale = kE4M3[scale_code];
    std::array<float, 16> out{};
    for (int i = 0; i < 16; ++i) {
        out[i] = kFp4Levels[codes[i]] * scale;
    }
    return out;
}

inline bool bitwise_equal(float a, float b) {
    uint32_t ua, ub;
    __builtin_memcpy(&ua, &a, 4);
    __builtin_memcpy(&ub, &b, 4);
    return ua == ub;
}

}  // namespace ink
