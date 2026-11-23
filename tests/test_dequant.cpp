#include "ink.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/vectors.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    if (data.size() < 4) return 2;
    uint32_t n = data[0] | (data[1] << 8) | (data[2] << 16) | (uint32_t(data[3]) << 24);
    const size_t need = 4 + size_t(n) * (16 + 1 + 16 * 4);
    if (data.size() != need) {
        std::fprintf(stderr, "bad vector size: got %zu want %zu\n", data.size(), need);
        return 2;
    }
    size_t off = 4;
    uint64_t mismatches = 0;
    for (uint32_t b = 0; b < n; ++b) {
        std::array<uint8_t, 16> codes{};
        for (int i = 0; i < 16; ++i) codes[i] = data[off++];
        uint8_t scale = data[off++];
        auto got = ink::dequant_block(codes, scale);
        for (int i = 0; i < 16; ++i) {
            uint32_t want;
            std::memcpy(&want, &data[off + i * 4], 4);
            uint32_t have;
            std::memcpy(&have, &got[i], 4);
            if (want != have) ++mismatches;
        }
        off += 16 * 4;
    }
    std::printf("blocks=%u mismatches=%llu\n", n, (unsigned long long)mismatches);
    return mismatches == 0 ? 0 : 1;
}
