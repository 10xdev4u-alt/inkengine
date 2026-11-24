#include "ink.hpp"
#include "ink_mxfp8.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <vector>

namespace {

struct Result {
    uint32_t blocks = 0;
    uint64_t values = 0;
    uint64_t mismatches = 0;
};

template <std::size_t N>
using Dequant = std::array<float, N> (*)(std::span<const uint8_t, N>, uint8_t);

template <std::size_t N>
bool check(const char* path, Dequant<N> dequant, Result& r) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    if (data.size() < 4) { std::fprintf(stderr, "short vector file %s\n", path); return false; }
    uint32_t n = data[0] | (data[1] << 8) | (data[2] << 16) | (uint32_t(data[3]) << 24);
    const size_t need = 4 + size_t(n) * (N + 1 + N * 4);
    if (data.size() != need) {
        std::fprintf(stderr, "bad vector size in %s: got %zu want %zu\n", path, data.size(), need);
        return false;
    }
    size_t off = 4;
    for (uint32_t b = 0; b < n; ++b) {
        std::array<uint8_t, N> codes{};
        for (std::size_t i = 0; i < N; ++i) codes[i] = data[off++];
        uint8_t scale = data[off++];
        auto got = dequant(codes, scale);
        for (std::size_t i = 0; i < N; ++i) {
            uint32_t want, have;
            std::memcpy(&want, &data[off + i * 4], 4);
            std::memcpy(&have, &got[i], 4);
            r.mismatches += (want != have);
        }
        off += N * 4;
    }
    r.blocks = n;
    r.values = uint64_t(n) * N;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* nvfp4_path = argc > 1 ? argv[1] : "tests/vectors.bin";
    const char* mxfp8_path = argc > 2 ? argv[2] : "tests/vectors_mxfp8.bin";
    Result nvfp4, mxfp8;
    if (!check<16>(nvfp4_path, ink::dequant_block, nvfp4)) return 2;
    if (!check<32>(mxfp8_path, ink::dequant_mxfp8_block, mxfp8)) return 2;

    std::printf("nvfp4 blocks=%u values=%llu mismatches=%llu\n",
                nvfp4.blocks, (unsigned long long)nvfp4.values,
                (unsigned long long)nvfp4.mismatches);
    std::printf("mxfp8 blocks=%u values=%llu mismatches=%llu\n",
                mxfp8.blocks, (unsigned long long)mxfp8.values,
                (unsigned long long)mxfp8.mismatches);
    return (nvfp4.mismatches == 0 && mxfp8.mismatches == 0) ? 0 : 1;
}
