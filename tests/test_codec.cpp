#include "ink_codec.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return {};
    }
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string read_text(const std::string& path) {
    const auto bytes = read_bytes(path);
    return std::string(bytes.begin(), bytes.end());
}

// Expected values printed by tools/make_codec_fixture.py (loopkit, rng 20261124).
constexpr std::array<uint8_t, 34> kExpectedCodes = {
    0, 13, 0, 11, 5, 7, 4, 10, 5, 9, 10, 12, 1, 13, 13, 11, 15,
    12, 11, 11, 4, 9, 4, 2, 9, 3, 1, 11, 7, 3, 3, 2, 3, 0,
};
constexpr std::array<uint8_t, 4> kExpectedScales = {48, 37, 47, 1};

std::string replace_once(std::string s, const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at != std::string::npos) s.replace(at, from.size(), to);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <header.json> <codes.bin> <scales.bin>\n", argv[0]);
        return 2;
    }
    const std::string header = read_text(argv[1]);
    const auto codes_payload = read_bytes(argv[2]);
    const auto scales_payload = read_bytes(argv[3]);
    if (header.empty() || codes_payload.empty() || scales_payload.empty()) return 2;

    ink::EncodedTensor tensor;
    const bool ok = ink::decode_stream(header, codes_payload, scales_payload, tensor);
    expect(ok, "valid deflate-v1 stream rejected");
    if (!ok) return 1;

    expect(tensor.codes.size() == kExpectedCodes.size(), "codes size");
    expect(tensor.scales.size() == kExpectedScales.size(), "scales size");
    expect(tensor.shape == std::vector<int64_t>({2, 17}), "shape");
    expect(tensor.block == 16, "block");
    if (tensor.codes.size() == kExpectedCodes.size()) {
        for (std::size_t i = 0; i < kExpectedCodes.size(); ++i) {
            if (tensor.codes[i] != kExpectedCodes[i]) {
                std::fprintf(stderr, "codes[%zu]: got %u want %u\n", i,
                             unsigned(tensor.codes[i]), unsigned(kExpectedCodes[i]));
                ++failures;
                break;
            }
        }
    }
    if (tensor.scales.size() == kExpectedScales.size()) {
        for (std::size_t i = 0; i < kExpectedScales.size(); ++i) {
            if (tensor.scales[i] != kExpectedScales[i]) {
                std::fprintf(stderr, "scales[%zu]: got %u want %u\n", i,
                             unsigned(tensor.scales[i]), unsigned(kExpectedScales[i]));
                ++failures;
                break;
            }
        }
    }
    std::printf("decoded element count: %zu (n=%llu, scales=%zu, block=%d)\n",
                tensor.codes.size(), (unsigned long long)kExpectedCodes.size(),
                tensor.scales.size(), tensor.block);

    ink::EncodedTensor scratch;
    expect(!ink::decode_stream(replace_once(header, "deflate-v1", "deflate-v2"),
                               codes_payload, scales_payload, scratch),
           "wrong codec accepted");

    auto truncated_codes = codes_payload;
    truncated_codes.pop_back();
    expect(!ink::decode_stream(header, truncated_codes, scales_payload, scratch),
           "truncated codes payload accepted");

    auto truncated_scales = scales_payload;
    truncated_scales.pop_back();
    expect(!ink::decode_stream(header, codes_payload, truncated_scales, scratch),
           "truncated scale payload accepted");

    // Declared size smaller than the real stream: must grow, then reject the
    // mismatch rather than silently decompressing into a short buffer.
    const std::string small_n = R"({"n":8,"n_scales":4,"shape":[2,17],"block":16,"codec":"deflate-v1"})";
    expect(!ink::decode_stream(small_n, codes_payload, scales_payload, scratch),
           "underdeclared n accepted");
    const std::string big_n = R"({"n":35,"n_scales":4,"shape":[2,17],"block":16,"codec":"deflate-v1"})";
    expect(!ink::decode_stream(big_n, codes_payload, scales_payload, scratch),
           "overdeclared n accepted");

    expect(!ink::decode_stream("{", codes_payload, scales_payload, scratch),
           "malformed header accepted");
    expect(!ink::decode_stream(R"({"n":34,"shape":[2,17],"block":16,"codec":"deflate-v1"})",
                               codes_payload, scales_payload, scratch),
           "header missing n_scales accepted");

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("codec tests passed");
    return 0;
}
