#pragma once

// Reader for loopkit's entropy-coded NVFP4 streams (codec "deflate-v1").
//
// The header is a small JSON object, e.g.
//   {"n":34,"n_scales":4,"shape":[2,17],"block":16,"codec":"deflate-v1"}
// Both byte payloads are raw DEFLATE/zlib streams produced by
// loopkit.nvfp4.ans_encode (zlib.compress(..., 9)).
//
// Requires zlib at link time (-lz). No third-party JSON library.

#include <zlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ink {

struct EncodedTensor {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<int64_t> shape;
    int block = 0;
};

namespace detail {

// Guard against hostile headers that claim an absurd output size.
inline constexpr uint64_t kMaxStreamBytes = uint64_t{1} << 30;

struct CodecHeader {
    int64_t n = -1;
    int64_t n_scales = -1;
    int64_t block = -1;
    std::vector<int64_t> shape;
    std::string codec;
    bool has_n = false;
    bool has_n_scales = false;
    bool has_block = false;
    bool has_shape = false;
    bool has_codec = false;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    bool parse(CodecHeader& out) {
        ws();
        if (!eat('{')) return false;
        ws();
        if (!eat('}')) {
            while (true) {
                std::string key;
                if (!string(key)) return false;
                ws();
                if (!eat(':')) return false;
                ws();
                if (!field(key, out)) return false;
                ws();
                if (eat('}')) break;
                if (!eat(',')) return false;
                ws();
            }
        }
        ws();
        return pos_ == text_.size();
    }

private:
    std::string_view text_;
    std::size_t pos_ = 0;

    void ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    bool eof() const { return pos_ == text_.size(); }

    bool eat(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; }
        return false;
    }

    bool literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) return false;
        pos_ += word.size();
        return true;
    }

    static bool digit(char c) { return c >= '0' && c <= '9'; }

    // JSON integers only: fractions and exponents are rejected for metadata.
    bool integer(int64_t& out) {
        const bool neg = eat('-');
        if (pos_ >= text_.size()) return false;
        uint64_t value = 0;
        if (text_[pos_] == '0') {
            ++pos_;
            if (pos_ < text_.size() && digit(text_[pos_])) return false;  // leading zero
        } else {
            if (text_[pos_] < '1' || text_[pos_] > '9') return false;
            while (pos_ < text_.size() && digit(text_[pos_])) {
                const uint64_t d = uint64_t(text_[pos_] - '0');
                constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
                if (value > (kMax - d) / 10) return false;
                value = value * 10 + d;
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            return false;
        }
        constexpr uint64_t kMaxI = uint64_t(std::numeric_limits<int64_t>::max());
        if (neg) {
            if (value > kMaxI + 1) return false;
            out = value == kMaxI + 1 ? std::numeric_limits<int64_t>::min() : -int64_t(value);
        } else {
            if (value > kMaxI) return false;
            out = int64_t(value);
        }
        return true;
    }

    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool codeUnit(uint32_t& cp) {
        if (text_.size() - pos_ < 4) return false;
        cp = 0;
        for (int i = 0; i < 4; ++i) {
            const int h = hex(text_[pos_++]);
            if (h < 0) return false;
            cp = (cp << 4) | uint32_t(h);
        }
        return true;
    }

    static void appendUtf8(std::string& s, uint32_t cp) {
        if (cp <= 0x7f) {
            s.push_back(char(cp));
        } else if (cp <= 0x7ff) {
            s.push_back(char(0xc0 | (cp >> 6)));
            s.push_back(char(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            s.push_back(char(0xe0 | (cp >> 12)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(char(0x80 | (cp & 0x3f)));
        } else {
            s.push_back(char(0xf0 | (cp >> 18)));
            s.push_back(char(0x80 | ((cp >> 12) & 0x3f)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(char(0x80 | (cp & 0x3f)));
        }
    }

    bool string(std::string& out) {
        if (!eat('"')) return false;
        out.clear();
        while (pos_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') return true;
            if (c < 0x20) return false;
            if (c != '\\') { out.push_back(char(c)); continue; }
            if (pos_ >= text_.size()) return false;
            const char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!codeUnit(cp)) return false;
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (!eat('\\') || !eat('u')) return false;
                        uint32_t low = 0;
                        if (!codeUnit(low) || low < 0xdc00 || low > 0xdfff) return false;
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                        return false;
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool intArray(std::vector<int64_t>& out) {
        if (!eat('[')) return false;
        out.clear();
        ws();
        if (eat(']')) return true;
        while (true) {
            ws();
            int64_t v = 0;
            if (!integer(v)) return false;
            out.push_back(v);
            ws();
            if (eat(']')) return true;
            if (!eat(',')) return false;
        }
    }

    bool skipNumber() {
        const std::size_t start = pos_;
        eat('-');
        if (pos_ >= text_.size()) return false;
        if (eat('0')) {
            // no further integer digits allowed
        } else {
            if (text_[pos_] < '1' || text_[pos_] > '9') return false;
            while (pos_ < text_.size() && digit(text_[pos_])) ++pos_;
        }
        if (eat('.')) {
            if (pos_ >= text_.size() || !digit(text_[pos_])) return false;
            while (pos_ < text_.size() && digit(text_[pos_])) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || !digit(text_[pos_])) return false;
            while (pos_ < text_.size() && digit(text_[pos_])) ++pos_;
        }
        return pos_ > start;
    }

    bool skipValue(unsigned depth = 0) {
        if (depth > 32) return false;
        ws();
        if (pos_ >= text_.size()) return false;
        switch (text_[pos_]) {
            case '{': {
                ++pos_;
                ws();
                if (eat('}')) return true;
                while (true) {
                    std::string key;
                    if (!string(key)) return false;
                    ws();
                    if (!eat(':')) return false;
                    if (!skipValue(depth + 1)) return false;
                    ws();
                    if (eat('}')) return true;
                    if (!eat(',')) return false;
                    ws();
                }
            }
            case '[': {
                ++pos_;
                ws();
                if (eat(']')) return true;
                while (true) {
                    if (!skipValue(depth + 1)) return false;
                    ws();
                    if (eat(']')) return true;
                    if (!eat(',')) return false;
                    ws();
                }
            }
            case '"': { std::string ignored; return string(ignored); }
            case 't': return literal("true");
            case 'f': return literal("false");
            case 'n': return literal("null");
            default: return skipNumber();
        }
    }

    bool field(const std::string& key, CodecHeader& out) {
        if (key == "codec") {
            if (out.has_codec || !string(out.codec)) return false;
            out.has_codec = true;
            return true;
        }
        if (key == "shape") {
            if (out.has_shape || !intArray(out.shape)) return false;
            out.has_shape = true;
            return true;
        }
        int64_t* target = nullptr;
        bool* seen = nullptr;
        if (key == "n") { target = &out.n; seen = &out.has_n; }
        else if (key == "n_scales") { target = &out.n_scales; seen = &out.has_n_scales; }
        else if (key == "block") { target = &out.block; seen = &out.has_block; }
        else return skipValue();
        if (*seen || !integer(*target)) return false;
        *seen = true;
        return true;
    }
};

// Inflate into a buffer sized from the header, growing on Z_BUF_ERROR. A stream
// that decompresses to more bytes than declared is never silently truncated:
// we grow until the real size is known, then require an exact match.
inline bool inflateExact(const std::vector<uint8_t>& src, int64_t expected,
                         std::vector<uint8_t>& out) {
    if (expected < 0 || uint64_t(expected) > kMaxStreamBytes) return false;
    std::size_t cap = std::max<std::size_t>(std::size_t(expected), 1);
    while (true) {
        out.resize(cap);
        uLongf dest_len = static_cast<uLongf>(cap);
        const Bytef* src_ptr = src.empty()
            ? reinterpret_cast<const Bytef*>("")
            : static_cast<const Bytef*>(src.data());
        const int rc = ::uncompress(out.data(), &dest_len, src_ptr,
                                    static_cast<uLong>(src.size()));
        if (rc == Z_OK) {
            out.resize(static_cast<std::size_t>(dest_len));
            return out.size() == static_cast<std::size_t>(expected);
        }
        if (rc != Z_BUF_ERROR || cap >= kMaxStreamBytes) return false;
        cap = std::min<std::size_t>(kMaxStreamBytes, cap > kMaxStreamBytes / 2
                                                       ? kMaxStreamBytes : cap * 2);
    }
}

}  // namespace detail

inline bool decode_stream(const std::string& json_header,
                          const std::vector<uint8_t>& payload,
                          const std::vector<uint8_t>& scale_payload,
                          EncodedTensor& out) {
    detail::CodecHeader header;
    if (!detail::JsonParser(json_header).parse(header)) return false;
    if (!header.has_n || !header.has_n_scales || !header.has_shape ||
        !header.has_block || !header.has_codec) {
        return false;
    }
    if (header.codec != "deflate-v1") return false;
    if (header.block <= 0 || header.block > std::numeric_limits<int>::max()) return false;
    for (int64_t dim : header.shape) if (dim < 0) return false;

    EncodedTensor decoded;
    decoded.block = static_cast<int>(header.block);
    decoded.shape = std::move(header.shape);
    if (!detail::inflateExact(payload, header.n, decoded.codes)) return false;
    if (!detail::inflateExact(scale_payload, header.n_scales, decoded.scales)) return false;
    out = std::move(decoded);
    return true;
}

}  // namespace ink
