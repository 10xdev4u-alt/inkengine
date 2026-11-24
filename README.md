# inkengine

Custom inference engine for **Inkling 2 Mini** (looped MoE, native NVFP4+MXFP8).
Week 0: a bit-exact C++ dequant anchor generated from `loopkit`.

> **Source fact-check (2026-11-24).** Upstream llama.cpp already provides the Inkling2 looped graph
> and converter (`LLM_ARCH_INKLING2_LOOP`, `src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`,
> `conversion/inkling2.py`) with cache slots
> `n_slot = n_phys + (n_passes-1)*(core_end-core_start)` and rank-r gain/forget loop controllers
> (`LLM_TENSOR_LOOP_Q_CTRL_*` etc., rank 64). This fork's contribution is union-resident MoE
> routing on top of upstream, not building the loop graph from scratch.

## Decision: fork llama.cpp, don't greenfield

On Kaggle T4 (sm_75, no FP4/FP8 tensor cores) llama.cpp's ggml path already sits
at the HBM bandwidth ceiling for decode. A greenfield engine would spend its time
re-deriving MoE/attention/TP and arrive at parity with nothing left to verify.
The genuinely new work is loop-specific and additive:

- **RORET** — route once, keep expert tiles resident across the 3 loop passes
  (in our simplified local model `gamma` starts at 0, so routing is likely stable; upstream gate
   is rank-r gain/forget controllers, not a scalar gamma).
- **PIBS** — pass-interleaved batching: fold `(token, pass)` pairs into one GEMM,
  M = batch × 3 instead of batch.
- **ODSL** — dequant to FP16 WMMA with a block-major scale plane and an FP32
  side-lane for outlier dimension 2027 **[unverified local hypothesis; not in primary sources]**.

See [docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) and
[docs/ENGINE_RISK.md](docs/ENGINE_RISK.md).

## Week 0 — bit-exact anchor (this repo)

The C++ dequant path must be **bitwise identical** to Python `loopkit`.
LUTs and 16,000 test values are generated from loopkit, not hand-typed:

```bash
PYTHONPATH=/tmp/build/loopkit/src python3 tools/gen_vectors.py
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

This checks signed zero, the 16-level E2M1 codebook, and the E4M3 block scale
byte-for-byte. The MXFP8 anchor covers the other native format: each 32-element
block stores E4M3 codes plus one E8M0 exponent byte, and `ink::dequant_mxfp8_block`
scales by 2^(exp-127). Its 500 blocks are generated from `loopkit`'s
`quantize_mxfp8`/`dequantize_mxfp8` into `tests/vectors_mxfp8.bin`, alongside a
constexpr E4M3 table in `include/ink_mxfp8.hpp`; the test compares all 16,000
MXFP8 float32 results bit-for-bit, including signed zeros and clamped exponents.
Regenerate the LUT headers and both vector files with the same
`PYTHONPATH=... python3 tools/gen_vectors.py` command above. No speed claims yet.

### Entropy-coded stream reader

`include/ink_codec.hpp` decodes loopkit's `ANSStream` payloads (`codec =
"deflate-v1"`): `codes` and `scales` are raw zlib streams, and the JSON header
carries `n`, `n_scales`, `shape`, `block`, and `codec`. The parser is
hand-written (no JSON dependency); decompression uses `uncompress()` with a
growable buffer and requires the inflated sizes to match `n` and `n_scales`
exactly. Unknown codecs, malformed headers, and truncated/undersized payloads
are rejected. The reader writes to the output tensor only on success.

The `codec_reader` test checks a fixture generated from `loopkit.ans_encode`
against the exact codes/scales printed by Python, plus rejection cases. To
regenerate the fixture after changing the generator:

```bash
PYTHONPATH=/tmp/build/loopkit/src python3 tools/make_codec_fixture.py
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Honesty

T4 has no FP4/FP8 tensor cores; FP4 is a storage win there, not a speed win.
All speed claims must be verified with `loopkit`'s CycleAudit and must be below
its 95% lower confidence bound.
