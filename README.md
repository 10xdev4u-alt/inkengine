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
byte-for-byte. The MXFP8 anchor covers the other native format: `ink::dequant_mxfp8_block`
scales E4M3 codes by 2^(exp-127) from one E8M0 exponent byte. Both constexpr E4M3
tables live in `include/ink_mxfp8.hpp`, generated from the same loopkit table.

### MXFP8: weights use 2-D tiles, activations use flat blocks

**Verified source fact (2026-11-24):** the shipped Inkling 2 Mini
[`hf_quant_config.json`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/hf_quant_config.json)
declares `weight_block_shape: [32, 32]` for the MXFP8 expert-weight group
(layers 20–23), while that same group's `required_activation_quantization`
declares `block_size: 32`. So one E8M0 exponent covers a **32×32 = 1024-element
weight tile**, not a flat 32-element run. Treating MXFP8 weights like activations
gives the wrong scale count and wrong dequantized values. `loopkit` keeps the
two paths separate: `quantize_mxfp8`/`dequantize_mxfp8` for flat activation blocks
of 32, `quantize_mxfp8_weights`/`dequantize_mxfp8_weights` for weight tiles
(`layout="2d"`, scales shaped `(ceil(H/32), ceil(W/32))`).

The generator emits three bit-exact vector files, all compared float32-bit-for-bit
(including signed zeros and clamped exponents):

| File | Values | Source path |
| --- | --- | --- |
| `tests/vectors.bin` | 16,000 (1000 × 16) | `quantize_nvfp4` |
| `tests/vectors_mxfp8.bin` | 16,000 (500 × 32) | `quantize_mxfp8` (activations, 1-D) |
| `tests/vectors_mxfp8_2d.bin` | 307,200 (300 × 1024) | `quantize_mxfp8_weights` (2-D tiles) |

The 2-D file starts with a uint32 tile count; each of the 300 tiles then stores
1024 row-major E4M3 code bytes, one shared E8M0 exponent byte, and 1024 expected
float32 values; `ink::dequant_mxfp8_tile` takes the codes, the exponent byte and
a 1024-float output buffer. `test_dequant` prints mismatch counts for `nvfp4`,
`mxfp8_1d` and `mxfp8_2d` and exits nonzero if any is nonzero (339,200 values
total). Regenerate the LUT headers and all three vector files with the same
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
