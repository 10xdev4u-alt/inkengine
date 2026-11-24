#!/usr/bin/env python3
"""Generate a deflate-v1 codec fixture from loopkit's ans_encode.

Run from the repo root with loopkit on the Python path:
    PYTHONPATH=/tmp/build/loopkit/src python3 tools/make_codec_fixture.py
"""
import json
from pathlib import Path

import numpy as np

from loopkit.nvfp4 import ans_encode, quantize_nvfp4

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def main() -> None:
    rng = np.random.default_rng(20261124)
    x = rng.standard_normal((2, 17)).astype(np.float32)
    x[0, 0] = -0.0
    x[0, 5] = 3.0
    x[1, 16] = 0.0
    t = quantize_nvfp4(x)
    s = ans_encode(t)

    header = {
        "n": s.n,
        "n_scales": s.n_scales,
        "shape": list(s.shape),
        "block": s.block,
        "codec": s.codec,
        "unused": {"freq": list(s.freq_table)[:2], "flag": True},
    }
    (TESTS / "fixture_header.json").write_text(json.dumps(header, sort_keys=True) + "\n")
    (TESTS / "fixture_codes.bin").write_bytes(s.payload)
    (TESTS / "fixture_scales.bin").write_bytes(s.scale_payload)

    print(f"n={s.n} n_scales={s.n_scales} shape={tuple(s.shape)} block={s.block}")
    print("codes:", ", ".join(str(int(v)) for v in t.codes.ravel()))
    print("scales:", ", ".join(str(int(v)) for v in t.scales.ravel()))


if __name__ == "__main__":
    main()
