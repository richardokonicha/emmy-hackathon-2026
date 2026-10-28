#!/usr/bin/env python3
"""Generate the negative-base pow pack used by dev/run_extra.py.

``torch.pow`` accepts a negative base when the exponent is an integer, so a
compiler that lowers pow to ``2 ** (y * log2(x))`` returns NaN for it.  The
pack clones the scalar-op pow case and negates its input, which puts half of
the bases below zero.

    .venv/bin/python dev/make_pow_neg.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dev" / "cases_oops" / "z08_pow"
DST = ROOT / "dev" / "cases_oops" / "z12_pow_neg"

# A mix of signs and magnitudes: half of the bases are negative.
X = np.array(
    [-2.0, -1.75, -1.5, -1.25, -1.0, -0.875, -0.75, -0.625,
     -0.5, -0.375, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125,
     0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125,
     1.25, 1.5, 1.75, 2.0, -0.1, 0.1, -0.2, 0.2,
     -0.3, 0.3, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8,
     -1.1, 1.1, -1.3, 1.3, -1.4, 1.4, -1.6, 1.6,
     -1.9, 1.9, -0.05, 0.05, -0.15, 0.15, -0.45, 0.45,
     -0.55, 0.55, -0.65, 0.65, -0.85, 0.85, -0.95, 0.95],
    dtype=np.float32,
)
EXPONENT = np.float32(3.0)


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for name in ("k_pow_1_pointwise.ir.txt", "k_pow_1_pointwise.ref.cu"):
        shutil.copy(SRC / name, DST / name)

    ir = json.loads((SRC / "k_pow_1_pointwise.ir.json").read_text())
    for node in ir["nodes"].values():
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        parts = body if isinstance(body, list) else [body]
        parts = [p.replace("ElementwiseImpl('abs')", "ElementwiseImpl('copy')") for p in parts]
        node["op_fields"]["body"] = parts if isinstance(body, list) else parts[0]
    (DST / "k_pow_1_pointwise.ir.json").write_text(json.dumps(ir, indent=2))

    meta = json.loads((SRC / "meta.json").read_text())
    meta["id"] = "z12_pow_neg"
    meta["code"] = "torch.pow(torch.randn(32), 3.0)"
    meta["rtol"] = 1e-5
    meta["atol"] = 1e-6
    kernel = meta["kernels"][0]
    for arg in kernel["args"]:
        if arg["name"] == "x":
            arg.pop("seed", None)
            arg["data"] = [float(v) for v in X]
        elif arg["name"] == "add_c1":
            arg["data"] = [0.0]
        elif arg["name"] == "pow_1_c1":
            arg["data"] = [float(EXPONENT)]
    (DST / "meta.json").write_text(json.dumps(meta, indent=1))

    expected = (X.astype(np.float64) ** float(EXPONENT)).astype(np.float32)
    np.savez(DST / "expected.npz", pow_1=expected)
    print(f"wrote {DST} ({(expected < 0).sum()} negative bases out of {expected.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
