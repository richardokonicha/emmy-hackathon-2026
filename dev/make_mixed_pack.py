#!/usr/bin/env python3
"""Generate the mixed-precision pack used by dev/run_extra.py.

The kernel adds an f32 buffer to an f64 buffer.  ``emmy`` stamps the Assign
with the *promoted* dtype (f64) but leaves the operands at their own
precision, so the compiler has to widen the f32 side before the add and
narrow the result again for the f32 destination.  Without those conversions
ptxas sees type-mismatched instructions.

    .venv/bin/python dev/make_mixed_pack.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "problems" / "p01_vector_add"
DST = ROOT / "dev" / "cases_oops" / "z13_mixed_f64"

F32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"
F64 = "DataType(name='f64', np=dtype('float64'), nbytes=8)"


def patch_body(text: str) -> str:
    """Make the x1 loads and the adds f64; x0 stays f32."""
    for tag in ("in1__u0", "in1__u1"):
        start = text.index(f"Load(names=('{tag}',)")
        marker = "), dtype="
        end = text.index(marker, start)
        text = text[:end] + marker + F64 + text[end + len(marker) + len(F32):]
    for name in ("v0__u0", "v0__u1"):
        text = text.replace(
            f"Assign(name='{name}', op=ElementwiseImpl('add'), "
            f"args=('in0__u{name[-1]}', 'in1__u{name[-1]}'), dtype={F32}",
            f"Assign(name='{name}', op=ElementwiseImpl('add'), "
            f"args=('in0__u{name[-1]}', 'in1__u{name[-1]}'), dtype={F64}",
        )
    return text


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for name in ("k_add_pointwise.ir.txt", "k_add_pointwise.ref.cu"):
        if (SRC / name).exists():
            shutil.copy(SRC / name, DST / name)

    ir = json.loads((SRC / "k_add_pointwise.ir.json").read_text())
    for node in ir["nodes"].values():
        # the graph's tensor dtype and the body's Load dtype have to agree:
        # the graph decides the address stride and the buffer layout
        for out in node.get("outputs", []):
            if out.get("name") == "x1":
                out["dtype"] = "f64"
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        parts = body if isinstance(body, list) else [body]
        parts = [patch_body(p) for p in parts]
        node["op_fields"]["body"] = parts if isinstance(body, list) else parts[0]
    (DST / "k_add_pointwise.ir.json").write_text(json.dumps(ir, indent=1))

    meta = json.loads((SRC / "meta.json").read_text())
    meta["id"] = "z13_mixed_f64"
    meta["code"] = "torch.randn(1024) + torch.randn(1024).double()"
    meta["rtol"] = 1e-6
    meta["atol"] = 1e-6
    kernel = meta["kernels"][0]
    for arg in kernel["args"]:
        if arg["name"] == "x1":
            arg["dtype"] = "f64"
    (DST / "meta.json").write_text(json.dumps(meta, indent=1))

    # the runner fills seeded buffers with default_rng(seed).standard_normal
    x0 = np.random.default_rng(0).standard_normal([1024]).astype(np.float32)
    x1 = np.random.default_rng(1).standard_normal([1024]).astype(np.float64)
    expected = (x0.astype(np.float64) + x1).astype(np.float32)
    np.savez(DST / "expected.npz", add=expected)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
