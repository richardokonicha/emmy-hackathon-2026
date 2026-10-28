#!/usr/bin/env python3
"""Generate the all-f64 packs used by dev/run_extra.py.

The open set is entirely f32, so every f64 path in the compiler — the 64-bit
load and store, the 64-bit arithmetic and the f64 reduction shuffle — had
never been run on the runner.  These packs clone two open problems and make
every buffer f64.

    .venv/bin/python dev/make_f64_packs.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "dev" / "cases_oops"

F32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"
F64 = "DataType(name='f64', np=dtype('float64'), nbytes=8)"


def to_f64(text: str) -> str:
    return text.replace(F32, F64).replace("dtype='float'", "dtype='double'")


def build(problem: str, dst_name: str, code: str) -> None:
    src = ROOT / "problems" / problem
    dst = CASES / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    ir_name = next(src.glob("*.ir.json")).name
    for extra in src.glob("*.ir.txt"):
        shutil.copy(extra, dst / extra.name)
    for extra in src.glob("*.ref.cu"):
        shutil.copy(extra, dst / extra.name)

    ir = json.loads((src / ir_name).read_text())
    for node in ir["nodes"].values():
        for out in node.get("outputs", []):
            out["dtype"] = "f64"
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        parts = body if isinstance(body, list) else [body]
        parts = [to_f64(p) for p in parts]
        node["op_fields"]["body"] = parts if isinstance(body, list) else parts[0]
    (dst / ir_name).write_text(json.dumps(ir, indent=1))

    meta = json.loads((src / "meta.json").read_text())
    meta["id"] = dst_name
    meta["code"] = code
    meta["rtol"] = 1e-6
    meta["atol"] = 1e-6
    kernel = meta["kernels"][0]
    for arg in kernel["args"]:
        arg["dtype"] = "f64"
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))

    # the runner fills seeded buffers with default_rng(seed).standard_normal
    inputs = {}
    for arg in kernel["args"]:
        if arg.get("output"):
            continue
        seed = int(arg.get("seed", 0))
        inputs[arg["name"]] = (np.random.default_rng(seed)
                               .standard_normal(arg["shape"])
                               .astype(np.float64))
    if len(inputs) == 1:
        data = next(iter(inputs.values()))
        outputs = {"sum_1": np.array([data.sum()], dtype=np.float64)}
    else:
        total = sum(inputs.values())
        outputs = {arg["name"]: total
                   for arg in kernel["args"] if arg.get("output")}
    np.savez(dst / "expected.npz", **outputs)
    print(f"wrote {dst}")


def build_pow_f64() -> None:
    """A pow whose *operand* is f64 but whose result is f32.

    emmy promotes ``pow`` to f32 (the overflow guard) but leaves the operand
    at its own precision, so the compiler has to narrow the f64 operand
    before the f32-only pow sequence runs.
    """
    src = CASES / "z08_pow"
    dst = CASES / "z16_pow_f64"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for extra in src.glob("*.ir.txt"):
        shutil.copy(extra, dst / extra.name)

    ir_name = "k_pow_1_pointwise.ir.json"
    ir = json.loads((src / ir_name).read_text())
    for node in ir["nodes"].values():
        for out in node.get("outputs", []):
            if out.get("name") == "x":
                out["dtype"] = "f64"
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        parts = body if isinstance(body, list) else [body]
        new = []
        for part in parts:
            # the x loads become f64
            for tag in ("in2__u0", "in2__u1"):
                start = part.index(f"Load(names=('{tag}',)")
                marker = "), dtype="
                end = part.index(marker, start)
                part = part[:end] + marker + F64 + part[end + len(marker) + len(F32):]
            # abs and add follow the x dtype; pow stays f32
            part = part.replace(
                f"Assign(name='v0__u0', op=ElementwiseImpl('abs'), "
                f"args=('in2__u0',), dtype={F32}",
                f"Assign(name='v0__u0', op=ElementwiseImpl('abs'), "
                f"args=('in2__u0',), dtype={F64}")
            part = part.replace(
                f"Assign(name='v0__u1', op=ElementwiseImpl('abs'), "
                f"args=('in2__u1',), dtype={F32}",
                f"Assign(name='v0__u1', op=ElementwiseImpl('abs'), "
                f"args=('in2__u1',), dtype={F64}")
            for tag in ("v1__u0", "v1__u1"):
                part = part.replace(
                    f"Assign(name='{tag}', op=ElementwiseImpl('add'), "
                    f"args=('in0__{tag[-1]}', 'v0__{tag[-1]}'), dtype={F32}",
                    f"Assign(name='{tag}', op=ElementwiseImpl('add'), "
                    f"args=('in0__{tag[-1]}', 'v0__{tag[-1]}'), dtype={F64}")
            new.append(part)
        node["op_fields"]["body"] = new if isinstance(body, list) else new[0]
    (dst / ir_name).write_text(json.dumps(ir, indent=1))

    meta = json.loads((src / "meta.json").read_text())
    meta["id"] = "z16_pow_f64"
    meta["code"] = "torch.pow(torch.randn(64).double().abs() + 0.5, 3.0)"
    meta["rtol"] = 1e-5
    meta["atol"] = 1e-5
    kernel = meta["kernels"][0]
    for arg in kernel["args"]:
        if arg["name"] == "x":
            arg["dtype"] = "f64"
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))

    x = np.random.default_rng(0).standard_normal([64]).astype(np.float64)
    base = np.abs(x).astype(np.float32) + np.float32(0.5)
    np.savez(dst / "expected.npz", pow_1=(base ** np.float32(3.0)).astype(np.float32))
    print(f"wrote {dst}")


def main() -> int:
    build("p01_vector_add", "z14_f64_pointwise",
          "torch.randn(1024).double() + torch.randn(1024).double()")
    build("r01_array_sum", "z15_f64_reduce",
          "torch.randn(1024).double().sum()")
    build_pow_f64()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
