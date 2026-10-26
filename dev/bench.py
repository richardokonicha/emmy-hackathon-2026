#!/usr/bin/env python3
"""Benchmark all problems on the runner endpoint with real warmup/iters timing.

For every problem: run compile.py, ptxas it on the runner, execute with
warmup=20 / iters=200, and check the outputs against expected.npz.
The organizers' hand-written example (examples/k_sum_1_reduce.ptx) is
benchmarked the same way for problems/r01_array_sum as the baseline.

    .venv/bin/python dev/bench.py            # all 15 open problems
    .venv/bin/python dev/bench.py r01_array_sum
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
RUN = "http://runner.riftstack.ai:48701"
COMPILER = ROOT / "compile.py"


def post(path: str, body: dict) -> dict:
    for _ in range(60):
        r = requests.post(f"{RUN}{path}", json=body, timeout=(10, 300))
        r.raise_for_status()
        j = r.json()
        if j.get("stage") != "busy":
            return j
        time.sleep(2)
    raise RuntimeError("runner stayed busy")


def run_kernel(cubin_id: str, entry: str, k: dict, run_args: list, need: int) -> dict:
    body = {
        "cubin_id": cubin_id, "entry": entry,
        "grid": k["grid"], "block": k["block"], "smem_bytes": k["smem_bytes"],
        "zero_outputs": k["zero_outputs"], "args": run_args,
        "warmup": 20, "iters": 200, "max_return": max(4096, need),
    }
    return post("/run", body)


def bench_problem(pdir: Path) -> tuple[bool, list[str]]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced: dict[str, list] = {}
    rows = []
    ok = True

    for k in meta["kernels"]:
        ir = next(pdir.glob(f"{k['entry']}.ir.json"))
        c0 = subprocess.run([sys.executable, str(COMPILER), str(ir)],
                            capture_output=True, text=True)
        if c0.returncode != 0:
            print(f"  [{k['entry']}] compiler crashed\n{c0.stderr}")
            return False, rows
        c = post("/compile", {"ptx": c0.stdout})
        if not c.get("ok"):
            print(f"  [{k['entry']}] ptxas rejected\n{c.get('stderr', '')[:400]}")
            return False, rows

        run_args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            run_args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        r = run_kernel(c["cubin_id"], k["entry"], k, run_args, need)
        if not r.get("ok"):
            print(f"  [{k['entry']}] launch failed: {r.get('stderr', '')[:300]}")
            return False, rows

        for name, out in r["outputs"].items():
            produced[name] = out["values"]
            if name in expected:
                got = np.array(out["values"], dtype=np.float32).reshape(expected[name].shape)
                if np.allclose(got, expected[name], rtol=meta["rtol"], atol=meta["atol"]):
                    err = float(np.max(np.abs(got - expected[name]))) if got.size else 0.0
                    rows.append(f"  {k['entry']:<26} PASS  err={err:.2e}  "
                                f"{r['median_us']:>8.2f} us  {c['registers'].get(k['entry'])} regs")
                else:
                    err = float(np.max(np.abs(got - expected[name])))
                    rows.append(f"  {k['entry']:<26} FAIL  err={err:.3e}")
                    ok = False
    return ok, rows


def bench_reference(ptx_path: Path, entry: str, pdir: Path) -> float:
    """Baseline: organizers' hand-written PTX compiled and timed identically."""
    meta = json.loads((pdir / "meta.json").read_text())
    k = next(k for k in meta["kernels"] if k["entry"] == entry)
    c = post("/compile", {"ptx": ptx_path.read_text()})
    if not c.get("ok"):
        print(f"baseline ptxas failed\n{c.get('stderr', '')[:400]}")
        return float("nan")
    r = run_kernel(c["cubin_id"], entry, k, k["args"], 4096)
    if not r.get("ok"):
        print(f"baseline launch failed: {r.get('stderr', '')[:300]}")
        return float("nan")
    return r["median_us"]


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = sorted(p for p in PROBLEMS.iterdir() if (p / "meta.json").exists())
    if only:
        dirs = [p for p in dirs if p.name == only]

    legacy = ROOT / "examples" / "k_sum_1_reduce.ptx"

    print(f"{'problem':<28} {'result':<5} {'our median (20 wu / 200 it)':>28}")
    all_ok = True
    for pdir in dirs:
        ok, rows = bench_problem(pdir)
        all_ok &= ok
        print(pdir.name + ("  " if ok else "  FAIL"))
        print("\n".join(rows))

        # endpoint-based baseline where the organizers shipped one: the worked
        # example bakes shape 1024 into its PTX, so it is only comparable to r01
        if pdir.name == "r01_array_sum" and legacy.exists():
            us = bench_reference(legacy, "k_sum_1_reduce", pdir)
            print(f"  {'baseline (examples/PTX)':<26}                "
                  f"{us:>8.2f} us")
    print("ALL PASS" if all_ok else "FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
