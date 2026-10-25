#!/usr/bin/env python3
"""Compile every open problem and ask the runner's /compile endpoint to ptxas it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "http://runner.riftstack.ai:48701/compile"


def main() -> int:
    failed = []
    for problem in sorted((ROOT / "problems").iterdir()):
        if not (problem / "meta.json").exists():
            continue
        ir = next(problem.glob("*.ir.json"))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "compile.py"), str(ir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            print(f"{problem.name:<24} COMPILER-FAIL")
            print(proc.stderr[-800:])
            failed.append(problem.name)
            continue
        resp = requests.post(URL, data=proc.stdout, timeout=60)
        text = resp.text.strip()
        ok = resp.status_code == 200 and "error" not in text.lower()
        print(f"{problem.name:<24} {'PTXAS-OK' if ok else 'PTXAS-FAIL'}")
        if not ok:
            print(text[:800])
            failed.append(problem.name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
