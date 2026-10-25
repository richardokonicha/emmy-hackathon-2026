#!/usr/bin/env python3
"""Compile and run every open problem through submit.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failed = []
    for problem in sorted((ROOT / "problems").iterdir()):
        if not (problem / "meta.json").exists():
            continue
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "submit.py"),
                problem.name,
                "--compiler",
                str(ROOT / "compile.py"),
            ],
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "PASS" in output
        print(f"{problem.name:<24} {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(output[-1200:])
            failed.append(problem.name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
