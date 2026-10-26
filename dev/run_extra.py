#!/usr/bin/env python3
"""Compile and run every generated variant in dev/cases through submit.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failed = []
    for problem in sorted((ROOT / "dev" / "cases").iterdir()):
        if not (problem / "meta.json").exists():
            continue
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "submit.py"),
                str(problem),
                "--compiler",
                str(ROOT / "compile.py"),
            ],
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "PASS" in output
        for line in output.splitlines():
            if "[k_" in line:
                print(f"{problem.name:<28} {line.strip()}")
        if not ok:
            print(output[-1500:])
            failed.append(problem.name)
    print(f"\n{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
