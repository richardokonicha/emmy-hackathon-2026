#!/usr/bin/env python3
"""Compile and run every generated variant through submit.py.

Walks both variant directories:

    dev/cases/       shape and op-mix variations of the five categories
    dev/cases_oops/  the scalar-op coverage pack

Every pack is compiled with compile.py and executed on the hosted runner, then checked
against its expected.npz within the problem's tolerance. This is the harness the video
quotes, so the number it prints is the number that is actually verified.

    .venv/bin/python dev/run_extra.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIRS = [ROOT / "dev" / "cases", ROOT / "dev" / "cases_oops"]


def packs():
    for directory in CASE_DIRS:
        if not directory.is_dir():
            continue
        for problem in sorted(directory.iterdir()):
            if (problem / "meta.json").exists():
                yield problem


def main() -> int:
    failed = []
    total = 0
    for problem in packs():
        total += 1
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
    print(f"\n{total - len(failed)}/{total} packs PASS"
          f"{'' if not failed else ' - FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
