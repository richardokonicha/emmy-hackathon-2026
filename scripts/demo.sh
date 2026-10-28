#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IR=problems/p00_scalar_add/k_add_pointwise.ir.json

echo "### 1. shapes and arg order live in the raw JSON + meta.json"
cat problems/p00_scalar_add/meta.json
echo
echo "### 2. compile.py: IR JSON -> PTX text (pure stdlib Python)"
$PY compile.py "$IR" > /tmp/emmy_demo.ptx
head -n 24 /tmp/emmy_demo.ptx
echo "   ... $(wc -l < /tmp/emmy_demo.ptx) lines of PTX in total"
echo
echo "### 3. the challenge runner assembles it with ptxas and checks vs the NumPy reference"
$PY submit.py r01_array_sum --compiler compile.py
