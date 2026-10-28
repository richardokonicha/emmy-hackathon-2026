#!/usr/bin/env python3
"""Generate the segment-indexed tree pack used by dev/run_extra.py.

Emmy's transposed ``coop-t`` cooperative reduce builds a *segment-indexed*
shared-memory tree: the slab holds ``length`` segments of ``scale`` slots,
the slot is ``t*scale + iv`` and the broadcast root is ``iv``.  None of the
55 open/generated kernels reaches that path -- every ``TreeHalve`` they
contain has ``inner=None`` -- so it is exercised here by hand.

The kernel is deliberately tiny: 128 threads, one element each, four
segments of 32 lanes.  Thread ``warp*32 + lane`` stages its value at
``warp*32 + lane``; the tree then halves the *segment* index, so lane ``i``
ends with ``x[i] + x[32+i] + x[64+i] + x[96+i]`` -- a strided sum, not the
flat total a segment-blind tree would produce.

    .venv/bin/python dev/make_segment_tree_pack.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "dev" / "cases_oops" / "z17_segment_tree"

THREADS = 128
SCALE = 32
SEGMENTS = THREADS // SCALE

F32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"

BODY = (
    "Tile(axes=(Axis(name='a0_co', extent=Dim({threads}), window=None),), body=("
    "Load(names=('in0',), input='x', index=(Var(name='a0_co'),), dtype={f32}),"
    "Smem(name='acc_smem', extents=({slots},), dtype='float', align=0),"
    "Write(output='acc_smem', index=(BinaryExpr(op='+', "
    "left=BinaryExpr(op='*', left=Var(name='warp'), "
    "right=Literal(value={scale}, dtype='int')), right=Var(name='lane')),), "
    "values=('in0',), value_dtype={f32}, atomic=False, swizzle='NONE'),"
    "Sync(barrier_id=0, count=None, warp=False),"
    "TreeHalve(bufs=('acc_smem',), state=('acc',), state_b=('acc__o',), "
    "combine_states=(Assign(name='acc', op=ElementwiseImpl('add'), "
    "args=('acc', 'acc__o'), dtype=None),), length={segments}, "
    "tid_var='warp', dtype={f32}, barrier_id=0, barrier_count=None, "
    "inner=('lane', {scale})),"
    "Cond(cond=BinaryExpr(op='==', left=Var(name='warp'), "
    "right=Literal(value=0, dtype='int')), body=("
    "Write(output='sum', index=(Var(name='lane'),), values=('acc',), "
    "value_dtype={f32}, atomic=False, swizzle='NONE'),), else_body=())"
    "), block_threads={threads}, aux_threads=0, raster_axes=None, "
    "raster_group=None, raster_orient='m')"
).format(threads=THREADS, slots=THREADS, scale=SCALE, segments=SEGMENTS, f32=F32)


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    ir = {
        "inputs": ["x"],
        "outputs": ["sum"],
        "nodes": {
            "x": {
                "op": "InputOp",
                "inputs": [],
                "outputs": [{"name": "x", "shape": [THREADS], "dtype": "f32"}],
            },
            "sum": {
                "op": "KernelOp",
                "inputs": ["x"],
                "outputs": [{"name": "sum", "shape": [SCALE], "dtype": "f32"}],
                "op_fields": {
                    "body": [BODY],
                    "name": "k_segment_tree",
                    "zero_delegated": [],
                    "serial": [],
                },
            },
        },
    }
    (DST / "k_segment_tree.ir.json").write_text(json.dumps(ir, indent=1))

    meta = {
        "id": "z17_segment_tree",
        "code": "x.reshape(4, 32).sum(dim=0)",
        "rtol": 1e-6,
        "atol": 1e-6,
        "graph_inputs": ["x"],
        "graph_outputs": ["sum"],
        "kernels": [
            {
                "node": "sum",
                "entry": "k_segment_tree",
                "arg_order": ["x", "sum"],
                "grid": [1, 1, 1],
                "block": [THREADS, 1, 1],
                "smem_bytes": 0,
                "zero_outputs": [],
                "args": [
                    {"name": "x", "dtype": "f32", "shape": [THREADS], "seed": 0},
                    {"name": "sum", "dtype": "f32", "shape": [SCALE], "output": True},
                ],
                "final": True,
            }
        ],
    }
    (DST / "meta.json").write_text(json.dumps(meta, indent=1))

    x = np.random.default_rng(0).standard_normal([THREADS]).astype(np.float32)
    expected = x.reshape(SEGMENTS, SCALE).sum(axis=0).astype(np.float32)
    np.savez(DST / "expected.npz", sum=expected)
    print(f"wrote {DST} (segment root = lane slot, {SEGMENTS} segments x {SCALE} lanes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
