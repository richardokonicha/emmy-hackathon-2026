# Emmy Hackathon 2026 — Kernel IR → PTX Compiler Design

**Date:** 2026-10-25
**Status:** Approved for planning
**Deadline:** 2026-10-27 23:59 AoE

## 1. Goal

Build a compiler that turns one Emmy Kernel IR JSON file directly into PTX:

```
python compile.py path/to/<entry>.ir.json > out.ptx
```

The evaluator runs this command on a hidden set of IRs in the same five categories as
the open problems, assembles the PTX with `ptxas`, launches it on an RTX 5090 under the
problem's `meta.json` geometry, and compares outputs within the problem tolerance.

Success targets, in judging order:

1. Correctness on the hidden set in all five categories (pointwise, reductions,
   matmul, fused, attention-like).
2. A clean, readable IR→PTX translation that judges can audit as our own work.
3. Reasonable generated-kernel speed (tiebreak only).

The MVP bar (categories 1–2) is treated as the first milestone, not the end goal.

## 2. Rules and constraints

- Emmy is used **only to load and inspect the IR** (`Graph.from_dict`). The compiler
  must not call `render_kernelop`, `emmy compile`, `nvcc`, or any existing compiler.
- The generated code is our own translation from the IR node tree to PTX.
- Runtime requirements: Python 3.12+, `emmy-ml==0.3.11`, standard library. No torch,
  no numpy, no GPU at compile time.
- Output must be PTX only (debug output goes to stderr), ASCII only.
- PTX header: `.version 8.0`, `.target sm_89`, `.address_size 64`.
- Entry name must be `KernelOp.name`; parameter order comes from the sibling
  `meta.json` record whose `entry` matches.
- Launch geometry and shared-memory size are taken from `meta.json`; the compiler never
  invents them.
- Compilation must stay in the seconds-per-kernel range.

## 3. Architecture

One self-contained `compile.py` at the repository root, organized as linear sections so
the translation is easy to audit:

1. **Loader** — read the IR JSON; load typed objects with `Graph.from_dict`; read the
   sibling `meta.json`; keep the raw JSON for tensor shapes/dtypes (the typed `Graph`
   node API does not expose shapes).
2. **Kernel metadata** — select the kernel record by entry name; resolve `arg_order`,
   argument shapes/dtypes/scalar flags, and the launch descriptor. If `meta.json` is
   missing, fall back to `KernelOp.inputs` plus graph outputs.
3. **PTX emitter** — register pools, label allocation, instruction buffer, module
   assembly. Registers are never reused; `ptxas` performs allocation.
4. **Expression compiler** — `Var`, `Literal`, `BinaryExpr`, `TernaryExpr`, `CastExpr`,
   `Builtin`, `FuncCallExpr`, `FlatIndex`; returns a register and dtype.
5. **Statement compiler** — one `emit_*` method per Kernel IR statement kind.
6. **Driver** — parse `argv[1]`, compile, print PTX to stdout; any unsupported node or
   lookup failure prints a diagnostic to stderr and exits non-zero (never partial PTX).

Development-only helpers live in `dev/` and are never imported by `compile.py`.

### Data flow

```
argv[1] .ir.json
  -> raw JSON (shapes/dtypes) + Graph.from_dict (typed body)
  -> sibling meta.json (entry, arg_order, args, grid/block/smem)
  -> KernelOp.body walk
  -> instruction buffer + register/shared declarations
  -> stdout PTX
```

## 4. Register and type model

Four virtual pools, declared with a final count:

| Pool | PTX | Purpose |
|---|---|---|
| float | `.reg .f32 %f<N>` | f32 loads, arithmetic, accumulators |
| integer | `.reg .b32 %r<N>` | indices, loop counters, shuffle carriers |
| address | `.reg .b64 %rd<N>` | global pointers and byte offsets |
| predicate | `.reg .pred %p<N>` | guards, comparisons, conditionals |

- Every IR SSA name maps to exactly one fresh register.
- Integer index arithmetic uses `s32` semantics; loads/stores compute byte offsets with
  `mul.wide.s32` into 64-bit addresses.
- Float and integer 32-bit registers are interchangeable bit containers via `mov.b32`.
- Literals are materialized into registers (`0fXXXXXXXX` for floats).
- No liveness analysis, no register coalescing.

## 5. Thread mapping (`Tile`)

- `gid = blockIdx.x * blockDim.x + threadIdx.x`; with `aux_threads`,
  `gid = blockIdx.x * block_threads + threadIdx.x % block_threads`.
- Always emit the `gid < N` guard and branch over the body.
- Decode each axis from inner strides: `axis[i] = (gid / prod(extents[i+1:])) % extent[i]`,
  with the outermost axis omitting the modulo.
- Implement `raster_axes` / `raster_group` / `raster_orient` block-tile swizzling,
  including the ragged final group.
- Widen `gid` and stride math to 64-bit only when `N > 2^31 - 1`.

## 6. Memory model

- **Global buffers** are identified by graph node name; element offset is flattened
  row-major using the buffer shape. Loads/stores use `ld.global` / `st.global`.
- **Vector IR loads/writes** (`width > 1`) are initially lowered as scalar
  loads/stores. Vector PTX (`ld/st.global.v2/v4`) is a later speed optimization.
- **Shared memory** is declared by `Smem`. If total shared bytes are ≤ 48 KiB, emit
  static module-scope `.shared` arrays. Otherwise emit one `.extern .shared` pool,
  assign each buffer a byte offset in declaration order, and address through the pool.
  This matches `meta.json` (`smem_bytes: 0` for static; the pool size otherwise).
- **Atomic writes** use `red.global.add.f32`.
- **`cp.async`** copies use `cp.async.cg.shared.global` for 16 bytes (`.ca` for 4/8),
  with `cp.async.commit_group` / `cp.async.wait_group`.
- Shared addresses are 32-bit byte addresses: static symbol address or pool base plus
  offset, then `add.s32` with the element byte offset.

## 7. Statement lowering

| IR node | Lowering |
|---|---|
| `Tile` | gid math, guard, axis decode, raster swizzle |
| `Load` | flatten index, address math, scalar `ld.global` per name |
| `Assign` | elementwise op table; result register by declared/inferred dtype |
| `Let` | materialize literal/index expression into a register |
| `Init` | seed a register with the identity literal |
| `Accum` | seed identity before loop (if `seed`); fold `op(base, value)` inside |
| `Loop` / `StridedLoop` | labels + counter register + `setp`/branch; optional `end` bound |
| `Cond` | predicate + conditional branch around body/else body |
| `Select` | reverse-order predicated moves (`selp`/`@p mov`) |
| `Write` | flatten index, address math, scalar `st.global` per value; atomic variant |
| `Smem` | record declaration; static or dynamic pool layout |
| `Sync` | `bar.sync 0`; named barriers when `barrier_id != 0`; `bar.warp.sync` if `warp` |
| `WarpShuffle` | `activemask.b32` + `shfl.sync.bfly.b32` ladder, then `combine_states` |
| `TreeHalve` | unrolled shared-memory halving tree with `bar.sync`, then broadcast root; support the segment-indexed `inner` form |
| `CpAsyncCopy/Commit/Wait` | `cp.async.*` with shared-byte and global-byte addresses |
| `ZeroPrologue`, `Pre`, `Carry`, tensor-core/TMA nodes, non-empty `KernelOp.serial` | not present in the open or hidden set; fail loudly if seen |

Expression operators: `+ - * / %` for integer index math; `+ - * /` for floats;
comparisons to predicates; logical `&& || !`; ternary to `selp`.

Elementwise op table (initial): `add`, `subtract`, `multiply`, `divide`, `maximum`,
`minimum`, `exp` (`mul` by log2(e) + `ex2.approx`), `exp_fast`, `sqrt`, `rsqrt`,
`reciprocal`, `relu`, `abs`, `neg`, `copy`, `where`.
Unsupported ops fail loudly with the op name. Fusing `multiply`+`Accum(add)` into
`fma.rn.f32` is a later speed optimization, not part of the correctness path.

## 8. Error handling

- Unknown statement/expression type, unknown buffer, unknown op, or missing metadata:
  print a one-line diagnostic plus the offending node to stderr; exit code 1.
- Never print partial PTX on failure: build the module in memory first.
- Assertions are not used for user-facing errors; explicit exceptions are.

## 9. Testing and validation

- `submit.py <problem> --compiler compile.py` for the 15 open problems.
- `dev/run_all.py`: runs all 15, prints a PASS/FAIL table, exits non-zero on failure.
- Fast syntax loop: POST generated PTX to the runner's `/compile` endpoint before
  running; `ptxas` stderr is the feedback.
- Hidden-set simulation:
  - install CPU torch in the dev environment;
  - write `dev/expressions_extra.py` with same-category expressions of different
    shapes and ops;
  - generate IRs with `tools/generate.py`, compile and run them through the same
    pipeline;
  - compare against expected arrays from the generator.
- Regression: run the full suite before each commit and after every lowering change.
- Edge cases: partial warps, block sizes not multiples of 32, N not divisible by the
  block, rectangular matmul, shared memory above 48 KiB, atomic accumulation.

## 10. Milestones

"Day" means a working block, not a full calendar day; the absolute deadline is
2026-10-27 23:59 AoE.

| When | Deliverable |
|---|---|
| Oct 25, first block | `compile.py` skeleton + expressions + `Tile`/`Load`/`Assign`/`Write`; p00–p03 pass |
| Oct 25, second block | reductions (`StridedLoop`, `Accum`, `WarpShuffle`, `Smem`, `Sync`, `TreeHalve`, `Cond`); r01–r04 pass; `dev/run_all.py` |
| Oct 26, first block | matmul (`Loop`, `Init`, `cp.async`, dynamic shared, atomics); m01/m02 pass |
| Oct 26, second block | fused + attention (`Let`, multi-accumulator combines, 2-D loops, rasterization); f01–f03 and a01/a02 pass |
| Oct 27 | generated hidden-like variants, edge cases, speed pass, code cleanup |
| Before deadline | public repo with `compile.py`, 3–5 minute video in the submitter's own voice, submission |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PTX syntax/semantic errors | short compile-only loop against the runner; start from the worked example's forms |
| Static vs dynamic shared memory mismatch | choose by total > 48 KiB and honor `meta.json.smem_bytes` |
| Partial warps / non-warp-multiple blocks | use `activemask.b32` instead of a fixed shuffle mask |
| `cp.async` addressing/alignment | reuse the IR's flattened indices; 16-byte copies use `.cg` |
| Hidden ops outside the open set | broad elementwise table; fail loudly with the op name; generate variants |
| Time limit | category order is strict; keep a passing state after each category; no optimization before correctness |
| Video ineligibility | explain each lowering while building; practice the walkthrough before recording |

## 12. Definition of done

- `python compile.py <ir>` works for every open problem and for generated variants.
- All 15 open problems pass via `submit.py --compiler`.
- No call to any existing compiler; translation lives in `compile.py`.
- Full regression suite green.
- Public repository contains `compile.py`, tests/dev tools, and the design/plan docs.
- 3–5 minute video recorded and submission entered before the deadline.
