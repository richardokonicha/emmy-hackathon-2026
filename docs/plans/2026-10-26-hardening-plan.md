# Plan — hardening the Kernel IR → PTX compiler for the hidden set

**Date:** 2026-10-26 · **Deadline:** 2026-10-27 23:59 AoE
**Baseline:** 50/50 kernels correct on the hosted runner (15 open + 26 `dev/cases` + 9 `dev/cases_oops`).

The compiler is already correct on everything we can measure. This plan closes the gaps
between *what we pass* and *what the task explicitly asks for*, removes the eligibility
blockers around the submission itself, and turns the repo from "works" into "reads like
a compiler" for the code-review tiebreak.

## Ground rules

1. **Nothing regresses.** Every code change must keep all 50 kernels green on the runner.
2. **Test first.** Each behaviour change gets a failing unit test before the implementation.
3. **The contract is the spec.** `TASK.md` and `meta.json` are the authority — never the IR's
   own argument order, never a guess at geometry.
4. **Safe by default.** Where an optimisation could change results, prefer the slower,
   provably-correct lowering.

---

## Phase A — Eligibility and judge-facing surface (no compiler changes)

### A1. Submission repo
The remote points at the organizers' repository. The submission must be a public repo of our
own. Create it with `gh repo create`, repoint `origin`, push.

### A2. `dev/run_extra.py` covers every generated variant
`run_extra.py` iterates `dev/cases` only; the nine `dev/cases_oops` packs are never run by
any committed harness, so the "ALL PASS" line on camera under-reports what we actually
verify. Extend it to walk both directories and print the true total.

### A3. Video script facts match the repo
The script says "51 kernel packs … all 51 pass" and "about 0.8 seconds per kernel".
The real numbers are **50** and **~1 second** (measured min 0.89 s / median 1.03 s /
max 1.66 s, of which ~0.75 s is `import emmy`). Correct the script so every on-camera
claim is reproducible by running the committed harness.

### A4. Root `README.md`
Judges land on the repository root. The current `README.md` is the organizers' task
description — it never mentions our compiler. Replace it with a solution README
(what it is, architecture, how to run, what passes, video link) and preserve the
original verbatim as `docs/ORGANIZERS-README.md`.

### A5. `requirements.txt`
`compile.py` needs only `emmy-ml==0.3.11` (stdlib otherwise). Pin it. `numpy`/`requests`
are for `submit.py` and the dev harness — list them separately.

---

## Phase B — Compiler conformance to `TASK.md`

### B1. Vector loads and stores  ← highest-value code change
`TASK.md` requires it explicitly:

> a grouped write `buf[i] = (v0, v1, v2, v3)` is a **vector store** `st.global.v4.f32`
> (v2 for pairs)

and lists "unrolled bodies + **vector stores/loads**" as what category 1 forces.

Our corpus contains **17 grouped writes and 2 grouped loads**; every one of them is
currently lowered to a run of scalar `ld.global.f32` / `st.global.f32`. Emmy's own
reference CUDA emits `float2` / `float4` for the same IR.

Implementation:
- `Write` with 2 or 4 values, f32/f16, non-atomic, contiguous, and a flat index that is a
  known multiple of the vector width → one `st.global.vN.f32 [addr], {v0, v1, …};`
- `Load` with 2 or 4 names under the same conditions → one `ld.global.vN.f32 {v0, v1, …}, [addr];`
- Everything else keeps the scalar path. Alignment is checked, not assumed: the flat index
  must be provably divisible by the vector width, otherwise we would fault.
- Atomics stay scalar (`red.global.add.vN` needs a newer target than `sm_89`).

### B2. `SetMaxNReg`
The only *unmarked* statement in `TASK.md`'s reference table that the compiler would reject
with `unsupported statement`. Two lines of PTX: `setmaxnreg.{inc,dec}.sync.aligned.u32 N;`

### B3. `meta.json` discovery
`_find_meta` reads `meta.json` from the IR's own directory and, if absent, silently falls
back to `kernel.inputs + kernel.outputs` — the exact argument order `TASK.md` warns is
*not* the contract. A misplaced file would produce a wrong-signature kernel that fails
silently.
- Search the IR's directory, then its parents, then the problem root.
- If no `meta.json` is found anywhere, **fail loudly** instead of guessing.

### B4. Scalar register typing for f16 / f64
`stmt_load` / `stmt_write` place any float in a `.f32` register, so `ld.global.f16 %f1` is
rejected by ptxas (*"Unexpected instruction types specified for 'ld'"* — verified against
the runner). Give f16 its own `.b16` pool and route f64 through `.f64`.

### B5. Axis-driven accumulator seeding
`seed_accums` currently seeds any `Accum` that is a *direct child* of a `seed=True` loop.
The invariant "`Accum.axes` equals the enclosing loop's axis" holds for all 113 accumulators
in our corpus, but the rule is implicit. Make it explicit: seed an accumulator only in the
loop that owns its axis.

### B6. Bounded loop unrolling
`unroll` is ignored. The reference CUDA emits `#pragma unroll` on the attention `a1` loop
and the matmul `_ki` loop; ours stays rolled (attention: 98.85 µs / 128 registers; matmul:
100.83 µs). Honour the hint with a **bounded** unroll factor, and only when it is provably
safe: static extent, divisible by the factor, and a body small enough not to explode the
instruction count. Default off where the gain is unclear.

---

## Phase C — Code-review quality (first tiebreak)

### C1. Real test suite
Five string-presence assertions do not survive a code review. Add:
- golden-PTX regression tests for the 15 open problems (snapshots, so an unintended change
  to the lowering is caught before it reaches the runner);
- negative tests: an unsupported IR node must exit non-zero with a message on stderr and
  nothing on stdout;
- contract tests: entry name and parameter order come from `meta.json`, not the IR;
- an ASCII + stdout-purity test over every kernel we can compile.

### C2. Compiler readability
1,529 lines in one file with one 30-method class is hard to review. Add a short
architecture banner at the top of `compile.py` and section markers, rather than a risky
file split this close to the deadline.

---

## Phase D — Verification and submission

1. Unit suite green.
2. All 50 kernels compile; pure ASCII; stdout is PTX only; stderr silent.
3. All 50 kernels **pass on the hosted runner**.
4. `dev/bench.py` — compare against the organizers' hand-written PTX baseline.
5. `dev/run_all.py` and `dev/run_extra.py` — the committed harnesses a judge will run.
6. Create the public repo, push, verify the URL is public and `compile.py` is at its root.

## Risks

| Risk | Mitigation |
|---|---|
| Vectorisation mis-aligns and faults | Require a provably divisible flat index; otherwise stay scalar |
| Unrolling blows up instruction count | Bounded factor, small bodies only, gated by a size check |
| A change breaks one of the 50 kernels | Full runner sweep after every phase; revert on any regression |
| Repo surgery costs time | Phase A/B are independent of Phase C; submit-ready after Phase B |
