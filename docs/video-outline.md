# Video outline — own-voice walkthrough (target: 4:00, hard ceiling 5:00)

This is a beat sheet with what to show on screen and speaking cues - rewrite in
your own words before recording; the walkthrough must sound like you.

Prep before recording:
- Repo clean, `git log --oneline` visible (shows incremental work).
- Two terminal panes: left `vim compile.py` scrolled to the relevant lines, right shell.
- Have `dev/run_all.py` output and one `dev/cases` IR handy.

---

## 0:00-0:25 Hook: what this compiler is
- On screen: `compile.py` header + one PTX line range from `examples/k_sum_1_reduce.ptx`.
- Say: "This repo is a compiler that turns Emmy Kernel IR JSON straight into NVIDIA
  PTX assembly - no CUDA, no nvcc, no rendering pass. I'm using emmy-ml only to
  load the IR with Graph.from_dict and then lower every node myself into PTX that
  ptxas accepts."

## 0:25-1:00 The problem space
- On screen: `ls problems`, the 5 category prefixes (p*, r*, m*, f*, a*).
- Say: "The open set has fifteen kernels: pointwise, reductions, matmuls, fused
  normalization/softmax, and attention. I paste seven hand-written characteristics
  of the IR into the design doc, then... " (keep short - summarize)
  Key points to hit:
  - one `KernelOp` per file, entry name must match the `KernelOp.name`;
  - shapes live in the raw JSON, not the typed graph - hence loading meta
    tensors from the *raw* JSON;
  - parameter order comes from `meta.json`, not the IR.

## 1:00-1:50 Architecture of the lowering
- On screen: scroll to the `Compiler` class - point at the four main layers:
  1) `emit_prologue` - params, `cvta.to.global`, tid/lane/warp decode;
  2) register pools per type: `.reg .f32 %f<N>` literals/scalars as f32,
     `.reg .b32 %r<N>` for idx and int ops, `.reg .b64 %rd<N>` for 64-bit pointers,
     `.reg .pred %p<N>` for conditions;
  3) statement walk: dispatch table `{"Tile": self.stmt_tile, ...}` - one method
     per IR node - this is the whole compiler shape (show the ~12-line dict);
  4) smem: static `.shared` under 48 KiB, and `.extern .shared` dynamic pool with
     runtime-aligned offsets when bigger (`m01_matmul_512`).
- Say: "Pointwise kernels are ~20 lines of PTX. Everything interesting is in
  Tile/loop reductions."

## 1:50-2:40 The hard parts (SSA vs loops, reductions, attention)
- On screen: start at `stmt_accum`, then jump to `stmt_tree_halve` briefly.
- Say:
  1) Loops break SSA: accumulation must write back *in place*. Show loop
     `counter` copied with `mov.s32` so the start variable (`a0_co`) survives.
     - Talk about the bug: fresh virtual register per SSA name is wrong inside a
       loop, the accumulator must carry values across iterations.
  2) Cooperative reductions: `WarpShuffle` (butterfly shuffles down to 1 value) +
     smem staging, then `TreeHalve` with `bar.sync` halving until one thread.
  3) Attention: online softmax; setp/comparisons produce predicates, `Select`
     statement maps `select` expr to conditional `mov.@p`.

## 2:40-3:10 Live demo
- On screen: run for real:
  - `.venv/bin/python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > /tmp/r01.ptx`
  - `.venv/bin/python submit.py r01_array_sum --compiler compile.py` - PASS output.
- Say: "One command per problem. The runner ptxas's my PTX and compares against
  expected.npz. All fifteen pass; hidden eval runs your compiler the same way."

## 3:10-3:50 Hardening against the hidden set
- On screen: `dev/run_extra.py` last-page output (35/35 packs PASS).
- Say: "The hidden set is the same categories with different shapes and ops. I
  couldn't run torch.compile on this Mac, so I ported the generator inside a
  small Ubuntu VM, generated 35 kernel variants by tracing new torch expressions
  that stress odd shapes - 257, 33x777, 8x8 attention, layernorm - and used those
  packs to fuzz my compiler. It caught three real bugs I never would have found
  with the open set alone:
  - a `Select` branch bug where only the last branch ran;
  - unsupported comparisons (`x > 0` in a `where`); and
  - string-valued expressions after Init inside loop bodies.
  Also added rasterization and `ZeroPrologue` support before they were needed."

## 3:50-4:15 Closing
- On screen: `git log --oneline`
- Say: "Every artifact in this repo is committed incrementally: the design doc,
  a plan, then code, then hardening. The compiler passes fifteen open problems
  and thirty-five unseen-shape variants; the whole submission is compile.py."

---

# If the clock runs over 5:00, cut from the bottom up:
1. closing, 2. demo REPL/second problem, 3. architecture bullet 4 dynamic smem.

Do NOT cut in the middle of: demo, variant bug list, loop-accumulator story.

---

## Screen-recording cheat sheet
```
ls problems && ls dev/cases && git log --oneline
# pane 1 prose: str one ir.json in vim
.venv/bin/python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > /tmp/r01.ptx
head -c 600 /tmp/r01.ptx
.venv/bin/python submit.py r01_array_sum --compiler compile.py
.venv/bin/python dev/run_extra.py
```
