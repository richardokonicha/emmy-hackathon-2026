# Own-voice video script (target 4:40, hard ceiling 5:00)

Full word-for-word draft. Read it out loud once, then paraphrase anything that
doesn't sound like you — the walkthrough must sound natural, not memorized.
Speaking pace assumed: ~160 words/min. Bracketed lines are [what to show].

---

## 0:00–0:30 — Hook: what this thing is

[Show: README top + one screenful of generated PTX side by side with an .ir.json]

"Say you have an idea for a GPU kernel — some math on arrays — and an IR file
that describes it. Not Python. Not CUDA C+. A JSON wiring diagram. My submission
is `compile.py`: a program that reads that diagram and prints out NVIDIA PTX —
the assembly the GPU driver actually feeds to the chip. No nvcc, no CUDA source,
no borrowed renderer. The only thing I use emmy-ml for is `Graph.from_dict` to
load the JSON — every line of actual PTX in this repo was lowered by code I
wrote myself. Think of IR as a wiring diagram, PTX as step-by-step instructions
for an army of very literal, very fast sous-chefs, and my compiler as the
translator between them."

## 0:30–1:10 — The problem space

[Show: `ls problems`, then the five category prefixes scrolling]

"The open set is fifteen kernels in five families: pointwise math, reductions,
matmuls, fused stuff like softmax and layernorm, and attention. Hidden set?
Same five families, different shapes and kernels. The rules are strict, and
three of them shaped the whole design. One: the entry must be named after the
KernelOp itself. Two: shapes don't survive into the typed IR — they live in the
raw JSON — so I read tensors straight from the raw text. Three: the parameter
order comes from a separate `meta.json`, not from the IR. Miss any of those and
ptxas either rejects you or — worse — you write to the wrong buffer."

## 1:10–2:00 — Architecture of the lowering

[Show: scroll to the `Compiler` class, then the ~12-line dispatch dict]

"The compiler is a tiny interpreter. A prologue decodes thread IDs and turns
kernel params into global-memory pointers. Then a dispatch dict maps each IR
node type — Tile, Load, Store, Reduce — to one method. Pointwise kernels compile
to about twenty lines of PTX. Registers are grouped in pools by type: f32 floats,
u32 ints, 64-bit pointers, and predicates — boolean one-bit registers. Shared
memory exists two ways: statically declared blocks under 48 kilobytes, and one
dynamically sized pool for big matmuls where every array gets an offset carved
at runtime, like assigning parking spots in a garage by counting how long each
car is."

## 2:00–2:50 — The hard part: SSA vs loops

[Show: `stmt_accum`, the mov-back instructions in the generated loop PTX]

"Here's the bug I'm proudest of fixing. The IR is SSA — every variable is
assigned exactly once — like a lab notebook where you never erase, you just add
a new page. Loops break that. A running sum is the same notebook page being
overwritten every iteration; if you follow SSA literally you get a fresh page
every time and at the end you're holding page one — an empty accumulator. The
fix feels backwards by design: the compiler must choose to violate SSA in
exactly one place, and write accumulators back in place. Randomness of another
kind: comparisons in the IR produce predicate registers, and `Select` statements
become predicated `mov`s — this one had a real bug where only the last branch
ever ran."

## 2:50–3:30 — Reductions and attention

[Show: `stmt_tree_halve`, then an attention kernel's PTX with `shfl` and `bar.sync`]

"Big reductions run in two phases. Within a warp, butterfly shuffles: thirty-two
lanes exchange values in a shrinking circle, like a knockout tournament, until
one lane holds the sum. Across warps, each lane's partial lands in shared
memory, then threads pair off, subtracting the working set in half each turn
with a barrier between rounds — a tree that halves. Attention compiles an online
softmax: we track the running max and rescale as we stream — like a live
leaderboard that keeps re-ranking as new scores arrive, never pausing the game."

## 3:30–4:05 — Live demo

[Show: run both commands for real, show PASS and the 20 microsecond number]

"Real command, real runner:

```
.venv/bin/python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > /tmp/r01.ptx
.venv/bin/python submit.py r01_array_sum --compiler compile.py
```

ptxas accepts it, the runner executes it on a GPU, and it matches the expected
output. Compile time: under one second per kernel. All fifteen open problems
pass."

## 4:05–4:40 — Hardening against the hidden set

[Show: `dev/run_extra.py` finishing with ALL PASS over ~36 cases]

"The hidden set is the same song in a different key, so I fuzzed for it. I
couldn't run torch.compile on this Mac, so I set up a small Ubuntu VM, generated
thirty-six kernel packs by tracing new torch expressions — odd shapes like
777-wide rows, 9-by-17 softmax, 8-by-8 attention, plus every scalar op family:
tanh, gelu, erf, log, pow, sin, softplus. That fuzzing caught real bugs the open
set never could: the Select branch bug, unsupported comparisons in a ternary,
string-valued expressions inside loop bodies, and a double-conversion bug in
pow where I converted to the wrong base — exp versus two-to-the-power. Every one
of those thirty-six packs now passes through my compiler end to end."

## 4:40–5:00 — Close

[Show: `git log --oneline`]

"Everything is incremental and committed: design doc, plan, the first passing
fifteen, then the hidden-set hardening, then scalar-op coverage. The whole
submission is one file at the repo root: `compile.py`. Thanks for watching."

---

# Timing failsafes

- Running long? Cut, in this order: the dynamic-shared-memory parking-garage
  analogy (1:30), the `Select` sentence in 2:00–2:50, the last two fuzz bullet
  examples in 4:05–4:40. Never cut: the loop/accumulator story, the live demo.
- Running short? Add one concrete PTX line by reading at 2:30 ("this `mov.s32`
  here is the write-back that keeps the tally alive").
