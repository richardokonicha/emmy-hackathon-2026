# Own-voice video script (target 4:55, hard ceiling 5:00)

Full word-for-word draft, written for an audience that has never seen a
compiler, an IR, or a GPU instruction set. Every technical term is explained
in the same breath it is used. Read it out loud once, then paraphrase anything
that doesn't sound like you — the walkthrough must sound natural, not
memorized. Speaking pace assumed: ~175 words/min. Bracketed lines are
[what to show].

---

## 0:00–0:35 — Hook: the job, in one sentence

[Show: README top; then split screen — a slice of .ir.json left, generated PTX right]

"Here's my hackathon submission in one sentence: it's a translator — 1,500
lines of Python that reads a description of a GPU program and perfectly
reproduces it as NVIDIA PTX, the low-level assembly that the GPU driver
actually loads onto the chip. Nothing in between: no CUDA compiler, no nvcc,
no borrowed code. Quick vocab, thirty seconds. A *kernel* is a function that
runs on a GPU — I'll call it a recipe. An *IR*, intermediate representation,
is a structured wiring diagram of that recipe, and my input is that diagram as
JSON — written by another program, not by me. *PTX* is the assembly the chip
wants: instructions like 'multiply these numbers'. And *ptxas* is NVIDIA's
assembler — think strict grammar checker — that turns PTX into the final
binary. If ptxas rejects a single line, you fail. First rule of the task: I may
only read the IR JSON; every instruction of PTX must come out of my own
lowering logic."

## 0:35–1:15 — The world my compiler walks into

[Show: `ls problems` — five letter prefixes; highlight one file's meta.json]

"Fifteen sample kernels come with the contest, in five families — and the
hidden graded set is the same five families with different shapes and math:
same song, different key. Three rules shaped everything I did. One: the
function name must be copied from the recipe file. Two: the diagram loses
shape information — sizes live only in sidecar metadata, so my compiler reads
them straight from raw JSON, not from the parsed diagram. Three: the exact
parameter order comes from a separate metadata file. Mix that order up and you
don't crash loud — you silently read thewrong buffers and every number is
garbage. GPU context in one breath: a GPU runs thousands of small cooks at
once, each with its own notepad called a *register*; a *warp* is 32 cooks who
work in lockstep like a stadium wave."

## 1:15–2:05 — What the compiler actually does

[Show: scroll the `Compiler` class, freeze on the ~12-line dispatch dict]

"The whole thing is a mini interpreter in four layers. Layer one, the prologue:
each cook figures out who it is — a lane ID, a warp ID — and turns its
parameters into real memory addresses, because on a GPU your inputs are just
big parked arrays of numbers. Layer two: register pools, one pool per
notepad type — floats, integers, 64-bit addresses, and *predicates*,
one-bit true/false flags. Layer three — this is the entire compiler shape — a
dispatch table with one method per diagram node. Load, store, tile, reduce:
twelve lines of Python that decide everything. Layer four, shared memory: a
whiteboard that a group of cooks read together. Small boards are declared
beforehand; for big matmuls, one large board is carved at runtime — every
array gets a parking spot by measuring how long each car is. A pointwise
kernel — multiply element zero by element zero, and so on — is only about
twenty lines of PTX. Everything hard hides in loops."

## 2:05–2:55 — The hard part: SSA vs loops (keep this in!)

[Show: stmt_accum, then the mov-back lines inside the generated loop PTX]

"Here's the bug I'm proudest of. The diagram is written in *SSA*, static
single assignment: every variable gets its value exactly once — imagine a lab
notebook where you never erase; you grab a new page for every calculation.
Loops break that. A running total is one notebook page overwritten every
iteration — like a sports scoreboard you update each point. My first draft
followed SSA literally: each loop turn made a brand-new page, so at the end I
was reading page one — an empty total. The fix feels illegal by SSA rules: in
exactly one place, the compiler must let two iterations share a page and write
the total back in place. That's the code you see here. Debugging it was eerie:
no crash, no warning, just quietly wrong sums — the worst kind."

## 2:55–3:35 — Reductions and attention, with enough context

[Show: stmt_tree_halve; then attention PTX with shfl and bar.sync]

"A *reduction* collapses many numbers into one — a sum of ten thousand values
becomes a single one. In parallel, you split the work like change-making:
within a warp, butterfly *shuffles* make the 32 lanes exchange values in a
shrinking circle, like a knockout tournament — each round, half the lanes
drop out, one lane ends up holding the whole sum. Then partial sums from each
warp land on the shared whiteboard, and cooks pair off, halving the workers
each round, with a *barrier* — a checkpoint nobody crosses until the whole
group arrives. Attention compiles an *online softmax*: softmax turns raw
scores into probabilities, and 'online' means we track a running maximum and
re-scale as we stream, never pausing the game — like a live leaderboard that
re-ranks as new scores arrive."

## 3:35–4:10 — Live demo

[Show: run both commands for real; show PASS and the ~20 microsecond line]

"Real commands, real server:

    .venv/bin/python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json
    .venv/bin/python submit.py r01_array_sum --compiler compile.py

ptxas accepts my PTX, the runner executes it on an actual GPU, and the output
matches the expected answer to within a rounding hair. Compile time: about
one second per kernel — comfortably inside the 'seconds' budget. All fifteen
open problems pass, and the string comparison and integer branches — easy to
forget — pass too."

## 4:10–4:35 — Hardening against the hidden set

[Show: dev/run_extra.py finishing with "35/35 packs PASS"; keep it tight]

"For the hidden set I fuzz-tested. I couldn't run torch locally, so I stood up
a tiny Ubuntu VM, generated 35 kernel packs by tracing new torch
expressions — odd shapes like rows of 777, 9-by-17 softmax, 8-by-8 attention,
plus every scalar op family: tanh, gelu, erf, log, pow, sin, softplus. Fuzzing
caught real bugs the samples never showed: the Select branch bug where only
the last branch ever ran, unsupported comparisons inside a ternary,
string-valued names inside loop bodies, and a wrong-base bug in pow —
exp-of versus two-to-the-power-of. All 35 pass end to end."

## 4:35–5:00 — Close

[Show: git log --oneline, then README headers]

"Everything is committed step by step — design doc, plan, first fifteen
passing, hardening, op coverage. The whole submission is one file at the
repo root: compile.py. Thanks for watching."

---

# Timing failsafes

Already at the ceiling. If you run long over any section, trim the *extra*
analogies first — they are decoration, not content:
1. parking-garage shared memory + stadium-wave warp (1:15–2:05),
2. leaderboard re-ranking (3:35),
3. the integer/string branch sentence in the demo (4:10).
Never under any clock: the SSA/loop story (2:05), the live demo (3:35),
the fuzz-coverage sentence (4:10). If desperate, compress the close to one
sentence and let `git log` speak.

# Glossary (for you, not for camera)

IR/SSA/predicate/register/warp/shuffle/barrier/reduction/matmul/ptxas — all
defined inline in the script above so an uninformed viewer never hits an
undefined word. If a term feels heavy while rehearsing, replace its clause,
not the concept.
