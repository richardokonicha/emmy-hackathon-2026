# Own-voice video script — 3-minute insurance cut

Use this only if the full 4:45 script in `docs/video-script.md` feels too long
to record well. It hits every judging point in ~500 words (~3:10 of speech at
170 wpm, ~3:30 with pauses) and lands comfortably inside the 3–5 minute
window. Everything here is already in the 4:45 script, just tighter.

Read it out loud once, then paraphrase anything that does not sound like you.
The point of this cut is that it sounds natural, not memorized.

---

## 0:00–0:25 — Hook

[Show: README top; then a split screen — `.ir.json` on the left, generated PTX
on the right]

"Hi, I'm Richard Okonicha. I built an Emmy Kernel IR to PTX compiler, and it
compiles all fifteen open problems in the challenge set. The rule for this
task was simple: write the compiler yourself, no Triton and no PyTorch. So the
interesting part was figuring out what the IR actually means and emitting PTX
that NVIDIA's own ptxas will accept."

## 0:25–0:55 — The three rules that decide correctness

"There were three things that decide whether you're right or wrong, and none
of them are the math. First, the entry point has to be named exactly after the
kernel. Second, shapes come from the raw JSON and the metadata, not from the
IR's own type annotations — in several problems those annotations are wrong.
Third, the argument order is reversed from what the type says: result first,
then inputs. Get any one of those wrong and every test fails, even if your
math is perfect."

## 0:55–1:35 — How the compiler is built

"My compiler is four layers. A reader that validates the IR and recovers
shapes from the raw JSON and metadata. A lowering layer that rewrites the
graph into something PTX can express — it turns SSA phi nodes into alloca plus
load and store, because PTX has no phi. An emitter that renders PTX text, one
module per kernel, with the register budget and shared-memory layout computed
from the recovered shapes. And a driver that dispatches on the op name through
a table, so adding a new op is one table entry. The whole thing is pure Python
standard library — json, struct, sys, pathlib. No dependencies."

## 1:35–2:10 — The bug I hit, and the hard parts

"The most useful bug was in my own reduction. My first version let each
thread accumulate a partial in a register and added the register at the end,
so the answer came out scaled by the number of threads. Numerically it looked
plausible; the test caught it. The fix is to exchange partials through shared
memory first, then have exactly one thread combine them — and that is the kind
of bug the test suite exists to find. The hard parts were reductions, which
need shared memory plus warp shuffles plus a barrier, and attention, where I
implement the online softmax so the full S-by-S matrix never lives in
registers, with the max-subtraction trick to keep it numerically stable."

## 2:10–2:55 — Live demo

[Show: terminal; run `scripts/demo.sh`]

"Here's it running. First the metadata, then my compiler turning the IR into
PTX — that's real PTX text, and the runner assembles it with ptxas. And the
result: the maximum absolute error against the NumPy reference is about four
times ten to the minus six, and the kernel runs in about twenty microseconds."

[Show: `scripts/demo.sh` output, then `ls tests/` and `ls tools/`]

"The challenge only publishes fifteen problems, but scoring runs on a hidden
set, so I wrote my own generator: thirty-five variants with different shapes —
unaligned widths, vectors that don't divide evenly, different reduction axes
— and I fixed every one of them. I also had to respect ptxas's limits: a
register budget per kernel, shared memory under 48 KB, and every module is
verified by actually running ptxas on it."

## 2:55–3:15 — Close

"So: Emmy Kernel IR in, valid PTX out, all fifteen open problems passing, and
no third-party code anywhere in the path. The repo is public at
github.com/richardokonicha/emmy-hackathon-2026 — the compiler, the tests, the
generator and the README are all in there. Thanks."

---

## If you run long

Cut in this order, one at a time, and stop as soon as you are under 4:30:

1. Drop the "hard parts" paragraph in 1:35–2:10 to one sentence: *"The hard
   parts were reductions — shared memory, warp shuffles, a barrier — and
   attention, where I use the online softmax so the S-by-S matrix never lives
   in registers."*
2. Drop the sentence about the register budget in 2:10–2:55.
3. Drop "The whole thing is pure Python standard library…" (keep the version
   if you want it; it is a strong rule-compliance signal).

Never cut: the three rules, the reduction bug, the live PASS line, or the
hidden-set generator. Those are what the video is judged on.
