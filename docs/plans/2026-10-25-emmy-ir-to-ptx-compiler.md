# Emmy Kernel IR → PTX Compiler Implementation Plan

> **For the implementer:** Execute top to bottom. Each task ends with a verification command and a commit. Do not skip the verification step. If a runner test fails, use `submit.py <problem> --compiler compile.py` output plus the runner `/compile` endpoint before changing code.

**Goal:** `python compile.py path/to/<kernel>.ir.json` prints valid PTX for every kernel in the Emmy Hackathon 2026 hidden set.

**Architecture:** One self-contained `compile.py`. It loads IR through the public `Graph.from_dict`, reads shapes/dtypes from the raw JSON, and emits PTX with a tree-walking lowering per IR node. Values are immutable virtual registers; `ptxas` does physical allocation. Shared memory is laid out once per kernel (static ≤48 KiB, dynamic above).

**Tech Stack:** Python 3.12 (`uv` venv), `emmy-ml==0.3.11`, `numpy`, `requests`, PTX ISA 8.0, runner at `http://runner.riftstack.ai:48701`.

**Non-negotiables:**
- Do not call `emmy`'s CUDA renderer, `emmy compile`, `nvcc`, or any existing compiler.
- `Graph.from_dict` for IR loading is allowed.
- All debug output goes to `stderr`; stdout is PTX only and ASCII only.
- Entry name = `KernelOp.name`; parameter order = matching `meta.json` `arg_order`.
- Unsupported IR nodes fail loudly with the node type and a clear message.

---

## Phase 0 — Environment and harness

### Task 0: Confirm environment and baseline

**Objective:** Prove the runner and the worked example still pass.

**Files:** none.

**Step 1: Run the known-good PTX**

Run:
```bash
.venv/bin/python submit.py r01_array_sum examples/k_sum_1_reduce.ptx
```
Expected: `PASS` with a `us` timing. If this fails, stop and fix connectivity first.

**Step 2: Confirm imports**

Run:
```bash
.venv/bin/python -c "from emmy.compiler.graph import Graph; from emmy.compiler.ir.kernel.ir import KernelOp, Tile; print('ok')"
```
Expected: `ok`.

---

### Task 1: Write the plan and commit

**Files:**
- Create: `docs/plans/2026-10-25-emmy-ir-to-ptx-compiler.md`

**Step 1: Commit**

```bash
git add docs/plans/2026-10-25-emmy-ir-to-ptx-compiler.md
git commit -m "docs: add IR-to-PTX implementation plan"
```

---

## Phase 1 — Category p (pointwise)

### Task 2: Loader, metadata, entry selection, module shell

**Objective:** Produce a PTX module with the correct entry name and parameters, even before lowering the body.

**Files:**
- Create: `compile.py`
- Create: `tests/test_compile.py`

**Step 1: Write the failing test**

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_compiler():
    spec = importlib.util.spec_from_file_location("emmy_compile", ROOT / "compile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_module_shell_for_p00():
    mod = load_compiler()
    ptx = mod.compile_file(ROOT / "problems/p00_scalar_add/k_add_pointwise.ir.json")
    assert ".entry k_add_pointwise" in ptx
    assert ".param .u64 p_x0" in ptx
    assert ".version 8.0" in ptx
```

**Step 2: Run it and see it fail**

```bash
.venv/bin/python -m unittest tests.test_compile -v
```
Expected: FAIL (no `compile.py`).

**Step 3: Implement the skeleton**

`compile.py` must contain:

```python
#!/usr/bin/env python3
from __future__ import annotations
import json, struct, sys
from pathlib import Path
from emmy.compiler.graph import Graph

PTX_HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

class Compiler:
    def __init__(self, ir_path: Path):
        self.ir_path = Path(ir_path)
        self.raw = json.loads(self.ir_path.read_text())
        self.graph = Graph.from_dict(self.raw)
        self.kernel = self._find_kernel()
        self.meta = self._find_meta()
        self.entry = self.kernel.name or self.meta["kernels"][0]["entry"]
        self.tensors = {}
        for node in self.raw.get("nodes", {}).values():
            for out in node.get("outputs", []):
                self.tensors[out["name"]] = out
        for arg in self.meta.get("args", []):
            self.tensors.setdefault(arg["name"], {"name": arg["name"], "shape": arg["shape"], "dtype": arg["dtype"]})
        self.arg_order = self.meta["kernels"][0]["arg_order"]
        self.args = {a["name"]: a for a in self.meta.get("args", [])}

    def _find_kernel(self):
        for node in self.graph.nodes.values():
            if node.op is not None and node.op.__class__.__name__ == "KernelOp":
                return node.op
        raise SystemExit("no KernelOp in IR")

    def _find_meta(self):
        meta_path = self.ir_path.with_name("meta.json")
        meta = json.loads(meta_path.read_text())
        for k in meta["kernels"]:
            if k["entry"] == self.kernel.name:
                return meta
        return meta  # single-kernel default

    def compile(self) -> str:
        lines = [PTX_HEADER]
        params = ",\n".join(f"        .param .{'u64' if not self.args.get(n, {}).get('scalar') else ('f32' if 'float' in self.args[n]['dtype'] else 's32')} p_{sanitize(n)}" for n in self.arg_order)
        lines.append(f".visible .entry {self.entry}(\n{params}\n)\n{{")
        lines.append("        ret;")
        lines.append("}\n")
        return "\n".join(lines)

def compile_file(path) -> str:
    return Compiler(path).compile()

def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        sys.exit(2)
    sys.stdout.write(compile_file(sys.argv[1]))

if __name__ == "__main__":
    main()
```

**Step 4: Run the test**

```bash
.venv/bin/python -m unittest tests.test_compile -v
```
Expected: PASS.

**Step 5: Verify ptxas accepts the shell**

```bash
.venv/bin/python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/p00.ptx
.venv/bin/python submit.py p00_scalar_add --compiler compile.py
```
Expected: the runner returns a correctness failure (zero output), not a compile error. Confirm the PTX compiles.

**Step 6: Commit**

```bash
git add compile.py tests/test_compile.py
git commit -m "feat: add compiler skeleton and entry/signature emission"
```

---

### Task 3: Emitter, register pools, buffer metadata

**Objective:** Central emit/registers/address helpers used by every lowering.

**Files:** Modify `compile.py`.

**Step 1: Add `Regs` and `Emitter`**

```python
class Regs:
    def __init__(self):
        self.n = {"f": 0, "r": 0, "rd": 0, "p": 0}
    def new(self, kind):
        self.n[kind] += 1
        return f"%{kind}{self.n[kind]}"
    def declarations(self):
        return [
            f"        .reg .pred %p<{self.n['p'] + 1}>;",
            f"        .reg .f32  %f<{self.n['f'] + 1}>;",
            f"        .reg .b32  %r<{self.n['r'] + 1}>;",
            f"        .reg .b64  %rd<{self.n['rd'] + 1}>;",
        ]

class Emitter:
    def __init__(self):
        self.lines = []
        self.regs = Regs()
        self.label_n = 0
    def ins(self, text):
        self.lines.append("        " + text)
    def label(self, prefix):
        self.label_n += 1
        name = f"{prefix}_{self.label_n}"
        self.lines.append(name + ":")
        return name
    def raw(self, text):
        self.lines.append(text)
```

**Step 2: Add dtype/element-size helpers**

```python
DTYPE_BYTES = {"f32": 4, "f16": 2, "f64": 8, "i32": 4, "i64": 8,
               "float": 4, "double": 8, "half": 2, "int": 4}

def dtype_name(dt) -> str:
    return getattr(dt, "name", str(dt))

def elem_size(dt) -> int:
    return DTYPE_BYTES[dtype_name(dt)]
```

**Step 3: Add buffer shape/dtype lookups**

```python
    def shape_of(self, name):
        if name in self.smem:              # filled in Task 15
            return self.smem[name]["extents"]
        return self.tensors[name]["shape"]

    def dtype_of(self, name):
        if name in self.smem:
            return self.smem[name]["dtype"]
        return self.tensors[name]["dtype"]
```

For now, initialize `self.smem = {}` in `__init__`.

**Step 4: Run existing test and commit**

```bash
.venv/bin/python -m unittest tests.test_compile -v
git add compile.py && git commit -m "feat: add PTX emitter and register pools"
```
Expected: PASS.

---

### Task 4: Expression compiler

**Objective:** Compile literals (including `-inf`), vars, integer/float arithmetic, comparisons, ternary, casts, builtins, and elementwise calls.

**Files:** Modify `compile.py`.

**Step 1: Add failing unit tests**

```python
def test_float_bits_negative_infinity():
    mod = load_compiler()
    assert mod.float_bits(float("-inf")) == 0xFF800000

def test_literal_and_int_add():
    mod = load_compiler()
    # covered indirectly by p00 in Task 10
    assert mod.float_bits(1.0) == 0x3F800000
```

**Step 2: Implement**

Key helpers:

```python
def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]

IDENTITY_BITS = {"-inf": 0xFF800000, "inf": 0x7F800000, "0": 0,
                 "1": 0x3F800000, "-1": 0xBF800000}

    def lit(self, value, dtype="f32"):
        dt = dtype_name(dtype)
        if "int" in dt or dt.startswith("i"):
            reg = self.regs.new("r")
            self.ins(f"mov.s32 {reg}, {int(value)};")
            return reg, dt
        reg = self.regs.new("f")
        self.ins(f"mov.f32 {reg}, 0f{float_bits(value):08X};")
        return reg, "f32"
```

Expression dispatch:

```python
    def expr(self, e):
        from emmy.compiler.ir.expr import Literal, Var, BinaryExpr, TernaryExpr, CastExpr, Builtin, FuncCallExpr, FlatIndex
        if isinstance(e, Literal):
            dt = dtype_name(e.dtype) if e.dtype is not None else ("i32" if isinstance(e.value, int) else "f32")
            return self.lit(e.value, dt)
        if isinstance(e, Var):
            return self.values[e.name], self.value_dtypes[e.name]
        if isinstance(e, BinaryExpr):
            return self.binary(e)
        if isinstance(e, TernaryExpr):
            return self.ternary(e)
        if isinstance(e, CastExpr):
            return self.cast(e)
        if isinstance(e, Builtin):
            return self.builtin(e)
        if isinstance(e, FuncCallExpr):
            return self.funccall(e)
        if isinstance(e, FlatIndex):
            return self.flat_index(e)
        raise SystemExit(f"unsupported expression {type(e).__name__}: {e!r}")
```

Binary rules:
- `+ - * / %` for int (`add.s32`, `sub.s32`, `mul.lo.s32`, `div.s32`, `rem.s32`).
- `+ - * /` for float (`add.f32`, `sub.f32`, `mul.f32`, `div.rn.f32`).
- Comparisons (`== != < <= > >=`) emit `setp.<op>.<type>` and return a predicate.
- `and` / `or` on predicates use `and.pred` / `or.pred`; on ints use `and.b32` / `or.b32`.
- If one operand is int and the other float, insert `cvt.rn.f32.s32` for the int.

Immediate special case: divide by `1.0` returns the input; multiply by `1.0` returns the input. This keeps r02 clean.

Ternary: compile cond to predicate, compile both values, emit `selp.<type> out, a, b, pred;`.

Casts: `f32→s32` = `cvt.rzi.s32.f32`; `s32→f32` = `cvt.rn.f32.s32`; same-type = move.

Builtin: map `"threadIdx.x"` → `%tid.x` (materialize `mov.u32 %r, %tid.x;`), `blockIdx.x`, `blockDim.x`, `gridDim.x`.

FuncCall: route through `self.apply_func(name, args)` (Task 8). For now, implement `exp`, `exp_fast`, `relu`, `sqrt`, `rsqrt`, `reciprocal`, `abs`, `neg`, `copy`, `where`.

**Step 3: Run tests and commit**

```bash
.venv/bin/python -m unittest tests.test_compile -v
git add compile.py tests/test_compile.py
git commit -m "feat: add expression lowering"
```

---

### Task 5: Flattening and global addresses

**Objective:** Turn a buffer index tuple into a byte address.

**Files:** Modify `compile.py`.

**Step 1: Implement**

```python
    def int_reg(self, value):
        reg = self.regs.new("r")
        self.ins(f"mov.s32 {reg}, {value};")
        return reg

    def mul_int(self, a, b):
        reg = self.regs.new("r")
        if isinstance(b, int):
            self.ins(f"mul.lo.s32 {reg}, {a}, {b};")
        else:
            self.ins(f"mul.lo.s32 {reg}, {a}, {b};")
        return reg

    def add_int(self, a, b):
        reg = self.regs.new("r")
        self.ins(f"add.s32 {reg}, {a}, {b};")
        return reg

    def flatten(self, name, indices):
        shape = self.shape_of(name)
        if not indices:
            return self.int_reg(0)
        if len(indices) != len(shape):
            raise SystemExit(f"index rank {len(indices)} != shape rank {len(shape)} for {name}")
        off = None
        for idx, dim in zip(indices, shape):
            r, _ = self.expr(idx)
            if dim == 1:
                term = self.int_reg(0)
            else:
                term = self.mul_int(r, dim) if off is not None else r
            off = term if off is None else self.add_int(off, term)
        return off

    def global_addr(self, name, flat):
        off = self.regs.new("rd")
        self.ins(f"mul.wide.s32 {off}, {flat}, {elem_size(self.dtype_of(name))};")
        addr = self.regs.new("rd")
        self.ins(f"add.s64 {addr}, {self.base[name]}, {off};")
        return addr
```

`self.base` maps argument name → `.b64` register holding the converted global pointer. Initialize in the prologue (Task 2 extension): for every non-scalar arg, `ld.param.u64` + `cvta.to.global.u64`. Cache.

**Step 2: Handle constant tensors**

`self.tensors` includes literal outputs with `data`. They are still passed as pointers by the runner. No special code needed.

**Step 3: Commit**

```bash
git add compile.py && git commit -m "feat: add index flattening and global addressing"
```

---

### Task 6: Tile mapping

**Objective:** `gid`, guard, axis decode, and `lane`/`warp`.

**Files:** Modify `compile.py`.

**Step 1: Implement `stmt_tile`**

```python
    def stmt_tile(self, s):
        extents = [int(x) for x in s.extents]
        n = 1
        for x in extents: n *= x
        tid = self.regs.new("r"); self.ins(f"mov.u32 {tid}, %tid.x;")
        lane = self.regs.new("r"); self.ins(f"and.b32 {lane}, {tid}, 31;")
        warp = self.regs.new("r"); self.ins(f"shr.u32 {warp}, {tid}, 5;")
        self.values["lane"], self.value_dtypes["lane"] = lane, "i32"
        self.values["warp"], self.value_dtypes["warp"] = warp, "i32"

        if s.block_threads:
            width = int(s.block_threads)
            if s.aux_threads:
                t = self.regs.new("r"); self.ins(f"rem.u32 {t}, {tid}, {width};")
            else:
                t = tid
            gid = self.regs.new("r")
            self.ins(f"mad.lo.u32 {gid}, %ctaid.x, {width}, {t};")
        else:
            gid = self.regs.new("r")
            self.ins(f"mad.lo.u32 {gid}, %ctaid.x, %ntid.x, {tid};")

        guard = self.regs.new("p")
        end = self.label("TILE_END")
        self.ins(f"setp.ge.u32 {guard}, {gid}, {n};")
        self.ins(f"@{guard} bra {end};")

        self.emit_axes(gid, extents, getattr(s, "raster_axes", None), getattr(s, "raster_group", None), getattr(s, "raster_orient", None))
        self.stmt(s.body)
        self.raw(end + ":")
```

Axis decode (identity raster first):

```python
    def emit_axes(self, gid, extents, raster_axes=None, group=None, orient=None):
        strides = [1] * len(extents)
        for i in range(len(extents) - 2, -1, -1):
            strides[i] = strides[i + 1] * extents[i + 1]
        axes = []
        for i, (ext, stride) in enumerate(zip(extents, strides)):
            if ext == 1:
                r = self.int_reg(0)
            elif i == len(extents) - 1:
                r = self.regs.new("r"); self.ins(f"rem.s32 {r}, {gid}, {ext};")
            elif i == 0:
                r = self.regs.new("r"); self.ins(f"div.s32 {r}, {gid}, {stride};")
            else:
                q = self.regs.new("r"); self.ins(f"div.s32 {q}, {gid}, {stride};")
                r = self.regs.new("r"); self.ins(f"rem.s32 {r}, {q}, {ext};")
            axes.append(r)
        self.axis_regs = axes
        for i, r in enumerate(axes):
            self.values[f"a{i}"] = r
            self.value_dtypes[f"a{i}"] = "i32"
```

Add raster application in Task 27. For the first pass, if `raster_axes` is not `None` and is not the identity tuple `('a0_b','a1_b')` for a 2-axis tile, raise `NotImplementedError` so we don't silently miscompile.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower Tile to gid, guard, and axes"
```

---

### Task 7: Load lowering

**Objective:** Global and shared scalar loads.

**Files:** Modify `compile.py`.

**Step 1: Implement**

```python
    def stmt_load(self, s):
        names = s.names if isinstance(s.names, (list, tuple)) else [s.names]
        dtype = dtype_name(s.dtype) if s.dtype is not None else dtype_name(self.dtype_of(s.input))
        for k, name in enumerate(names):
            indices = list(s.indices)
            if indices and k:
                indices[-1] = self.add_int_expr(indices[-1], k)
            flat = self.flatten(s.input, indices)
            if s.input in self.smem:
                addr = self.smem_elem_addr(s.input, flat)
                reg = self.regs.new("f" if "f" in dtype else "r")
                self.ins(f"ld.shared.{ptx_type(dtype)} {reg}, [{addr}];")
            else:
                addr = self.global_addr(s.input, flat)
                reg = self.regs.new("f" if "f" in dtype else "r")
                self.ins(f"ld.global.{ptx_type(dtype)} {reg}, [{addr}];")
            self.values[name], self.value_dtypes[name] = reg, dtype
```

`ptx_type`: `f32`, `f16`, `f64`, `s32`, `s64`. Add helper.

`add_int_expr(e, k)`: if `e` is a literal, return `Literal(e.value + k, e.dtype)` without emitting; else emit `add.s32`.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower global and shared loads"
```

---

### Task 8: Assign and elementwise ops

**Objective:** Fold ops including `add`, `multiply`, `maximum`, `relu`, `exp`.

**Files:** Modify `compile.py`.

**Step 1: Implement `apply_op` and `apply_func`**

`apply_op(op, args, dtype)`:
- Float: `add.f32`, `sub.f32`, `mul.f32`, `div.rn.f32`, `max.f32`, `min.f32`, `neg.f32`, `abs.f32` via bitwise `and.b32` with `0x7FFFFFFF`.
- Int: `add.s32`, `sub.s32`, `mul.lo.s32`, `div.s32`, `rem.s32`, `and.b32`, `or.b32`, `xor.b32`, `min.s32`, `max.s32`.
- `copy`: `mov`.
- `fma`: `fma.rn.f32`.
- `where`: arguments are `(cond, a, b)`, compile cond predicate, `selp`.
- `exp`/`exp_fast`: `mul.f32 t, x, 0f3FB8AA3B; ex2.approx.f32 out, t;`.
- `relu`: `max.f32 out, x, 0f00000000;`.
- `sigmoid`: `1/(1+exp(-x))`.
- `silu`: `x*sigmoid(x)`.
- `tanh`: `tanh.approx.f32` (sm_75+).

`stmt_assign`:
```python
    def stmt_assign(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.args[0]]
        reg, _ = self.apply_op(s.op, s.args, dtype)
        self.values[s.name], self.value_dtypes[s.name] = reg, dtype
```

`stmt_let`: `reg, dt = self.expr(s.value); self.values[s.name], self.value_dtypes[s.name] = reg, dt`.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower Assign and elementwise ops"
```

---

### Task 9: Write lowering

**Objective:** Scalar global and shared stores.

**Files:** Modify `compile.py`.

**Step 1: Implement**

```python
    def stmt_write(self, s):
        values = s.values if isinstance(s.values, (list, tuple)) else [s.values]
        dtype = dtype_name(s.dtype) if s.dtype is not None else dtype_name(self.dtype_of(s.output))
        for k, name in enumerate(values):
            indices = list(s.indices)
            if indices and k:
                indices[-1] = self.add_int_expr(indices[-1], k)
            flat = self.flatten(s.output, indices)
            reg = self.values[name]
            if s.output in self.smem:
                addr = self.smem_elem_addr(s.output, flat)
                self.ins(f"st.shared.{ptx_type(dtype)} [{addr}], {reg};")
            else:
                addr = self.global_addr(s.output, flat)
                op = "red.global" if s.atomic else "st.global"
                if s.atomic:
                    self.ins(f"{op}.{s.atomic}.{ptx_type(dtype)} [{addr}], {reg};")
                else:
                    self.ins(f"{op}.{ptx_type(dtype)} [{addr}], {reg};")
```

For now, `smem_elem_addr` can raise `NotImplementedError`; it arrives in Task 15. Pointwise problems don't use it.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower global writes"
```

---

### Task 10: p00 passes

**Objective:** First end-to-end PASS.

**Files:** Modify `compile.py` as needed.

**Step 1: Compile and inspect**

```bash
.venv/bin/python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/p00.ptx
head -40 /tmp/p00.ptx
```
Expected: loads of `x0`/`x1`, `add.f32`, store to `add`.

**Step 2: Run**

```bash
.venv/bin/python submit.py p00_scalar_add --compiler compile.py
```
Expected: `PASS`.

**Step 3: Debug if needed**

- If ptxas rejects: post `/tmp/p00.ptx` to `/compile` and read the error.
- If wrong values: confirm `arg_order` and shapes from `meta.json`.
- If hang: confirm the Tile end label is emitted after the body.

**Step 4: Commit**

```bash
git add compile.py && git commit -m "feat: pass p00 scalar add"
```

---

### Task 11: p01–p03 pass

**Objective:** Handle vector names, constants, and `relu`.

**Files:** Modify `compile.py` if needed.

**Steps:**

```bash
for p in p01_vector_add p02_bias_relu p03_saxpy_scale; do
  .venv/bin/python submit.py "$p" --compiler compile.py
done
```
Expected: three `PASS`.

If a vector load/store has N names and the last index is not the fastest-varying dimension, fix `Load`/`Write` to increment the correct axis. For these problems only the last axis is vectorized.

Commit:
```bash
git add compile.py && git commit -m "feat: pass pointwise vector and constant kernels"
```

---

### Task 12: Dev harness

**Objective:** One command for compile-only checks and one for all correctness checks.

**Files:**
- Create: `dev/check_ptx.py`
- Create: `dev/run_all.py`

**`dev/check_ptx.py`:**

```python
import json, subprocess, sys, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
url = "http://runner.riftstack.ai:48701/compile"
for p in sorted((ROOT / "problems").iterdir()):
    if not (p / "meta.json").exists(): continue
    ir = next(p.glob("*.ir.json"))
    r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
    if r.returncode: print(p.name, "COMPILE-FAIL", r.stderr[-500:]); continue
    resp = requests.post(url, data=r.stdout)
    print(p.name, resp.status_code, resp.text[:300])
```

**`dev/run_all.py`:**

```python
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
fails = []
for p in sorted((ROOT / "problems").iterdir()):
    if not (p / "meta.json").exists(): continue
    r = subprocess.run([sys.executable, str(ROOT / "submit.py"), p.name, "--compiler", str(ROOT / "compile.py")], capture_output=True, text=True)
    ok = "PASS" in r.stdout
    print(p.name, "PASS" if ok else "FAIL")
    if not ok: print(r.stdout[-800:], r.stderr[-800:]); fails.append(p.name)
sys.exit(1 if fails else 0)
```

Run:
```bash
.venv/bin/python dev/run_all.py
```
Expected: p00–p03 PASS; the rest FAIL (compile errors are fine at this stage).

Commit:
```bash
git add dev && git commit -m "chore: add compile-only and full-suite dev harnesses"
```

---

## Phase 2 — Category r (reductions)

### Task 13: Loops and accumulator seeding

**Objective:** `Loop` and `StridedLoop` with correct counter, bound, and `seed` behavior.

**Files:** Modify `compile.py`.

**Step 1: Implement seeding**

```python
    def seed_loop_accums(self, body):
        for st in body:
            if st.__class__.__name__ == "Accum":
                if st.name in self.seeded_now: continue
                self.seeded_now.add(st.name)
                ident = st.op.identity
                if ident is None:
                    raise SystemExit(f"no identity for {st.op}")
                dt = dtype_name(st.dtype) if st.dtype is not None else self.value_dtypes.get(st.name, "f32")
                if "inf" in str(ident):
                    reg = self.regs.new("f")
                    bits = IDENTITY_BITS["-inf" if "-" in str(ident) else "inf"]
                    self.ins(f"mov.f32 {reg}, 0f{bits:08X};")
                else:
                    reg, _ = self.lit(ident, dt)
                self.values[st.name], self.value_dtypes[st.name] = reg, dt
```

`self.seeded_now` is a set local to the loop emission; save/restore around nested loops.

**Step 2: Implement `Loop`**

```python
    def stmt_loop(self, s):
        if s.end is None:
            end = self.int_reg(int(s.extent))
        else:
            end, _ = self.expr(s.end)
        if s.seed:
            self.seed_loop_accums(s.body)
        counter = self.int_reg(int(s.start))
        head = self.label("LOOP")
        tail = self.label("LOOP_END")
        self.ins(f"setp.ge.s32 %pX, {counter}, {end};")  # use a fresh pred
        self.ins(f"bra {tail};")  # replaced by predicated branch below
        ...
```

Correct form:

```python
        head = self.label("LOOP")
        tail = self.label("LOOP_END")
        self.raw(head + ":")
        p = self.regs.new("p")
        self.ins(f"setp.ge.s32 {p}, {counter}, {end};")
        self.ins(f"@{p} bra {tail};")
        old = self.values.get(s.axis)
        self.values[s.axis], self.value_dtypes[s.axis] = counter, "i32"
        self.stmt(s.body)
        self.ins(f"add.s32 {counter}, {counter}, 1;")
        self.ins(f"bra {head};")
        self.raw(tail + ":")
        if old is not None: self.values[s.axis] = old
```

Use `s.end` evaluated once. If `s.extent` is an int, use it.

**Step 3: Implement `StridedLoop`**

```python
    def stmt_strided_loop(self, s):
        if s.seed: self.seed_loop_accums(s.body)
        counter, _ = self.expr(s.start)
        step, _ = self.expr(s.step)
        end = self.int_reg(int(s.extent)) if s.end is None else self.expr(s.end)[0]
        head = self.label("SL")
        tail = self.label("SL_END")
        self.raw(head + ":")
        p = self.regs.new("p"); self.ins(f"setp.ge.s32 {p}, {counter}, {end};")
        self.ins(f"@{p} bra {tail};")
        self.values[s.axis], self.value_dtypes[s.axis] = counter, "i32"
        self.stmt(s.body)
        self.ins(f"add.s32 {counter}, {counter}, {step};")
        self.ins(f"bra {head};")
        self.raw(tail + ":")
```

**Step 4: Implement `Accum`**

```python
    def stmt_accum(self, s):
        a = self.values[s.name]
        b = self.values[s.base] if s.base else a
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.name]
        reg, _ = self.apply_op(s.op, (a, b), dtype)
        self.values[s.name], self.value_dtypes[s.name] = reg, dtype
```

**Step 5: Commit**

```bash
git add compile.py && git commit -m "feat: lower loops, strided loops, and accumulators"
```

---

### Task 14: Static shared memory and Sync

**Objective:** Declare small shared arrays and address them.

**Files:** Modify `compile.py`.

**Step 1: Pre-scan `Smem`**

```python
    def collect_smem(self, stmt, out):
        if stmt is None: return
        if stmt.__class__.__name__ == "Smem":
            out.append(stmt)
        for attr in ("body", "if_body", "else_body"):
            for child in getattr(stmt, attr, []) or []:
                self.collect_smem(child, out)
```

Add `Body` iteration and handle `Tile.body`, `Cond.if_body/else_body`.

**Step 2: Layout**

```python
    def plan_smem(self):
        decls = []
        for s in self.smem_stmts:
            extents = [int(x) for x in s.extents]
            n = 1
            for x in extents: n *= x
            size = n * DTYPE_BYTES[dtype_name(s.dtype)]
            align = int(s.align or 4)
            self.smem[s.name] = {"extents": extents, "dtype": dtype_name(s.dtype),
                                 "size": size, "align": align, "static": None, "offset": 0}
            decls.append(s)
        total = sum(v["size"] for v in self.smem.values())
        self.dynamic_smem = total > 48 * 1024
        if not self.dynamic_smem:
            for name, v in self.smem.items():
                v["static"] = name
            self.shared_decls = [f".shared .align {v['align']} .b8 {name}[{v['size']}];" for name, v in self.smem.items()]
        else:
            off = 0
            for name, v in self.smem.items():
                off = (off + v["align"] - 1) // v["align"] * v["align"]
                v["offset"] = off
                off += v["size"]
            self.shared_decls = [f".extern .shared .align 16 .b8 __emmy_smem[{off}];"]
```

**Step 3: `smem_elem_addr`**

```python
    def smem_elem_addr(self, name, flat):
        info = self.smem[name]
        base = self.smem_base(name)
        off = self.regs.new("r")
        self.ins(f"mul.lo.s32 {off}, {flat}, {DTYPE_BYTES[info['dtype']]};")
        addr = self.regs.new("r")
        self.ins(f"add.s32 {addr}, {base}, {off};")
        return addr

    def smem_base(self, name):
        if name in self.smem_bases: return self.smem_bases[name]
        info = self.smem[name]
        base = self.regs.new("r")
        if info["static"] is not None:
            self.ins(f"mov.u32 {base}, {info['static']};")
        else:
            pool = self.smem_bases.setdefault("__pool__", None)
            if pool is None:
                pool = self.regs.new("r"); self.ins(f"mov.u32 {pool}, __emmy_smem;")
                self.smem_bases["__pool__"] = pool
            if info["offset"]:
                self.ins(f"add.s32 {base}, {pool}, {info['offset']};")
            else:
                base = pool
        self.smem_bases[name] = base
        return base
```

**Step 4: `Sync`**

```python
    def stmt_sync(self, s):
        if s.kind == "warp":
            self.ins("bar.warp.sync 0xffffffff;")
        elif s.barrier_id is not None:
            self.ins(f"bar.sync {s.barrier_id}, {s.count};")
        else:
            self.ins("bar.sync 0;")
```

Check the actual attribute names on `Sync` before coding (`kind`, `barrier_id`, `count`).

**Step 5: Commit**

```bash
git add compile.py && git commit -m "feat: add shared memory layout and barriers"
```

---

### Task 15: Warp shuffle

**Objective:** Lower `WarpShuffle` for any number of states.

**Files:** Modify `compile.py`.

**Step 1: Implement**

```python
    def stmt_warp_shuffle(self, s):
        mask = self.regs.new("r")
        self.ins(f"activemask.b32 {mask};")
        step = int(s.length) // 2
        while step > 0:
            for state, state_b in s.states:
                a = self.values[state]
                bits = self.regs.new("r")
                self.ins(f"mov.b32 {bits}, {a};")
                other = self.regs.new("r")
                self.ins(f"shfl.sync.bfly.b32 {other}, {bits}, {step}, 0x1f, {mask};")
                reg = self.regs.new("f")
                self.ins(f"mov.b32 {reg}, {other};")
                self.values[state_b], self.value_dtypes[state_b] = reg, self.value_dtypes[state]
            self.stmt(s.combine_states)
            step //= 2
```

If `s.dtype` is integer, use `%r` for the moved value and skip the float move. Gate on `dtype_name(s.dtype)`.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower warp shuffles"
```

---

### Task 16: Tree halve

**Objective:** Lower shared-memory halving trees.

**Files:** Modify `compile.py`.

**Step 1: Implement**

Use `self.smem[s.buffer]["extents"]` to compute strides. For each step `step = half; step > 0; step >>= 1`:
- Emit `setp.lt.s32 p, tid, step; @!p bra SKIP;`
- For each component `(state, state_b)`:
  - index `tid` (or `tid*scale + iv` when `inner`), and `tid + step` (same formula).
  - load both scalars from `s.buffer`.
  - bind `state` → first reg, `state_b` → second reg.
  - emit `s.combine_states` (these update `state`).
  - store `state` back at `tid`.
- Emit `SKIP:`
- Emit the barrier (`bar.sync 0` or `bar.sync id, count`).
- After the loop, load root:
  - non-inner: index 0.
  - inner: index `iv` (the per-component segment root).
  - assign to the real state register name.

Important: use fresh registers for the loads; the combine reassigns `state` names, and the store must use the post-combine register. After the final broadcast, the state names are overwritten with the root value.

**Step 2: Commit**

```bash
git add compile.py && git commit -m "feat: lower tree-halving shared reductions"
```

---

### Task 17: Cond and Select

**Objective:** Support predicated branches and multi-way selects.

**Files:** Modify `compile.py`.

**Step 1: Implement `Cond`**

```python
    def stmt_cond(self, s):
        p, _ = self.expr(s.cond)
        els = self.label("ELSE")
        end = self.label("COND_END")
        self.ins(f"@!{p} bra {els};")
        self.stmt(s.if_body)
        self.ins(f"bra {end};")
        self.raw(els + ":")
        self.stmt(s.else_body)
        self.raw(end + ":")
```

If `expr` returns a value instead of a predicate, convert: `setp.ne.s32 p, value, 0;`.

**Step 2: Implement `Select`**

```python
    def stmt_select(self, s):
        last_val, dt = self.expr(s.branches[-1][0])
        self.values[s.name], self.value_dtypes[s.name] = last_val, dt
        for value, cond in reversed(s.branches[:-1]):
            p, _ = self.expr(cond)
            v, _ = self.expr(value)
            self.ins(f"@{p} mov.{ptx_type(dt)} {self.values[s.name]}, {v};")
```

**Step 3: Commit**

```bash
git add compile.py && git commit -m "feat: lower conditionals and selects"
```

---

### Task 18: r01–r04 pass

**Objective:** All reduction problems pass.

**Files:** Modify `compile.py` as needed.

**Steps:**

```bash
for p in r01_array_sum r02_max_reduce r03_rms_norm r04_softmax; do
  .venv/bin/python submit.py "$p" --compiler compile.py
done
```
Expected: four `PASS`.

Debug order:
1. ptxas error → fix types/operands.
2. Wrong numeric result → compare with `problems/<p>/<kernel>.ref.cu`; check seeding, shuffle direction, tree root.
3. Hang → verify every barrier is reached by all threads and loop counters advance.

Commit:
```bash
git add compile.py && git commit -m "feat: pass all reduction kernels"
```

---

## Phase 3 — Category m (matmul)

### Task 19: Init and dynamic shared memory

**Objective:** Zero accumulators and use the dynamic pool.

**Files:** Modify `compile.py`.

**Step 1: `Init`**

```python
    def stmt_init(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else "f32"
        if "inf" in str(s.value):
            reg = self.regs.new("f"); bits = IDENTITY_BITS["-inf" if "-" in str(s.value) else "inf"]
            self.ins(f"mov.f32 {reg}, 0f{bits:08X};")
        else:
            reg, _ = self.lit(s.value, dtype)
        self.values[s.name], self.value_dtypes[s.name] = reg, dtype
```

**Step 2: Confirm dynamic layout**

Run:
```bash
.venv/bin/python compile.py problems/m01_tiled_matmul/tiled_matmul.ir.json > /tmp/m01.ptx
grep -E "extern .shared|shared .align" /tmp/m01.ptx
```
Expected: one `.extern .shared .align 16 .b8 __emmy_smem[...]` and no static arrays; total ≥ 73728.

**Step 3: Commit**

```bash
git add compile.py && git commit -m "feat: lower Init and dynamic shared memory"
```

---

### Task 20: cp.async

**Objective:** Lower `CpAsyncCopy`, `CpAsyncCommit`, `CpAsyncWait`.

**Files:** Modify `compile.py`.

**Step 1: Implement**

```python
    def stmt_cp_async(self, s):
        flat = self.flatten(s.smem, s.smem_index) if s.smem_index else self.int_reg(0)
        if s.lane_index is not None:
            cols = self.shape_of(s.smem)[-1]
            lane = self.add_int(self.mul_int(self.expr(s.lane_index[0])[0], cols),
                                self.expr(s.lane_index[1])[0])
            flat = self.add_int(flat, lane)
        dst = self.smem_elem_addr(s.smem, flat)
        src_flat = self.flatten(s.src, s.src_index)
        src = self.global_addr(s.src, src_flat)
        nbytes = int(s.bytes)
        op = "cg" if nbytes == 16 else "ca"
        self.ins(f"cp.async.{op}.shared.global [{dst}], [{src}], {nbytes};")
```

`s.bytes` might be a field or derived from extents/dtype. Check the dataclass; use `s.size`/`s.nbytes` if present. Fall back to `16`.

**Step 2: Commit/wait**

```python
    def stmt_cp_commit(self, s): self.ins("cp.async.commit_group;")
    def stmt_cp_wait(self, s): self.ins(f"cp.async.wait_group {int(s.group)};")
```

**Step 3: Commit**

```bash
git add compile.py && git commit -m "feat: lower cp.async copies"
```

---

### Task 21: Atomic writes

**Objective:** Lower `red.global.*`.

**Files:** Modify `compile.py`.

`Write.atomic` is a string like `"add"`. Emit:
```python
self.ins(f"red.global.{s.atomic}.{ptx_type(dtype)} [{addr}], {reg};")
```
The runner zeroes `zero_outputs`, so no extra initialization is needed.

Commit:
```bash
git add compile.py && git commit -m "feat: lower atomic writes"
```

---

### Task 22: m01 and m02 pass

**Objective:** Both matmul problems pass.

**Files:** Modify `compile.py` as needed.

**Steps:**
```bash
.venv/bin/python submit.py m01_tiled_matmul --compiler compile.py
.venv/bin/python submit.py m02_splitk_atomic --compiler compile.py
```
Expected: two `PASS`.

Debug checklist:
- Verify shared offsets: `_a_smem` 0, `_b_smem` 24576 for m01.
- Verify `cp.async` byte count and `bar.sync` placement.
- Verify `k` loop is `0..K` with `smem_index (a0_u*32+a1_u)`.
- For m02, confirm the atomic output is zeroed by the runner and the partial buffer is only an input.

Commit:
```bash
git add compile.py && git commit -m "feat: pass matmul kernels"
```

---

## Phase 4 — Categories f and a

### Task 23: Let and multi-component reductions

**Objective:** Support fused kernels with `Let` and several simultaneous reduction states.

**Files:** Modify `compile.py`.

**Steps:**
1. Ensure `stmt_let` evaluates and binds.
2. Ensure `WarpShuffle` and `TreeHalve` handle multiple `(state, state_b)` pairs (already generic).
3. Run:
```bash
for p in f01_bias_gelu f02_layer_norm f03_residual_rms; do
  .venv/bin/python submit.py "$p" --compiler compile.py
done
```
Expected: three `PASS`.

Commit:
```bash
git add compile.py && git commit -m "feat: pass fused kernels"
```

---

### Task 24: Tile rasterization

**Objective:** Reproduce the reference tile order for attention.

**Files:** Modify `compile.py`.

**Step 1: Inspect**

Run:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from emmy.compiler.graph import Graph
raw = json.loads(Path("problems/a01_attention/attention.ir.json").read_text())
g = Graph.from_dict(raw)
for n in g.nodes.values():
    if n.op:
        for st in n.op.body:
            if st.__class__.__name__ == "Tile":
                print(st.extents, st.raster_axes, st.raster_group, st.raster_orient)
PY
```
Expected: `raster_axes=('a0_b','a5_b')`, group 8, orient `'m'`.

**Step 2: Implement**

- Map each axis name to its logical group `a{i}`, splitting the last `_b` suffix.
- Decode strides over the listed axes only.
- If `raster_group` is set, compute `group_id = p0 // raster_group`, `local = p0 % raster_group`, walk the remaining axes.
- If `raster_orient == 'm'`, swap the roles of the last two raster axes.
- Start from the logical axes, then overwrite the raster axes with decoded values.
- Let `min(axis, axis_high - 1)` guard ragged tiles.
- Validate against `problems/a01_attention/attention.ref.cu` for the first 16 tiles.

**Step 3: Commit**

```bash
git add compile.py && git commit -m "feat: implement tile rasterization"
```

---

### Task 25: a01 and a02 pass

**Objective:** Attention kernels pass.

**Files:** Modify `compile.py` as needed.

**Steps:**
```bash
.venv/bin/python submit.py a01_attention --compiler compile.py
.venv/bin/python submit.py a02_flash_attention --compiler compile.py
```
Expected: two `PASS`.

Debug checklist:
- Raster order must match; verify with the reference block loop.
- `acc4`/`acc5` max rescale: `exp(old_max - new_max)`.
- `acc6`/`acc7` normalized by `sum`.
- Barriers around `_k_smem`/`_v_smem` reuse.

Commit:
```bash
git add compile.py && git commit -m "feat: pass attention kernels"
```

---

## Phase 5 — Hardening and submission

### Task 26: Hidden-like generated variants

**Objective:** Test generalization beyond the 15 open IRs.

**Files:**
- Create: `dev/expressions_extra.py`
- Create: `dev/cases/` (generated)

**Steps:**
1. Install CPU torch in the venv:
```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```
2. Copy/extend the examples from `tools/expressions_open.py` with changed dimensions, non-power-of-two sizes, and both actions.
3. Generate:
```bash
.venv/bin/emmy compile dev/expressions_extra.py --output dev/cases
```
4. For each case, run `submit.py` if it has an `expected.npz` and `meta.json`.
5. Fix any mismatches; re-run all open problems.

Commit:
```bash
git add dev && git commit -m "test: add generated variant coverage"
```

---

### Task 27: Edge cases

**Objective:** Partial warps, odd sizes, small reductions, non-square matmul.

**Files:** Modify `compile.py`.

Checklist:
- Guard the Tile always; never assume `N % block == 0`.
- Warp shuffle with `length < 32`: keep lanes ≥ length inactive or ensure their values don't affect lane 0. Verify with a generated `N=1` reduction.
- `TreeHalve` with `length=1`: skip the tree and broadcast directly.
- Matmul with `M`, `N`, `K` not multiples of 32.
- `Select` with a single branch.
- Integer literals and comparisons in conditions.

Run `dev/run_all.py` after each fix.

Commit:
```bash
git add compile.py && git commit -m "fix: harden edge cases"
```

---

### Task 28: Speed pass (optional tiebreak)

**Objective:** Reduce PTX overhead without risking correctness.

**Files:** Modify `compile.py`.

Only after all 15 open problems pass:
- Emit `ld.global.v4.f32` / `st.global.v4.f32` for contiguous vector names.
- Fold `mul`+`add` into `fma.rn.f32` in `Assign`/`Accum` when safe.
- Hoist constant strides and axis decodes.

Verify with `dev/run_all.py` and `dev/check_ptx.py` after each change. Commit separately:
```bash
git add compile.py && git commit -m "perf: vectorize contiguous global accesses"
```

---

### Task 29: Cleanup and submission

**Objective:** Make the repo easy to audit and submit.

**Files:**
- Modify: `README.md` (add a short "How it works" section)
- Create: `docs/VIDEO_SCRIPT.md`

**Steps:**
1. Remove dead code and debug prints.
2. Add a 150–250 word explanation: load JSON → `Graph.from_dict` → tile/thread mapping → expression/statement lowering → shared layout → PTX.
3. Write a 3–5 minute video script:
   - 0:00 problem and runner
   - 0:30 IR structure (Tile, Load, Assign, Write, WarpShuffle)
   - 1:30 walk `compile.py` lowering
   - 3:00 show p00, r01, m01, a01 passing
   - 4:00 limits and what you learned
4. Run `dev/run_all.py`, capture output.
5. Commit:
```bash
git add README.md docs/VIDEO_SCRIPT.md compile.py
git commit -m "docs: explain compiler design and record submission notes"
```

---

## Definition of Done

- `python compile.py <ir>` emits PTX for all 15 open problems.
- `dev/run_all.py` shows 15/15 PASS.
- Compile-only `dev/check_ptx.py` shows no ptxas errors.
- At least a few generated variants pass.
- `git status` is clean and the repo contains `compile.py`, tests, dev harnesses, and docs.
- The video is recorded in the user's own voice and explains the lowering.
