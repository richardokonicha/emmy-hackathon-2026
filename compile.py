#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX compiler.

Usage: python compile.py path/to/<kernel>.ir.json > out.ptx

The IR is loaded through the public ``Graph.from_dict`` API.  Shapes and
dtypes are read from the raw JSON because the typed graph does not expose
them.  Every IR node is lowered directly to PTX; no existing CUDA/PTX
renderer is used.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FlatIndex,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    Smem,
    Sync,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Select,
    StridedLoop,
    Write,
)

PTX_HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"
SMEM_STATIC_LIMIT = 48 * 1024

_DTYPE_ALIASES = {
    "float": "f32",
    "double": "f64",
    "half": "f16",
    "int": "s32",
    "bool": "s32",
}


def dtype_name(dt) -> str:
    name = getattr(dt, "name", str(dt))
    return _DTYPE_ALIASES.get(name, name)


def dtype_bytes(dt) -> int:
    name = dtype_name(dt)
    if name in ("f32", "s32", "u32"):
        return 4
    if name in ("f16", "u16"):
        return 2
    if name in ("f64", "s64", "u64"):
        return 8
    raise SystemExit(f"unsupported dtype size: {name}")


def is_float(dt) -> bool:
    return dtype_name(dt).startswith("f")


def ptx_type(dt) -> str:
    name = dtype_name(dt)
    if name in ("f32", "f16", "f64"):
        return name
    if name in ("s32", "i32", "int"):
        return "s32"
    if name in ("s64", "i64"):
        return "s64"
    if name in ("b32", "u32"):
        return "b32"
    raise SystemExit(f"unsupported PTX type: {name}")


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class Regs:
    """Virtual register pool.  ptxas does the physical allocation."""

    def __init__(self):
        self.n = {"f": 0, "r": 0, "rd": 0, "p": 0}

    def new(self, kind: str) -> str:
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

    def ins(self, text: str):
        self.lines.append("        " + text)

    def raw(self, text: str):
        self.lines.append(text)

    def label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}_{self.label_n}"


class Compiler:
    def __init__(self, ir_path: Path):
        self.ir_path = Path(ir_path)
        self.raw = json.loads(self.ir_path.read_text())
        self.graph = Graph.from_dict(self.raw)
        self.kernel = self._find_kernel()
        self.meta = self._find_meta()
        self.entry = self.kernel.name or self._kernel_meta()["entry"]
        self.kmeta = self._kernel_meta()

        self.tensors: dict[str, dict] = {}
        for node in self.raw.get("nodes", {}).values():
            for out in node.get("outputs", []):
                self.tensors[out["name"]] = out
        for arg in self.kmeta.get("args", []):
            self.tensors.setdefault(
                arg["name"],
                {"name": arg["name"], "shape": arg.get("shape", []), "dtype": arg.get("dtype", "f32")},
            )

        self.arg_order = list(self.kmeta["arg_order"])
        self.args = {a["name"]: a for a in self.kmeta.get("args", [])}

        self.emitter = Emitter()
        self.values: dict[str, str] = {}
        self.value_dtypes: dict[str, str] = {}
        self.base: dict[str, str] = {}
        self.special: dict[str, str] = {}
        self.smem: dict[str, dict] = {}
        self.smem_bases: dict[str, str] = {}
        self.shared_decls: list[str] = []
        self.dynamic_smem = False

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def _find_kernel(self) -> KernelOp:
        for node in self.graph.nodes.values():
            if isinstance(node.op, KernelOp):
                return node.op
        raise SystemExit("no KernelOp found in IR")

    def _find_meta(self) -> dict:
        meta_path = self.ir_path.with_name("meta.json")
        if not meta_path.exists():
            return {"kernels": [{"entry": self.kernel.name, "arg_order": list(self.kernel.inputs) + list(self.kernel.outputs), "args": []}]}
        return json.loads(meta_path.read_text())

    def _kernel_meta(self) -> dict:
        for k in self.meta.get("kernels", []):
            if k.get("entry") == self.kernel.name:
                return k
        return self.meta["kernels"][0]

    def shape_of(self, name: str):
        if name in self.smem:
            return self.smem[name]["extents"]
        if name not in self.tensors:
            raise SystemExit(f"unknown buffer {name!r}")
        return self.tensors[name].get("shape", [])

    def dtype_of(self, name: str):
        if name in self.smem:
            return self.smem[name]["dtype"]
        if name not in self.tensors:
            raise SystemExit(f"unknown buffer {name!r}")
        return self.tensors[name].get("dtype", "f32")

    # ------------------------------------------------------------------
    # shared memory planning
    # ------------------------------------------------------------------
    @staticmethod
    def _children(stmt):
        if isinstance(stmt, (Body, list, tuple)):
            for child in stmt:
                yield child
            return
        for attr in ("body", "else_body"):
            child = getattr(stmt, attr, None)
            if child is not None:
                yield from child

    def _walk(self, stmt):
        for child in self._children(stmt):
            yield child
            yield from self._walk(child)

    def plan_smem(self):
        smem_stmts = [s for s in self._walk(self.kernel.body) if isinstance(s, Smem)]
        if not smem_stmts:
            return
        total = 0
        for s in smem_stmts:
            n = 1
            for e in s.extents:
                n *= int(e)
            size = n * dtype_bytes(s.dtype)
            align = int(s.align or 4)
            self.smem[s.name] = {
                "extents": [int(e) for e in s.extents],
                "dtype": dtype_name(s.dtype),
                "size": size,
                "align": align,
                "offset": 0,
                "static": None,
            }
            total += size
        self.dynamic_smem = total > SMEM_STATIC_LIMIT
        if not self.dynamic_smem:
            for name, info in self.smem.items():
                info["static"] = name
                self.shared_decls.append(
                    f".shared .align {info['align']} .b8 {name}[{info['size']}];"
                )
        else:
            off = 0
            for name, info in self.smem.items():
                align = info["align"]
                off = (off + align - 1) // align * align
                info["offset"] = off
                off += info["size"]
            self.shared_decls.append(".extern .shared .align 16 .b8 __emmy_smem[];")

    def _materialize_smem_base(self, name: str) -> str:
        if name in self.smem_bases:
            return self.smem_bases[name]
        info = self.smem[name]
        base = self.emitter.regs.new("r")
        if info["static"] is not None:
            self.emitter.ins(f"mov.u32 {base}, {info['static']};")
        else:
            pool = self.smem_bases.get("__pool__")
            if pool is None:
                pool = self.emitter.regs.new("r")
                self.emitter.ins(f"mov.u32 {pool}, __emmy_smem;")
                self.smem_bases["__pool__"] = pool
            if info["offset"]:
                self.emitter.ins(f"add.s32 {base}, {pool}, {info['offset']};")
            else:
                base = pool
        self.smem_bases[name] = base
        return base

    def smem_base(self, name: str) -> str:
        return self._materialize_smem_base(name)

    def smem_elem_addr(self, name: str, flat: str) -> str:
        info = self.smem[name]
        off = self.emitter.regs.new("r")
        self.emitter.ins(f"mul.lo.s32 {off}, {flat}, {dtype_bytes(info['dtype'])};")
        addr = self.emitter.regs.new("r")
        self.emitter.ins(f"add.s32 {addr}, {self.smem_base(name)}, {off};")
        return addr

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def int_reg(self, value: int) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.s32 {reg}, {int(value)};")
        return reg

    def fconst(self, value: float) -> str:
        reg = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {reg}, 0f{float_bits(value):08X};")
        return reg

    def mul_int(self, a: str, b) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mul.lo.s32 {reg}, {a}, {b};")
        return reg

    def add_int(self, a: str, b) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"add.s32 {reg}, {a}, {b};")
        return reg

    @staticmethod
    def const_int(e):
        if isinstance(e, Literal):
            name = dtype_name(e.dtype)
            if name.startswith("s") or name.startswith("i") or name == "int":
                return int(e.value)
            if isinstance(e.value, int):
                return int(e.value)
            return None
        if isinstance(e, BinaryExpr):
            a = Compiler.const_int(e.left)
            b = Compiler.const_int(e.right)
            if a is None or b is None:
                return None
            if e.op == "+":
                return a + b
            if e.op == "-":
                return a - b
            if e.op == "*":
                return a * b
            if e.op == "/" and b:
                return a // b
            if e.op == "%" and b:
                return a % b
        return None

    def add_const_expr(self, e, k: int):
        c = self.const_int(e)
        if c is not None:
            return Literal(c + k, "int")
        return BinaryExpr("+", e, Literal(k, "int"))

    def flatten(self, name: str, indices) -> str:
        shape = self.shape_of(name)
        indices = tuple(indices or ())
        if not indices:
            return self.int_reg(0)
        if len(indices) != len(shape):
            raise SystemExit(
                f"index rank {len(indices)} != shape rank {len(shape)} for {name}"
            )
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        acc_reg = None
        acc_const = 0
        for idx, stride in zip(indices, strides):
            c = self.const_int(idx)
            if c is not None:
                acc_const += c * stride
                continue
            reg, _ = self.expr(idx)
            if stride != 1:
                reg = self.mul_int(reg, stride)
            acc_reg = reg if acc_reg is None else self.add_int(acc_reg, reg)
        if acc_reg is None:
            return self.int_reg(acc_const)
        if acc_const:
            acc_reg = self.add_int(acc_reg, acc_const)
        return acc_reg

    def global_addr(self, name: str, flat: str) -> str:
        if name not in self.base:
            raise SystemExit(f"no pointer parameter for buffer {name!r}")
        off = self.emitter.regs.new("rd")
        self.emitter.ins(f"mul.wide.s32 {off}, {flat}, {dtype_bytes(self.dtype_of(name))};")
        addr = self.emitter.regs.new("rd")
        self.emitter.ins(f"add.s64 {addr}, {self.base[name]}, {off};")
        return addr

    def to_pred(self, e) -> str:
        reg, dt = self.expr(e)
        if dt == "pred":
            return reg
        p = self.emitter.regs.new("p")
        if is_float(dt):
            self.emitter.ins(f"setp.ne.f32 {p}, {reg}, 0f00000000;")
        else:
            self.emitter.ins(f"setp.ne.s32 {p}, {reg}, 0;")
        return p

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def lit(self, value, dtype="f32"):
        name = dtype_name(dtype)
        if name.startswith("s") or name.startswith("i") or name == "int":
            reg = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.s32 {reg}, {int(value)};")
            return reg, "s32"
        reg = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {reg}, 0f{float_bits(value):08X};")
        return reg, "f32"

    def expr(self, e):
        if isinstance(e, Literal):
            name = dtype_name(e.dtype)
            if name.startswith("s") or name.startswith("i") or name == "int" or isinstance(e.value, int):
                return self.lit(e.value, "s32")
            return self.lit(e.value, "f32")
        if isinstance(e, Var):
            if e.name not in self.values:
                raise SystemExit(f"unknown variable {e.name!r}")
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
            regs = [self.expr(a)[0] for a in e.args]
            return self.op_on_regs(e.name, regs, "f32")
        if isinstance(e, FlatIndex):
            return self.flatten(e.buffer, e.index), "s32"
        raise SystemExit(f"unsupported expression {type(e).__name__}: {e!r}")

    def coerce(self, reg, dt, target):
        if dtype_name(dt) == dtype_name(target):
            return reg
        if is_float(target) and not is_float(dt):
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"cvt.rn.f32.s32 {out}, {reg};")
            return out
        if not is_float(target) and is_float(dt):
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"cvt.rzi.s32.f32 {out}, {reg};")
            return out
        return reg

    def binary(self, e: BinaryExpr):
        a, adt = self.expr(e.left)
        b, bdt = self.expr(e.right)
        op = e.op

        if op in ("==", "!=", "<", "<=", ">", ">="):
            target = "f32" if (is_float(adt) or is_float(bdt)) else "s32"
            a = self.coerce(a, adt, target)
            b = self.coerce(b, bdt, target)
            p = self.emitter.regs.new("p")
            kind = "f32" if target == "f32" else "s32"
            cmp = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
            self.emitter.ins(f"setp.{cmp}.{kind} {p}, {a}, {b};")
            return p, "pred"

        if op in ("and", "or"):
            if adt == "pred" and bdt == "pred":
                p = self.emitter.regs.new("p")
                self.emitter.ins(f"{op}.pred {p}, {a}, {b};")
                return p, "pred"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"{op}.b32 {out}, {a}, {b};")
            return out, "s32"

        target = "f32" if (is_float(adt) or is_float(bdt)) else "s32"
        a = self.coerce(a, adt, target)
        b = self.coerce(b, bdt, target)

        if target == "f32":
            ptxop = {
                "+": "add.f32",
                "-": "sub.f32",
                "*": "mul.f32",
                "/": "div.rn.f32",
                "maximum": "max.f32",
                "minimum": "min.f32",
                "max": "max.f32",
                "min": "min.f32",
            }.get(op)
            if ptxop is None:
                raise SystemExit(f"unsupported float binary op {op!r}")
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"{ptxop} {out}, {a}, {b};")
            return out, "f32"

        ptxop = {
            "+": "add.s32",
            "-": "sub.s32",
            "*": "mul.lo.s32",
            "/": "div.s32",
            "%": "rem.s32",
            "and": "and.b32",
            "or": "or.b32",
            "xor": "xor.b32",
            "maximum": "max.s32",
            "minimum": "min.s32",
            "max": "max.s32",
            "min": "min.s32",
        }.get(op)
        if ptxop is None:
            raise SystemExit(f"unsupported integer binary op {op!r}")
        out = self.emitter.regs.new("r")
        self.emitter.ins(f"{ptxop} {out}, {a}, {b};")
        return out, "s32"

    def ternary(self, e: TernaryExpr):
        p = self.to_pred(e.cond)
        a, adt = self.expr(e.if_true)
        b, bdt = self.expr(e.if_false)
        target = "f32" if (is_float(adt) or is_float(bdt)) else "s32"
        a = self.coerce(a, adt, target)
        b = self.coerce(b, bdt, target)
        out = self.emitter.regs.new("f" if target == "f32" else "r")
        self.emitter.ins(f"selp.{ptx_type(target)} {out}, {a}, {b}, {p};")
        return out, target

    def cast(self, e: CastExpr):
        reg, src = self.expr(e.expr)
        dst = dtype_name(e.dtype)
        if dst == src:
            return reg, src
        if dst == "f32" and not is_float(src):
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"cvt.rn.f32.s32 {out}, {reg};")
            return out, "f32"
        if dst == "s32" and is_float(src):
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"cvt.rzi.s32.f32 {out}, {reg};")
            return out, "s32"
        raise SystemExit(f"unsupported cast {src} -> {dst}")

    def builtin(self, e: Builtin):
        if e.name in self.special:
            return self.special[e.name], "s32"
        special = {
            "threadIdx.x": "%tid.x",
            "threadIdx.y": "%tid.y",
            "threadIdx.z": "%tid.z",
            "blockIdx.x": "%ctaid.x",
            "blockIdx.y": "%ctaid.y",
            "blockIdx.z": "%ctaid.z",
            "blockDim.x": "%ntid.x",
            "blockDim.y": "%ntid.y",
            "blockDim.z": "%ntid.z",
            "gridDim.x": "%nctaid.x",
        }.get(e.name)
        if special is None:
            raise SystemExit(f"unsupported builtin {e.name!r}")
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {reg}, {special};")
        self.special[e.name] = reg
        return reg, "s32"

    # ------------------------------------------------------------------
    # elementwise ops
    # ------------------------------------------------------------------
    def op_on_regs(self, op, args, dtype="f32", dest=None):
        name = getattr(op, "name", str(op))
        dt = dtype_name(dtype)
        if name == "where":
            cond, a, b = args
            p = self.emitter.regs.new("p")
            self.emitter.ins(f"setp.ne.s32 {p}, {cond}, 0;")
            out = self.emitter.regs.new("f" if is_float(dt) else "r")
            self.emitter.ins(f"selp.{ptx_type(dt)} {out}, {a}, {b}, {p};")
            return out, dt

        if name in ("exp", "exp_fast"):
            x = args[0]
            t = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {t}, {x}, 0f3FB8AA3B;")
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"ex2.approx.f32 {out}, {t};")
            return out, "f32"

        if name == "relu":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"max.f32 {out}, {args[0]}, 0f00000000;")
            return out, "f32"

        if name in ("sigmoid", "silu"):
            x = args[0]
            neg = self.emitter.regs.new("f")
            self.emitter.ins(f"neg.f32 {neg}, {x};")
            t = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {t}, {neg}, 0f3FB8AA3B;")
            ex = self.emitter.regs.new("f")
            self.emitter.ins(f"ex2.approx.f32 {ex}, {t};")
            den = self.emitter.regs.new("f")
            self.emitter.ins(f"add.f32 {den}, {ex}, 0f3F800000;")
            sig = self.emitter.regs.new("f")
            self.emitter.ins(f"rcp.rn.f32 {sig}, {den};")
            if name == "sigmoid":
                return sig, "f32"
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {out}, {x}, {sig};")
            return out, "f32"

        if name == "tanh":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"tanh.approx.f32 {out}, {args[0]};")
            return out, "f32"

        if name == "reciprocal":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"rcp.rn.f32 {out}, {args[0]};")
            return out, "f32"

        if name in ("absolute", "abs"):
            if is_float(dt):
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"abs.f32 {out}, {args[0]};")
                return out, "f32"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"abs.s32 {out}, {args[0]};")
            return out, "s32"

        if name == "neg":
            if is_float(dt):
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"neg.f32 {out}, {args[0]};")
                return out, "f32"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"neg.s32 {out}, {args[0]};")
            return out, "s32"

        if name == "copy":
            out = self.emitter.regs.new("f" if is_float(dt) else "r")
            self.emitter.ins(f"mov.{ptx_type(dt)} {out}, {args[0]};")
            return out, dt

        if name in ("sqrt", "rsqrt"):
            out = self.emitter.regs.new("f")
            ptx = "sqrt.rn.f32" if name == "sqrt" else "rsqrt.approx.f32"
            self.emitter.ins(f"{ptx} {out}, {args[0]};")
            return out, "f32"

        if name == "fma":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"fma.rn.f32 {out}, {args[0]}, {args[1]}, {args[2]};")
            return out, "f32"

        if name in ("from_f8e4m3", "from_f8e5m2"):
            raise SystemExit(f"unsupported fp8 decode op {name}")

        if is_float(dt):
            ptxop = {
                "add": "add.f32",
                "subtract": "sub.f32",
                "multiply": "mul.f32",
                "divide": "div.rn.f32",
                "maximum": "max.f32",
                "minimum": "min.f32",
                "max": "max.f32",
                "min": "min.f32",
            }.get(name)
            if ptxop is None:
                raise SystemExit(f"unsupported float op {name!r}")
            out = dest or self.emitter.regs.new("f")
            self.emitter.ins(f"{ptxop} {out}, {args[0]}, {args[1]};")
            return out, "f32"

        ptxop = {
            "add": "add.s32",
            "subtract": "sub.s32",
            "multiply": "mul.lo.s32",
            "divide": "div.s32",
            "maximum": "max.s32",
            "minimum": "min.s32",
            "max": "max.s32",
            "min": "min.s32",
            "and": "and.b32",
            "or": "or.b32",
            "xor": "xor.b32",
        }.get(name)
        if ptxop is None:
            raise SystemExit(f"unsupported integer op {name!r}")
        out = dest or self.emitter.regs.new("r")
        self.emitter.ins(f"{ptxop} {out}, {args[0]}, {args[1]};")
        return out, "s32"

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def stmt(self, s):
        if s is None:
            return
        if isinstance(s, (Body, list, tuple)):
            for child in s:
                self.stmt(child)
            return
        method = {
            "Tile": self.stmt_tile,
            "Load": self.stmt_load,
            "Assign": self.stmt_assign,
            "Let": self.stmt_let,
            "Accum": self.stmt_accum,
            "Write": self.stmt_write,
            "Init": self.stmt_init,
            "Cond": self.stmt_cond,
            "Loop": self.stmt_loop,
            "StridedLoop": self.stmt_strided_loop,
            "Smem": self.stmt_smem,
            "Sync": self.stmt_sync,
            "WarpShuffle": self.stmt_warp_shuffle,
            "TreeHalve": self.stmt_tree_halve,
            "CpAsyncCopy": self.stmt_cp_async,
            "CpAsyncCommit": self.stmt_cp_commit,
            "CpAsyncWait": self.stmt_cp_wait,
            "Select": self.stmt_select,
        }.get(type(s).__name__)
        if method is None:
            raise SystemExit(f"unsupported statement {type(s).__name__}: {s!r}")
        method(s)

    def stmt_smem(self, s):
        return

    def stmt_tile(self, s):
        extents = []
        for axis in s.axes:
            if not axis.extent.is_static:
                raise SystemExit(f"dynamic tile axis {axis.name!r} is unsupported")
            extents.append(int(axis.extent.as_static()))
        n = 1
        for e in extents:
            n *= e

        tid = self.special.get("threadIdx.x")
        if tid is None:
            tid = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {tid}, %tid.x;")
            self.special["threadIdx.x"] = tid

        cta = self.builtin(Builtin("blockIdx.x"))[0]
        gid = self.emitter.regs.new("r")
        if s.block_threads:
            width = int(s.block_threads)
            if s.aux_threads:
                t = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {t}, {tid}, {width};")
            else:
                t = tid
            self.emitter.ins(f"mad.lo.u32 {gid}, {cta}, {width}, {t};")
        else:
            ntid = self.builtin(Builtin("blockDim.x"))[0]
            self.emitter.ins(f"mad.lo.u32 {gid}, {cta}, {ntid}, {tid};")

        end = self.emitter.label("TILE_END")
        guard = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ge.u32 {guard}, {gid}, {n};")
        self.emitter.ins(f"@{guard} bra {end};")

        strides = [1] * len(extents)
        for i in range(len(extents) - 2, -1, -1):
            strides[i] = strides[i + 1] * extents[i + 1]
        for i, (axis, extent, stride) in enumerate(zip(s.axes, extents, strides)):
            if extent == 1:
                reg = self.int_reg(0)
            elif i == len(extents) - 1:
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.s32 {reg}, {gid}, {extent};")
            elif i == 0:
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"div.s32 {reg}, {gid}, {stride};")
            else:
                q = self.emitter.regs.new("r")
                self.emitter.ins(f"div.s32 {q}, {gid}, {stride};")
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.s32 {reg}, {q}, {extent};")
            self.values[axis.name] = reg
            self.value_dtypes[axis.name] = "s32"

        self.stmt(s.body)
        self.emitter.raw(end + ":")

    def stmt_load(self, s):
        names = s.names if getattr(s, "names", None) else (getattr(s, "name", None),)
        input_name = s.input or getattr(s, "name", None)
        if input_name is None:
            raise SystemExit(f"Load without input: {s!r}")
        dtype = dtype_name(s.dtype) if s.dtype is not None else dtype_name(self.dtype_of(input_name))
        index = tuple(s.index or ())
        for k, name in enumerate(names):
            if name is None:
                continue
            idx = list(index)
            if k and idx:
                idx[-1] = self.add_const_expr(idx[-1], k)
            flat = self.flatten(input_name, idx)
            if input_name in self.smem:
                addr = self.smem_elem_addr(input_name, flat)
                reg = self.emitter.regs.new("f" if is_float(dtype) else "r")
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {reg}, [{addr}];")
            else:
                addr = self.global_addr(input_name, flat)
                reg = self.emitter.regs.new("f" if is_float(dtype) else "r")
                self.emitter.ins(f"ld.global.{ptx_type(dtype)} {reg}, [{addr}];")
            self.values[name] = reg
            self.value_dtypes[name] = dtype

    def stmt_assign(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.args[0]]
        regs = []
        for name in s.args:
            if name not in self.values:
                raise SystemExit(f"Assign {s.name}: unknown value {name!r}")
            regs.append(self.values[name])
        reg, dt = self.op_on_regs(s.op, regs, dtype)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dt

    def stmt_let(self, s):
        reg, dt = self.expr(s.value)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dtype_name(s.dtype) if s.dtype is not None else dt

    def stmt_accum(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.name]
        if s.name not in self.values:
            raise SystemExit(f"Accum {s.name}: accumulator was not seeded")
        acc = self.values[s.name]
        a = self.values[s.base] if s.base else acc
        b = self.values[s.value]
        reg, dt = self.op_on_regs(s.op, (a, b), dtype, dest=acc)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dt

    def stmt_write(self, s):
        values = s.values if getattr(s, "values", None) else (getattr(s, "value", None),)
        output = s.output or getattr(s, "name", None)
        if output is None:
            raise SystemExit(f"Write without output: {s!r}")
        dtype = dtype_name(s.value_dtype) if s.value_dtype is not None else dtype_name(self.dtype_of(output))
        index = tuple(s.index or ())
        for k, name in enumerate(values):
            if name is None:
                continue
            if name not in self.values:
                raise SystemExit(f"Write {output}: unknown value {name!r}")
            idx = list(index)
            if k and idx:
                idx[-1] = self.add_const_expr(idx[-1], k)
            flat = self.flatten(output, idx)
            reg = self.values[name]
            if output in self.smem:
                addr = self.smem_elem_addr(output, flat)
                self.emitter.ins(f"st.shared.{ptx_type(dtype)} [{addr}], {reg};")
            else:
                addr = self.global_addr(output, flat)
                if s.atomic:
                    op = "add" if s.atomic is True else str(s.atomic)
                    self.emitter.ins(f"red.global.{op}.{ptx_type(dtype)} [{addr}], {reg};")
                else:
                    self.emitter.ins(f"st.global.{ptx_type(dtype)} [{addr}], {reg};")

    def stmt_init(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else "f32"
        if is_float(dtype):
            reg = self.fconst(float(s.identity))
        else:
            reg = self.int_reg(int(s.identity))
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dtype

    def stmt_cond(self, s):
        p = self.to_pred(s.cond)
        els = self.emitter.label("ELSE")
        end = self.emitter.label("COND_END")
        self.emitter.ins(f"@!{p} bra {els};")
        self.stmt(s.body)
        self.emitter.ins(f"bra {end};")
        self.emitter.raw(els + ":")
        self.stmt(s.else_body)
        self.emitter.raw(end + ":")

    def seed_accums(self, body):
        seen = set()
        for st in body:
            if isinstance(st, Accum) and st.name not in seen:
                seen.add(st.name)
                dtype = dtype_name(st.dtype) if st.dtype is not None else "f32"
                identity = st.op.identity
                if identity is None:
                    raise SystemExit(f"no identity for accumulator op {st.op.name}")
                if is_float(dtype):
                    reg = self.fconst(float(identity))
                else:
                    reg = self.int_reg(int(identity))
                self.values[st.name] = reg
                self.value_dtypes[st.name] = dtype

    def stmt_loop(self, s):
        if not s.axis.extent.is_static:
            raise SystemExit(f"dynamic loop axis {s.axis.name!r} is unsupported")
        extent = int(s.axis.extent.as_static())
        if s.seed:
            self.seed_accums(s.body)
        counter = self.int_reg(0)
        head = self.emitter.label("LOOP")
        tail = self.emitter.label("LOOP_END")
        self.emitter.raw(head + ":")
        p = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ge.s32 {p}, {counter}, {extent};")
        self.emitter.ins(f"@{p} bra {tail};")
        old = self.values.get(s.axis.name)
        self.values[s.axis.name] = counter
        self.value_dtypes[s.axis.name] = "s32"
        self.stmt(s.body)
        self.emitter.ins(f"add.s32 {counter}, {counter}, 1;")
        self.emitter.ins(f"bra {head};")
        self.emitter.raw(tail + ":")
        if old is not None:
            self.values[s.axis.name] = old
        else:
            self.values.pop(s.axis.name, None)
            self.value_dtypes.pop(s.axis.name, None)

    def stmt_strided_loop(self, s):
        if s.seed:
            self.seed_accums(s.body)
        start, _ = self.expr(s.start)
        counter = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.s32 {counter}, {start};")
        step, _ = self.expr(s.step)
        if s.end is None:
            if not s.axis.extent.is_static:
                raise SystemExit(f"dynamic strided loop axis {s.axis.name!r} is unsupported")
            end = self.int_reg(int(s.axis.extent.as_static()))
        else:
            end, _ = self.expr(s.end)
        head = self.emitter.label("SL")
        tail = self.emitter.label("SL_END")
        self.emitter.raw(head + ":")
        p = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ge.s32 {p}, {counter}, {end};")
        self.emitter.ins(f"@{p} bra {tail};")
        old = self.values.get(s.axis.name)
        self.values[s.axis.name] = counter
        self.value_dtypes[s.axis.name] = "s32"
        self.stmt(s.body)
        self.emitter.ins(f"add.s32 {counter}, {counter}, {step};")
        self.emitter.ins(f"bra {head};")
        self.emitter.raw(tail + ":")
        if old is not None:
            self.values[s.axis.name] = old
        else:
            self.values.pop(s.axis.name, None)
            self.value_dtypes.pop(s.axis.name, None)

    def stmt_sync(self, s):
        if s.warp:
            self.emitter.ins("bar.warp.sync 0xffffffff;")
        elif s.count is not None:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)}, {int(s.count)};")
        else:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)};")

    def stmt_warp_shuffle(self, s):
        dtype = dtype_name(s.dtype)
        mask = self.emitter.regs.new("r")
        self.emitter.ins(f"activemask.b32 {mask};")
        step = int(s.length) // 2
        while step > 0:
            for state, state_b in zip(s.state, s.state_b):
                a = self.values[state]
                bits = self.emitter.regs.new("r")
                self.emitter.ins(f"mov.b32 {bits}, {a};")
                other = self.emitter.regs.new("r")
                self.emitter.ins(
                    f"shfl.sync.bfly.b32 {other}, {bits}, {step}, 0x1f, {mask};"
                )
                if is_float(dtype):
                    reg = self.emitter.regs.new("f")
                    self.emitter.ins(f"mov.b32 {reg}, {other};")
                else:
                    reg = other
                self.values[state_b] = reg
                self.value_dtypes[state_b] = dtype
            self.stmt(s.combine_states)
            step //= 2

    def _smem_index(self, buf: str, index_expr: int) -> str:
        return self.int_reg(index_expr)

    def stmt_tree_halve(self, s):
        dtype = dtype_name(s.dtype)
        tid = self.values[s.tid_var]
        inner = s.inner
        bufs = list(s.bufs)
        states = list(s.state)
        state_bs = list(s.state_b)
        if len(bufs) == 1 and len(states) > 1:
            bufs = bufs * len(states)

        def slot_expr(component: int, plus: int):
            if inner is None:
                return self.add_int(tid, plus) if plus else tid
            _, scale = inner
            base = self.mul_int(tid, int(scale))
            if plus:
                base = self.add_int(base, int(plus) * int(scale))
            if component:
                base = self.add_int(base, component)
            return base

        step = int(s.length) // 2
        while step > 0:
            p = self.emitter.regs.new("p")
            skip = self.emitter.label("TREE_SKIP")
            self.emitter.ins(f"setp.lt.s32 {p}, {tid}, {step};")
            self.emitter.ins(f"@!{p} bra {skip};")
            for comp, (buf, state, state_b) in enumerate(zip(bufs, states, state_bs)):
                a_idx = slot_expr(comp, 0)
                b_idx = slot_expr(comp, step)
                a_addr = self.smem_elem_addr(buf, a_idx)
                b_addr = self.smem_elem_addr(buf, b_idx)
                a = self.emitter.regs.new("f" if is_float(dtype) else "r")
                b = self.emitter.regs.new("f" if is_float(dtype) else "r")
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {a}, [{a_addr}];")
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {b}, [{b_addr}];")
                self.values[state] = a
                self.value_dtypes[state] = dtype
                self.values[state_b] = b
                self.value_dtypes[state_b] = dtype
            self.stmt(s.combine_states)
            for comp, (buf, state) in enumerate(zip(bufs, states)):
                a_idx = slot_expr(comp, 0)
                a_addr = self.smem_elem_addr(buf, a_idx)
                self.emitter.ins(
                    f"st.shared.{ptx_type(dtype)} [{a_addr}], {self.values[state]};"
                )
            self.emitter.raw(skip + ":")
            self._tree_barrier(s)
            step //= 2

        for comp, (buf, state) in enumerate(zip(bufs, states)):
            root = self.int_reg(comp) if inner is not None else self.int_reg(0)
            addr = self.smem_elem_addr(buf, root)
            reg = self.emitter.regs.new("f" if is_float(dtype) else "r")
            self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {reg}, [{addr}];")
            self.values[state] = reg
            self.value_dtypes[state] = dtype

    def _tree_barrier(self, s):
        if s.barrier_count is not None:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)}, {int(s.barrier_count)};")
        else:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)};")

    def stmt_cp_async(self, s):
        if s.swizzle not in (None, "NONE"):
            raise SystemExit(f"unsupported cp.async swizzle {s.swizzle!r}")
        if s.smem_index:
            flat = self.flatten(s.smem, s.smem_index)
        else:
            flat = self.int_reg(0)
        if s.lane_index is not None:
            cols = self.shape_of(s.smem)[-1]
            row, _ = self.expr(s.lane_index[0])
            col, _ = self.expr(s.lane_index[1])
            lane = self.add_int(self.mul_int(row, cols), col)
            flat = self.add_int(flat, lane)
        dst = self.smem_elem_addr(s.smem, flat)
        src_flat = self.flatten(s.src, s.src_index)
        src = self.global_addr(s.src, src_flat)
        nbytes = int(s.nbytes)
        op = "cg" if nbytes == 16 else "ca"
        self.emitter.ins(
            f"cp.async.{op}.shared.global [{dst}], [{src}], {nbytes};"
        )

    def stmt_cp_commit(self, s):
        self.emitter.ins("cp.async.commit_group;")

    def stmt_cp_wait(self, s):
        self.emitter.ins(f"cp.async.wait_group {int(s.group)};")

    def stmt_select(self, s):
        branches = list(s.branches)
        if not branches:
            raise SystemExit(f"Select {s.name} has no branches")
        last_val, dt = self.expr(branches[-1].value)
        dtype = dtype_name(dt)
        self.values[s.name] = last_val
        self.value_dtypes[s.name] = dtype
        for branch in reversed(branches[:-1]):
            p = self.to_pred(branch.select)
            v, vdt = self.expr(branch.value)
            v = self.coerce(v, vdt, dtype)
            self.emitter.ins(f"@{p} mov.{ptx_type(dtype)} {self.values[s.name]}, {v};")

    # ------------------------------------------------------------------
    # module
    # ------------------------------------------------------------------
    def _param_decl(self, name: str, arg: dict) -> str:
        dtype = dtype_name(arg.get("dtype", "f32"))
        scalar = bool(arg.get("scalar")) or len(arg.get("shape", [])) == 0
        if scalar:
            return f"        .param .{ptx_type(dtype)} p_{sanitize(name)}"
        return f"        .param .u64 p_{sanitize(name)}"

    def emit_prologue(self):
        for name in self.arg_order:
            arg = self.args.get(name, {"name": name, "dtype": self.dtype_of(name), "shape": self.shape_of(name)})
            dtype = dtype_name(arg.get("dtype", "f32"))
            scalar = bool(arg.get("scalar")) or len(arg.get("shape", [])) == 0
            pname = f"p_{sanitize(name)}"
            if scalar:
                if is_float(dtype):
                    reg = self.emitter.regs.new("f")
                    self.emitter.ins(f"ld.param.f32 {reg}, [{pname}];")
                else:
                    reg = self.emitter.regs.new("r")
                    self.emitter.ins(f"ld.param.s32 {reg}, [{pname}];")
                self.values[name] = reg
                self.value_dtypes[name] = dtype
            else:
                raw = self.emitter.regs.new("rd")
                base = self.emitter.regs.new("rd")
                self.emitter.ins(f"ld.param.u64 {raw}, [{pname}];")
                self.emitter.ins(f"cvta.to.global.u64 {base}, {raw};")
                self.base[name] = base

        tid = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {tid}, %tid.x;")
        self.special["threadIdx.x"] = tid
        lane = self.emitter.regs.new("r")
        self.emitter.ins(f"and.b32 {lane}, {tid}, 31;")
        warp = self.emitter.regs.new("r")
        self.emitter.ins(f"shr.u32 {warp}, {tid}, 5;")
        self.values["lane"] = lane
        self.value_dtypes["lane"] = "s32"
        self.values["warp"] = warp
        self.value_dtypes["warp"] = "s32"

        for name in self.smem:
            self._materialize_smem_base(name)

    def compile(self) -> str:
        self.plan_smem()
        self.emit_prologue()
        self.stmt(self.kernel.body)
        self.emitter.ins("ret;")

        params = ",\n".join(
            self._param_decl(name, self.args.get(name, {"dtype": self.dtype_of(name), "shape": self.shape_of(name)}))
            for name in self.arg_order
        )
        out = [PTX_HEADER]
        if self.shared_decls:
            out.append("\n".join(self.shared_decls) + "\n")
        out.append(f".visible .entry {self.entry}(\n{params}\n)\n{{")
        out.extend(self.emitter.regs.declarations())
        out.append("")
        out.extend(self.emitter.lines)
        out.append("}\n")
        return "\n".join(out)


def compile_file(path) -> str:
    return Compiler(path).compile()


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        sys.exit(2)
    try:
        ptx = compile_file(sys.argv[1])
    except Exception as exc:  # noqa: BLE001 - report the real failure on stderr
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(ptx)


if __name__ == "__main__":
    main()
