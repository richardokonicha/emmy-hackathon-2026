import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_compiler():
    spec = importlib.util.spec_from_file_location("emmy_compile", ROOT / "compile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ir_path(problem: str) -> Path:
    return next((ROOT / "problems" / problem).glob("*.ir.json"))


def dev_ir_path(problem: str) -> Path:
    for directory in (ROOT / "dev" / "cases", ROOT / "dev" / "cases_oops"):
        found = list(directory.glob(problem + "/*.ir.json"))
        if found:
            return found[0]
    raise AssertionError(f"no dev IR for {problem}")


def rewrite_write_index(src: Path, dst: Path, old: str, new: str) -> Path:
    """Copy a problem package, rewriting only the Write statement's index text.

    The IR ``body`` field is a list of statement strings; locate the statement
    containing ``Write(`` and patch its index arguments textually.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        statements = body if isinstance(body, list) else [body]
        for i, stmt in enumerate(statements):
            if "Write(" not in stmt:
                continue
            head, sep, tail = stmt.rpartition("Write(")
            assert sep, "no Write statement in the IR"
            statements[i] = head + "Write(" + tail.replace(old, new, 1)
        node["op_fields"]["body"] = statements if isinstance(body, list) else statements[0]
    ir.write_text(json.dumps(doc, indent=1))
    return ir


class CompilerTests(unittest.TestCase):
    def test_float_bits(self):
        mod = load_compiler()
        self.assertEqual(mod.float_bits(1.0), 0x3F800000)
        self.assertEqual(mod.float_bits(float("-inf")), 0xFF800000)
        self.assertEqual(mod.float_bits(0.0), 0)

    def test_p00_ptx_shape(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p00_scalar_add"))
        self.assertIn(".version 8.0", ptx)
        self.assertIn(".entry k_add_pointwise", ptx)
        self.assertIn(".param .u64 p_x0", ptx)
        self.assertIn("ld.global.f32", ptx)
        self.assertIn("add.f32", ptx)
        self.assertIn("st.global.v2.f32", ptx)
        self.assertTrue(ptx.endswith("}\n"))

    def test_r01_has_shared_and_shuffle(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("r01_array_sum"))
        self.assertIn(".shared .align 4 .b8 acc0_smem[16];", ptx)
        self.assertIn("shfl.sync.bfly.b32", ptx)
        self.assertIn("bar.sync 0;", ptx)

    def test_m01_uses_dynamic_shared_and_cp_async(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m01_matmul_512"))
        self.assertIn(".extern .shared", ptx)
        self.assertIn("cp.async.cg.shared.global", ptx)
        self.assertIn("cp.async.wait_group 2;", ptx)

    def test_all_open_problems_compile(self):
        mod = load_compiler()
        for problem in sorted((ROOT / "problems").iterdir()):
            if not (problem / "meta.json").exists():
                continue
            ptx = mod.compile_file(ir_path(problem.name))
            self.assertIn(".visible .entry", ptx, problem.name)
            self.assertIn("ret;", ptx, problem.name)


class VectorMemoryTests(unittest.TestCase):
    """TASK.md: a grouped write ``buf[i] = (v0, v1, v2, v3)`` is a vector store."""

    def test_grouped_write_becomes_one_vector_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p01_vector_add"))
        self.assertEqual(ptx.count("st.global"), 1)
        self.assertIn("st.global.v2.f32", ptx)

    def test_vector_store_carries_both_values(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p01_vector_add"))
        self.assertRegex(ptx, r"st\.global\.v2\.f32 \[%rd\d+\], \{%f\d+, %f\d+\};")

    def test_two_dimensional_grouped_write_becomes_vector_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m01_matmul_512"))
        self.assertIn("st.global.v2.f32", ptx)
        self.assertNotIn("st.global.f32", ptx)

    def test_grouped_load_becomes_vector_load(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("a01_attention"))
        self.assertIn("ld.global.v2.f32", ptx)
        self.assertRegex(ptx, r"ld\.global\.v2\.f32 \{%f\d+, %f\d+\}, \[%rd\d+\];")

    def test_single_value_write_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("a01_attention"))
        self.assertIn("st.global.f32", ptx)
        self.assertNotIn("st.global.v1.f32", ptx)

    def test_atomic_grouped_write_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m02_matmul_rect"))
        self.assertIn("red.global.add.f32", ptx)
        self.assertNotIn("red.global.add.v2.f32", ptx)

    def test_unaligned_grouped_write_stays_scalar(self):
        mod = load_compiler()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "case"
            ir = rewrite_write_index(
                ROOT / "problems" / "p01_vector_add",
                dst,
                "right=Literal(value=0, dtype='int')",
                "right=Literal(value=1, dtype='int')",
            )
            ptx = mod.compile_file(ir)
        self.assertNotIn("st.global.v2.f32", ptx)
        self.assertIn("st.global.f32", ptx)

    def test_odd_row_extent_keeps_grouped_write_scalar(self):
        mod = load_compiler()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "case"
            ir = rewrite_write_index(
                ROOT / "problems" / "p01_vector_add",
                dst,
                "right=Literal(value=0, dtype='int')",
                "right=Literal(value=1, dtype='int')",
            )
            meta = json.loads((dst / "meta.json").read_text())
            for arg in meta["kernels"][0]["args"]:
                if arg["name"] == "add":
                    arg["shape"] = [1023]
            (dst / "meta.json").write_text(json.dumps(meta, indent=1))
            ptx = mod.compile_file(ir)
        self.assertNotIn("st.global.v2.f32", ptx)


class WideVectorTests(unittest.TestCase):
    """TASK.md: a grouped write of 4 values is a v4 vector store."""

    ALIGNED4 = "BinaryExpr(op='*', left=Var(name='a0'), right=Literal(value=4, dtype='int'))"
    UNALIGNED4 = (
        "BinaryExpr(op='+', left=BinaryExpr(op='*', left=Var(name='a0'), "
        "right=Literal(value=4, dtype='int')), right=Literal(value=1, dtype='int'))"
    )

    def test_four_wide_group_is_one_v4_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 256))
        self.assertIn("st.global.v4.f32", ptx)
        self.assertEqual(ptx.count("st.global"), 1)
        self.assertNotIn("st.global.f32", ptx)

    def test_four_wide_group_is_one_v4_load(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 256))
        self.assertIn("ld.global.v4.f32", ptx)
        self.assertEqual(ptx.count("ld.global"), 2)

    def test_unaligned_four_wide_group_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.UNALIGNED4, 256))
        self.assertNotIn("st.global.v4.f32", ptx)
        self.assertIn("st.global.f32", ptx)

    def test_size_not_divisible_by_width_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 254))
        self.assertNotIn("st.global.v4.f32", ptx)
        self.assertIn("st.global.f32", ptx)


class ContractTests(unittest.TestCase):
    def test_entry_name_comes_from_the_ir(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("r01_array_sum"))
        self.assertIn(".visible .entry k_sum_1_reduce(", ptx)

    def test_parameter_order_comes_from_meta_json(self):
        mod = load_compiler()
        meta = json.loads((ROOT / "problems" / "p01_vector_add" / "meta.json").read_text())
        order = meta["kernels"][0]["arg_order"]
        mod_ = load_compiler()
        ptx = mod_.compile_file(ir_path("p01_vector_add"))
        params = re.findall(r"\.param \.\w+ (p_\w+)", ptx)
        self.assertEqual(params, [f"p_{name}" for name in order])

    def test_output_is_pure_ascii(self):
        mod = load_compiler()
        for problem in sorted((ROOT / "problems").iterdir()):
            if not (problem / "meta.json").exists():
                continue
            ptx = mod.compile_file(ir_path(problem.name))
            ptx.encode("ascii")

    def test_unsupported_statement_fails_loudly(self):
        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))

        class _CarryStatement:
            pass

        with self.assertRaises(SystemExit):
            comp.stmt(_CarryStatement())


F32_DTYPE = "DataType(name='f32', np=dtype('float32'), nbytes=4)"

BODY_TMPL = (
    "Load(names=('in0__u0',), input='x0', index=({idx},), dtype=" + F32_DTYPE + "), "
    "Assign(name='v0__u0', op=ElementwiseImpl('add'), args=('in0__u0', 'in0__u0'), "
    "dtype=" + F32_DTYPE + "), "
    "Write(output='add', index=({idx},), values=('v0__u0',), value_dtype=" + F32_DTYPE
    + ", atomic=False, swizzle='NONE')"
)


def _body(idx):
    return ", ".join(
        [
            BODY_TMPL.format(idx=idx),
        ]
    )


def _tile(inner, axis_extent=4):
    return (
        "Tile(axes=(Axis(name='a0', extent=Dim(%d), window=None),), body=(%s,))"
        % (axis_extent, inner)
    )


def _wide_body(width, idx, out_len=256):
    """A width-N grouped load/add/write over one buffer."""
    parts = []
    for buf in ("x0", "x1"):
        names = ", ".join("'in%s__u%d'" % (buf, k) for k in range(width)) + ","
        parts.append(
            "Load(names=(%s), input='%s', index=(%s,), dtype=%s)"
            % (names, buf, idx, F32_DTYPE)
        )
    for k in range(width):
        parts.append(
            "Assign(name='v0__u%d', op=ElementwiseImpl('add'), "
            "args=('inx0__u%d', 'inx1__u%d'), dtype=%s)" % (k, k, k, F32_DTYPE)
        )
    values = ", ".join("'v0__u%d'" % k for k in range(width)) + ","
    parts.append(
        "Write(output='add', index=(%s,), values=(%s), value_dtype=%s, atomic=False, swizzle='NONE')"
        % (idx, values, F32_DTYPE)
    )
    return ", ".join(parts)


def _wide_ir(width, index_expr, out_len=256):
    """Build a synthetic package whose 'add' output has `out_len` elements."""
    td = Path(tempfile.mkdtemp())
    dst = td / "case"
    shutil.copytree(ROOT / "problems" / "p01_vector_add", dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] == "KernelOp":
            node["op_fields"]["body"] = [
                _tile(_wide_body(width, index_expr, out_len), axis_extent=out_len // width)
            ]
        for o in node.get("outputs", []):
            if o["name"] == "add":
                o["shape"] = [out_len]
    ir.write_text(json.dumps(doc, indent=1))
    meta_path = dst / "meta.json"
    meta = json.loads(meta_path.read_text())
    for kern in meta["kernels"]:
        for arg in kern["args"]:
            arg["shape"] = [out_len]
    meta_path.write_text(json.dumps(meta, indent=1))
    return ir


def synthetic_ir(body_str: str) -> Path:
    """Copy p01 into a temp dir with the KernelOp body replaced."""
    td = Path(tempfile.mkdtemp())
    dst = td / "case"
    shutil.copytree(ROOT / "problems" / "p01_vector_add", dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] == "KernelOp":
            node["op_fields"]["body"] = [body_str]
    ir.write_text(json.dumps(doc, indent=1))
    return ir


class UnrollTests(unittest.TestCase):
    """Unrolling must be honoured AND bounded (a wide loop must stay rolled)."""

    def _loop_body(self, extent):
        inner = (
            "Loop(axis=Axis(name='i', extent=Dim(%d), window=None), body=(%s,), "
            "unroll=True, seed=False)" % (extent, _body("Var(name='i')"))
        )
        return _tile(inner)

    def test_small_loop_is_unrolled(self):
        mod = load_compiler()
        ptx = mod.compile_file(synthetic_ir(self._loop_body(8)))
        self.assertEqual(ptx.count("ld.global"), 8)
        self.assertNotIn("LOOP_", ptx)

    def test_wide_loop_stays_rolled(self):
        mod = load_compiler()
        ptx = mod.compile_file(synthetic_ir(self._loop_body(4096)))
        self.assertEqual(ptx.count("ld.global"), 1)
        self.assertIn("LOOP_", ptx)

    def test_strided_loop_honours_unroll(self):
        mod = load_compiler()
        inner = (
            "StridedLoop(axis=Axis(name='i', extent=Dim(8), window=None), "
            "start=Literal(value=0, dtype='int'), step=Literal(value=1, dtype='int'), "
            "body=(%s,), unroll=True, end=None, seed=False)" % _body("Var(name='i')")
        )
        ptx = mod.compile_file(synthetic_ir(_tile(inner)))
        self.assertEqual(ptx.count("ld.global"), 8)
        self.assertNotIn("SL_", ptx)


class DtypeTests(unittest.TestCase):
    def test_float16_bits_are_half_precision(self):
        mod = load_compiler()
        self.assertEqual(mod.float16_bits(1.0), 0x3C00)
        self.assertEqual(mod.float16_bits(0.5), 0x3800)
        self.assertEqual(mod.float16_bits(-2.0), 0xC000)

    def test_float64_bits(self):
        mod = load_compiler()
        self.assertEqual(mod.float64_bits(1.0), 0x3FF0000000000000)

    def test_f64_setp_uses_a_register_immediate(self):
        """ptxas rejects a 64-bit immediate in setp.ne.f64."""
        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))
        comp.values["probe"] = comp.emitter.regs.new("d")
        comp.value_dtypes["probe"] = "f64"
        comp.to_pred(mod.Var("probe"))
        emitted = [l.strip() for l in comp.emitter.lines if "setp" in l]
        self.assertEqual(len(emitted), 1, emitted)
        self.assertIn("setp.ne.f64", emitted[0])
        self.assertNotIn("0x0000000000000000", emitted[0])
        self.assertTrue(any("mov.f64" in l for l in comp.emitter.lines), comp.emitter.lines)

    def test_zero_prologue_uses_full_address_register(self):
        """global_addr returns a register name, not a sequence: [0] took one char."""
        from emmy.compiler.ir.stmt.leaves import ZeroPrologue

        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))
        comp.emit_prologue()
        comp.stmt_zero_prologue(ZeroPrologue(dst="add", words=16))
        stores = [l.strip() for l in comp.emitter.lines if "st.global.u32" in l]
        self.assertTrue(stores, comp.emitter.lines)
        # the address operand must be a whole register, e.g. "[%rd9 + %r3]"
        self.assertRegex(stores[0], r"st\.global\.u32 \[%rd\d+( \+ %r\d+)?\], 0;")


if __name__ == "__main__":
    unittest.main()
