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


if __name__ == "__main__":
    unittest.main()
