import importlib.util
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
        self.assertIn("st.global.f32", ptx)
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


if __name__ == "__main__":
    unittest.main()
