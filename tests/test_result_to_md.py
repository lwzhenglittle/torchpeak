import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("result_to_md", ROOT / "scripts" / "result_to_md.py")
result_to_md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(result_to_md)


def doc(rows):
    return {
        "meta": {
            "format_version": 2,
            "label": "dev-x",
            "name": "DEV X",
            "backend": "cuda 12.4",
            "torch": "2.5.0",
            "platform": "Linux-x86_64",
            "compute_capability": [8, 6],
            "compute_units": 40,
            "total_mem_gb": 16.0,
            "compare_compile": True,
            "quick": False,
            "runtime": {"triton": "triton==3.1.0"},
            "args": {"tests": "probe,gemm", "amp": "fp16"},
            "errors": {"total": 0},
            "peak_measured": {"tflops:fp16": 30.0, "device_bw:all": 400.0},
        },
        "results": rows,
    }


class ResultToMdTests(unittest.TestCase):
    def test_render_groups_in_file_order_with_peak_units(self):
        rows = [
            {"test": "gemm", "config": "4096x4096x4096 fp16", "value": 30.0, "unit": "TFLOPS", "cv_pct": 1.5},
            {"test": "probe", "config": "sdpa_flash", "value": 0.0, "unit": "probe", "status": "fail",
             "error": "RuntimeError: no kernel", "error_kind": "runtime"},
            {"test": "gemm", "config": "512x512x512 fp16", "value": 12.5, "unit": "TFLOPS", "cv_pct": 2.0},
        ]
        md = result_to_md.render(doc(rows))
        self.assertIn("# dev-x", md)
        self.assertIn("| PyTorch | 2.5.0 |", md)
        self.assertIn("| matmul fp16 | 30 | TFLOPS |", md)
        self.assertIn("| device stream | 400 | GB/s |", md)
        # section order follows first appearance, rows keep their original order
        self.assertLess(md.index("## gemm"), md.index("## probe"))
        self.assertLess(md.index("4096x4096x4096 fp16"), md.index("512x512x512 fp16"))
        self.assertIn("| probe | sdpa_flash | runtime | RuntimeError: no kernel |", md)

    def test_memory_wall_renders_matrix(self):
        row = {"test": "memory_wall", "config": "read mountain 2x2", "value": 900.0, "unit": "GB/s",
               "value_basis": "largest working set, stride 1",
               "mountain": {"op": "read", "dtype": "fp32 random", "sizes": ["4 KiB", "16 KiB"],
                            "strides": [1, 2], "gbs": [[900.0, 850.5], [800.0, 700.25]]}}
        md = result_to_md.render(doc([row]))
        self.assertIn("| Stride | 4 KiB | 16 KiB |", md)
        self.assertIn("| 1 | 900 | 850.5 |", md)
        self.assertIn("| 2 | 800 | 700.2 |", md)

    def test_check_detects_missing_and_stale_files(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "dev-x.json"
            src.write_text(json.dumps(doc([{"test": "probe", "config": "sdpa_math", "value": 1.0,
                                           "unit": "probe", "status": "pass"}])))
            self.assertEqual(result_to_md.main(["--check", str(src)]), 1)  # missing
            self.assertEqual(result_to_md.main([str(src)]), 0)
            self.assertEqual(result_to_md.main(["--check", str(src)]), 0)
            src.write_text(src.read_text().replace("DEV X", "DEV Y"))  # now stale
            self.assertEqual(result_to_md.main(["--check", str(src)]), 1)


if __name__ == "__main__":
    unittest.main()
