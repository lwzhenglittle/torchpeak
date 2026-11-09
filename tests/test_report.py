import tempfile
import unittest
from pathlib import Path

from torchpeak.report import (annotate_compile_speedup, annotate_efficiency, row_metric,
                              write_html_report)


class ReportTests(unittest.TestCase):
    def test_bandwidth_dimensions_and_spec_denominators(self):
        rows = [
            {"test": "bandwidth", "config": "read 1024MiB", "value": 90.0, "unit": "GB/s"},
            {"test": "bandwidth", "config": "read 16MiB", "value": 100.0, "unit": "GB/s"},
            {"test": "bandwidth", "config": "h2d 16MiB", "value": 10.0, "unit": "GB/s"},
            {"test": "memory_wall", "config": "read mountain", "value": 50.0, "unit": "GB/s"},
            {"test": "gemv", "config": "fp32 m1", "value": 80.0, "unit": "GB/s"},
            {"test": "gpt_decode", "config": "fp32", "value": 2.0,
             "approx_weight_bw_gbs": 70.0},
        ]
        measured = annotate_efficiency(rows, {"bw_gbs": 200.0, "host_bw_gbs": 20.0})
        self.assertEqual(row_metric(rows[0]), ("device_bw", None, 90.0))
        self.assertEqual(row_metric(rows[2]), ("host_bw", None, 10.0))
        self.assertEqual(rows[0]["eff_spec_pct"], 45.0)
        self.assertEqual(rows[2]["eff_spec_pct"], 50.0)
        # envelope: only the DRAM-size STREAM row; the L2-resident 16MiB read must not raise it
        self.assertEqual(measured[("device_bw", None)], 90.0)
        self.assertEqual(measured[("host_bw", None)], 10.0)
        # non-envelope rows keep per-row spec efficiency but use the DRAM ceiling as denominator
        self.assertEqual(rows[3]["eff_spec_pct"], 25.0)
        self.assertEqual(rows[4]["eff_spec_pct"], 40.0)
        self.assertEqual(rows[4]["peak_measured"], 90.0)
        self.assertEqual(rows[5]["eff_spec_pct"], 35.0)

    def test_device_bw_envelope_excludes_l2_gemv_and_models(self):
        rows = [
            {"test": "bandwidth", "config": "read 16MiB", "value": 100.0, "unit": "GB/s"},
            {"test": "bandwidth", "config": "copy 1024MiB", "value": 60.0, "unit": "GB/s"},
            {"test": "gemv", "config": "fp32 m1", "value": 80.0, "unit": "GB/s"},
            {"test": "gpt_decode", "config": "fp32", "value": 2.0,
             "approx_weight_bw_gbs": 70.0},
        ]
        measured = annotate_efficiency(rows, {"bw_gbs": 200.0})
        self.assertEqual(measured, {("device_bw", None): 60.0})  # copy@DRAM wins; 16MiB/gemv/decode out
        self.assertEqual(rows[0]["peak_measured"], 60.0)  # L2 read exceeds the DRAM ceiling visibly

    def test_gemv_only_run_has_no_device_bw_ceiling(self):
        rows = [{"test": "gemv", "config": "fp32 m1 k4096 n11008", "value": 80.0, "unit": "GB/s"}]
        measured = annotate_efficiency(rows, {"bw_gbs": 200.0})
        self.assertEqual(measured, {})
        self.assertEqual(rows[0]["eff_spec_pct"], 40.0)  # per-row spec ratio still annotated
        self.assertNotIn("peak_measured", rows[0])

    def test_compile_speedup_and_regression(self):
        rows = [
            {"test": "train", "config": "bert fp32 eager", "value": 100.0, "unit": "tok/s"},
            {"test": "train", "config": "bert fp32 compile", "value": 80.0, "unit": "tok/s"},
        ]
        annotate_compile_speedup(rows)
        self.assertEqual(rows[1]["speedup_vs_eager"], 0.8)
        self.assertTrue(rows[1]["compile_regress"])

        rows[1].update(config="bert fp32 compile-max-autotune", value=120.0)
        annotate_compile_speedup(rows)
        self.assertEqual(rows[1]["speedup_vs_eager"], 1.2)
        self.assertFalse(rows[1]["compile_regress"])

    def test_html_report_is_self_contained_and_escaped(self):
        rows = [{"test": "probe", "config": "<probe>", "value": 0.0, "unit": "probe",
                 "status": "fail", "error": "<bad>", "error_kind": "unsupported"}]
        meta = {"label": "<label>", "runtime": {}, "tuned": {"enabled": False}}
        with tempfile.TemporaryDirectory(dir=".") as outdir:
            path = Path(outdir) / "report.html"
            write_html_report(meta, rows, path)
            html = path.read_text()
        self.assertIn("&lt;label&gt;", html)
        self.assertIn("&lt;bad&gt;", html)
        self.assertNotIn("<script", html)
        self.assertNotIn("<probe>", html)


if __name__ == "__main__":
    unittest.main()
