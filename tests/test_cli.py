import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torchpeak.cli import main


class CliTests(unittest.TestCase):
    def run_with_error(self, kind, extra_args=()):
        def fake_gpt(results, dev, args):
            results.append({"test": "gpt", "config": "synthetic", "value": None, "unit": "tok/s",
                            "status": "error", "error": "synthetic error", "error_kind": kind})

        with tempfile.TemporaryDirectory(dir=".") as outdir:
            argv = ["torchpeak", "--device", "cpu", "--tests", "gpt", "--no-report",
                    "--no-compare-compile", "--label", "test", "--outdir", outdir, *extra_args]
            with patch.object(sys, "argv", argv), patch("torchpeak.cli.test_gpt", fake_gpt):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = main()
            result = json.loads((Path(outdir) / "bench_test.json").read_text())
        return code, result

    def test_internal_error_returns_two(self):
        code, result = self.run_with_error("internal")
        self.assertEqual(code, 2)
        self.assertEqual(result["meta"]["format_version"], 2)
        self.assertEqual(result["meta"]["errors"]["internal"], 1)

    def test_unsupported_is_allowed_by_default(self):
        code, result = self.run_with_error("unsupported")
        self.assertEqual(code, 0)
        self.assertEqual(result["meta"]["errors"]["by_kind"], {"unsupported": 1})

    def test_strict_errors_returns_one(self):
        code, result = self.run_with_error("unsupported", ("--strict-errors",))
        self.assertEqual(code, 1)
        self.assertTrue(result["meta"]["args"]["strict_errors"])


if __name__ == "__main__":
    unittest.main()
