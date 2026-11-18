import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torchpeak.cli import KNOWN_TESTS, TEST_ALIASES, main


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

    def run_argv(self, argv):
        """Drive main() with fixed argv; returns the SystemExit code (usage errors exit 2)."""
        with patch.object(sys, "argv", ["torchpeak", *argv]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    main()
        return cm.exception.code

    def test_unknown_test_id_exits_two(self):
        self.assertEqual(self.run_argv(["--tests", "bogus", "--device", "cpu"]), 2)

    def test_hf_without_model_exits_two(self):
        self.assertEqual(self.run_argv(["--tests", "hf", "--device", "cpu"]), 2)

    def test_hf_decode_is_an_alias(self):
        self.assertEqual(TEST_ALIASES.get("hf_decode"), "hf")

    def test_known_tests_match_the_documented_list(self):
        """--tests ids drift silently: an id in the parser help but missing from KNOWN_TESTS (or the
        reverse) would either reject a valid test or run nothing without a word."""
        buf = io.StringIO()
        with patch.object(sys, "argv", ["torchpeak", "--help", "--lang", "en"]):
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                main()
        # argparse wraps mid-token (e.g. "gemv_in\nt8"), so compare against a whitespace-free copy
        flat = re.sub(r"\s+", "", buf.getvalue())
        m = re.search(r"comma-separated:probe,([a-z0-9_,]+);", flat)
        self.assertIsNotNone(m, "the --tests help line changed shape")
        documented = {"probe"} | {name for name in m.group(1).split(",") if name}
        self.assertEqual(documented, set(KNOWN_TESTS))


if __name__ == "__main__":
    unittest.main()
