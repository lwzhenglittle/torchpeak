import contextlib
import io
import unittest
from unittest.mock import patch

from torchpeak import harness
from torchpeak.harness import (ESCALATE_CV_PCT, OUTLIER_CAP, STABLE_CV_PCT,
                               mad_outliers, measure, rounds_for_cv)


def run_measure(values, strict=False, cooldown=0.0, error=None):
    """Drive measure() with a fake device and patched runtime hooks; return the recorded row."""
    it = iter(values)

    def fn():
        if error is not None:
            raise error
        return next(it)

    results = []
    with patch.object(harness, "reset_peak"), \
            patch.object(harness, "peak_gb", return_value=None), \
            patch.object(harness, "peak_reserved_gb", return_value=None), \
            patch.object(harness, "empty_cache"), \
            patch.object(harness.time, "sleep") as sleep, \
            contextlib.redirect_stdout(io.StringIO()):
        measure(results, dev=object(), test="t", config="c", unit="x", fn=fn,
                strict=strict, cooldown=cooldown)
    return results[0], sleep


class RoundTiers(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(rounds_for_cv(0.0), 3)
        self.assertEqual(rounds_for_cv(STABLE_CV_PCT), 3)
        self.assertEqual(rounds_for_cv(STABLE_CV_PCT + 0.01), 5)
        self.assertEqual(rounds_for_cv(ESCALATE_CV_PCT), 5)
        self.assertEqual(rounds_for_cv(ESCALATE_CV_PCT + 0.01), 7)
        self.assertEqual(rounds_for_cv(90.0), 7)

    def test_outlier_caps(self):
        self.assertEqual(OUTLIER_CAP[3], 0)
        self.assertEqual(OUTLIER_CAP[5], 1)
        self.assertEqual(OUTLIER_CAP[7], 2)


class MadOutliers(unittest.TestCase):
    def test_no_data_at_n3(self):
        self.assertEqual(mad_outliers([1.0, 1.0, 500.0], 1), [])

    def test_clean_data_untouched(self):
        self.assertEqual(mad_outliers([10.0, 10.1, 10.2, 10.3], 1), [])

    def test_flag_and_cap(self):
        vals = [10.0, 10.1, 10.2, 10.3, 100.0]
        out = mad_outliers(vals, 1)
        self.assertEqual(out, [4])  # med 10.2, MAD 0.1, modified z ~605 >> 3.5
        self.assertEqual(mad_outliers([10.0, 10.1, 10.2, 10.3, 100.0, 200.0], 2), [5, 4])

    def test_ties_break_farther_first(self):
        # 106 and 94 are equidistant; only the farther-ranked one (first index) is dropped at cap 1
        self.assertEqual(mad_outliers([100.0, 106.0, 94.0, 100.0, 101.0], 1), [1])

    def test_majority_identical_is_conservative(self):
        # MAD=0 falls back to meanAD; at n=5 with 4 identical points the z ceiling is ~3.37 < 3.5
        self.assertEqual(mad_outliers([5.0, 5.0, 5.0, 5.0, 500.0], 1), [])
        # but at n=7 it still fires
        vals = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 500.0]
        self.assertEqual(mad_outliers(vals, 2), [6])


class MeasurePolicy(unittest.TestCase):
    def test_stable_stops_at_three(self):
        row, _ = run_measure([10.0, 10.1, 10.2])
        self.assertEqual(len(row["trials"]), 3)
        self.assertEqual(row["n_used"], 3)
        self.assertNotIn("outliers", row)
        self.assertTrue(row["stable"])
        self.assertEqual(row["value"], 10.1)
        self.assertNotIn("iqr_pct", row)  # IQR needs n>=4

    def test_escalates_and_recovers(self):
        # first-3 CV ~6% -> n=5; 106 is an Iglewicz-Hoaglin outlier; kept CV 3.2% -> STABLE
        row, _ = run_measure([100.0, 106.0, 94.0, 100.0, 101.0])
        self.assertEqual(len(row["trials"]), 5)
        self.assertEqual(row["n_used"], 4)
        self.assertEqual(row["outliers"], [106.0])
        self.assertTrue(row["stable"])
        self.assertEqual(row["value"], 100.0)
        self.assertIn("iqr_pct", row)

    def test_strict_runs_seven_rounds(self):
        row, _ = run_measure([10.0] * 9, strict=True)
        self.assertEqual(len(row["trials"]), 7)
        self.assertTrue(row["stable"])

    def test_unstable_reported_with_full_trials(self):
        # 6% -> n=5, provisional removal doesn't help (mz<3.5) -> n=7, still ~29% -> UNSTABLE
        row, _ = run_measure([100.0, 106.0, 94.0, 130.0, 70.0, 60.0, 140.0])
        self.assertEqual(len(row["trials"]), 7)
        self.assertEqual(row["n_used"], 7)
        self.assertFalse(row["stable"])
        self.assertGreater(row["cv_pct"], STABLE_CV_PCT)

    def test_grows_from_five_to_seven_and_recovers(self):
        # 6% -> n=5, kept CV ~13% after provisional removal -> n=7; final removal drops 2 -> 2.9% STABLE
        row, _ = run_measure([100.0, 106.0, 94.0, 100.0, 130.0, 100.0, 101.0])
        self.assertEqual(len(row["trials"]), 7)
        self.assertEqual(row["n_used"], 5)
        self.assertEqual(row["outliers"], [106.0, 130.0])
        self.assertTrue(row["stable"])
        self.assertEqual(row["value"], 100.0)

    def test_high_cv_goes_straight_to_seven(self):
        row, _ = run_measure([100.0, 140.0, 60.0, 100.0, 100.0, 100.0, 100.0])
        self.assertEqual(len(row["trials"]), 7)

    def test_cooldown_sleeps_after_row(self):
        _, sleep = run_measure([10.0, 10.1, 10.2], cooldown=2.0)
        sleep.assert_called_once_with(2.0)

    def test_no_cooldown_by_default(self):
        _, sleep = run_measure([10.0, 10.1, 10.2])
        sleep.assert_not_called()

    def test_error_row_recorded(self):
        row, _ = run_measure([], error=ValueError("boom"))
        self.assertIsNone(row["value"])
        self.assertEqual(row["error_kind"], "runtime")

class RowPrinting(unittest.TestCase):
    ROW = dict(test="gemm", config="fp32 1024x1024", value=3.06, unit="TFLOPS",
               trials=[3.0, 3.1, 3.1], n_used=3, cv_pct=1.08, stable=True)

    def run_capture(self, fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        return buf.getvalue()

    def test_head_then_tail_is_one_line(self):
        harness._row_pending = False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            harness.print_row_head("gemm", "fp32 1024x1024")
            harness.print_row(self.ROW, head="")
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(f"  {'gemm':<16} fp32 1024x1024"))
        self.assertIn("3.06 TFLOPS", lines[0])

    def test_full_row_terminates_dangling_head(self):
        harness._row_pending = False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            harness.print_row_head("gemm", "fp32 1024x1024")
            harness.print_row(self.ROW)
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].rstrip().endswith("fp32 1024x1024"))
        self.assertTrue(lines[1].startswith("  gemm"))


if __name__ == "__main__":
    unittest.main()
