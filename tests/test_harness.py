import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchpeak.harness import (classify_error, coefficient_variation, compare_numerics,
                               dispersion, error_summary, measure, measure_compile_pair,
                               split_pair_config, timeit)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.dev = torch.device("cpu")

    def test_timeit_includes_warmup_and_returns_positive_time(self):
        calls = []
        elapsed = timeit(lambda: calls.append(1), self.dev, warmup=2, min_time=0)
        self.assertEqual(len(calls), 3)
        self.assertGreater(elapsed, 0)

    def test_dispersion_and_coefficient_variation(self):
        stats = dispersion([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats, {"q1": 1.25, "q3": 3.75, "iqr_pct": 100.0})
        self.assertEqual(coefficient_variation([2.0, 2.0, 2.0]), 0.0)

    def test_numerics_equal_nan_and_mismatch(self):
        reference = [torch.tensor([1.0, float("nan")])]
        candidate = [reference[0].clone()]
        max_abs, _, rtol, atol = compare_numerics(reference, candidate, "fp32")
        self.assertFalse(torch.isnan(torch.tensor(max_abs)))
        self.assertEqual((rtol, atol), (1e-4, 1e-4))
        candidate[0][0] = 2.0
        with self.assertRaises(AssertionError):
            compare_numerics(reference, candidate, "fp32")

    def test_error_classification_and_summary(self):
        self.assertEqual(classify_error(NameError("F")), "internal")
        self.assertEqual(classify_error(NotImplementedError()), "unsupported")
        self.assertEqual(classify_error(RuntimeError("CUDA out of memory")), "resource")
        self.assertEqual(classify_error(AssertionError()), "numerical")
        self.assertEqual(classify_error(RuntimeError("backend failed")), "runtime")
        results = [{"error": "x", "error_kind": "internal"},
                   {"error": "y", "error_kind": "unsupported"}, {"value": 1.0}]
        self.assertEqual(error_summary(results),
                         {"total": 2, "by_kind": {"internal": 1, "unsupported": 1}, "internal": 1})

    def test_measure_marks_internal_error(self):
        results = []

        def broken():
            raise NameError("F")

        measure(results, self.dev, "synthetic", "case", "x/s", broken)
        self.assertIsNone(results[0]["value"])
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["error_kind"], "internal")

    def test_compile_pair_configs_and_full_step_memory_sampling(self):
        self.assertEqual(split_pair_config("model fp32 eager"), ("model fp32", "eager"))
        self.assertEqual(split_pair_config("model fp32 compile-max-autotune"),
                         ("model fp32", "compile"))
        self.assertEqual(split_pair_config("model fp32"), (None, None))

        events = []

        def make_case(compiled, scale):
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.ones(1, 2)
            name = "compile_step" if compiled else "eager_step"

            def step():
                events.append(name)
                optimizer.zero_grad(set_to_none=True)
                model(x).sum().backward()
                optimizer.step()

            def check():
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = out.sum()
                loss.backward()
                return loss.detach(), out.detach()

            return step, check

        def reset_peak(_):
            events.append("reset")

        def peak_gb(_):
            events.append("peak")
            return 1.0

        args = SimpleNamespace(amp="fp32", compare_compile=True, compile_mode="default", min_time=0,
                               cooldown=0)
        results = []
        with patch("torchpeak.harness.reset_peak", reset_peak), patch("torchpeak.harness.peak_gb", peak_gb):
            measure_compile_pair(results, self.dev, "synthetic", "case", "x/s", make_case, 1, args)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["value"] > 0 for row in results))
        for name in ("eager_step", "compile_step"):
            index = events.index(name)
            self.assertEqual(events[index - 1:index + 2], ["reset", name, "peak"])
        self.assertTrue(all(row["peak_mem_gb"] == 1.0 for row in results))


if __name__ == "__main__":
    unittest.main()
