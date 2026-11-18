"""TPU adapter tests: precision policy, spec derivation, accelerator routing, compile backend.

No TPU hardware is required — the device-facing pieces are injected, so these pin the contract the
TorchTPU path relies on (raw torch.tpu APIs raise on PyTorch >= 2.11; fp32 is an MXU-pass rate).
"""
import types
import unittest
from unittest.mock import patch

import torch

from torchpeak import runtime
from torchpeak.harness import is_oom, tcompile
from torchpeak.peaks import tpu_spec
from torchpeak.report import annotate_efficiency


class _FakeAccelerator:
    def __init__(self, dev):
        self._dev = dev
        self.calls = []

    def current_accelerator(self):
        return self._dev

    def is_available(self):
        return True

    def synchronize(self):
        self.calls.append("synchronize")

    def max_memory_allocated(self):
        self.calls.append("max_memory_allocated")
        return 4 * 10 ** 9


class TpuSpecTests(unittest.TestCase):
    def test_bundled_table_matches_tpu_generations(self):
        from torchpeak.peaks import load_peak_entries, match_peak
        entries, _ = load_peak_entries(None)
        for name in ("TPU v4", "TPU v5e", "TPU v5p", "TPU v6e"):
            spec = match_peak({"name": name}, entries)
            self.assertIsNotNone(spec, name)
            modes = spec["tflops_fp32_mode"]
            self.assertLess(modes["highest"], modes["high"])
            self.assertLess(modes["high"], modes["medium"])
            self.assertAlmostEqual(modes["medium"], spec["tflops"]["bf16"])
            self.assertEqual(spec["tflops"]["fp32"], modes["highest"])

    def test_spec_is_derived_from_device_properties(self):
        info = {"name": "TPU v6e", "device_properties": {
            "name": "TPU v6e", "bf16_flops_per_device": 918e12, "hbm_bytes_per_device": 32e9}}
        spec = tpu_spec(info)
        self.assertEqual(spec["device"], "TPU v6e")
        self.assertEqual(spec["tflops"]["bf16"], 918.0)
        self.assertEqual(spec["tflops"]["fp32"], 153.0)          # 6-pass highest
        self.assertEqual(spec["tflops_fp32_mode"]["high"], 306.0)  # 3-pass
        self.assertEqual(spec["tflops_fp32_mode"]["medium"], 918.0)  # 1 bf16 pass
        self.assertEqual(spec["bw_gbs"], 32.0)

    def test_no_properties_means_no_derived_spec(self):
        self.assertIsNone(tpu_spec({"name": "TPU"}))
        self.assertIsNone(tpu_spec({}))


class PrecisionPolicyTests(unittest.TestCase):
    def test_tpu_measures_fp32_in_two_modes_others_in_one(self):
        self.assertEqual(runtime.fp32_precision_modes(types.SimpleNamespace(type="tpu")),
                         ("highest", "medium"))
        self.assertEqual(runtime.fp32_precision_modes(types.SimpleNamespace(type="cuda")), (None,))
        self.assertEqual(runtime.default_matmul_precision(types.SimpleNamespace(type="tpu")), "medium")
        self.assertIsNone(runtime.default_matmul_precision(types.SimpleNamespace(type="cuda")))

    def test_matmul_precision_scope_restores_previous_mode(self):
        prev = torch.get_float32_matmul_precision()
        with runtime.matmul_precision("highest"):
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertEqual(torch.get_float32_matmul_precision(), prev)

    def test_efficiency_uses_per_mode_peak(self):
        spec = {"tflops": {"fp32": 150.0},
                "tflops_fp32_mode": {"highest": 150.0, "medium": 900.0}}
        rows = [dict(test="gemm", config="fp32 1024x1024 mp=highest", value=100.0, unit="TFLOPS",
                     matmul_precision="highest"),
                dict(test="gemm", config="fp32 1024x1024 mp=medium", value=600.0, unit="TFLOPS",
                     matmul_precision="medium")]
        annotate_efficiency(rows, spec)
        # 100/150, 600/900: each row divides by its own hardware mode, not by one shared fp32 number
        self.assertEqual(rows[0]["peak_spec"], 150.0)
        self.assertAlmostEqual(rows[0]["eff_spec_pct"], 66.7)
        self.assertEqual(rows[1]["peak_spec"], 900.0)
        self.assertAlmostEqual(rows[1]["eff_spec_pct"], 66.7)
        # the measured ceilings stay separate, so "medium" does not inflate the "highest" row
        self.assertEqual(rows[0]["peak_measured"], 100.0)
        self.assertEqual(rows[1]["peak_measured"], 600.0)
        self.assertAlmostEqual(rows[0]["eff_measured_pct"], 100.0)
        self.assertAlmostEqual(rows[1]["eff_measured_pct"], 100.0)


class AcceleratorRoutingTests(unittest.TestCase):
    def test_tpu_uses_accelerator_not_legacy_module(self):
        dev = types.SimpleNamespace(type="tpu", index=0)
        fake = _FakeAccelerator(dev)
        fake_torch = types.SimpleNamespace(accelerator=fake, tpu=types.SimpleNamespace(
            synchronize=lambda: (_ for _ in ()).throw(RuntimeError("legacy torch.tpu API"))))
        with patch.object(runtime, "torch", fake_torch):
            runtime.sync(dev)  # would raise if it fell through to torch.tpu.synchronize
            self.assertEqual(runtime.peak_gb(dev), 4.0)
        self.assertEqual(fake.calls, ["synchronize", "max_memory_allocated"])

    def test_other_backends_keep_the_legacy_path(self):
        dev = types.SimpleNamespace(type="cuda", index=0)
        calls = []
        fake_torch = types.SimpleNamespace(
            accelerator=_FakeAccelerator(types.SimpleNamespace(type="cuda", index=0)),
            cuda=types.SimpleNamespace(synchronize=lambda: calls.append("cuda")))
        with patch.object(runtime, "torch", fake_torch):
            runtime.sync(dev)
        self.assertEqual(calls, ["cuda"])

    def test_auto_prefers_the_current_accelerator(self):
        dev = torch.device("xla")  # a registered non-legacy device type, like "tpu" after backend load
        fake = _FakeAccelerator(dev)
        with patch.object(runtime.torch, "accelerator", fake):
            self.assertEqual(runtime.pick_device("auto").type, "xla")

    def test_explicit_index_is_made_current(self):
        dev = types.SimpleNamespace(type="tpu", index=1)
        fake = _FakeAccelerator(types.SimpleNamespace(type="tpu", index=0))
        fake.current_device_index = lambda: 0
        fake.set_device_index = lambda i: setattr(fake, "_dev", types.SimpleNamespace(type="tpu", index=i))
        with patch.object(runtime, "torch", types.SimpleNamespace(accelerator=fake)):
            runtime.select_device(dev)
        self.assertEqual(fake._dev.index, 1)


class CompileAndOomTests(unittest.TestCase):
    def test_tpu_compile_backend_is_passed_and_mode_skipped(self):
        seen = {}
        with patch("torch.compile", lambda m, **kw: seen.update(kw) or m):
            tcompile(object(), types.SimpleNamespace(compile_backend="tpu",
                                                     compile_mode="max-autotune"))
        self.assertEqual(seen, {"backend": "tpu"})

    def test_inductor_mode_still_passed_for_other_backends(self):
        seen = {}
        with patch("torch.compile", lambda m, **kw: seen.update(kw) or m):
            tcompile(object(), types.SimpleNamespace(compile_backend=None,
                                                     compile_mode="max-autotune"))
        self.assertEqual(seen, {"mode": "max-autotune"})

    def test_xla_out_of_memory_is_retryable(self):
        self.assertTrue(is_oom(RuntimeError("RESOURCE_EXHAUSTED: Out of memory while allocating 4 GiB")))
        self.assertTrue(is_oom(RuntimeError("HBM OOM")))
        self.assertFalse(is_oom(RuntimeError("invalid shape")))


if __name__ == "__main__":
    unittest.main()
