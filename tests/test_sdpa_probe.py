"""SDPA probe input contract: fused kernels are half-precision only.

Probing FLASH_ATTENTION/CUDNN_ATTENTION with fp32 tensors fails on every CUDA device (Flash checks
the dtype in flash_api.cpp; cuDNN's check_dtypes_low_precision accepts only half/bfloat16 on sm>=80),
so the probe dtype is pinned here.
"""
import types
import unittest

import torch

from torchpeak.probes import sdpa_probe_dtype


class SdpaProbeDtypeTests(unittest.TestCase):
    def test_fused_backends_are_probed_at_half_precision(self):
        cuda = types.SimpleNamespace(type="cuda")
        for backend in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
            self.assertEqual(sdpa_probe_dtype(cuda, backend), torch.float16, backend)

    def test_math_and_cpu_stay_fp32(self):
        cuda = types.SimpleNamespace(type="cuda")
        cpu = types.SimpleNamespace(type="cpu")
        self.assertEqual(sdpa_probe_dtype(cuda, "MATH"), torch.float32)
        self.assertEqual(sdpa_probe_dtype(cpu, "FLASH_ATTENTION"), torch.float32)


if __name__ == "__main__":
    unittest.main()
