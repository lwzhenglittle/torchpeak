import unittest

from torchpeak.ops import GEMV_DRAM_K, GEMV_DRAM_TARGET_MIB, gemv_dram_n


class GemvDramShape(unittest.TestCase):
    def test_weight_matrix_exceeds_every_last_level_cache(self):
        # DRAM regime for int8 (1 B), fp16/bf16 (2 B) and fp32 (4 B): the weight matrix alone must
        # clear the target, which in turn clears the largest last-level cache shipping today (256 MiB).
        for dt_size in (1, 2, 4):
            n = gemv_dram_n(dt_size)
            weight_bytes = GEMV_DRAM_K * n * dt_size
            self.assertEqual(n % 64, 0)
            self.assertGreaterEqual(weight_bytes, GEMV_DRAM_TARGET_MIB << 20)
            self.assertGreater(weight_bytes, 256 << 20)


if __name__ == "__main__":
    unittest.main()
