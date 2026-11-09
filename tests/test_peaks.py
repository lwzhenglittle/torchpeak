import json
import tempfile
import unittest
from pathlib import Path

from torchpeak.peaks import load_peak_entries, match_peak


class PeakTests(unittest.TestCase):
    def test_match_is_case_insensitive_and_ordered(self):
        entries = [{"match": "A100-SXM4", "device": "specific"},
                   {"match": "a100", "device": "generic"}]
        self.assertEqual(match_peak({"name": "NVIDIA A100-SXM4-80GB"}, entries)["device"], "specific")
        self.assertEqual(match_peak({"name": "NVIDIA A100-PCIE"}, entries)["device"], "generic")
        self.assertIsNone(match_peak({"name": "unknown"}, entries))

    def test_user_entries_override_bundled_entries(self):
        user = [{"match": "review-device", "device": "user entry"}]
        with tempfile.TemporaryDirectory(dir=".") as outdir:
            path = Path(outdir) / "peaks.json"
            path.write_text(json.dumps(user))
            entries, source = load_peak_entries(str(path))
        self.assertEqual(entries[0]["device"], "user entry")
        self.assertTrue(source.endswith("torchpeak/peaks.json"))
        self.assertEqual(match_peak({"name": "review-device"}, entries)["device"], "user entry")


if __name__ == "__main__":
    unittest.main()
