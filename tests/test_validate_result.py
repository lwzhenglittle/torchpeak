import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validate_result", ROOT / "scripts" / "validate_result.py")
validate_result = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_result)


def payload():
    meta = {"format_version": 2, "torch": "2.5.0", "platform": "Linux-x86_64",
            "stability_policy": {}, "errors": {"total": 0}, "label": "dev-x"}
    return {"meta": meta, "results": [{"test": "probe", "config": "sdpa_math", "value": 1.0, "unit": "probe"}]}


class ValidateLayoutTests(unittest.TestCase):
    def write(self, root, relpath):
        p = Path(root) / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload()))
        return p

    def test_one_directory_per_device(self):
        with tempfile.TemporaryDirectory() as d:
            ok = self.write(d, "community/dev-x/dev-x.json")
            self.assertEqual(validate_result.validate(ok)[0], [])
            tuned = self.write(d, "community/dev-x/dev-x-tuned.json")
            self.assertEqual(validate_result.validate(tuned)[0], [])
            wrong = self.write(d, "community/other/dev-x.json")
            self.assertIn("expected results/community/dev-x/", validate_result.validate(wrong)[0][0])
            flat = self.write(d, "community/dev-x.json")
            self.assertIn("expected results/community/dev-x/", validate_result.validate(flat)[0][0])

    def test_files_outside_the_corpus_skip_the_directory_rule(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "bench/dev-x.json")
            self.assertEqual(validate_result.validate(p)[0], [])


if __name__ == "__main__":
    unittest.main()
