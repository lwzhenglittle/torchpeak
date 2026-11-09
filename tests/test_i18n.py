import ast
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from torchpeak import i18n
from torchpeak.i18n import detect, language, set_language, t

PKG = Path(__file__).resolve().parent.parent / "torchpeak"
ZH = PKG / "locales" / "zh.json"
PLACEHOLDER = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)[^}]*}")


def msgids_in_source():
    """AST-scan the first-argument constant of every t("...") call (implicit string concatenation is merged into one Constant by the parser)."""
    keys = []
    for py in sorted(PKG.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and not node.keywords is None
                    and isinstance(node.func, ast.Name) and node.func.id == "t"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.append((py.name, node.args[0].value))
    return keys


class I18nTests(unittest.TestCase):
    def setUp(self):
        set_language("en")

    def tearDown(self):
        set_language("en")

    def test_fallback_and_format(self):
        self.assertEqual(t("plain {a} + {b}", a=1, b=2), "plain 1 + 2")
        set_language("zh")
        self.assertEqual(t("results written to {path}", path="out.json"), "结果已写入 out.json")
        self.assertEqual(t("key not in any catalog"), "key not in any catalog")  # missing key falls back to English

    def test_unknown_language_falls_back_to_english(self):
        set_language("fr")
        self.assertEqual(language(), "en")
        self.assertEqual(t("results written to {path}", path="x"), "results written to x")

    def test_detect_from_env(self):
        with patch.dict(os.environ, {"TORCHPEAK_LANG": "zh"}, clear=False):
            self.assertEqual(detect(), "zh")
        with patch.dict(os.environ, {"TORCHPEAK_LANG": "en", "LC_ALL": "zh_CN.UTF-8"}, clear=False):
            self.assertEqual(detect(), "en")  # TORCHPEAK_LANG takes precedence over LC_ALL
        with patch.dict(os.environ, {"LC_ALL": "", "LANG": "zh_CN.UTF-8"}, clear=False):
            self.assertEqual(detect(), "zh")
        with patch.dict(os.environ, {"LC_ALL": "", "LANG": "en_US.UTF-8"}, clear=False):
            self.assertEqual(detect(), "en")
        with patch.dict(os.environ, {"LC_ALL": "", "LANG": ""}, clear=False):
            self.assertEqual(detect(), "en")

    def test_zh_catalog_covers_every_msgid(self):
        catalog = json.loads(ZH.read_text())
        missing = [(f, k) for f, k in msgids_in_source() if k not in catalog]
        self.assertEqual(missing, [], f"zh.json 缺少 msgid: {missing}")

    def test_zh_catalog_placeholders_match(self):
        catalog = json.loads(ZH.read_text())
        bad = {k: v for k, v in catalog.items() if set(PLACEHOLDER.findall(k)) != set(PLACEHOLDER.findall(v))}
        self.assertEqual(bad, {}, f"占位符不一致: {bad}")

    def test_no_stale_catalog_entries(self):
        used = {k for _, k in msgids_in_source()}
        catalog = json.loads(ZH.read_text())
        stale = sorted(set(catalog) - used)
        self.assertEqual(stale, [], f"zh.json 含源码已不存在的 msgid: {stale}")


if __name__ == "__main__":
    unittest.main()
