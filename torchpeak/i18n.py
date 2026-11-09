"""torchpeak.i18n — localization of runtime messages.

English is the source language: user-facing literals are written in English and passed
through t(); a language catalog (torchpeak/locales/<lang>.json) maps English → target
language. Missing keys fall back to English, so a stale catalog can never break a run.

Language selection order: --lang > $TORCHPEAK_LANG > $LC_ALL/$LANG (zh* → zh) > en.
"""
import json
import os

LANG_ENV = "TORCHPEAK_LANG"
DEFAULT_LANG = "en"

_catalog = {}
_lang = DEFAULT_LANG


def _locales_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")


def available_languages():
    """Built-in en plus the languages present in the locales directory."""
    langs = [DEFAULT_LANG]
    d = _locales_dir()
    if os.path.isdir(d):
        langs += sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
    return langs


def detect():
    """Detect via TORCHPEAK_LANG > LC_ALL > LANG; zh* → zh, anything else (including unset) → en."""
    for env in (LANG_ENV, "LC_ALL", "LANG"):
        v = os.environ.get(env)
        if v:
            return "zh" if v.lower().split(".")[0].split("_")[0] == "zh" else DEFAULT_LANG
    return DEFAULT_LANG


def set_language(lang):
    """Load a language catalog; unknown language or broken file silently falls back to English."""
    global _catalog, _lang
    _catalog, _lang = {}, DEFAULT_LANG
    if not lang or lang == DEFAULT_LANG or lang == "auto":
        if lang == "auto":
            set_language(detect())
        return
    path = os.path.join(_locales_dir(), f"{lang}.json")
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _catalog, _lang = data, lang
    except (OSError, ValueError):
        pass


def language():
    return _lang


def t(msg, **fmt):
    """Look up a translation; missing keys fall back to the English source; fmt supplies placeholders (str.format semantics)."""
    text = _catalog.get(msg, msg)
    return text.format(**fmt) if fmt else text
