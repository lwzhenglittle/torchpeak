#!/usr/bin/env python3
"""Validate community-contributed result files (results/community/*.json).

Hard failures (exit 1): unparseable JSON, wrong format_version, incomplete meta,
bad filename slug, oversized file, or no measured rows.
Soft warnings (stderr, exit 0): hostname-looking label, home-path leakage.
"""
import json
import re
import sys
from pathlib import Path

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*(-tuned)?\.json$")
MAX_BYTES = 2 * 1024 * 1024
META_KEYS = ("torch", "platform", "stability_policy", "errors", "results")


def validate(path):
    name = Path(path).name
    if not SLUG.match(name):
        return [f"filename {name!r} is not a device slug like 'gtx1660ti.json' or 'a100-sxm4-80gb.json'"], []
    p = Path(path)
    if p.stat().st_size > MAX_BYTES:
        return [f"{name}: {p.stat().st_size} bytes exceeds the {MAX_BYTES} cap"], []
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return [f"{name}: not valid JSON: {e}"], []
    meta = d.get("meta") or {}
    rows = d.get("results")
    errs, warns = [], []
    if meta.get("format_version") != 2:
        errs.append(f"{name}: meta.format_version must be 2, got {meta.get('format_version')!r}")
    for k in META_KEYS:
        if k not in meta and k not in d:
            errs.append(f"{name}: missing meta.{k}" if k != "results" else f"{name}: missing results")
    if not isinstance(rows, list) or not rows:
        errs.append(f"{name}: results is empty")
    elif not any(r.get("value") is not None for r in rows if isinstance(r, dict)):
        errs.append(f"{name}: no row carries a measured value")
    label = meta.get("label", "")
    if "_" in label:
        warns.append(f"{name}: label {label!r} looks like hostname_device; prefer --label <device-slug>")
    blob = json.dumps(d)
    for pat in ("/home/", "/Users/", "C:\\\\Users\\\\"):
        if pat in blob:
            warns.append(f"{name}: contains a home path ({pat}); rerun from a neutral working directory")
            break
    return errs, warns


def main(argv):
    files = [f for f in argv if Path(f).exists()]  # tolerate an unmatched shell glob
    if not files:
        print("no result files to validate", file=sys.stderr)
        return 0
    failed = False
    for f in files:
        errs, warns = validate(f)
        for w in warns:
            print(f"WARN {w}", file=sys.stderr)
        if errs:
            failed = True
            for e in errs:
                print(f"FAIL {e}", file=sys.stderr)
        else:
            print(f"OK   {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
