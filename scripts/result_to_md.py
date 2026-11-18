#!/usr/bin/env python3
"""Render result JSON files as human-readable Markdown tables.

Usage:
    python scripts/result_to_md.py results/community/*/*.json     # write <name>.md beside each JSON
    python scripts/result_to_md.py --check results/community/*/*.json
    python scripts/result_to_md.py --stdout results/community/gtx1660ti/gtx1660ti.json

--check exits 1 when a .md is missing or stale (CI gate).
"""
import argparse
import json
import sys
from pathlib import Path

PEAK_LABELS = {
    "tflops": ("matmul", "TFLOPS"),
    "tops": ("matmul", "TOPS"),
    "vec_tflops": ("vector", "TFLOPS"),
    "vec_packed": ("packed vector", "TFLOPS"),
    "device_bw": ("device stream", "GB/s"),
    "host_bw": ("host link", "GB/s"),
}


def peak_label(key):
    kind, _, dtype = key.partition(":")
    name, unit = PEAK_LABELS.get(kind, (kind, "—"))
    return (f"{name} {dtype}" if dtype and dtype != "all" else name), unit


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    av = abs(v)
    if av >= 10_000:
        s = f"{v:,.0f}"
    elif av >= 100:
        s = f"{v:,.1f}"
    else:
        s = f"{v:,.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def cell(row):
    """Short note for a measurement row."""
    if row.get("error"):
        first = str(row["error"]).splitlines()[0]
        return "error: " + esc(first[:90])
    if row.get("compile_regress"):
        return "compile regression"
    if row.get("speedup_vs_eager") is not None:
        return f"compile ×{row['speedup_vs_eager']:g}"
    if row.get("numerical_check") in (False, "fail"):
        return "numerical mismatch"
    if row.get("cache_resident"):
        return "L2-resident (cache bandwidth)"
    return ""


def render(doc):
    meta = doc["meta"]
    rows = doc["results"]
    out = []
    label = meta.get("label") or "result"
    out.append(f"# {label}")
    out.append("")

    caps = meta.get("compute_capability")
    cc = ".".join(str(x) for x in caps) if caps else "—"
    args = meta.get("args") or {}
    rt = meta.get("runtime") or {}

    out.append("| Field | Value |")
    out.append("| --- | --- |")
    out.append(f"| Device | {esc(meta.get('name') or '—')}, {fmt(meta.get('compute_units'))} CUs, compute capability {cc} |")
    out.append(f"| Memory | {fmt(meta.get('total_mem_gb'))} GB |")
    out.append(f"| Backend | {esc(meta.get('backend') or '—')} |")
    out.append(f"| PyTorch | {esc(meta.get('torch') or '—')} |")
    out.append(f"| Triton | {esc(rt.get('triton') or '—')} |")
    out.append(f"| Platform | {esc(meta.get('platform') or '—')} |")
    out.append(f"| Suite | {esc(args.get('tests') or '—')} · amp {esc(args.get('amp') or '—')}"
               f" · {'compare_compile' if meta.get('compare_compile') else 'no compare_compile'}"
               f" · {'quick' if meta.get('quick') else 'full scale'} |")
    measured = sum(1 for r in rows if r.get("value") is not None)
    errs = (meta.get("errors") or {}).get("total", 0)
    out.append(f"| Rows | {len(rows)} total, {measured} measured, {errs} errors |")
    if meta.get("redaction"):
        out.append("| Provenance | vendor identifiers redacted, measurements untouched (see `meta.redaction`) |")
    out.append("")

    peaks = meta.get("peak_measured") or {}
    if peaks:
        out.append("## Peak (measured)")
        out.append("")
        out.append("| Metric | Value | Unit |")
        out.append("| --- | --- | --- |")
        for k in sorted(peaks):
            label, unit = peak_label(k)
            out.append(f"| {esc(label)} | {fmt(peaks[k])} | {esc(unit)} |")
        out.append("")

    groups = {}
    order = []
    for r in rows:
        t = r.get("test") or "?"
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(r)

    for t in order:
        grp = groups[t]
        out.append(f"## {t} ({len(grp)})")
        out.append("")
        if any(r.get("mountain") for r in grp):
            for r in grp:
                mt = r.get("mountain") or {}
                sizes, strides, gbs = mt.get("sizes") or [], mt.get("strides") or [], mt.get("gbs") or []
                out.append(f"{esc(mt.get('op') or 'read')} mountain, {esc(mt.get('dtype') or '—')} — GB/s, "
                           f"rows = stride, columns = working set (peak {fmt(r.get('value'))} {esc(r.get('unit') or '')}, "
                           f"{esc(r.get('value_basis') or '—')})")
                out.append("")
                if not gbs:
                    continue
                out.append("| Stride | " + " | ".join(esc(x) for x in sizes) + " |")
                out.append("| --- |" + " --- |" * len(sizes))
                for si, stride in enumerate(strides):
                    out.append(f"| {esc(stride)} | " + " | ".join(fmt(v) for v in gbs[si]) + " |")
                out.append("")
        elif t == "probe":
            out.append("| Config | Result | Status | Note |")
            out.append("| --- | --- | --- | --- |")
            for r in grp:
                out.append(f"| {esc(r.get('config'))} | {fmt(r.get('value'))} | {esc(r.get('status') or 'ok')} | {cell(r)} |")
        else:
            out.append("| Config | Value | Unit | vs meas. peak | CV % | Note |")
            out.append("| --- | --- | --- | --- | --- | --- |")
            for r in grp:
                eff = r.get("eff_measured_pct")
                out.append(
                    "| {} | {} | {} | {} | {} | {} |".format(
                        esc(r.get("config")),
                        fmt(r.get("value")),
                        esc(r.get("unit") or "—"),
                        f"{eff:.1f}%" if eff is not None else "—",
                        fmt(r.get("cv_pct")),
                        cell(r),
                    )
                )
        out.append("")

    failed = [r for r in rows if r.get("status") in ("fail", "error") or r.get("error")]
    if failed:
        out.append("## Failures")
        out.append("")
        out.append("| Test | Config | Kind | Message |")
        out.append("| --- | --- | --- | --- |")
        for r in failed:
            msg = str(r.get("error") or r.get("status") or "").splitlines()[0]
            out.append(f"| {esc(r.get('test'))} | {esc(r.get('config'))} | {esc(r.get('error_kind') or r.get('status') or '—')} | {esc(msg[:120])} |")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--check", action="store_true", help="fail if <name>.md is missing or stale")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    a = ap.parse_args(argv)

    stale = []
    for p in a.paths:
        src = Path(p)
        if not src.is_file():
            stale.append(f"{src}: no such file")
            continue
        with open(src) as f:
            text = render(json.load(f))
        if a.stdout:
            sys.stdout.write(text)
            continue
        dst = src.with_suffix(".md")
        if a.check:
            current = dst.read_text() if dst.is_file() else None
            if current != text:
                stale.append(f"{dst}: {'missing' if current is None else 'stale'}")
        else:
            dst.write_text(text)
            print(f"wrote {dst}")
    if a.check and stale:
        sys.stderr.write("markdown summaries out of date (run scripts/result_to_md.py <json>):\n")
        for s in stale:
            sys.stderr.write(f"  {s}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
