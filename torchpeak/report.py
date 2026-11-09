"""torchpeak.report — see the package root docstring."""

import torch
from .harness import split_pair_config
from .i18n import t

EFFICIENCY_KINDS = ("tflops", "vec_tflops", "vec_packed", "tops", "device_bw", "host_bw")
HOST_OPS = {"h2d", "d2h", "h2d_pinned", "d2h_pinned"}
STREAM_OPS = {"read", "write", "copy"}
DRAM_SIZE_MIB = 256  # working sets >= this are beyond any current GPU L2 (largest today ~50 MB)


def is_dram_stream_row(row):
    """Device-local STREAM op (read/write/copy) at a DRAM-regime size. Only such rows enter the
    device_bw envelope / measured ceiling; L2-resident sizes, GEMV proxies, memory-wall cells and
    model-level weight bandwidth stay per-row metrics (eff_spec_pct) without shaping the ceiling."""
    if row.get("test") != "bandwidth":
        return False
    parts = (row.get("config") or "").split()
    return (len(parts) == 2 and parts[0] in STREAM_OPS and parts[1].endswith("MiB")
            and int(parts[1][:-3]) >= DRAM_SIZE_MIB)


def envelope_metric(row):
    m = row_metric(row)
    if m and m[0] in EFFICIENCY_KINDS and (m[0] != "device_bw" or is_dram_stream_row(row)):
        return m
    return None


def row_metric(row):
    """Return (metric kind, precision key, value); device memory and host interconnect bandwidth use different kinds."""
    t, cfg = row.get("test"), row.get("config") or ""
    if row.get("value") is None:
        return None
    if t == "gemm":
        return ("tflops", cfg.split()[0], row["value"])
    if t == "vec_flops":
        kind = "vec_packed" if "triton-native" in cfg else "vec_tflops"
        return (kind, cfg.split()[0], row["value"])
    if t == "gemm_int8":
        return ("tops", "int8", row["value"])
    if t in ("gemv", "gemv_int8", "memory_wall"):
        return ("device_bw", None, row["value"])
    if t == "bandwidth":
        kind = "host_bw" if cfg.split()[0] in HOST_OPS else "device_bw"
        return (kind, None, row["value"])
    if row.get("approx_tflops") is not None:
        dtype = next((d for d in ("fp32", "fp16", "bf16") if d in cfg), None)
        return ("tflops", dtype, row["approx_tflops"]) if dtype else None
    if row.get("approx_weight_bw_gbs") is not None:
        return ("device_bw", None, row["approx_weight_bw_gbs"])
    return None


def annotate_efficiency(results, spec):
    """Core metric: software efficiency = measured / peak in the same dimension; measured ceilings are
    computed per metric kind and precision. Writes peak_spec / eff_spec_pct / peak_measured / eff_measured_pct in place."""
    measured = {}
    for r in results:
        m = envelope_metric(r)
        if m:
            k = (m[0], m[1])
            if k not in measured or m[2] > measured[k]:
                measured[k] = m[2]
    spec_tf = (spec or {}).get("tflops") or {}
    spec_int8 = (spec or {}).get("int8_tops")
    spec_bw = (spec or {}).get("bw_gbs")
    spec_host_bw = (spec or {}).get("host_bw_gbs")
    for r in results:
        m = row_metric(r)
        if not m:
            continue
        kind, key, value = m
        if kind == "tops":
            spec_peak = spec_int8
        elif kind == "device_bw":
            spec_peak = spec_bw
        elif kind == "host_bw":
            spec_peak = spec_host_bw
        elif kind == "vec_tflops":
            spec_peak = (spec or {}).get("vec_tflops", {}).get(key)
        elif kind == "vec_packed":
            spec_peak = (spec or {}).get("vec_packed_tflops", {}).get(key)
        else:
            spec_peak = spec_tf.get(key)
        meas_peak = measured.get((kind, key))
        if spec_peak:
            r["peak_spec"] = spec_peak
            r["eff_spec_pct"] = round(value / spec_peak * 100, 1)
        if meas_peak:
            r["peak_measured"] = round(meas_peak, 3)
            r["eff_measured_pct"] = round(value / meas_peak * 100, 1)
    return measured


def annotate_compile_speedup(results):
    """Pair compare_compile rows by config suffix; compile rows get speedup_vs_eager, and <1 is also flagged compile_regress."""
    pair = {}
    for r in results:
        cfg = r.get("config") or ""
        stem, variant = split_pair_config(cfg)
        if stem is not None:
            pair.setdefault((r["test"], stem, r["unit"]), {})[variant] = r
    for pair_rows in pair.values():
        c, e = pair_rows.get("compile"), pair_rows.get("eager")
        if c and e and c.get("value") and e.get("value"):
            sp = c["value"] / e["value"]
            c["speedup_vs_eager"] = round(sp, 3)
            c["compile_regress"] = sp < 1.0


def print_efficiency_summary(results, spec):
    best = {}
    for r in results:
        m = envelope_metric(r)
        if m:
            k = f"{m[0]}:{m[1] or 'all'}"
            if k not in best or m[2] > best[k][1]:
                best[k] = (r, m[2])
    model_rows = [r for r in results if r.get("approx_tflops") is not None or r.get("approx_weight_bw_gbs") is not None]
    if not best and not model_rows:
        return
    print(t("\n== Software efficiency (measured throughput / peak, primary metric)"))
    if spec:
        print(t("   Device: {device}", device=spec["device"]))
    else:
        print(t("   Device: no built-in peak-table match for this device; "
                "--peaks FILE can supply one; the table below reports measured upper bounds only"))
    for k, (r, v) in best.items():
        s = f"{r.get('eff_spec_pct', 'n/a'):>5}% spec" if "eff_spec_pct" in r else "  n/a spec"
        m2 = f"{r.get('eff_measured_pct', 0):>5.1f}% meas" if "eff_measured_pct" in r else ""
        print(t("   {k:<14} best {v:>10.1f}   {s} {m2}", k=k, v=v, s=s, m2=m2))
    for r in model_rows:
        v = r.get("approx_tflops") if r.get("approx_tflops") is not None else r["approx_weight_bw_gbs"]
        unit = "TFLOPS" if r.get("approx_tflops") is not None else "GB/s"
        s = f"{r['eff_spec_pct']:>5.1f}% spec" if "eff_spec_pct" in r else "  n/a spec"
        m2 = f"{r['eff_measured_pct']:>5.1f}% meas" if "eff_measured_pct" in r else ""
        tag = f"  speedup {r['speedup_vs_eager']:.2f}x" if r.get("speedup_vs_eager") is not None else ""
        reg = "  REGRESS" if r.get("compile_regress") else ""
        print(f"   {r['test']:<16} {r['config']:<34} {v:>8.1f} {unit:<7} {s} {m2}{tag}{reg}")

# ---------------------------------------------------------------- HTML report
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _svg_bars(items, width=780, row_h=26):
    """Horizontal bar chart. items: (label, value, note, color); values normalized to the max."""
    if not items:
        return ""
    label_w, plot_w = 320, width - 320 - 100
    vmax = max(v for _, v, *_ in items) or 1.0
    parts = []
    for i, it in enumerate(items):
        label, value = it[0], it[1]
        note, color = (it[2] if len(it) > 2 else ""), (it[3] if len(it) > 3 else "#2563eb")
        w = max(2.0, value / vmax * plot_w)
        y = i * row_h
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 17}" text-anchor="end" font-size="12" fill="#444">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y + 4}" width="{w:.0f}" height="{row_h - 10}" rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 6:.0f}" y="{y + 17}" font-size="12" fill="#333">{value:.5g}'
            f'{"  " + _esc(note) if note else ""}</text>')
    return (f'<svg viewBox="0 0 {width} {len(items) * row_h}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="system-ui,sans-serif">' + "".join(parts) + "</svg>")


def _svg_mountain(mt):
    """CSAPP memory mountain heatmap: x=working-set size, y=stride, color=read throughput (blue low, red high)."""
    sizes, strides, gbs = mt["sizes"], mt["strides"], mt["gbs"]
    cell, lw, th = 76, 64, 22
    w, h = lw + cell * len(sizes), th + cell * len(strides)
    flat = [g for row in gbs for g in row]
    lo, hi = min(flat), max(flat)

    def color(v):
        t = (v - lo) / (hi - lo) if hi > lo else 0.5
        return f"hsl({220 * (1 - t):.0f},72%,55%)"

    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    for xi, s in enumerate(sizes):
        parts.append(f'<text x="{lw + xi * cell + cell / 2}" y="14" text-anchor="middle" font-size="11" '
                     f'fill="#666">{_esc(s)}</text>')
    for yi, s in enumerate(strides):
        parts.append(f'<text x="{lw - 8}" y="{th + yi * cell + cell / 2 + 4}" text-anchor="end" font-size="11" '
                     f'fill="#666">stride {s}</text>')
        for xi in range(len(sizes)):
            v = gbs[xi][yi]
            parts.append(f'<rect x="{lw + xi * cell}" y="{th + yi * cell}" width="{cell - 2}" height="{cell - 2}" '
                         f'rx="3" fill="{color(v)}"/>'
                         f'<text x="{lw + xi * cell + cell / 2 - 1}" y="{th + yi * cell + cell / 2 + 4}" '
                         f'text-anchor="middle" font-size="11" fill="#fff">{v:.0f}</text>')
    parts.append(f'<text x="4" y="14" font-size="11" fill="#666">{_esc(mt.get("dtype", "fp32"))} read GB/s</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_html_report(meta, results, path):
    """Self-contained static HTML (inline SVG, no external deps/JS) written alongside the JSON."""
    def val(row, key, fmt=None):
        v = row.get(key)
        return ("-" if v is None else (fmt % v if fmt else v))

    m_rows = [r for r in results if r["test"] == "probe"]
    ok = [r for r in m_rows if r.get("status") == "pass"]
    bad = [r for r in m_rows if r.get("status") == "fail"]
    chips = "".join(f'<span class="chip ok" title="{_esc(r["config"])} {t("available")}">✓ {_esc(r["config"])}</span>'
                    for r in ok) + "".join(
        f'<span class="chip bad" title="{_esc(r.get("error", ""))}">✗ {_esc(r["config"])}</span>' for r in bad)

    # Efficiency bars: best microbenchmark per dimension + model-derived rows
    best = {}
    for r in results:
        m = row_metric(r)
        if m and m[0] in EFFICIENCY_KINDS:
            k = f"{m[0]}:{m[1] or 'all'}"
            if k not in best or m[2] > best[k][1]:
                best[k] = (r, m[2])
    bars = []
    for k, (r, v) in best.items():
        note = f"{r['eff_spec_pct']:.0f}{t('% spec')}" if r.get("eff_spec_pct") is not None else t("no spec peak")
        bars.append((t("{kind} best {test} {config}", kind=k, test=r["test"], config=r["config"]), v, note))
    model_rows = [r for r in results if r.get("approx_tflops") is not None or r.get("approx_weight_bw_gbs") is not None]
    for r in model_rows:
        v = r.get("approx_tflops") if r.get("approx_tflops") is not None else r["approx_weight_bw_gbs"]
        unit = "TFLOPS" if r.get("approx_tflops") is not None else "GB/s"
        note = f"{r['eff_spec_pct']:.0f}{t('% spec')}" if r.get("eff_spec_pct") is not None else ""
        if r.get("speedup_vs_eager") is not None:
            note += f" · compile {r['speedup_vs_eager']:.2f}x" + (" REGRESS" if r.get("compile_regress") else "")
        color = "#dc2626" if r.get("compile_regress") else "#2563eb"
        bars.append((f"{r['test']} {r['config'][:40]}", v, f"{unit} {note}", color))

    # Bandwidth comparison bars (including pinned vs pageable)
    bw_rows = [r for r in results if r["test"] in ("bandwidth", "gemv", "gemv_int8") and r.get("value") is not None]
    bw_bars = [(f"{r['test']} {r['config']}", r["value"],
                f"{r['eff_spec_pct']:.0f}{t('% spec')}" if r.get("eff_spec_pct") is not None else "",
                "#0d9488" if r["test"] != "bandwidth" else "#2563eb") for r in bw_rows]

    # Compile speedup table
    pair = {}
    for r in results:
        stem, variant = split_pair_config(r.get("config") or "")
        if stem is not None:
            pair.setdefault((r["test"], stem), {})[variant] = r
    speedup_html = ""
    if pair:
        trs = ""
        for (test, stem), pr in pair.items():
            e, c = pr.get("eager"), pr.get("compile")
            if not (e and c and e.get("value") and c.get("value")):
                continue
            sp = c["value"] / e["value"]
            style = ' style="color:#dc2626"' if sp < 1 else ""
            trs += (f'<tr><td>{_esc(test)}</td><td>{_esc(stem)}</td><td>{e["value"]}</td><td>{c["value"]}</td>'
                    f'<td{style}><b>{sp:.3f}x</b>{" REGRESS" if sp < 1 else ""}</td>'
                    f'<td>{val(c, "compile_cold_s", "%.1f")}s / {val(c, "compile_hot_s", "%.1f")}s</td>'
                    f'<td>{val(c, "dynamo_graph_breaks")}</td></tr>')
        speedup_html = (f'<div class="card"><h2>{t("compile speedup")}</h2><table><tr><th>test</th><th>config</th>'
                        f'<th>eager</th><th>compile</th><th>speedup</th><th>{t("compile cold/hot")}</th><th>breaks</th></tr>'
                        f"{trs}</table></div>")

    mountain_html = ""
    for r in results:
        if r.get("mountain"):
            mountain_html = (f'<div class="card"><h2>{t("memory mountain (CSAPP)")}</h2>'
                             f'<p class="hint">{t("Read throughput vs working set × stride; ridges = cache levels; performance collapses when large strides cross cache lines.")}</p>'
                             f"{_svg_mountain(r['mountain'])}</div>")
            break

    trs = ""
    for r in results:
        err = f' title="{_esc(r.get("error", ""))}" style="color:#dc2626"' if r.get("value") is None else ""
        eff = val(r, "eff_spec_pct", "%.0f%%")
        trs += (f'<tr{err}><td>{_esc(r["test"])}</td><td>{_esc(r["config"])}</td><td>{val(r, "value")}</td>'
                f'<td>{_esc(r.get("unit", ""))}</td><td>{eff}</td>'
                f'<td>{val(r, "eff_measured_pct", "%.0f%%")}</td><td>{val(r, "cv_pct", "%.1f%%")}</td>'
                f'<td>{val(r, "peak_mem_gb", "%.2f")}</td></tr>')

    u = meta.get("usability") or {}
    rt = meta.get("runtime") or {}
    peak = meta.get("peak_spec") or {}
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>torchpeak — {_esc(meta["label"])}</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}}
.wrap{{max-width:980px;margin:0 auto;padding:24px}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;margin:14px 0}}
h1{{font-size:20px;margin:4px 0}} h2{{font-size:16px;margin:2px 0 10px;color:#374151}}
.kv{{display:grid;grid-template-columns:150px 1fr;gap:2px 12px;font-size:13px}}
.kv b{{color:#6b7280;font-weight:500}}
.chip{{display:inline-block;margin:3px 4px 3px 0;padding:2px 10px;border-radius:99px;font-size:12.5px}}
.chip.ok{{background:#dcfce7;color:#166534}} .chip.bad{{background:#fee2e2;color:#991b1b;cursor:help}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
td,th{{border-bottom:1px solid #f0f0f0;padding:4px 8px;text-align:left;white-space:nowrap}}
th{{color:#6b7280;font-weight:500}} .hint{{font-size:12.5px;color:#6b7280;margin:4px 0}}
details summary{{cursor:pointer;font-size:14px;color:#374151}}
</style></head><body><div class="wrap">
<div class="card"><h1>torchpeak · {_esc(meta["label"])}</h1>
<div class="kv">
<b>{t("Device")}</b><span>{_esc(meta.get("name"))} [{_esc(meta.get("backend"))}] · {_esc(meta.get("total_mem_gb"))}GB</span>
<b>{t("Software stack")}</b><span>torch {_esc(meta.get("torch"))} · triton {_esc(rt.get("triton"))} · driver {_esc(meta.get("driver_version") or rt.get("cuda_driver"))} · governor {_esc(rt.get("cpu_governor"))}</span>
<b>{t("Config")}</b><span>amp={_esc(meta.get("amp"))} · quick={_esc(meta.get("quick"))} · strict={_esc(meta.get("strict"))} · compile={_esc(meta.get("compare_compile"))}</span>
<b>{t("Track / tuning")}</b><span>{_esc(t("tuned") if (meta.get("tuned") or {}).get("enabled") else t("out of the box"))} · ocloc shim: {_esc(rt.get("ocloc_shim"))}</span>
<b>{t("Peak baseline")}</b><span>{_esc(peak.get("device", t("no match, measured upper bound only")))} · {t("measured upper bound")} {_esc(meta.get("peak_measured"))}</span>
</div></div>
<div class="card"><h2>{t("Feature availability {passed}/{total} (hover for failure reasons)", passed=u.get("passed", 0), total=u.get("total", len(m_rows)))}</h2>{chips}</div>
<div class="card"><h2>{t("Software efficiency (measured / peak)")}</h2>{_svg_bars(bars)}</div>
{mountain_html}
<div class="card"><h2>{t("Bandwidth / GEMV (GB/s)")}</h2>{_svg_bars(bw_bars)}</div>
{speedup_html}
<div class="card"><details open><summary>{t("All results")}</summary><table>
<tr><th>test</th><th>config</th><th>value</th><th>unit</th><th>{t("Spec efficiency")}</th><th>{t("Measured efficiency")}</th><th>cv</th><th>peak GB</th></tr>
{trs}</table></details></div>
</div></body></html>"""
    with open(path, "w") as f:
        f.write(html)


