"""
torchpeak — cross-platform PyTorch benchmark (cross-stack, peak-normalized).

One codebase runs the same workloads on NVIDIA (CUDA) / AMD (ROCm) / Intel (XPU) /
Apple (MPS) / Huawei (NPU) / Moore Threads (MUSA) / CPU (or any backend implementing
the torch device API). The primary metric is software efficiency = measured / hardware
peak (spec and measured-ceiling denominators side by side); layered measurement
(feature probes → operator microbenchmarks → whole models) locates gaps in operators,
memory traffic, or compilation.

Tests (pure PyTorch; nothing to download except optional extras):
  probe           usability matrix (autocast dtypes / SDPA backends / GradScaler /
                  channels_last / int8 mm / compile fwd+bwd / pin_memory), pass rate
  gemm            square + LLM non-square matmul (fp32 / fp16 / bf16 TFLOPS, int8 TOPS),
                  adaptive 3/5/7 rounds by CV tier (STABLE <= 5% after outlier removal)
  vec             vector-ALU throughput: a compile-fused FMA chain; compare with gemm to tell
                  whether GEMM actually reaches matrix units
  bandwidth       read / write / copy / h2d / d2h GB/s, one L2-resident and one DRAM size
  memory_wall     CSAPP memory mountain: read throughput over working-set × stride (HTML heatmap;
                  opt-in via --tests)
  resnet          ResNet50 inference bs1 / bs32 img/s (needs torchvision)
  train           classic model training (SGD/AdamW + AMP, synthetic data):
                  resnet50, mobilenet_v3_large, efficientnet_b0, vit_b_16, vgg16 (torchvision),
                  bert_base (pure PyTorch, seq 128); batch halves automatically on OOM
  gpt             GPT-2-small-scale Transformer training + prefill tokens/s with a TFLOPS estimate
  llama           1B-class Llama (RMSNorm / SwiGLU / RoPE): long-sequence prefill + bs1 KV-cache decode
  moe             small sparse-MoE training: dynamic routing forces graph breaks / recompiles
  decode          GPT-2-medium bs1 KV-cache token decode tok/s + effective weight bandwidth (≈ llama.cpp tg)
  hf_decode       optional: HuggingFace small-LLM decode tok/s (needs transformers, use --hf-model)

compare_compile mode (default): AB/BA alternated eager / compile; dual numerical checks
(vs eager at same precision + vs fp32 reference); cold/hot compile time; peak memory
(allocated + reserved), median/CV/IQR, dynamo graph breaks and graph count, compile/eager
speedup (< 1 flagged REGRESS).

Usage:
  python -m torchpeak                         # full scale, eager/compile compare by default
  python -m torchpeak --quick                 # small scale, fast smoke
  python -m torchpeak --strict                # fixed 7 rounds, median + IQR dispersion
  python -m torchpeak --cooldown 5            # idle seconds between subtests (default 2)
  python -m torchpeak --strict-errors         # exit 1 on any failed/unsupported row
  python -m torchpeak --device cpu            # CPU baseline on the same machine
  python -m torchpeak --amp bf16 --compile    # switch precision / compile-only
  python -m torchpeak --amp all               # run model tests in FP32 / FP16 / BF16
  python -m torchpeak --compile-mode max-autotune   # default / max-autotune / reduce-overhead
  python -m torchpeak --tuned                 # tuned track: TF32, cudnn.benchmark, recorded in meta
  python -m torchpeak --peaks my.json         # extend/override the built-in hardware peak table
  python -m torchpeak --lang zh               # Chinese output (default auto: env / system locale)
  python -m torchpeak --hf-model Qwen/Qwen2.5-0.5B-Instruct
  python -m torchpeak --compare results/bench_*.json  # side-by-side across machines

Output: results/bench_<label>.json (machine-readable, meta.format_version marks the schema)
plus a same-named .html (self-contained report: usability matrix, efficiency bars,
memory-mountain heatmap, compile speedup table); --outdir changes the directory,
--no-report disables the HTML.
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

import torch
from .runtime import AMP, device_info, ensure_ocloc_on_path, pick_device, runtime_info
from .i18n import available_languages, detect, set_language, t
from .peaks import load_peak_entries, match_peak
from .harness import (ESCALATE_CV_PCT, OUTLIER_CAP, STABLE_CV_PCT, error_summary,
                      split_pair_config)
from .report import annotate_compile_speedup, annotate_efficiency, print_efficiency_summary, write_html_report
from .probes import test_probe
from .ops import test_bandwidth, test_gemm, test_memory_wall, test_vec
from .models import (_model_vram_estimate, test_gpt, test_gpt_decode, test_hf_decode,
                     test_llama, test_moe, test_resnet, test_train)

# ---------------------------------------------------------------- Compare mode
def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    labels = [r["meta"]["label"] for r in runs]
    keys = []
    for r in runs:
        for row in r["results"]:
            k = (row["test"], row["config"], row["unit"])
            if k not in keys:
                keys.append(k)
    lines = []
    for k in keys:
        bs_values = {row["bs_used"] for r in runs for row in r["results"]
                     if (row["test"], row["config"], row["unit"]) == k and row.get("bs_used") is not None}
        cells = []
        for r in runs:
            row = next((row for row in r["results"]
                        if (row["test"], row["config"], row["unit"]) == k), None)
            if row is None or row.get("value") is None:
                cells.append("-")
                continue
            cell = f"{row['value']:.2f}"
            if row.get("bs_used") is not None:
                cell += f" bs{row['bs_used']}" + ("!" if len(bs_values) > 1 else "")
            if row.get("eff_spec_pct") is not None:
                cell += f" ({row['eff_spec_pct']:.0f}{t('% peak')})"
            cells.append(cell)
        lines.append((k, cells))
    w = max([12] + [len(l) for l in labels] + [len(c) for _, cells in lines for c in cells])
    print(f"{'test':<16} {'config':<24} {'unit':<8}" + "".join(f"{l:>{w + 2}}" for l in labels))
    for k, cells in lines:
        print(f"{k[0]:<16} {k[1]:<24} {k[2]:<8}" + "".join(f"{c:>{w + 2}}" for c in cells))

    # compile speedup: pair compare_compile eager/compile rows by config suffix
    by_stem = {}
    for r in runs:
        lbl = r["meta"]["label"]
        for row in r["results"]:
            cfg = row.get("config") or ""
            stem, variant = split_pair_config(cfg)
            if stem is not None:
                by_stem.setdefault((row["test"], stem), {}).setdefault(lbl, {})[variant] = row
    header_printed = False
    for (test, stem), by_label in by_stem.items():
        cells, any_pair = [], False
        for r in runs:
            pair = by_label.get(r["meta"]["label"], {})
            c, e = pair.get("compile"), pair.get("eager")
            if c and e and c.get("value") and e.get("value"):
                sp = c["value"] / e["value"]
                cells.append(f"{sp:.2f}x" + ("!" if sp < 1 else ""))
                any_pair = True
            else:
                cells.append("-")
        if any_pair:
            if not header_printed:
                print(t("\n== compile speedup (! marks compile regressions)"))
                header_printed = True
            print(f"{test:<16} {stem:<24}" + "".join(f"{c:>{w + 2}}" for c in cells))
    print(t("Note: only rows with identical config (both --quick or both full scale, same precision) "
            "compare directly; train rows must also match bs. ! marks a differing batch or a compile "
            "fallback. (N% peak) is spec-denominator software efficiency."))

def _argv_lang(argv):
    """--lang must take effect before argparse is built (help text is localized too), so scan argv early."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def main():
    set_language(_argv_lang(sys.argv[1:]) or "auto")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", help=t("device selection: auto / cuda / xpu / mps / cpu"))
    p.add_argument("--amp", default="fp16",
                   help=t("model-test precision: fp32/fp16/bf16, comma-separated or all (default fp16)"))
    p.add_argument("--quick", action="store_true", help=t("small sizes, fast run"))
    p.add_argument("--strict", action="store_true",
                   help=t("strict mode: fixed 7 rounds, median + IQR dispersion, longer warm-up/timing"))
    p.add_argument("--cooldown", type=float, default=2.0,
                   help=t("idle seconds between subtests for DVFS/thermal recovery (0 disables)"))
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", action="store_true",
                               help=t("run resnet/train/gpt with torch.compile only"))
    compile_group.add_argument("--compare-compile", action="store_true", default=None,
                               help=t("AB/BA alternated eager vs compile compare (default on)"))
    compile_group.add_argument("--no-compare-compile", action="store_false", dest="compare_compile",
                               help=t("eager only, no compile compare"))
    p.add_argument("--tests", default="probe,gemm,vec,bandwidth,resnet,gpt,llama,moe,decode",
                   help=t("comma-separated: probe,gemm,vec,gemv,int8_gemm,gemv_int8,bandwidth,memory_wall,"
                          "resnet,train,gpt,llama,moe,decode,hf; gemm includes GEMV by default"))
    p.add_argument("--train-models", default="resnet50,mobilenet_v3_large,efficientnet_b0,vit_b_16,bert_base",
                   help=t("models for the train test, comma-separated or all (adds vgg16)"))
    p.add_argument("--train-bs", type=int, default=None,
                   help=t("unified train batch size (default per model, halves automatically on OOM)"))
    p.add_argument("--hf-model", default=None, help=t("optional: HF model name, enables the hf_decode test"))
    p.add_argument("--new-tokens", type=int, default=128, help=t("tokens to generate in decode tests"))
    p.add_argument("--min-time", type=float, default=None,
                   help=t("minimum timed seconds per round (default 1 s; 3 s under strict/compile-compare)"))
    p.add_argument("--label", default=None, help=t("result label (default hostname_device)"))
    p.add_argument("--outdir", default="results", help=t("output directory (default results, created if missing)"))
    p.add_argument("--peaks", default=None, metavar="JSON",
                   help=t("peak-table JSON path (entries take priority over the built-in table), "
                          "spec denominator of software efficiency"))
    p.add_argument("--compile-mode", default="default", choices=("default", "max-autotune", "reduce-overhead"),
                   help=t("torch.compile mode (with --compile / the default --compare-compile)"))
    p.add_argument("--tuned", action="store_true",
                   help=t("tuned track: apply one-line knobs (TF32, cudnn.benchmark, ...) and record them "
                          "in meta; default is the out-of-the-box track"))
    p.add_argument("--compare", nargs="+", metavar="JSON", help=t("compare several result files side by side"))
    p.add_argument("--no-report", action="store_true", help=t("skip the HTML report (JSON is still written)"))
    p.add_argument("--strict-errors", action="store_true",
                   help=t("exit 1 on any failed/unsupported row; internal errors always exit 2"))
    p.add_argument("--no-vram-check", action="store_true",
                   help=t("disable the pre-run VRAM check (by default oversized model tests are downgraded "
                          "to quick presets based on total device memory)"))
    p.add_argument("--lang", default="auto", choices=["auto"] + available_languages(),
                   help=t("output language: auto / en / zh (auto = $TORCHPEAK_LANG, then system locale)"))
    a = p.parse_args()

    if a.compare:
        compare(a.compare)
        return

    if a.compare_compile is None:
        a.compare_compile = not a.compile

    amp_names = ["fp32", "fp16", "bf16"] if a.amp == "all" else [v.strip() for v in a.amp.split(",")]
    if not amp_names or any(v not in AMP for v in amp_names):
        p.error(t("--amp must be fp32/fp16/bf16, a comma-separated list, or all"))
    amp_names = list(dict.fromkeys(amp_names))

    if a.min_time is None:
        a.min_time = 3.0 if a.strict or a.compare_compile else 1.0

    dev = pick_device(a.device)
    ocloc_shim = ensure_ocloc_on_path() if dev.type == "xpu" else None
    try:  # silence torch's internal glog-style ERROR spam (compile failures dump a full screen of traceback); per-row errors still go to JSON
        import logging
        torch._logging.set_logs(inductor=logging.CRITICAL, dynamo=logging.ERROR)
    except Exception:
        pass
    tuned_applied = []
    if a.tuned:
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            tuned_applied += ["cuda.matmul.allow_tf32=True", "cudnn.allow_tf32=True"]
        torch.backends.cudnn.benchmark = True
        tuned_applied.append("cudnn.benchmark=True")
    info = device_info(dev)
    label = a.label or f"{socket.gethostname()}_{dev.type}"
    meta = dict(format_version=2, label=label, torch=torch.__version__, python=sys.version.split()[0], platform=platform.platform(),
                amp=a.amp, amp_list=amp_names, quick=a.quick, strict=a.strict, compile=a.compile,
                compare_compile=a.compare_compile, args=vars(a).copy(), num_threads=torch.get_num_threads(),
                num_interop_threads=torch.get_num_interop_threads(), runtime=runtime_info(), **info)
    meta["tuned"] = {"enabled": a.tuned, "applied": tuned_applied}
    meta["stability_policy"] = dict(stable_cv_pct=STABLE_CV_PCT, escalate_cv_pct=ESCALATE_CV_PCT,
                                    rounds=[3, 5, 7], outlier_rule="Iglewicz-Hoaglin modified z>3.5",
                                    outlier_cap={str(k): v for k, v in OUTLIER_CAP.items()},
                                    escalation=("n=3 stops at CV<=5%; >5% extends to 5; still >5% "
                                                "after outlier removal extends to 7; initial CV>10% "
                                                "extends to 7 directly"),
                                    cooldown_s=a.cooldown)
    meta["runtime"]["ocloc_shim"] = ocloc_shim
    compile_mode = "compare" if a.compare_compile else str(a.compile).lower()
    print(f"== {label}: {info['name']} [{info['backend']}]  torch {torch.__version__}"
          f"  amp={a.amp} quick={a.quick} strict={a.strict} compile={compile_mode}")
    if ocloc_shim:
        print(t("== ocloc shim: {path} (Triton-XPU needs the bare name ocloc)", path=ocloc_shim))
    if a.tuned:
        print(t("== tuned track enabled: {knobs}", knobs=tuned_applied))

    tests = set(a.tests.split(","))
    if a.hf_model:
        tests.add("hf")

    # VRAM preflight: full-scale runs OOM whole pairs on low-memory cards (a compare pair keeps ~3 fp32
    # instances alive), so estimate from total device memory first and downgrade models that don't fit to quick.
    a.quick_tests = set()
    vram_check = {"total_gb": info.get("total_mem_gb"), "threshold_gb": None, "downgraded": {}, "notes": []}
    vram = info.get("total_mem_gb")
    if vram and not a.no_vram_check:
        vram_check["threshold_gb"] = round(vram * 0.85, 1)
        for name in ("llama", "gpt", "moe", "decode", "resnet", "train"):
            if name not in tests:
                continue
            est = _model_vram_estimate(name, a.quick, a.compare_compile)
            if est <= vram * 0.85:
                continue
            if not a.quick and name in ("llama", "gpt", "moe"):
                a.quick_tests.add(name)
                vram_check["downgraded"][name] = {"estimate_gb": round(est, 1), "preset": "quick"}
                print(t("== VRAM pre-check: {test} full run needs ~{est:.1f} GB > {vram} GB × 85%, "
                        "downgraded to the quick preset (--no-vram-check disables)",
                        test=name, est=est, vram=vram), flush=True)
            else:
                vram_check["notes"].append(f"{name}: estimate ~{est:.1f} GB vs device {vram} GB, may OOM")
                print(t("== VRAM pre-check: warning {test} estimate ~{est:.1f} GB vs {vram} GB, may OOM",
                        test=name, est=est, vram=vram), flush=True)
    meta["vram_check"] = vram_check

    results = []
    if "probe" in tests:
        test_probe(results, dev, a)
        passed = sum(1 for r in results if r["test"] == "probe" and r.get("status") == "pass")
        total = sum(1 for r in results if r["test"] == "probe")
        meta["usability"] = {"passed": passed, "total": total,
                             "pass_rate_pct": round(passed / total * 100, 1) if total else None}
    if tests.intersection(("gemm", "gemv", "int8_gemm", "gemv_int8")):
        test_gemm(results, dev, a)
    if "vec" in tests:
        test_vec(results, dev, a)
    if "bandwidth" in tests:
        test_bandwidth(results, dev, a)
    if "memory_wall" in tests:
        test_memory_wall(results, dev, a)
    for amp in amp_names:
        a.amp = amp
        if "resnet" in tests:
            test_resnet(results, dev, a)
        if "train" in tests:
            test_train(results, dev, a)
        if "gpt" in tests:
            test_gpt(results, dev, a)
        if "llama" in tests:
            test_llama(results, dev, a)
        if "moe" in tests:
            test_moe(results, dev, a)
        if "decode" in tests:
            test_gpt_decode(results, dev, a)
        if "hf" in tests and a.hf_model:
            test_hf_decode(results, dev, a)

    peak_entries, peaks_source = load_peak_entries(a.peaks)
    spec = match_peak(info, peak_entries)
    meta["peaks_source"] = peaks_source
    print(t("== peak table: {source}", source=peaks_source or t("not found; measured-ceiling denominator only")))
    measured = annotate_efficiency(results, spec)
    annotate_compile_speedup(results)
    meta["peak_spec"] = spec
    meta["peak_measured"] = {f"{kind}:{key or 'all'}": round(v, 3) for (kind, key), v in measured.items()}
    meta["errors"] = error_summary(results)

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"bench_{label}.json")
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    print_efficiency_summary(results, spec)
    print("\n" + t("results written to {path}", path=out))
    if not a.no_report:
        try:
            html_path = out.rsplit(".", 1)[0] + ".html"
            write_html_report(meta, results, html_path)
            print(t("report written to {path}", path=html_path))
        except Exception as e:
            print(t("HTML report generation failed (JSON unaffected): {err}",
                    err=f"{type(e).__name__}: {e}"))

    if meta["errors"]["internal"]:
        print(t("{n} internal error(s) found, see the error/error_kind fields in the JSON",
                n=meta["errors"]["internal"]), file=sys.stderr)
        return 2
    if a.strict_errors and meta["errors"]["total"]:
        print(t("strict-errors mode: {n} failed/unsupported row(s)", n=meta["errors"]["total"]),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
