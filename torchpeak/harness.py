"""torchpeak.harness — see the package root docstring."""
import contextlib
import os
import shutil
import statistics
import tempfile
import time

import torch.nn as nn
import torch
from .i18n import t
from .runtime import AMP, empty_cache, peak_gb, peak_reserved_gb, reset_peak, sync

# ---------------------------------------------------------------- stability policy
STABLE_CV_PCT = 5.0     # STABLE after outlier handling; above this the row is tagged UNSTABLE
ESCALATE_CV_PCT = 10.0  # initial-CV tiers: <=5% stop at n=3, 5-10% extend to n=5, >10% extend to n=7
OUTLIER_CAP = {3: 0, 5: 1, 7: 2}  # max points removed per round count (never at n=3)


def rounds_for_cv(cv_pct):
    """Round count for a first-pass CV of n=3: the adaptive escalation tiers."""
    if cv_pct <= STABLE_CV_PCT:
        return 3
    if cv_pct <= ESCALATE_CV_PCT:
        return 5
    return 7


def mad_outliers(values, cap):
    """Indices of statistical outliers by Iglewicz-Hoaglin modified z-score (0.6745*|x-med|/MAD > 3.5).
    At most `cap` indices (farthest from the median first); MAD=0 falls back to meanAD; no data at n<4."""
    if len(values) < 4 or cap <= 0:
        return []
    med = statistics.median(values)
    devs = [abs(v - med) for v in values]
    mad = statistics.median(devs) or statistics.mean(devs)  # majority-identical fallback
    if mad == 0:
        return []
    order = sorted(range(len(values)), key=lambda i: -devs[i])
    return [i for i in order[:cap] if 0.6745 * devs[i] / mad > 3.5]


def is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "out_of_device_memory" in s or "out_of_host_memory" in s


def classify_error(e):
    if isinstance(e, (NameError, AttributeError, TypeError, SyntaxError)):
        return "internal"
    if isinstance(e, (ImportError, NotImplementedError)):
        return "unsupported"
    if is_oom(e):
        return "resource"
    if isinstance(e, AssertionError):
        return "numerical"
    return "runtime"


def error_fields(e, limit=200):
    return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:limit]}",
            "error_kind": classify_error(e)}


def error_summary(results):
    counts = {}
    for row in results:
        if row.get("error"):
            kind = row.get("error_kind", "runtime")
            counts[kind] = counts.get(kind, 0) + 1
    return {"total": sum(counts.values()), "by_kind": counts,
            "internal": counts.get("internal", 0)}

def compile_modes(a):
    return (False, True) if a.compare_compile else (a.compile,)


def tcompile(m, a):
    """Compile per --compile-mode; default is torch.compile's default mode."""
    if getattr(a, "compile_mode", "default") == "default":
        return torch.compile(m)
    return torch.compile(m, mode=a.compile_mode)


def compile_config(config, a, enabled):
    if not a.compare_compile:
        return config
    tag = "compile"
    if enabled and getattr(a, "compile_mode", "default") != "default":
        tag += f"-{a.compile_mode}"
    return f"{config} {tag if enabled else 'eager'}"


def split_pair_config(cfg):
    """Split a compare_compile config into (stem, variant); variant is eager or compile[-mode]."""
    stem, _, tag = cfg.rpartition(" ")
    if tag == "eager":
        return stem, "eager"
    if tag.startswith("compile"):
        return stem, "compile"
    return None, None


def model_precision(amp):
    return "fp32" if amp == "fp32" else f"{amp} autocast"


def autocast(dev, amp):
    if amp == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev.type, dtype=AMP[amp])


def timeit(fn, dev, warmup=3, min_time=1.0, max_iters=200):
    """Return mean time per call (seconds). Warm up first, then grow the iteration count until the total time >= min_time."""
    for _ in range(warmup):
        fn()
    sync(dev)
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync(dev)
        dt = time.perf_counter() - t0
        if dt >= min_time or n >= max_iters:
            return dt / n
        n = min(max_iters, max(n * 2, int(n * min_time / max(dt, 1e-6)) + 1))


# ---------------------------------------------------------------- Result recording
def measure(results, dev, test, config, unit, fn, strict=False, repeats=1, cooldown=0.0):
    reset_peak(dev)
    print_row_head(test, config)
    try:
        need = 7 if strict else max(repeats, 3)  # strict = fixed 7 rounds; default escalates by CV tier
        trials = []
        while len(trials) < need:
            out = fn()
            extra = {}
            if isinstance(out, tuple):
                out, extra = out
            trials.append((out, extra))
            if not strict and len(trials) == 3:
                cv = coefficient_variation([v for v, _ in trials])
                need = rounds_for_cv(cv)
                if need > 3:
                    print(f" n→{need}", end="", flush=True)  # inline: rounds extended, detail in the tail
            elif not strict and len(trials) == 5 and need == 5:
                # provisional n=5 verdict: still >5% after outlier removal -> grow to n=7
                vals = [v for v, _ in trials]
                drop = set(mad_outliers(vals, OUTLIER_CAP[5]))
                kept = [v for i, v in enumerate(vals) if i not in drop]
                if coefficient_variation(kept) > STABLE_CV_PCT:
                    need = 7
                    print(" n→7", end="", flush=True)
        values = [out for out, _ in trials]
        drop = set(mad_outliers(values, OUTLIER_CAP.get(len(values), 2)))
        kept = [v for i, v in enumerate(values) if i not in drop]
        med = statistics.median(kept)
        out_idx = min((i for i in range(len(values)) if i not in drop), key=lambda i: abs(values[i] - med))
        r = dict(test=test, config=config, value=round(med, 3), unit=unit, peak_mem_gb=peak_gb(dev),
                 peak_reserved_gb=peak_reserved_gb(dev), **trials[out_idx][1])
        r["trials"] = [round(v, 3) for v in values]
        r["n_used"] = len(kept)
        if drop:
            r["outliers"] = [round(values[i], 3) for i in sorted(drop)]
        mean = statistics.mean(kept)
        r["cv_pct"] = round(statistics.stdev(kept) / mean * 100, 2) if mean else 0.0
        r["stable"] = r["cv_pct"] <= STABLE_CV_PCT
        r.update(dispersion(kept))
    except Exception as e:
        r = dict(test=test, config=config, value=None, unit=unit, **error_fields(e))
    results.append(r)
    print_row(r, head="")
    empty_cache(dev)
    if cooldown:
        time.sleep(cooldown)

_row_pending = False


def print_row_head(test, config, pad=24):
    """Print the test/config columns when a subtest starts; the value tail completes the line later."""
    global _row_pending
    print(f"  {test:<16} {config:<{pad}}", end="", flush=True)
    _row_pending = True


def _end_row_pending():
    global _row_pending
    if _row_pending:
        print()
        _row_pending = False


def print_row(r, head=None):
    """Print a result row. head=None prints the full line (terminating any dangling row head first);
    head=str completes a line started by print_row_head with that extra config text."""
    global _row_pending
    if head is None:
        _end_row_pending()
    else:
        print(head, end="")
        _row_pending = False
    v = "ERROR" if r.get("value") is None else f"{r['value']:.2f}"
    extra = ""
    if "cv_pct" in r:
        n_used = r.get("n_used", len(r["trials"]))
        removed = len(r["trials"]) - n_used
        extra += f"  cv {r['cv_pct']:.2f}% n={n_used}" + (f"(-{removed})" if removed else "")
        if "iqr_pct" in r:
            extra += f" iqr {r['iqr_pct']:.2f}%"
        if not r["stable"]:
            extra += " UNSTABLE"
    if "bs_used" in r:
        extra += f"  bs{r['bs_used']}"
    if "approx_tflops" in r:
        extra += f"  (~{r['approx_tflops']:.2f} TFLOPS)"
    if "approx_weight_bw_gbs" in r:
        extra += f"  (~{r['approx_weight_bw_gbs']:.1f} GB/s weights)"
    if "vs_torch_path_ratio" in r:
        extra += f"  vs torch x{r['vs_torch_path_ratio']}"
    if "compile_cold_s" in r:
        extra += f"  compile cold {r['compile_cold_s']:.1f}s hot {r['compile_hot_s']:.1f}s"
    elif "compile_time_s" in r:
        extra += f"  compile {r['compile_time_s']:.2f}s"
    if "dynamo_graph_breaks" in r:
        extra += f"  breaks {r['dynamo_graph_breaks']} graphs {r['dynamo_unique_graphs']}"
    if "mem_ratio_vs_eager" in r and r["mem_ratio_vs_eager"]:
        extra += f"  mem x{r['mem_ratio_vs_eager']}"
    if r.get("numerical_check") == "passed":
        extra += "  numeric OK"
    if r.get("peak_mem_gb") is not None:
        extra += f"  peak {r['peak_mem_gb']} GB"
        if r.get("peak_reserved_gb") is not None:
            extra += f" (+{r['peak_reserved_gb']} res)"
    if r.get("error"):
        extra += f"  [{r.get('error_kind', 'runtime')}] {r['error']}"
    if head is None:
        print(f"  {r['test']:<16} {r['config']:<24} {v:>10} {r['unit']:<8}{extra}", flush=True)
    else:
        print(f" {v:>10} {r['unit']:<8}{extra}", flush=True)


def coefficient_variation(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean * 100 if mean else 0.0


def compare_numerics(reference, candidate, amp):
    tolerances = {"fp32": (1e-4, 1e-4), "fp16": (2e-2, 1e-1), "bf16": (5e-2, 5e-1)}
    rtol, atol = tolerances[amp]
    if len(reference) != len(candidate):
        raise AssertionError(t("numerical check tensor count mismatch: {got} != {want}",
                               got=len(reference), want=len(candidate)))
    max_abs, max_rel = 0.0, 0.0
    for ref, out in zip(reference, candidate):
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, equal_nan=True)  # matching NaNs on both sides count as equal
        ref_float, out_float = ref.detach().float(), out.detach().float()
        diff = (out_float - ref_float).abs()
        if diff.numel():
            max_abs = max(max_abs, diff.max().item())
            max_rel = max(max_rel, (diff / ref_float.abs().clamp_min(atol)).max().item())
    return max_abs, max_rel, rtol, atol


def measure_compile_pair(results, dev, test, config, unit, make_case, scale, a,
                         extra_fn=None, allow_oom_retry=False):
    while True:
        retry = False
        try:
            _measure_compile_pair_once(results, dev, test, config, unit, make_case, scale, a,
                                       extra_fn, allow_oom_retry)
            empty_cache(dev)
            if a.cooldown:
                time.sleep(a.cooldown)
            return
        except Exception as e:
            if allow_oom_retry and scale > 1 and is_oom(e):
                retry = True
                scale //= 2
            else:
                for enabled in (False, True):
                    row = dict(test=test, config=compile_config(config, a, enabled), value=None, unit=unit,
                               **error_fields(e))
                    if allow_oom_retry:
                        row["bs_used"] = scale
                    results.append(row)
                    print_row(row)
                empty_cache(dev)
                if a.cooldown:
                    time.sleep(a.cooldown)
                return
        if retry:
            empty_cache(dev)
            _end_row_pending()
            print(t("  {config}: OOM, retrying the whole pair at scale {scale}",
                    config=config, scale=scale), flush=True)


def dispersion(values):
    """Interquartile range (reproducibility spec: report median and IQR; empty when too few samples)."""
    if len(values) < 4:
        return {}
    q1, _, q3 = statistics.quantiles(values, n=4)
    med = statistics.median(values)
    return {"q1": round(q1, 3), "q3": round(q3, 3), "iqr_pct": round((q3 - q1) / med * 100, 2) if med else None}


def dynamo_counters_snapshot():
    """torch._dynamo counter snapshot: graph breaks / frame coverage / graph count (fallback-rate metrics)."""
    try:
        from torch._dynamo.utils import counters
        stats = counters.get("stats", {})
        frames = counters.get("frames", {})
        breaks = sum(counters.get("graph_break", {}).values()) + sum(counters.get("unimplemented", {}).values())
        return {"graph_breaks": breaks, "unique_graphs": stats.get("unique_graphs", 0),
                "frames_total": frames.get("total", 0), "frames_ok": frames.get("ok", 0)}
    except Exception:
        return {}


def dynamo_counters_delta(before):
    after = dynamo_counters_snapshot()
    return {f"dynamo_{k}": after.get(k, 0) - before.get(k, 0) for k in ("graph_breaks", "unique_graphs",
                                                                        "frames_total", "frames_ok")}


def _measure_compile_pair_once(results, dev, test, config, unit, make_case, scale, a,
                               extra_fn, allow_oom_retry):
    print_row_head(test, config, pad=18)  # config + " eager"/" compile" lands at the usual column
    seed = 123456789
    fp32_ref = None
    if a.amp != "fp32":
        # Primary numerical-check baseline: fp32 reference (same seed, same inputs) measuring the low-precision stack's own error
        saved_amp = a.amp
        try:
            a.amp = "fp32"
            torch.manual_seed(seed)
            _, fp32_check = make_case(False, scale)
            fp32_ref = fp32_check()
            sync(dev)
            del _, fp32_check
            empty_cache(dev)
        finally:
            a.amp = saved_amp

    torch.manual_seed(seed)
    eager_step, eager_check = make_case(False, scale)
    torch.manual_seed(seed)
    compile_step, compile_check = make_case(True, scale)
    torch.manual_seed(seed)
    reference = eager_check()
    sync(dev)

    # Dual compile-time scopes: cold = first step with a brand-new Inductor cache dir; hot = first step of recompiling a wrapper in the same dir
    prev_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    cache_dir = tempfile.mkdtemp(prefix="torchpeak_inductor_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    dyn_before = dynamo_counters_snapshot()
    try:
        torch.manual_seed(seed)
        reset_peak(dev)
        sync(dev)
        t0 = time.perf_counter()
        candidate = compile_check()
        sync(dev)
        compile_cold = time.perf_counter() - t0

        torch.manual_seed(seed)
        hot_step, hot_check = make_case(True, scale)
        torch.manual_seed(seed)
        sync(dev)
        t0 = time.perf_counter()
        hot_check()
        sync(dev)
        compile_hot = time.perf_counter() - t0
        del hot_step, hot_check

        max_abs, max_rel, rtol, atol = compare_numerics(reference, candidate, a.amp)
        has_nan = any(torch.isnan(t).any().item() for t in reference
                      if torch.is_tensor(t) and t.is_floating_point())
        fp32_abs = fp32_rel = None
        fp32_status = None
        if fp32_ref is not None:
            try:  # upcast low-precision outputs to fp32 before comparing; a failure is recorded, not fatal
                fp32_abs, fp32_rel, _, _ = compare_numerics(fp32_ref, [t.float() for t in reference], a.amp)
                fp32_status = "passed"
            except AssertionError as e:
                fp32_status = f"failed: {str(e)[:120]}"
        del reference, candidate, fp32_ref

        # Run a full step on its own so training peak includes optimizer state and parameter updates, not just the check's fwd/bwd.
        reset_peak(dev)
        eager_step()
        sync(dev)
        eager_peak, eager_peak_res = peak_gb(dev), peak_reserved_gb(dev)
        reset_peak(dev)
        compile_step()
        sync(dev)
        compile_peak, compile_peak_res = peak_gb(dev), peak_reserved_gb(dev)

        for i in range(10):  # interleaved warmup; both sides are done compiling/tracing, so no first-step special case
            steps = (eager_step, compile_step) if i % 2 == 0 else (compile_step, eager_step)
            for step in steps:
                step()
        sync(dev)

        eager_values, compile_values, orders = [], [], []
        while len(orders) < 9:
            if len(orders) % 2 == 0:
                eager_t = timeit(eager_step, dev, warmup=0, min_time=a.min_time)
                compile_t = timeit(compile_step, dev, warmup=0, min_time=a.min_time)
                orders.append("AB")
            else:
                compile_t = timeit(compile_step, dev, warmup=0, min_time=a.min_time)
                eager_t = timeit(eager_step, dev, warmup=0, min_time=a.min_time)
                orders.append("BA")
            eager_values.append(scale / eager_t)
            compile_values.append(scale / compile_t)
            if len(orders) >= 5:
                stable = (coefficient_variation(eager_values) <= 2.0
                          and coefficient_variation(compile_values) <= 2.0)
                if stable:
                    break

        stable = (coefficient_variation(eager_values) <= 2.0
                  and coefficient_variation(compile_values) <= 2.0)
        dyn_delta = dynamo_counters_delta(dyn_before)
        for enabled, values in ((False, eager_values), (True, compile_values)):
            value = statistics.median(values)
            extra = extra_fn(value) if extra_fn else {}
            if allow_oom_retry:
                extra["bs_used"] = scale
            row = dict(test=test, config=compile_config(config, a, enabled), value=round(value, 3), unit=unit,
                       peak_mem_gb=eager_peak if not enabled else compile_peak,
                       peak_reserved_gb=eager_peak_res if not enabled else compile_peak_res,
                       trials=[round(v, 3) for v in values], cv_pct=round(coefficient_variation(values), 2),
                       **dispersion(values),
                       stable=stable, ab_order=orders, seed=seed, warmup_steps=10,
                       round_min_time_s=a.min_time, cv_threshold_pct=2.0, numerical_check="passed",
                       numerical_randomness="disabled for check only",
                       numerical_max_abs_error=max_abs, numerical_max_rel_error=max_rel,
                       numerical_rtol=rtol, numerical_atol=atol,
                       numerical_nan_outputs=has_nan, **extra)
            if not enabled and fp32_status is not None:
                row["numerical_vs_fp32"] = fp32_status
                row["numerical_fp32_max_abs_error"] = fp32_abs
                row["numerical_fp32_max_rel_error"] = fp32_rel
            if enabled:
                row["compile_cold_s"] = round(compile_cold, 3)
                row["compile_hot_s"] = round(compile_hot, 3)
                row["compile_time_scope"] = ("cold=fresh Inductor cache dir, first step "
                                             "(includes the check step's backward); "
                                             "hot=second compile of a wrapper in the same dir, first step")
                if compile_peak is not None and eager_peak:
                    row["mem_ratio_vs_eager"] = round(compile_peak / eager_peak, 3)
                row.update(dyn_delta)
            results.append(row)
            print_row(row, head=" eager" if not enabled else None)
    finally:
        if prev_cache is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = prev_cache
        shutil.rmtree(cache_dir, ignore_errors=True)

