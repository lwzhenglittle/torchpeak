# torchpeak

**One command, every PyTorch backend: operator throughput, memory bandwidth, model-level speed,
and eager-vs-`torch.compile` A/B — each number normalized to the hardware's peak rate.**

torchpeak runs an identical, fully synthetic workload set on NVIDIA (CUDA), AMD (ROCm), Intel (XPU),
Apple (MPS), Huawei Ascend (NPU), Moore Threads (MUSA), CPU — or any other backend that
implements the torch device API — and reports throughput two ways: absolute
(TFLOPS / GB/s / img/s / tok/s) and **efficiency = measured / hardware peak**, using two denominators
(vendor spec table *and* this run's measured ceiling) so results stay comparable even where specs
are missing or optimistic. Tests are layered — usability probes → operator microbenchmarks → whole
models — so a slow row can be traced back to the operator library, memory traffic, or compilation.

## Quickstart

Requirements: Python ≥ 3.9, PyTorch ≥ 2.4 (any official wheel, any backend). Nothing else —
all weights and inputs are synthesized with fixed seeds, so there are **no downloads and no logins**.

```bash
pip install -e .          # from a checkout; installs the `torchpeak` entry point
torchpeak --quick         # small sizes, ~5–10 min; picks the GPU automatically, falls back to CPU
```

Without installing, run from the repo root: `python -m torchpeak --quick`.

While it runs you'll see the usability matrix first, then one line per measurement:

```text
== usability matrix: 11/13 available
   ✓ autocast_fp16  autocast_bf16  bf16_matmul  sdpa_flash  sdpa_math  grad_scaler
   ✗ sdpa_cudnn — RuntimeError: No viable backend for scaled_dot_product_attention ...
  gemv             fp32 m1 k1024 n1024          232.38 GB/s      cv 41.83% n=3 UNSTABLE
  ...
== Software efficiency (measured throughput / peak, primary metric)
   device_bw:all       best      232.4   n/a spec 100.0% meas
results written to results/bench_<host>_<device>.json
report written to results/bench_<host>_<device>.html
```

When it finishes:

- open `results/bench_<host>_<device>.html` — a self-contained report (usability matrix,
  efficiency bars, memory-mountain heatmap, compile-speedup table); a real
  [GTX 1660 Ti run](results/community/gtx1660ti.json) is in the community corpus (JSON —
  the HTML report regenerates from it)
- `results/bench_<host>_<device>.json` — machine-readable rows with per-row trials,
  efficiency fields and raw errors (`meta.format_version` marks the schema)

Exit codes: `0` clean, `1` with `--strict-errors` when any row failed, `2` on internal
errors (JSON/HTML are written before any non-zero exit).

Common next steps:

```bash
python -m torchpeak --device cpu            # CPU baseline on the same machine
python -m torchpeak --strict                # fixed 7 rounds, median + IQR
python -m torchpeak --amp all               # fp32 / fp16 / bf16 for model tests
python -m torchpeak --compare results/bench_*.json   # side-by-side across runs/machines
```

## What it measures

| Layer | Tests | What they exercise |
| --- | --- | --- |
| 1. Usability probes | `probe` — 13 one-shot checks: autocast dtypes, SDPA backends, GradScaler, channels_last, int8 mm, compile fwd/bwd, pin_memory | Which code paths exist and run at all; pass rate plus per-feature error text |
| 2. Operators | `gemm` (fp32/fp16/bf16/int8; a small square, a large square and an LLM non-square shape), `gemv` (reported as GB/s — it is bandwidth-bound), `vec` (vector-ALU FMA chain; torch path **and** a raw-Triton packed-fp16 kernel), `bandwidth` (read/write/copy + h2d/d2h, one L2-resident and one DRAM size) | Whether the operator library reaches tensor cores, packed math and DRAM; separating compute-bound from bandwidth-bound behavior |
| 3. Models | `resnet` (infer), `gpt` (train + prefill), `llama` (1B-class RMSNorm/SwiGLU/RoPE: long-context prefill + bs1 KV-cache decode), `moe` (top-k sparse routing → dynamic shapes, graph breaks), `decode` (GPT-2-medium bs1 KV-cache decode ≈ llama.cpp tg), `hf_decode` (optional, any HF model via `--hf-model`); opt-in via `--tests`: `train` (5-model CV zoo), `memory_wall` (CSAPP memory mountain) | End-to-end throughput; per-test VRAM estimates with automatic downgrade to quick presets when the device is too small, OOM auto-halves the batch |

## Measurement methodology

**Timing.** Every row: fixed warm-up, then adaptive iteration count until the accumulated time
exceeds `--min-time` (1 s default, 3 s under `--strict`/compile-compare). Rounds are adaptive: a
3-round baseline is extended to 5 rounds when CV exceeds 5%, and to 7 rounds when the 5-round result
is still above 5% after outlier removal (an initial CV above 10% goes straight to 7); at most one
outlier (two at n=7) is removed by the Iglewicz-Hoaglin rule before the median, CV and IQR are
recomputed — rows still above CV 5% are tagged `UNSTABLE`, and removed points stay visible in the
JSON (`trials`, `n_used`, `outliers`). `--strict` runs a fixed 7 rounds. A `--cooldown` idle gap
(default 2 s) separates subtests so DVFS/thermal state from one subtest does not depress the next.
Vector-ALU tests spin ≥ 2 s before timing so shared-power iGPUs reach steady clocks (short kernels
otherwise swing ±40% with DVFS state).

**compile A/B (default on).** Eager vs `torch.compile` with AB/BA alternation to cancel ordering
effects. Per pair it records: speedup (`REGRESS` when < 1.0), cold-cache vs hot-cache compile time,
peak memory (allocated + reserved, sampled over a complete step including optimizer state), and
`torch._dynamo` counters (graph breaks, unique graphs, frame coverage).

**Numerics.** Dual-reference checks: candidate vs eager at the same precision, *and* vs an fp32
reference, with per-dtype tolerances. Identical NaN on both sides passes and is flagged separately
(`numerical_nan_outputs`).

**Peak normalization.** Efficiency uses two denominators:

- *spec* — `peaks.json` shipped next to the package (entries carry `basis` and `sources`).
  `--peaks FILE` prepends overrides; `$TORCHPEAK_PEAKS` relocates the table. Matching is
  ordered substring on the device name, so specific entries (`a100-sxm4-80gb`) win over generic
  ones (`a100`). Peak-table fixes and new devices are plain JSON PRs — no code change.
- *measured ceiling* — the best value this run achieved per (metric, dtype), so unknown devices
  still get a consistent column. Device DRAM bandwidth and host-interconnect bandwidth
  (`host_bw_gbs`) are separate dimensions; H2D/D2H rows are never divided by the DRAM peak.
  The `device_bw` ceiling is an envelope over STREAM ops (read/write/copy) at DRAM-regime sizes
  only — L2-resident sizes, GEMV, the memory mountain and model-level weight bandwidth keep their
  per-row ratios but never raise the ceiling.

**Accounting details that matter.**

- fp16/bf16 elementwise math is promoted to fp32 under torch `opmath` semantics (verified in
  Triton TTIR), so `vec` reports two rows per dtype: the torch path (fp32-rate ceiling) and a
  `triton-native` kernel that keeps arithmetic in fp16 to expose the packed-half2 rate
  (Arc B390: 6.4 torch vs 13.4 native vs 15.36 packed peak; GTX 1660 Ti: 10.9 native = 100% of spec).
- The memory-mountain headline is the largest-working-set stride-1 cell (DRAM regime); smaller
  cache-resident cells stay in the heatmap but never serve as the DRAM-efficiency numerator.
- The bandwidth headline (`device_bw`) only counts read/write/copy at the DRAM size for the same
  reason — the 16 MiB (L2-regime) row can exceed the ceiling and that gap is the cache effect.
- GEMV is scored in GB/s over bytes moved (read W + read x + write y), not TFLOPS.

## CLI reference

```text
--device auto|cuda|xpu|mps|cpu     device selection (default auto)
--amp fp32|fp16|bf16|all           model-test precision (default fp16)
--quick                            small sizes, fast smoke run
--strict / --strict-errors         fixed 7 rounds + IQR / exit 1 on any error row
--cooldown SEC                     idle seconds between subtests (default 2, 0 disables)
--compile / --compare-compile / --no-compare-compile
                                   compile-only / AB-BA pair (default) / eager-only
--compile-mode default|max-autotune|reduce-overhead
--tests gemm,vec,bandwidth,...     subset selection (see --help for the full list)
--train-models resnet50,...        train-zoo subset, or all (+vgg16)
--train-bs N                       unified batch size (default per model, OOM halves)
--tuned                            apply one-line knobs (TF32, cudnn.benchmark), recorded in meta
--peaks my.json / $TORCHPEAK_PEAKS spec-peak table override
--hf-model Qwen/Qwen2.5-0.5B-Instruct   optional real-weight decode row
--label / --outdir / --no-report   output naming; --compare a.json b.json for multi-run tables
--lang auto|en|zh                 output language ($TORCHPEAK_LANG, then system locale)
```

Each error row carries `error_kind` (`internal`, `unsupported`, `resource`, `numerical`,
`runtime`, `usage`); `meta.errors` aggregates them. Unsupported hardware features are recorded
without failing the run; internal errors always exit 2.

Console messages and the HTML report follow `--lang` (default auto: `$TORCHPEAK_LANG`, then the
system locale — `zh*` locales get Chinese, everything else English). Messages are plain-JSON
catalogs in `torchpeak/locales/` — adding a language is a data-file PR, and missing keys fall
back to English.

## Compatibility status

Main tests use only torch operators and the standard per-backend module interface, so any backend
that implements the torch device API can run them; missing backend-specific capabilities (Triton
target, `torch._int_mm`, Inductor) degrade to recorded error rows, not a crashed run.

| Backend | Status |
| --- | --- |
| CUDA / ROCm | tested (GTX 1660 Ti, cu132; ROCm via `cuda` device type) |
| Intel XPU | tested (Arc B390 iGPU; auto adds a user-level `ocloc` shim when the distro ships only `ocloc-<ver>`, recycled on exit incl. SIGTERM) |
| CPU | tested (launch path, torch.compile C++ backend, honest per-row errors where the stack can't do fp16 backward) |
| MPS / NPU / MUSA | code paths present, not yet hardware-verified |

Methodology and implementation details: [`docs/design.md`](docs/design.md) —
中文版：[`docs/design-zh.md`](docs/design-zh.md).

## 中文摘要

torchpeak 是跨平台 PyTorch 性能基准：同一份 torch 代码跑遍 CUDA / ROCm / XPU / MPS / NPU / MUSA / CPU
及其他任何能跑 torch 的设备，主指标是**软件效率 = 实测吞吐 / 硬件峰值**（规格口径 + 实测上限口径并列）。
三层测量：功能可用性
矩阵、算子级微基准（含 CSAPP memory mountain 与 torch 路径 vs 原生 Triton 的向量算力对照）、整模型
（含 1B Llama、稀疏 MoE、KV-cache 解码，全部合成数据、零下载）。默认 AB/BA 对比 eager 与
torch.compile，记录加速比、冷/热编译时间、显存、数值双口径与 graph break。稳定性策略：3 轮
基线按 CV 分档扩到 5/7 轮，Iglewicz-Hoaglin 剔离群点后 CV≤5% 记 STABLE；子项间默认冷却
2 秒。输出
`results/bench_*.json` + 自包含可视化 HTML。提示与报告语言由 `--lang` / `$TORCHPEAK_LANG` /
系统语言自动决定（中/英）。快速开始：`pip install -e . && torchpeak --quick`。

## Contributing
### Submit your results

Run the default suite on your device and PR the JSON into `results/community/` — one file per
device, named after it (`gtx1660ti.json`, `arc-b390.json`, `a100-sxm4-80gb.json`; append `-tuned`
for `--tuned` runs). Use a device-only label:

```bash
python -m torchpeak --label gtx1660ti            # full default suite
```

Commit only the JSON (HTML regenerates from it); `scripts/validate_result.py` runs in CI and
checks schema, filename and size. Measured values from these files feed `peaks.json` entries
(`basis: "measured, see results/community/<file>"`) — a result PR plus a peak-table entry in the
same PR is the ideal contribution. Compare across machines with
`python -m torchpeak --compare results/community/a.json results/community/b.json`.

- Peak-table corrections (new devices, wrong specs) are plain JSON PRs against `peaks.json`.
- Vendor submissions are welcome and must be labelled: put `--tuned` results in a separate file and
  state the applied settings (they are recorded in `meta.tuned` automatically).
- Raw logs + JSON together; include `meta` unchanged.
- Run `python -m unittest discover -s tests -v` before submitting code changes.
- CI runs all built-in benchmark types on CPU with quick shapes and fp32. The optional HuggingFace
  download test is excluded; probe failures for genuinely unsupported CPU features remain valid
  results, while internal errors and missing benchmark groups fail CI.

## Disclaimer

torchpeak is an independent project, **not affiliated with or endorsed by the PyTorch Foundation**.
“PyTorch” is a trademark of The Linux Foundation. Benchmark results depend on software and driver
versions; always compare runs recorded with the same `meta.format_version` and configuration.

## License

Apache-2.0 — see [LICENSE](LICENSE).
