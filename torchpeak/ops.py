"""torchpeak.ops — see the package root docstring."""
import time

import torch
from .i18n import t
from .runtime import AMP, empty_cache, sync
from .harness import measure, print_row, timeit

# ---------------------------------------------------------------- tests
def test_gemm(results, dev, a):
    squares = (256, 512) if a.quick else (1024, 8192)
    shapes = [(n, n, n) for n in squares]
    shapes += [(128, 512, 1024)] if a.quick else [(1024, 4096, 11008)]  # Llama FFN projection
    gemv_shapes = [(1, 1024, 1024)] if a.quick else [(1, 4096, 11008)]
    int8_shapes = [(512, 512, 512)] if a.quick else [(4096, 4096, 4096)]
    # cuBLASLt int8 (torch._int_mm) on CUDA requires m > 16 and 16-alignment, so the int8 "GEMV"
    # runs as a skinny m=32 batch instead of m=1; bytes are counted for the actual shape
    int8_gemv_shapes = [(32, 1024, 1024)] if a.quick else [(32, 4096, 11008)]
    gemv_min_time = max(a.min_time, 0.2 if a.quick else 3.0)
    tests = set(a.tests.split(","))
    gemm_selected = "gemm" in tests
    gemv_selected = gemm_selected or "gemv" in tests
    int8_gemm_selected = gemm_selected or "int8_gemm" in tests
    int8_gemv_selected = gemv_selected or int8_gemm_selected or "gemv_int8" in tests

    def measure_float(test, name, m, k, n, min_time=None):
        min_time = a.min_time if min_time is None else min_time

        def run():
            dt = AMP[name]
            x = torch.randn(m, k, device=dev, dtype=dt)
            y = torch.randn(k, n, device=dev, dtype=dt)
            t = timeit(lambda: x @ y, dev, min_time=min_time, warmup=10 if a.strict else 3,
                       max_iters=10000)
            return 2 * m * k * n / t / 1e12

        shape = f"{m}x{n}" if m == k == n else f"m{m} k{k} n{n}"
        measure(results, dev, test, f"{name} {shape}", "TFLOPS", run, strict=a.strict, repeats=3,
                cooldown=a.cooldown)

    def measure_int8(test, m, k, n, min_time=None):
        min_time = a.min_time if min_time is None else min_time

        def run():
            x = torch.randint(-128, 127, (m, k), device=dev, dtype=torch.int8)
            y = torch.randint(-128, 127, (k, n), device=dev, dtype=torch.int8)
            t = timeit(lambda: torch._int_mm(x, y), dev, min_time=min_time,
                       warmup=10 if a.strict else 3, max_iters=10000)
            return 2 * m * k * n / t / 1e12

        shape = f"{m}x{n}" if m == k == n else f"m{m} k{k} n{n}"
        measure(results, dev, test, f"int8 {shape}", "TOPS", run, strict=a.strict, repeats=3,
                cooldown=a.cooldown)

    def measure_gemv(test, name, m, k, n, dt_size, min_time):
        def run():
            if name == "int8":
                x = torch.randint(-128, 127, (m, k), device=dev, dtype=torch.int8)
                y = torch.randint(-128, 127, (k, n), device=dev, dtype=torch.int8)
                op = lambda: torch._int_mm(x, y)
            else:
                x = torch.randn(m, k, device=dev, dtype=AMP[name])
                y = torch.randn(k, n, device=dev, dtype=AMP[name])
                op = lambda: x @ y
            t = timeit(op, dev, min_time=min_time, warmup=10 if a.strict else 3, max_iters=10000)
            bytes_moved = (m * k + k * n + m * n) * dt_size
            return bytes_moved / t / 1e9

        shape = f"m{m} k{k} n{n}"
        measure(results, dev, test, f"{name} {shape}", "GB/s", run, strict=a.strict, repeats=3,
                cooldown=a.cooldown)

    if gemm_selected:
        for name in ("fp32", "fp16", "bf16"):
            for m, k, n in shapes:
                measure_float("gemm", name, m, k, n)
    if gemv_selected:
        # GEMV is bandwidth-bound: report GB/s from bytes moved (read W + read x + write y), not TFLOPS
        for name, dt_size in (("fp32", 4), ("fp16", 2), ("bf16", 2)):
            for m, k, n in gemv_shapes:
                measure_gemv("gemv", name, m, k, n, dt_size, gemv_min_time)
    if int8_gemm_selected:
        for m, k, n in int8_shapes:
            measure_int8("gemm_int8", m, k, n)
    if int8_gemv_selected:
        for m, k, n in int8_gemv_shapes:
            measure_gemv("gemv_int8", "int8", m, k, n, 1, gemv_min_time)


def test_vec(results, dev, a):
    """Vector ALU throughput (non-matrix path), comparing two paths:
    1) torch.compile-fused FMA chain (256 FMAs/element, arithmetic intensity ~170 FLOP/B): the ceiling reachable under torch semantics;
       fp16/bf16 are promoted to fp32 by opmath (confirmed in TTIR), so the ceiling equals the fp32 rate;
    2) triton-native row: a native Triton kernel bypasses Inductor, keeping arithmetic in its original precision domain,
       and measures the hardware's packed half2 vector rate (B390 measured fp16 ≈1.8× fp32, unreachable on the torch path).
    Versus gemm: fp16/bf16 gemm ≈ vec value ⇒ the library is not dispatching GEMM to XMX/tensor cores.
    Note: there is no Triton backend on CPU, so the triton-native row reports an error as-is; on CPU the torch row measures
    Inductor C++ codegen quality (weak at auto-vectorizing long chains), not the hardware vector peak."""
    chain = 256
    numel = 2 * 1024 * 1024  # 8MiB fp32 / 4MiB fp16: stays in L2/LLC, removing the DRAM constraint
    fp32_value = {}
    for name in ("fp32", "fp16", "bf16"):
        def run(name=name):
            t0 = torch.randn(numel, device=dev, dtype=AMP[name])

            def fma_chain(t):
                for _ in range(chain):
                    t = t * 1.0001 + 0.0001  # serial dependency; cannot be folded or CSE'd away
                return t

            f = torch.compile(fma_chain)
            # iGPU DVFS ramp: spin for ~2s first so clocks reach steady state; otherwise a short kernel can vary ±40% (B390: 4.1↔6.8)
            warm_until = time.perf_counter() + max(2.0, a.min_time)
            while time.perf_counter() < warm_until:
                f(t0)
            sync(dev)
            tt = timeit(lambda: f(t0), dev, min_time=a.min_time, warmup=10 if a.strict else 3)
            value = 2 * numel * chain / tt / 1e12
            if name == "fp32":
                fp32_value["v"] = value
                return value
            ratio = round(value / fp32_value["v"], 3) if fp32_value.get("v") else None
            # ratio ≈1.0 means fp16/bf16 are promoted to fp32 math (torch opmath semantics); see the peak table's basis
            return value, {"vs_fp32_ratio": ratio,
                           "promoted_to_fp32_suspected": ratio is not None and 0.9 <= ratio <= 1.1}

        measure(results, dev, "vec_flops", f"{name} chain{chain}", "TFLOPS", run,
                strict=a.strict, repeats=3, cooldown=a.cooldown)

    # Native Triton probe: bypasses Inductor's opmath promotion, keeping arithmetic in the fp16/bf16 domain,
    # to ask whether the hardware actually delivers the packed half2 vector rate (B390 measured 11.8 vs 6.4 on
    # the torch path, i.e. 2× packing is effective). Triton ships with torch, so it is not a new dependency;
    # when the platform backend is unavailable, an error row is recorded as-is.
    packed_kernel = None
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _packed_fma(ptr, out, numel, BLOCK: tl.constexpr, CHAIN: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < numel
            x = tl.load(ptr + offs, mask=mask)
            c1 = tl.full((), 1.0001, x.dtype)  # constants match the input dtype: arithmetic never leaves the original precision domain
            c2 = tl.full((), 0.0001, x.dtype)
            for _ in tl.static_range(CHAIN):
                x = x * c1 + c2
            tl.store(out + offs, x, mask=mask)

        packed_kernel = _packed_fma
    except Exception:
        pass

    torch_values = {r["config"].split()[0]: r["value"] for r in results
                    if r["test"] == "vec_flops" and "triton" not in r["config"] and r.get("value")}
    grid = ((numel + 1023) // 1024,)
    for name in ("fp32", "fp16", "bf16"):
        def run_packed(name=name):
            if packed_kernel is None:
                raise RuntimeError(t("triton unavailable (platform without a native Triton backend)"))
            x = torch.randn(numel, device=dev, dtype=AMP[name])
            y = torch.empty_like(x)
            launch = lambda: packed_kernel[grid](x, y, numel, BLOCK=1024, CHAIN=chain)
            warm_until = time.perf_counter() + max(1.0, a.min_time)
            while time.perf_counter() < warm_until:
                launch()
            sync(dev)
            sec = timeit(launch, dev, min_time=a.min_time, warmup=3)
            value = 2 * numel * chain / sec / 1e12
            tv = torch_values.get(name)
            # ratio >1.5 ⇒ how much packed throughput torch semantics (opmath promotion to fp32) costs
            extra = {"vs_torch_path_ratio": round(value / tv, 2)} if tv else {}
            return value, extra

        measure(results, dev, "vec_flops", f"{name} chain{chain} triton-native", "TFLOPS",
                run_packed, strict=a.strict, repeats=3, cooldown=a.cooldown)


def test_bandwidth(results, dev, a):
    sizes = (16, 512) if a.quick else (16, 1024)  # one L2-resident size + one DRAM size
    ops = ["read", "write", "copy"]
    if dev.type in ("cuda", "xpu"):
        ops += ["h2d", "d2h"]

    for size_mib in sizes:
        for op in ops:
            def run(size_mib=size_mib, op=op):
                nbytes = size_mib * 2 ** 20
                if op == "read":
                    x = torch.empty(nbytes // 4, dtype=torch.float32, device=dev).fill_(1)
                    step = x.sum
                elif op == "write":
                    x = torch.empty(nbytes, dtype=torch.uint8, device=dev)
                    step = lambda: x.fill_(1)
                elif op == "copy":
                    x = torch.empty(nbytes, dtype=torch.uint8, device=dev).fill_(1)
                    y = torch.empty_like(x)
                    step = lambda: y.copy_(x)
                elif op == "h2d":
                    x = torch.empty(nbytes, dtype=torch.uint8).fill_(1)
                    y = torch.empty(nbytes, dtype=torch.uint8, device=dev)
                    step = lambda: y.copy_(x)
                else:
                    x = torch.empty(nbytes, dtype=torch.uint8, device=dev).fill_(1)
                    y = torch.empty(nbytes, dtype=torch.uint8)
                    step = lambda: y.copy_(x)
                t = timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
                traffic = 2 * nbytes if op == "copy" else nbytes
                return traffic / t / 1e9
            measure(results, dev, "bandwidth", f"{op} {size_mib}MiB", "GB/s", run, strict=a.strict,
                    cooldown=a.cooldown)


def test_memory_wall(results, dev, a):
    """CSAPP classic memory mountain: read throughput as a matrix over working-set size × stride.
    The full matrix is stored in the mountain field; the main value uses only the stride-1 cell of the
    largest working set, so cache throughput of smaller working sets is not taken for DRAM bandwidth."""
    sizes_kib = (256, 1024, 4096, 16384, 65536) if a.quick else (64, 256, 1024, 4096, 16384, 65536, 262144)
    strides = (1, 2, 4, 8, 16, 32)  # in fp32 elements; 16 elements = one 64B cache line
    cell_min_time = max(0.1, min(a.min_time, 0.5))
    matrix = []
    best = 0.0
    largest_working_set_gbs = 0.0
    for ws_kib in sizes_kib:
        n = ws_kib * 1024 // 4
        buf = torch.randn(n, dtype=torch.float32, device=dev)  # random data keeps compression/prefetch from helping
        row_gbs = []
        for stride in strides:
            view = buf[::stride]

            def step():
                return view.sum()

            sec = timeit(step, dev, min_time=cell_min_time, warmup=3 if not a.strict else 6, max_iters=10000)
            gbs = ws_kib * 1024 / sec / 1e9  # CSAPP convention: one logical read of the whole working set / time
            row_gbs.append(round(gbs, 1))
            if stride == 1:
                best = max(best, gbs)
                largest_working_set_gbs = gbs
        matrix.append(row_gbs)
        del buf
    sizes_label = [f"{ws // 1024}MiB" if ws >= 1024 else f"{ws}KiB" for ws in sizes_kib]
    # compact console matrix: rows = stride, cols = working set
    print(t("  memory mountain (GB/s), rows=stride cols=working set:"), flush=True)
    header = "         " + "".join(f"{s:>9}" for s in sizes_label)
    print(header, flush=True)
    for si, stride in enumerate(strides):
        cells = "".join(f"{matrix[wi][si]:>9.1f}" for wi in range(len(sizes_kib)))
        print(f"  s{stride:<4}" + cells, flush=True)
    r = dict(test="memory_wall", config=f"read mountain {len(sizes_kib)}x{len(strides)}",
             value=round(largest_working_set_gbs, 1), unit="GB/s",
             value_basis="largest working set, stride 1", best_stride1_gbs=round(best, 1),
             mountain={"op": "read", "dtype": "fp32 random",
                       "sizes": sizes_label, "strides": list(strides), "gbs": matrix})
    results.append(r)
    print_row(r)
    empty_cache(dev)


