"""torchpeak.probes — See the package-root docstring (torchpeak.cli)."""
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from .runtime import empty_cache, sync
from .i18n import t
from .harness import autocast, error_fields

# ---------------------------------------------------------------- feature availability matrix
def test_probe(results, dev, a):
    """Third evaluation layer: feature probes. Each answers yes/no plus an error message, aggregated into a pass rate; "unavailable" is itself a finding."""
    def autocast_dtype(dtype):
        def run():
            m = nn.Linear(16, 16).to(dev)
            with torch.autocast(device_type=dev.type, dtype=dtype):
                out = m(torch.randn(4, 16, device=dev))
            assert out.dtype == dtype, t("output {got} != {want}", got=out.dtype, want=dtype)
        return run

    def bf16_matmul():
        x = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
        assert (x @ x).dtype == torch.bfloat16

    def sdpa_backend(backend_name):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        def run():
            backend = getattr(SDPBackend, backend_name)
            q = torch.randn(1, 2, 8, 16, device=dev)
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, q, q)
        return run

    def grad_scaler():
        m = nn.Linear(8, 8).to(dev)
        scaler = torch.amp.GradScaler(dev.type, enabled=True)
        scaler.scale(m(torch.randn(2, 8, device=dev)).sum()).backward()
        scaler.step(torch.optim.SGD(m.parameters(), lr=0.01))
        scaler.update()

    def channels_last():
        x = torch.randn(1, 3, 8, 8, device=dev).to(memory_format=torch.channels_last)
        w = torch.randn(4, 3, 3, 3, device=dev).to(memory_format=torch.channels_last)
        out = F.conv2d(x, w)
        assert out.is_contiguous(memory_format=torch.channels_last), t("channels_last not preserved")

    def int8_mm():
        x = torch.randint(-128, 127, (32, 32), device=dev, dtype=torch.int8)  # cuBLAS requires m>16
        y = torch.randint(-128, 127, (32, 32), device=dev, dtype=torch.int8)
        assert torch._int_mm(x, y).dtype == torch.int32

    def compile_step(with_backward):
        def run():
            m = nn.Linear(16, 16).to(dev)
            cm = torch.compile(m)
            loss = cm(torch.randn(4, 16, device=dev)).sum()
            if with_backward:
                loss.backward()
            sync(dev)
        return run

    def host_pin_memory():
        assert torch.empty(1024, dtype=torch.uint8).pin_memory().is_pinned()

    probes = [
        ("autocast_fp16", autocast_dtype(torch.float16)),
        ("autocast_bf16", autocast_dtype(torch.bfloat16)),
        ("bf16_matmul", bf16_matmul),
        ("sdpa_flash", sdpa_backend("FLASH_ATTENTION")),
        ("sdpa_efficient", sdpa_backend("EFFICIENT_ATTENTION")),
        ("sdpa_cudnn", sdpa_backend("CUDNN_ATTENTION")),
        ("sdpa_math", sdpa_backend("MATH")),
        ("grad_scaler", grad_scaler),
        ("channels_last", channels_last),
        ("int8_mm", int8_mm),
        ("torch_compile_fwd", compile_step(False)),
        ("torch_compile_bwd", compile_step(True)),
        ("host_pin_memory", host_pin_memory),
    ]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # probes run silently; exceptions go into results instead of spamming output
        for name, fn in probes:
            try:
                fn()
                r = dict(test="probe", config=name, value=1.0, unit="probe", status="pass")
            except Exception as e:
                fields = error_fields(e, limit=160)
                fields["status"] = "fail"
                r = dict(test="probe", config=name, value=0.0, unit="probe", **fields)
            results.append(r)
    empty_cache(dev)

    # after all probes, print the yes/no list once and pause a few seconds before continuing
    ok = [r["config"] for r in results if r["test"] == "probe" and r["status"] == "pass"]
    bad = [r for r in results if r["test"] == "probe" and r["status"] == "fail"]
    if not ok and not bad:
        return
    print(t("== usability matrix: {ok}/{total} available", ok=len(ok), total=len(ok) + len(bad)), flush=True)
    for i in range(0, len(ok), 6):
        print("   ✓ " + "  ".join(ok[i:i + 6]), flush=True)
    for r in bad:
        reason = (r.get("error") or "")[:100]
        print(f"   ✗ {r['config']} — {reason}", flush=True)
    print(t("   benchmarks start in 3 s ..."), flush=True)
    time.sleep(3)

