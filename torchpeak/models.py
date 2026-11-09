"""torchpeak.models — See the package-root docstring (torchpeak.cli)."""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .i18n import t
from .runtime import AMP, empty_cache
from .harness import (autocast, compile_config, compile_modes, is_oom, measure,
                     measure_compile_pair, model_precision, print_row, tcompile,
                     timeit)

def _model_vram_estimate(test, quick, compare):
    """Rough peak-VRAM estimate (GB) for this test in full/quick mode, used for pre-run checks.
    The compare pair path keeps the fp32 reference (released first), eager, compile, and hot-wrapper instances alive at once;
    the hot phase holds ~3 models; activations are approximated from residuals + the math SDPA T² score matrix."""
    headroom = 0.5
    if test == "llama":
        L, d, h, bs, T = (4, 512, 8, 2, 512) if quick else (16, 2048, 16, 4, 2048)
        vocab = 50304
        dff = int(8 * d / 3)
        params = vocab * d + L * (4 * d * d + 3 * d * dff + 2 * d) + d
        act = (bs * T * d * 2 + bs * h * T * T * 4) / 1e9
    elif test == "gpt":
        L, d, h, bs, T = (6, 512, 8, 4, 256) if quick else (12, 768, 12, 8, 512)
        vocab = 50304
        params = vocab * d + L * (4 * d * d + 8 * d * d) + d
        act = (bs * T * d * 2 + bs * h * T * T * 4) / 1e9
    elif test == "moe":
        L, d, bs, T, E = (2, 256, 8, 128, 4) if quick else (4, 512, 8, 128, 8)
        params = 8192 * d + L * (4 * d * d + 4 * d * d + E * d * d)
        act = 0.2
    elif test == "decode":
        params = 50304 * 768 + 24 * (4 * 768 * 768 + 8 * 768 * 768)  # GPT-2 medium, one fp16 instance
        return params * 2 / 1e9 + 0.6
    elif test == "resnet":
        return 0.8
    elif test == "train":
        return 2.5  # largest case is vit_b_16; OOM has a halving fallback
    else:
        return 0.4  # microbenchmark: largest is a 1024 MiB buffer or an 8192² fp32 input
    instances = 3.0 if compare else 1.2
    return instances * params * 4 / 1e9 + act + headroom


@contextlib.contextmanager
def disable_training_randomness(model):
    saved = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            saved.append((module, "p", module.p))
            module.p = 0.0
        elif (isinstance(module, (nn.TransformerEncoderLayer, nn.MultiheadAttention))
              and not isinstance(module.dropout, nn.Module)):
            saved.append((module, "dropout", module.dropout))
            module.dropout = 0.0
        elif type(module).__name__ == "StochasticDepth":
            saved.append((module, "p", module.p))
            module.p = 0.0
    try:
        yield
    finally:
        for module, attr, value in saved:
            setattr(module, attr, value)

def test_resnet(results, dev, a):
    try:
        import torchvision
    except ImportError:
        results.append(dict(test="resnet50", config="-", value=None, unit="", status="skipped",
                            error="torchvision not installed", error_kind="unsupported"))
        print_row(results[-1])
        return

    def make():
        m = torchvision.models.resnet50(weights=None).to(dev).to(memory_format=torch.channels_last)
        return m

    def make_case(compile_enabled, bs):
        m = make().eval()
        if compile_enabled:
            m = tcompile(m, a)
        x = torch.randn(bs, 3, 224, 224, device=dev).to(memory_format=torch.channels_last)

        def step():
            with torch.no_grad(), autocast(dev, a.amp):
                return m(x)
        return step, step

    if a.compare_compile:
        for bs in (1, 32):
            measure_compile_pair(results, dev, "resnet50_infer", f"bs{bs} {model_precision(a.amp)}", "img/s",
                                 make_case, bs, a)
        return

    for compile_enabled in compile_modes(a):
        for bs in (1, 32):
            def run_inf(bs=bs, compile_enabled=compile_enabled):
                step, _ = make_case(compile_enabled, bs)
                return bs / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
            config = compile_config(f"bs{bs} {model_precision(a.amp)}", a, compile_enabled)
            measure(results, dev, "resnet50_infer", config, "img/s", run_inf, strict=a.strict,
                    cooldown=a.cooldown)


# ---------------------------------------------------------------- classic model training
class BertBase(nn.Module):
    """BERT-base architecture (12 layers / 768 / 12 heads / FFN 3072) with MLM-style cross-entropy over all tokens. Pure PyTorch, no transformers needed."""

    def __init__(self, vocab=30522, d=768, L=12, h=12, T=512):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.ln = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.1, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.ln(self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device)))
        return self.head(self.enc(x))


# name: (constructor, input kind, default bs, quick bs, unit)
def train_zoo():
    zoo = {"bert_base": (lambda: BertBase(), "tokens128", 32, 8, "seq/s")}
    try:
        import torchvision.models as tvm
        zoo = {
            "resnet50": (lambda: tvm.resnet50(), "image224", 64, 16, "img/s"),
            "mobilenet_v3_large": (lambda: tvm.mobilenet_v3_large(), "image224", 64, 16, "img/s"),
            "efficientnet_b0": (lambda: tvm.efficientnet_b0(), "image224", 64, 16, "img/s"),
            "vit_b_16": (lambda: tvm.vit_b_16(), "image224", 32, 8, "img/s"),
            "vgg16": (lambda: tvm.vgg16(), "image224", 32, 8, "img/s"),
            **zoo,
        }
    except ImportError:
        pass
    return zoo


def test_train(results, dev, a):
    zoo = train_zoo()
    names = list(zoo) if a.train_models == "all" else a.train_models.split(",")
    for name in names:
        if name not in zoo:
            results.append(dict(test="train", config=name, value=None, unit="", status="error",
                                error=f"unknown model (available: {','.join(zoo)}; torchvision installed?)",
                                error_kind="usage"))
            print_row(results[-1])
            continue
        ctor, kind, bs_full, bs_quick, unit = zoo[name]
        bs = a.train_bs or (bs_quick if a.quick else bs_full)

        def make_case(compile_enabled, bs):
            m = ctor().to(dev)
            if kind == "image224":
                m = m.to(memory_format=torch.channels_last)
                x = torch.randn(bs, 3, 224, 224, device=dev).to(memory_format=torch.channels_last)
                y = torch.randint(0, 1000, (bs,), device=dev)
                loss_fn = lambda out: F.cross_entropy(out, y)
                opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
            else:  # tokens128
                x = torch.randint(0, 30522, (bs, 128), device=dev)
                y = torch.randint(0, 30522, (bs, 128), device=dev)
                loss_fn = lambda out: F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            m.train()
            cm = tcompile(m, a) if compile_enabled else m
            scaler = torch.amp.GradScaler(dev.type, enabled=(a.amp == "fp16" and dev.type != "cpu"))

            def step():
                opt.zero_grad(set_to_none=True)
                with autocast(dev, a.amp):
                    loss = loss_fn(cm(x))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            def check():
                opt.zero_grad(set_to_none=True)
                with disable_training_randomness(m):
                    with autocast(dev, a.amp):
                        out = cm(x)
                        loss = loss_fn(out)
                    loss.backward()
                return loss.detach(), out.detach()
            return step, check

        if a.compare_compile:
            measure_compile_pair(results, dev, "train", f"{name} {model_precision(a.amp)}", unit, make_case, bs, a,
                                 allow_oom_retry=True)
            continue

        for compile_enabled in compile_modes(a):
            def run_once(bs, compile_enabled=compile_enabled):
                step, _ = make_case(compile_enabled, bs)
                return bs / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)

            def run(bs=bs, compile_enabled=compile_enabled):
                # halve the batch on OOM; the result config records the actual bs used
                while True:
                    try:
                        return run_once(bs, compile_enabled), {"bs_used": bs}
                    except Exception as e:
                        if not is_oom(e) or bs <= 1:
                            raise
                    # only after leaving the except block are the exception and its referenced frames/tensors released; then clear the cache
                    empty_cache(dev)
                    print(t("  {name}: bs{bs} OOM, halving to bs{halved} and retrying",
                            name=name, bs=bs, halved=bs // 2), flush=True)
                    bs //= 2

            # config omits bs so devices with reduced bs still line up in --compare; the actual bs is recorded in bs_used
            config = compile_config(f"{name} {model_precision(a.amp)}", a, compile_enabled)
            measure(results, dev, "train", config, unit, run, strict=a.strict, cooldown=a.cooldown)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, L, h, T):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # weight tying, as in GPT-2

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def test_gpt(results, dev, a):
    if a.quick or "gpt" in getattr(a, "quick_tests", ()):
        L, d, h, bs, T = 6, 512, 8, 4, 256
    else:
        L, d, h, bs, T = 12, 768, 12, 8, 512  # ≈ GPT-2 small (124M)
    vocab = 50304
    tag = f"L{L} d{d} bs{bs}x{T} {model_precision(a.amp)}"

    def build():
        m = GPT(vocab, d, L, h, T).to(dev)
        n_params = sum(p.numel() for p in m.parameters())
        return m, n_params

    train_n = {}

    def make_train_case(compile_enabled, scale):
        m, N = build()
        train_n["value"] = N
        bs_eff = max(1, scale // T)  # scale halves on OOM retry; rebuild the case at the smaller batch
        m.train()
        cm = tcompile(m, a) if compile_enabled else m
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler(dev.type, enabled=(a.amp == "fp16" and dev.type != "cpu"))
        x = torch.randint(0, vocab, (bs_eff, T), device=dev)
        y = torch.randint(0, vocab, (bs_eff, T), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with autocast(dev, a.amp):
                loss = F.cross_entropy(cm(x).view(-1, vocab), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        def check():
            opt.zero_grad(set_to_none=True)
            with disable_training_randomness(m):
                with autocast(dev, a.amp):
                    out = cm(x)
                    loss = F.cross_entropy(out.view(-1, vocab), y.view(-1))
                loss.backward()
            return loss.detach(), out.detach()
        return step, check

    if a.compare_compile:
        def train_extra(tps):
            N = train_n["value"]
            flops_per_tok = 6 * N + 12 * L * d * T
            return {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
        measure_compile_pair(results, dev, "gpt_train", tag, "tok/s", make_train_case, bs * T, a,
                             extra_fn=train_extra, allow_oom_retry=True)
    else:
        for compile_enabled in compile_modes(a):
            def run_train(compile_enabled=compile_enabled):
                step, _ = make_train_case(compile_enabled, bs * T)
                N = train_n["value"]
                tps = bs * T / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
                flops_per_tok = 6 * N + 12 * L * d * T  # MFU estimate formula from nanoGPT
                return tps, {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
            config = compile_config(tag, a, compile_enabled)
            measure(results, dev, "gpt_train", config, "tok/s", run_train, strict=a.strict,
                    cooldown=a.cooldown)

    prefill_n = {}

    def make_prefill_case(compile_enabled, scale):
        m, N = build()
        prefill_n["value"] = N
        bs_eff = max(1, scale // T)  # scale halves on OOM retry
        m.eval()
        cm = tcompile(m, a) if compile_enabled else m
        x = torch.randint(0, vocab, (bs_eff, T), device=dev)

        def step():
            with torch.no_grad(), autocast(dev, a.amp):
                return cm(x)
        return step, step

    if a.compare_compile:
        def prefill_extra(tps):
            N = prefill_n["value"]
            flops_per_tok = 2 * N + 4 * L * d * T
            return {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
        measure_compile_pair(results, dev, "gpt_prefill", tag, "tok/s", make_prefill_case, bs * T, a,
                             extra_fn=prefill_extra, allow_oom_retry=True)
    else:
        for compile_enabled in compile_modes(a):
            def run_prefill(compile_enabled=compile_enabled):
                step, _ = make_prefill_case(compile_enabled, bs * T)
                N = prefill_n["value"]
                tps = bs * T / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
                flops_per_tok = 2 * N + 4 * L * d * T
                return tps, {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
            config = compile_config(tag, a, compile_enabled)
            measure(results, dev, "gpt_prefill", config, "tok/s", run_prefill, strict=a.strict,
                    cooldown=a.cooldown)


def _block_decode(b, x, kc, vc, t):
    """Single-token forward: write the new k/v into slot t of the preallocated cache, then attend over [0, t]."""
    B, _, C = x.shape
    q, k, v = b.qkv(b.ln1(x)).split(C, dim=2)
    q, k, v = (z.view(B, 1, b.h, C // b.h).transpose(1, 2) for z in (q, k, v))
    kc[:, :, t:t + 1] = k
    vc[:, :, t:t + 1] = v
    y = F.scaled_dot_product_attention(q, kc[:, :, :t + 1], vc[:, :, :t + 1])
    x = x + b.proj(y.transpose(1, 2).reshape(B, 1, C))
    return x + b.out(F.gelu(b.fc(b.ln2(x))))


def test_gpt_decode(results, dev, a):
    """bs=1 greedy decode with a KV cache, equivalent to llama.cpp's tg — mainly measures memory bandwidth. No model download required."""
    if a.quick:
        L, d, h = 12, 768, 12    # ≈ GPT-2 small, fp16 weights ~0.25 GB
    else:
        L, d, h = 24, 1024, 16   # ≈ GPT-2 medium, fp16 weights ~0.7 GB, far larger than any L2/LLC
    vocab, P, N = 50304, 128, a.new_tokens
    dtype = AMP[a.amp]

    def run():
        m = GPT(vocab, d, L, h, P + N).to(dev, dtype).eval()
        hd = d // h
        kc = [torch.zeros(1, h, P + N, hd, device=dev, dtype=dtype) for _ in range(L)]
        vc = [torch.zeros(1, h, P + N, hd, device=dev, dtype=dtype) for _ in range(L)]
        for c in kc + vc:  # pretend P tokens have already been prefilled
            c[:, :, :P].normal_()

        @torch.no_grad()
        def gen():
            tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
            for i in range(N):
                t = P + i
                x = m.wte(tok) + m.wpe.weight[t].view(1, 1, d)
                for li, b in enumerate(m.blocks):
                    x = _block_decode(b, x, kc[li], vc[li], t)
                tok = m.head(m.ln(x)).argmax(-1)  # each step depends on the previous result, serial like real decoding
        tps = N / timeit(gen, dev, warmup=10 if a.strict else 1, min_time=a.min_time, max_iters=10)
        wbytes = sum(p.numel() * p.element_size() for p in m.parameters())
        return tps, {"approx_weight_bw_gbs": round(wbytes * tps / 1e9, 1),
                     "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
    measure(results, dev, "gpt_decode", f"L{L} d{d} bs1 {a.amp}", "tok/s", run, strict=a.strict,
            cooldown=a.cooldown)


# ---------------------------------------------------------------- Llama-style ~1B decoder
def apply_rope(x, theta, t0=0):
    """Rotary position embedding. x: (B, h, T, hd), theta: (hd/2,), starting at position t0 (decode computes only the new position)."""
    T = x.shape[-2]
    pos = torch.arange(t0, t0 + T, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, theta.to(x.device))  # (T, hd/2)
    cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)  # keep the input dtype so q/k and cache dtypes do not mix during decode
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class LlamaBlock(nn.Module):
    """RMSNorm + RoPE + SwiGLU, bias-free, matching the Llama architecture."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        dff = int(8 * d / 3)
        self.gate, self.up = nn.Linear(d, dff, bias=False), nn.Linear(d, dff, bias=False)
        self.down = nn.Linear(dff, d, bias=False)
        self.register_buffer("theta", 1.0 / (10000.0 ** (torch.arange(0, d // h, 2, dtype=torch.float32) / (d // h))),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class Llama(nn.Module):
    def __init__(self, vocab, d, L, h):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(LlamaBlock(d, h) for _ in range(L))
        self.ln = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # weight tying

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def _llama_block_decode(b, x, kc, vc, t):
    """Single-token forward: RoPE applies only to the current position; k/v are written into the preallocated cache."""
    B, _, C = x.shape
    hd = C // b.h
    q, k, v = b.qkv(b.ln1(x)).split(C, dim=2)
    q, k, v = (z.view(B, 1, b.h, hd).transpose(1, 2) for z in (q, k, v))
    q, k = apply_rope(q, b.theta, t), apply_rope(k, b.theta, t)
    kc[:, :, t:t + 1] = k
    vc[:, :, t:t + 1] = v
    y = F.scaled_dot_product_attention(q, kc[:, :, :t + 1], vc[:, :, :t + 1])
    x = x + b.proj(y.transpose(1, 2).reshape(B, 1, C))
    return x + b.down(F.silu(b.gate(x)) * b.up(x))


def test_llama(results, dev, a):
    """~1B Llama architecture: long-sequence prefill (GEMM/SDPA/RMSNorm) + bs1 KV-cache decode (bandwidth)."""
    if a.quick or "llama" in getattr(a, "quick_tests", ()):
        L, d, h, bs, T = 4, 512, 8, 2, 512     # ≈ 38M, smoke test
    else:
        L, d, h, bs, T = 16, 2048, 16, 4, 2048  # ≈ 0.9B
    vocab = 50304
    tag = f"L{L} d{d} bs{bs}x{T} {model_precision(a.amp)}"

    def build():
        return Llama(vocab, d, L, h).to(dev)

    n_params = {}

    def make_prefill(compile_enabled, scale):
        m = build()
        n_params["value"] = sum(p.numel() for p in m.parameters())
        bs_eff = max(1, scale // T)  # scale halves on OOM retry
        m.eval()
        cm = tcompile(m, a) if compile_enabled else m
        x = torch.randint(0, vocab, (bs_eff, T), device=dev)

        def step():
            with torch.no_grad(), autocast(dev, a.amp):
                return cm(x)
        return step, step

    if a.compare_compile:
        def prefill_extra(tps):
            N = n_params["value"]
            flops_per_tok = 2 * N + 4 * L * d * T
            return {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
        measure_compile_pair(results, dev, "llama_prefill", tag, "tok/s", make_prefill, bs * T, a,
                             extra_fn=prefill_extra, allow_oom_retry=True)
    else:
        for compile_enabled in compile_modes(a):
            def run_prefill(compile_enabled=compile_enabled):
                step, _ = make_prefill(compile_enabled, bs * T)
                N = n_params["value"]
                tps = bs * T / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
                flops_per_tok = 2 * N + 4 * L * d * T
                return tps, {"approx_tflops": round(tps * flops_per_tok / 1e12, 3), "params_m": round(N / 1e6, 1)}
            measure(results, dev, "llama_prefill", compile_config(tag, a, compile_enabled), "tok/s",
                    run_prefill, strict=a.strict, cooldown=a.cooldown)

    def run_decode():
        m = build().to(dev, AMP[a.amp]).eval()
        hd = d // h
        P, N = 128, a.new_tokens
        kc = [torch.zeros(1, h, P + N, hd, device=dev, dtype=AMP[a.amp]) for _ in range(L)]
        vc = [torch.zeros(1, h, P + N, hd, device=dev, dtype=AMP[a.amp]) for _ in range(L)]
        for c in kc + vc:
            c[:, :, :P].normal_()

        @torch.no_grad()
        def gen():
            tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
            for i in range(N):
                t = P + i
                x = m.wte(tok)
                for li, b in enumerate(m.blocks):
                    x = _llama_block_decode(b, x, kc[li], vc[li], t)
                tok = m.head(m.ln(x)).argmax(-1)
        tps = N / timeit(gen, dev, warmup=10 if a.strict else 1, min_time=a.min_time, max_iters=10)
        wbytes = sum(p.numel() * p.element_size() for p in m.parameters())
        return tps, {"approx_weight_bw_gbs": round(wbytes * tps / 1e9, 1),
                     "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
    measure(results, dev, "llama_decode", f"L{L} d{d} bs1 {a.amp}", "tok/s", run_decode, strict=a.strict,
            cooldown=a.cooldown)


# ---------------------------------------------------------------- small MoE (dynamic shapes / graph break)
class MoELayer(nn.Module):
    """Top-k sparse routing. The sparse path uses boolean indexing + index_add_, with data-dependent routing:
    this is exactly the behavior the compile graph-break / recompile metrics target. The dense path computes all experts then masks by weight,
    with no dynamic shapes, serving as a compilable control."""

    def __init__(self, d, n_experts, top_k, sparse=True):
        super().__init__()
        self.n_experts, self.top_k, self.sparse = n_experts, top_k, sparse
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_experts))

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        scores = F.softmax(self.gate(xf), dim=-1)
        top_w, top_i = torch.topk(scores, self.top_k, dim=-1)
        out = torch.zeros_like(xf)
        if self.sparse:
            for e, expert in enumerate(self.experts):
                mask = (top_i == e)
                idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]  # data-dependent shape
                if idx.numel():  # Python branch depends on tensor contents → graph break
                    w = (top_w * mask).sum(dim=-1, keepdim=True)[idx]  # this expert's own top-k weight
                    src = expert(xf[idx]) * w
                    out.index_add_(0, idx, src.to(out.dtype))
        else:
            for e, expert in enumerate(self.experts):
                mask = (top_i == e)  # (tokens, k)
                w = (top_w * mask).sum(dim=-1, keepdim=True)
                out = out + w * expert(xf)
        return out.reshape(shape)


class MoE(nn.Module):
    def __init__(self, vocab, d, L, h, n_experts, top_k, sparse):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        def block():
            attn = nn.TransformerEncoderLayer(d, h, d, dropout=0.0, activation="gelu", batch_first=True,
                                              norm_first=True)
            return nn.ModuleList([attn, MoELayer(d, n_experts, top_k, sparse)])
        self.blocks = nn.ModuleList(block() for _ in range(L))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.wte(idx)
        for attn, moe in self.blocks:
            x = x + attn(x)
            x = x + moe(x)
        return self.head(self.ln(x))


def test_moe(results, dev, a):
    if a.quick or "moe" in getattr(a, "quick_tests", ()):
        L, d, h, E, K = 2, 256, 4, 4, 1
    else:
        L, d, h, E, K = 4, 512, 8, 8, 2
    vocab, bs, T = 8192, 8, 128
    tag = f"sparse E{E}K{K} L{L} d{d} {model_precision(a.amp)}"

    def make_case(compile_enabled, scale):
        m = MoE(vocab, d, L, h, E, K, sparse=True).to(dev)
        bs_eff = max(1, scale // T)  # scale halves on OOM retry
        m.train()
        cm = tcompile(m, a) if compile_enabled else m
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, vocab, (bs_eff, T), device=dev)
        y = torch.randint(0, vocab, (bs_eff, T), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            with autocast(dev, a.amp):
                loss = F.cross_entropy(cm(x).view(-1, vocab), y.view(-1))
            loss.backward()
            opt.step()
            return loss

        def check():
            opt.zero_grad(set_to_none=True)
            with autocast(dev, a.amp):
                out = cm(x)
                loss = F.cross_entropy(out.view(-1, vocab), y.view(-1))
            loss.backward()
            return loss.detach(), out.detach()
        return step, check

    if a.compare_compile:
        measure_compile_pair(results, dev, "moe_train", tag, "tok/s", make_case, bs * T, a,
                             allow_oom_retry=True)
    else:
        for compile_enabled in compile_modes(a):
            def run(compile_enabled=compile_enabled):
                step, _ = make_case(compile_enabled, bs * T)
                return bs * T / timeit(step, dev, min_time=a.min_time, warmup=10 if a.strict else 3)
            measure(results, dev, "moe_train", compile_config(tag, a, compile_enabled), "tok/s", run,
                    strict=a.strict, cooldown=a.cooldown)


def test_hf_decode(results, dev, a):
    def run():
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float32 if a.amp == "fp32" else AMP[a.amp]
        ver = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
        kw = {"dtype": dtype} if ver >= (4, 56) else {"torch_dtype": dtype}
        tok = AutoTokenizer.from_pretrained(a.hf_model)
        model = AutoModelForCausalLM.from_pretrained(a.hf_model, **kw).to(dev).eval()
        ids = tok("Explain what a unified memory architecture is.", return_tensors="pt").to(dev)
        n = a.new_tokens

        def gen():
            with torch.no_grad():
                model.generate(**ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        t = timeit(gen, dev, warmup=1, min_time=0, max_iters=1)
        tps = n / t
        wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # bs=1 decode is essentially memory-bound: every token must read all the weights once
        return tps, {"approx_weight_bw_gbs": round(wbytes * tps / 1e9, 1),
                     "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    measure(results, dev, "hf_decode", f"{a.hf_model.split('/')[-1]} bs1 {a.amp}", "tok/s", run, strict=a.strict,
            cooldown=a.cooldown)

