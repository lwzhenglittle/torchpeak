"""torchpeak.runtime — see the package root docstring."""
import atexit
import contextlib
import importlib.metadata
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile

import torch

AMP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ---------------------------------------------------------------- device helpers
# Device types whose legacy `torch.<type>` module is an internal/deprecated shim with no usable
# synchronize/memory hooks (TorchTPU on PyTorch >= 2.11 raises when those are called). Those go
# through torch.accelerator; every other backend keeps its existing torch.<type> path.
ACCELERATOR_ONLY = {"tpu"}


def _accelerator(dev):
    """torch.accelerator if `dev` is the current accelerator; None for legacy backends or old torch."""
    if dev.type not in ACCELERATOR_ONLY:
        return None
    acc = getattr(torch, "accelerator", None)
    if acc is None or not hasattr(acc, "current_accelerator"):
        return None
    try:
        cur = acc.current_accelerator()
    except Exception:
        return None
    return acc if cur is not None and cur.type == dev.type else None


def select_device(dev):
    """Make `dev` the current accelerator device. Accelerator-routed hooks (synchronize, peak memory,
    empty_cache) take no index, so an explicit --device tpu:1 must also move the current index."""
    acc = _accelerator(dev)
    if acc is None or dev.index is None:
        return
    try:
        if acc.current_device_index() != dev.index:
            acc.set_device_index(dev.index)
    except Exception:
        pass


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    # torch.accelerator knows out-of-tree backends registered as PrivateUse1 (e.g. TPU), which the
    # per-vendor probes below cannot see. On machines with several accelerators it also encodes
    # PyTorch's own notion of the current one, which is what "auto" should mean.
    acc = getattr(torch, "accelerator", None)
    if acc is not None and hasattr(acc, "current_accelerator"):
        try:
            if acc.is_available():
                current = acc.current_accelerator()
                if current is not None and current.type != "cpu":
                    return torch.device(current.type, current.index)
        except Exception:
            pass
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:  # TorchTPU on a torch too old for torch.accelerator (its is_available raises on >= 2.11)
        try:
            if tpu.is_available():
                return torch.device("tpu")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    for ext, attr in (("torch_npu", "npu"), ("torch_musa", "musa")):  # domestic cards phase 2: lazy-load vendor extensions
        if getattr(torch, attr, None) is None:
            try:
                importlib.import_module(ext)
            except ImportError:
                pass
        m = getattr(torch, attr, None)
        if m is not None and m.is_available():
            return torch.device(attr)
    return torch.device("cpu")


def mod(dev):
    return getattr(torch, dev.type, None)  # torch.cuda / torch.xpu / torch.mps / torch.cpu / torch.tpu


def sync(dev):
    acc = _accelerator(dev)
    if acc is not None:
        try:
            acc.synchronize()
            return
        except Exception:
            pass  # fall through to the legacy hook so the failure surfaces instead of hiding it
    m = mod(dev)
    if m is not None and hasattr(m, "synchronize"):
        m.synchronize()


def reset_peak(dev):
    acc = _accelerator(dev)
    if acc is not None and hasattr(acc, "reset_peak_memory_stats"):
        try:
            acc.reset_peak_memory_stats()
            return
        except Exception:
            pass
    m = mod(dev)
    if m is not None and hasattr(m, "reset_peak_memory_stats"):
        try:
            m.reset_peak_memory_stats()
        except Exception:
            pass


def peak_gb(dev):
    for m in (_accelerator(dev), mod(dev)):
        if m is not None and hasattr(m, "max_memory_allocated"):
            try:
                return round(m.max_memory_allocated() / 1e9, 2)
            except Exception:
                return None
    return None


def peak_reserved_gb(dev):
    """Caching-allocator reserved bytes; on iGPU/UMA this diverges a lot from allocated, so record both.
    TorchTPU's max_memory_reserved is a small fixed runtime pool (observed 0.02-0.82 GB against 14.7 GB
    allocated), not a caching-allocator reserve, so the field is left empty there."""
    if dev.type == "tpu":
        return None
    for m in (_accelerator(dev), mod(dev)):
        if m is not None and hasattr(m, "max_memory_reserved"):
            try:
                return round(m.max_memory_reserved() / 1e9, 2)
            except Exception:
                return None
    return None


def empty_cache(dev):
    for m in (_accelerator(dev), mod(dev)):
        if m is not None and hasattr(m, "empty_cache"):
            try:
                m.empty_cache()
                return
            except Exception:
                return


def fp32_precision_modes(dev):
    """TPU has no fp32 matmul unit: float32 matmuls are MXU passes, and the stack default ("medium")
    is a single bf16 pass (roughly the bf16 rate). The primary fp32 row therefore runs at "highest"
    (6-pass, ~IEEE-32 accuracy, ~1/6 of the bf16 rate) and an additional row records the default.
    Other backends have a real fp32 unit, so they keep a single row."""
    return ("highest", "medium") if dev.type == "tpu" else (None,)


def default_matmul_precision(dev):
    """Precision the stack uses when torchpeak does not override it."""
    return "medium" if dev.type == "tpu" else None


@contextlib.contextmanager
def matmul_precision(mode):
    """Scope torch.set_float32_matmul_precision for one measurement; mode=None is a no-op."""
    if not mode:
        yield
        return
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(mode)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def device_info(dev):
    info = {"device_type": dev.type, "device_index": dev.index,
            "name": platform.processor() or platform.machine(), "backend": dev.type, "total_mem_gb": None}
    try:
        if dev.type == "cuda":
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            p = torch.cuda.get_device_properties(index)
            info.update(device_index=index, name=p.name, total_mem_gb=round(p.total_memory / 1e9, 1),
                        backend="rocm " + torch.version.hip if torch.version.hip else "cuda " + str(torch.version.cuda),
                        compute_capability=torch.cuda.get_device_capability(index),
                        compute_units=p.multi_processor_count)
        elif dev.type == "xpu":
            index = dev.index if dev.index is not None else torch.xpu.current_device()
            p = torch.xpu.get_device_properties(index)
            info.update(device_index=index, name=p.name, total_mem_gb=round(p.total_memory / 1e9, 1),
                        backend=f"xpu {torch.version.xpu or 'unknown'}",
                        driver_version=getattr(p, "driver_version", None),
                        platform_name=getattr(p, "platform_name", None), uuid=str(getattr(p, "uuid", "")),
                        compute_units=getattr(p, "max_compute_units", None),
                        last_level_cache_mb=round(getattr(p, "last_level_cache_size", 0) / 2 ** 20, 1),
                        is_integrated_gpu=getattr(p, "is_integrated_gpu", None))
        elif dev.type == "tpu":
            index = dev.index
            if index is None:
                acc = _accelerator(dev)
                index = acc.current_device_index() if acc is not None else None
            p = torch.tpu.get_device_properties(index)
            props = {k: getattr(p, k, None) for k in (
                "name", "devices_per_chip", "hbm_bytes_per_chip", "hbm_bytes_per_device",
                "bf16_flops_per_chip", "bf16_flops_per_device", "sparse_cores_per_device",
                "sparse_core_simd_width", "topology", "version",
                # deprecated in TorchTPU once torch.accelerator exists, but the only memory/size
                # fields older wheels expose (v5e run on torch_tpu 0.2.0 reported name only)
                "total_memory", "multi_processor_count")}
            props = {k: v for k, v in props.items() if v is not None}
            mem = (props.get("hbm_bytes_per_device") or props.get("hbm_bytes_per_chip")
                   or props.get("total_memory"))
            info.update(device_index=index, name=props.get("name", "TPU"),
                        total_mem_gb=round(mem / 1e9, 1) if mem else None,
                        backend="torch_tpu " + (_pkg_version("torch_tpu") or "unknown"),
                        compute_units=props.get("multi_processor_count"),
                        device_properties=props)
        elif dev.type in ("npu", "musa"):
            m = mod(dev)
            index = dev.index if dev.index is not None else m.current_device()
            p = m.get_device_properties(index)
            info.update(device_index=index, name=getattr(p, "name", dev.type),
                        total_mem_gb=round(getattr(p, "total_memory", 0) / 1e9, 1) or None,
                        backend=f"{dev.type} {getattr(torch.version, dev.type, None) or ''}".strip(),
                        compute_units=getattr(p, "multi_processor_count", None))
    except Exception as e:
        info["info_error"] = str(e)
    return info


# ---------------------------------------------------------------- vendor capability probes
# Some vendor stacks ship a torch extension or compiler whose mismatch with the installed torch is
# not a Python exception: the dynamic loader kills the process, or the codegen path faults, before
# any except-block can run. Rows driven through those paths are therefore gated by an out-of-process
# probe; the parent process only touches a path a child process survived.
_CANARY = r"""
import sys
import torch

dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[sys.argv[1]]
kind = sys.argv[2]
numel = 1 << 16
x = torch.randn(numel, device="cuda", dtype=dtype)

if kind == "compile":
    def chain(t):
        for _ in range(4):
            t = t * 1.0001 + 0.0001
        return t
    y = torch.compile(chain)(x)
elif kind == "triton":
    import triton
    import triton.language as tl

    @triton.jit
    def k(ptr, out, n, BLOCK: tl.constexpr, CHAIN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(ptr + offs, mask=m)
        c1 = tl.full((), 1.0001, v.dtype)
        c2 = tl.full((), 0.0001, v.dtype)
        for _ in tl.static_range(CHAIN):
            v = v * c1 + c2
        tl.store(out + offs, v, mask=m)

    y = torch.empty_like(x)
    k[(numel // 1024,)](x, y, numel, BLOCK=1024, CHAIN=4)
else:
    raise SystemExit("unknown kind: " + kind)

torch.cuda.synchronize()
print(float(y[0]))
"""

_PATH_OK = {}
_PATH_DETAIL = {}


def _canary_tail(stderr):
    """First line that looks like a real compiler error, else the last non-empty line."""
    lines = [ln.strip() for ln in stderr.decode(errors="replace").splitlines() if ln.strip()]
    for ln in lines:
        if "error" in ln.lower():
            return ln[:100]
    return (lines[-1] if lines else "")[:100]


def path_works(kind, dtype_name, timeout=120):
    """Return True when `kind` ('compile' for a small Inductor-fused chain, 'triton' for a small
    native Triton kernel) runs to completion on `dtype_name`. A vendor codegen crash (e.g. Triton
    emitting LLVM IR its own ComGR rejects) is a SIGSEGV inside the process, so the probe runs it in
    a throwaway subprocess — written to a real file, because triton.jit needs inspect.getsource().
    Off by default (--path-canary): on a working stack the probe produces nothing but ~30 s per
    dtype of extra compiles. Results are cached per process."""
    key = (kind, dtype_name)
    if key not in _PATH_OK:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(_CANARY)
                tmp = f.name
            p = subprocess.run([sys.executable, tmp, dtype_name, kind],
                               capture_output=True, timeout=timeout)
            rc = p.returncode
            _PATH_OK[key] = rc == 0
            if rc == 0:
                _PATH_DETAIL[key] = "ok"
            elif rc < 0:
                _PATH_DETAIL[key] = f"killed by signal {-rc} ({_canary_tail(p.stderr)})"
            else:
                _PATH_DETAIL[key] = f"exit {rc} ({_canary_tail(p.stderr)})"
        except (OSError, subprocess.TimeoutExpired) as e:
            _PATH_OK[key] = False
            _PATH_DETAIL[key] = type(e).__name__
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return _PATH_OK[key]


def path_failure(kind, dtype_name):
    """Human-readable reason recorded by the last path_works() probe for this (kind, dtype)."""
    return _PATH_DETAIL.get((kind, dtype_name), "not probed")


def find_ocloc():
    """Locate the ocloc compiler; returns (path or None, source). shutil.which only finds an unversioned ocloc on PATH;
    some distros (e.g. AOSC) install only versioned ocloc-<ver> without a symlink, so also scan /usr/bin etc. for the highest version;
    Debian/Ubuntu's intel-opencl-icd layout hides it in /usr/lib/x86_64-linux-gnu/intel-opencl/bin/."""
    path = shutil.which("ocloc")
    if path:
        return path, "path"
    libdir = "/usr/lib/x86_64-linux-gnu/intel-opencl/bin/ocloc"
    if os.access(libdir, os.X_OK):
        return libdir, "libdir"
    best, best_ver = None, (0,)
    for d in ("/usr/bin", "/usr/local/bin"):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            stem, sep, ver = name.partition("-")
            if stem != "ocloc" or not sep or not ver:
                continue
            try:
                key = tuple(int(p) for p in ver.split("."))
            except ValueError:
                continue
            if key > best_ver:
                best, best_ver = os.path.join(d, name), key
    return (best, "versioned") if best else (None, None)

def ensure_ocloc_on_path():
    """Triton-XPU spawns subprocesses with bare name 'ocloc' when producing zebin (triton/backends/intel/compiler.py),
    so XPU compile hits FileNotFoundError on distros with no symlink (e.g. AOSC ships only /usr/bin/ocloc-<ver>).
    This creates a user-level shim: ~/.cache/torchpeak/bin/ocloc -> real binary, prepended to this process's PATH,
    with cleanup registered: normal exit goes through atexit, and Ctrl-C/SIGTERM turned into SystemExit also goes through atexit;
    SIGKILL cannot be intercepted — the leftover symlink is compared by realpath on the next run, reused/retargeted, and cleaned at that exit.
    Returns the shim description (recorded in the fingerprint) or None. System-wide equivalent: sudo ln -s /usr/bin/ocloc-<ver> /usr/local/bin/ocloc"""
    path, source = find_ocloc()
    if not path or source not in ("versioned", "libdir"):
        return None
    shim_dir = os.path.join(os.path.expanduser("~"), ".cache", "torchpeak", "bin")
    link = os.path.join(shim_dir, "ocloc")
    target = os.path.realpath(path)
    try:
        if os.path.lexists(link) and os.path.realpath(link) != target:
            os.remove(link)
        if not os.path.lexists(link):
            os.makedirs(shim_dir, exist_ok=True)
            os.symlink(path, link)
        os.environ["PATH"] = shim_dir + os.pathsep + os.environ["PATH"]
    except OSError:
        return None

    global _SHIM_LINK, _SHIM_TARGET
    _SHIM_LINK, _SHIM_TARGET = link, target
    atexit.register(_shim_cleanup)

    def shim_signal(signum, frame):
        raise SystemExit(128 + signum)  # take the normal exit path so atexit reclaims the symlink

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            if signal.getsignal(sig) in (signal.SIG_DFL, signal.SIG_IGN):
                signal.signal(sig, shim_signal)
        except (ValueError, OSError):
            pass
    return f"{link} -> {path}"


_SHIM_LINK = None
_SHIM_TARGET = None


def _shim_cleanup():
    global _SHIM_LINK, _SHIM_TARGET
    link, target = _SHIM_LINK, _SHIM_TARGET
    _SHIM_LINK = _SHIM_TARGET = None
    if link and os.path.lexists(link):
        try:
            if os.path.realpath(link) == target:  # only remove our own link; leave it if it points elsewhere
                os.remove(link)
        except OSError:
            pass


def runtime_info():
    packages = {}
    for name in ("torchvision", "transformers", "torch_npu", "torch_musa", "torch_tpu", "libtpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    triton_version = None
    for pkg in ("triton", "pytorch-triton-xpu", "pytorch-triton"):
        try:
            triton_version = f"{pkg}=={importlib.metadata.version(pkg)}"
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    ocloc_path, ocloc_source = find_ocloc()
    ocloc_version = None
    if ocloc_path:
        try:
            out = subprocess.run([ocloc_path, "--version"], capture_output=True, text=True,
                                 timeout=10).stdout
            ocloc_version = next((ln.strip() for ln in out.splitlines() if ln.strip()), None)
        except (OSError, subprocess.TimeoutExpired):
            pass
    cuda_driver = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
            cuda_driver = out.splitlines()[0] if out else None
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
    cpu_governor = None
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            cpu_governor = f.read().strip()
    except OSError:
        pass
    relevant_env = {k: v for k, v in os.environ.items()
                    if k.startswith(("TORCH", "TRITON", "ZE_", "ONEAPI", "NEO", "HIP", "MIOPEN", "XPU",
                                     "XLA", "LIBTPU", "TPU_"))}
    eager_mode = None
    try:
        eager_mode = str(torch.tpu.eager_mode)
    except Exception:
        pass
    return {
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "xpu": getattr(torch.version, "xpu", None),
        "npu": getattr(torch.version, "npu", None),
        "musa": getattr(torch.version, "musa", None),
        "tpu": getattr(torch.version, "tpu", None),
        "triton": triton_version,
        "packages": packages,
        "ocloc_path": ocloc_path,
        "ocloc_source": ocloc_source,
        "ocloc_version": ocloc_version,
        "cuda_driver": cuda_driver,
        "cpu_governor": cpu_governor,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tpu_eager_mode": eager_mode,
        "relevant_env": relevant_env,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }

