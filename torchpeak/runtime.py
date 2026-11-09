"""torchpeak.runtime — see the package root docstring."""
import atexit
import importlib.metadata
import os
import platform
import shutil
import signal
import subprocess
import sys

import torch

AMP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ---------------------------------------------------------------- device helpers
def pick_device(name):
    if name != "auto":
        return torch.device(name)
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
    return getattr(torch, dev.type, None)  # torch.cuda / torch.xpu / torch.mps / torch.cpu


def sync(dev):
    m = mod(dev)
    if m is not None and hasattr(m, "synchronize"):
        m.synchronize()


def reset_peak(dev):
    m = mod(dev)
    if m is not None and hasattr(m, "reset_peak_memory_stats"):
        try:
            m.reset_peak_memory_stats()
        except Exception:
            pass


def peak_gb(dev):
    m = mod(dev)
    if m is not None and hasattr(m, "max_memory_allocated"):
        try:
            return round(m.max_memory_allocated() / 1e9, 2)
        except Exception:
            return None
    return None


def peak_reserved_gb(dev):
    """Caching-allocator reserved bytes; on iGPU/UMA this diverges a lot from allocated, so record both."""
    m = mod(dev)
    if m is not None and hasattr(m, "max_memory_reserved"):
        try:
            return round(m.max_memory_reserved() / 1e9, 2)
        except Exception:
            return None
    return None


def empty_cache(dev):
    m = mod(dev)
    if m is not None and hasattr(m, "empty_cache"):
        try:
            m.empty_cache()
        except Exception:
            pass


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
    for name in ("torchvision", "transformers", "torch_npu", "torch_musa"):
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
                    if k.startswith(("TORCH", "TRITON", "ZE_", "ONEAPI", "NEO", "HIP", "MIOPEN", "XPU"))}
    return {
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "xpu": getattr(torch.version, "xpu", None),
        "npu": getattr(torch.version, "npu", None),
        "musa": getattr(torch.version, "musa", None),
        "triton": triton_version,
        "packages": packages,
        "ocloc_path": ocloc_path,
        "ocloc_source": ocloc_source,
        "ocloc_version": ocloc_version,
        "cuda_driver": cuda_driver,
        "cpu_governor": cpu_governor,
        "relevant_env": relevant_env,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }

