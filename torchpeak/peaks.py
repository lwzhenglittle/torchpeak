"""torchpeak.peaks — see the package root docstring."""
import json
import os

# ---------------------------------------------------------------- hardware peak table (spec-sheet basis)
# Data lives in peaks.json (next to the script, or $TORCHPEAK_PEAKS / CWD), entry format:
#   {"match": device-name substring (case-insensitive),
#    "device": display name, "tflops": {dtype: matrix-path peak}, "vec_tflops": {dtype: vector-path peak},
#    "vec_packed_tflops": {dtype: packed-path peak}, "int8_tops": value, "bw_gbs": local bandwidth,
#    "host_bw_gbs": optional host interconnect bandwidth (H2D/D2H only; spec efficiency is not reported when missing),
#    "tflops_fp32_mode": optional {highest|high|medium: rate} for devices without a native fp32
#                        matmul unit (TPU: the MXU runs 1/3/6 bf16 passes per mode),
#    "basis": basis description, "sources": [sources]}
# Matching takes the first substring hit in entry order; more specific entries (a100-sxm4-80gb) precede generic ones (a100).
# Entries passed via --peaks precede the built-in table (matched first). Without peaks.json, fall back to the "measured-only ceiling" basis.
PEAKS_ENV = "TORCHPEAK_PEAKS"


def load_peak_entries(path=None):
    """Return (entries, peaks_source). Priority: --peaks entries > $TORCHPEAK_PEAKS > script-dir peaks.json > CWD."""
    user = []
    if path:
        with open(path) as f:
            user = json.load(f)
    source = None
    bundled = []
    candidates = []
    env_path = os.environ.get(PEAKS_ENV)
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(here, "peaks.json"), os.path.join(os.getcwd(), "peaks.json")]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                with open(cand) as f:
                    bundled = json.load(f)
                source = cand
                break
            except (OSError, ValueError):
                continue
    return user + bundled, source


def match_peak(info, entries):
    name = (info.get("name") or "").lower()
    for e in entries:
        if e.get("match", "").lower() in name:
            return e
    return None


def tpu_spec(info):
    """Spec denominator from the backend's own device properties, so a new TPU generation needs no
    static table entry. TorchTPU reports the dense bf16 MXU rate and HBM size per device; fp32 has no
    native matmul unit (binary32 is emulated by N bf16 MXU passes: 6 for "highest", 3 for "high",
    1 for the default "medium"), so fp32 has one peak per precision mode instead of one number."""
    props = (info or {}).get("device_properties") or {}
    bf16 = props.get("bf16_flops_per_device") or props.get("bf16_flops_per_chip")
    if not bf16:
        return None
    tf = bf16 / 1e12
    hbm = props.get("hbm_bytes_per_device") or props.get("hbm_bytes_per_chip")
    return {
        "match": "tpu",
        "device": props.get("name") or (info or {}).get("name") or "TPU",
        "tflops": {"bf16": round(tf, 3), "fp16": round(tf, 3), "fp32": round(tf / 6, 3)},
        "tflops_fp32_mode": {"highest": round(tf / 6, 3), "high": round(tf / 3, 3), "medium": round(tf, 3)},
        "bw_gbs": round(hbm / 1e9, 1) if hbm else None,
        "basis": ("torch.tpu.get_device_properties: bf16_flops_per_device is the 1-pass MXU rate; "
                  "fp32/fp16 matmuls have no native unit and run as MXU passes (6 for 'highest', "
                  "3 for 'high', 1 bf16 pass for the default 'medium'); bw = hbm_bytes_per_device"),
        "sources": ["torch.tpu.get_device_properties()",
                    "https://torchtpu.readthedocs.io/en/latest/numerical_precision/"],
    }


def load_peaks(info, path=None):
    entries, _ = load_peak_entries(path)
    return match_peak(info, entries)

