"""torchpeak.peaks — see the package root docstring."""
import json
import os

# ---------------------------------------------------------------- hardware peak table (spec-sheet basis)
# Data lives in peaks.json (next to the script, or $TORCHPEAK_PEAKS / CWD), entry format:
#   {"match": device-name substring (case-insensitive),
#    "device": display name, "tflops": {dtype: matrix-path peak}, "vec_tflops": {dtype: vector-path peak},
#    "vec_packed_tflops": {dtype: packed-path peak}, "int8_tops": value, "bw_gbs": local bandwidth,
#    "host_bw_gbs": optional host interconnect bandwidth (H2D/D2H only; spec efficiency is not reported when missing),
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


def load_peaks(info, path=None):
    entries, _ = load_peak_entries(path)
    return match_peak(info, entries)

