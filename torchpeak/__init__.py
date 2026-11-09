"""torchpeak — cross-platform PyTorch performance benchmark (single-command, cross-stack, peak-normalized)."""
from torchpeak.cli import main
from torchpeak.peaks import PEAKS_ENV, load_peak_entries, load_peaks, match_peak

__version__ = "0.1.0"
__all__ = ["main", "PEAKS_ENV", "load_peak_entries", "load_peaks", "match_peak", "__version__"]
