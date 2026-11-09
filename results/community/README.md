# Community results

Community-contributed benchmark runs, one JSON per device. See "Submit your results" in the
top-level README for the run recipe and rules; `scripts/validate_result.py` checks every file
in CI.

| File | Device | Notes |
| --- | --- | --- |
| [gtx1660ti.json](gtx1660ti.json) | NVIDIA GeForce GTX 1660 Ti (Turing, 6 GB) | full default suite, torch 2.15.0+cu132 |

Measured values here feed `peaks.json` entries: `basis: "measured, see results/community/<file>"`.
HTML reports are not stored — they regenerate from the JSON (`meta` + `results`).
