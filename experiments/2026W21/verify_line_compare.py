# %%
"""Resume sanity check: 60V verify line vs prior 60V baselines.

For every dataset (2D scans and line scans alike) reduces the data to the
SAME lumped |sin| fit on the y=3.0mm row (row 10 of 21 for the 2D scans;
the single row for line scans) via :func:`fit_axial` forced to 1D, so
P_1f / freq / R^2 / drive-current differences across runs are method-
independent (apples-to-apples even between a full 2D and a 1D line).

The :data:`DATASETS` list at the top is the single source of truth -- add
a new entry when a new verify scan arrives (same y=3.0mm convention).
Prints a per-frequency table and the peak summary per dataset; useful for
checking whether the device has drifted, been restored, or whether an
LDV-range / scaling change is hiding inside the metadata.

Usage::

    python experiments/2026W21/verify_line_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import CHANNEL_WIDTH  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.sweep_fit import fit_axial  # noqa: E402

OUT = ROOT / "experiments" / "2026W21" / "output"
FREQS = list(range(1890, 1911, 2))  # kHz
# (label, dirname, is_line). is_line=True for 1D verify lines; False for
# full 2D scans (we extract row 10 of 21 = y=3.0mm = y_start 2.0mm + 10*0.1mm).
DATASETS = [
    ("05-24 reference (2D row)",     "sample_101x21_fsweep_peak_60Vpp_20260524_223919", False),
    ("05-26 15:50 drifted (2D row)", "sample_101x21_fsweep_peak_60Vpp_20260526_155045", False),
    ("05-26 18:29 re-leveled (2D)",  "sample_101x21_fsweep_peak_60Vpp_20260526_182949", False),
    ("05-26 line 15:38",             "verify_peak60_line_y3_20260526_153834",           True),
    ("05-29 line 23:14 (2m/s mis)",  "verify_peak60_line_y3_20260529_231455",           True),
    ("05-30 line 00:04 (6m/s set)",  "verify_peak60_line_y3_20260530_000443",           True),
]
ROW = 10  # y_start 2.0mm + ROW*y_step (100um) = 3.0mm  -- must match the verify-line y


class _SubCache:
    """Minimal npz-like wrapper exposing .files / __getitem__ for fit_axial."""

    def __init__(self, d):
        self._d, self.files = d, list(d.keys())

    def __getitem__(self, k):
        return self._d[k]


def _fit_row(dirname: str, f_khz: int, is_line: bool):
    """Lumped |sin| fit at the y=3.0mm row for either a 1D line or a 2D scan."""
    c = np.load(OUT / dirname / "fft_cache" / f"_fft_cache_f{f_khz:04d}000.npz")
    if is_line:
        cl = c
    else:
        pos_y = np.asarray(c["pos_y"])
        y_target = np.unique(pos_y)[ROW]
        mask = pos_y == y_target
        n = len(pos_y)
        d = {k: (np.asarray(c[k])[mask] if np.asarray(c[k]).shape == (n,)
                 else np.asarray(c[k])) for k in c.files}
        d["n_x_meta"], d["n_y_meta"] = int(mask.sum()), 1
        cl = _SubCache(d)
    V = np.asarray(cl["voltage_1f"])
    rssi = np.asarray(cl["rssi"]) if "rssi" in cl.files else None
    valid = make_valid_mask(V, rssi)
    fit, _ = fit_axial(cl, valid, CHANNEL_WIDTH)
    I = np.asarray(cl["current_1f"])
    return fit.p1_mag / 1e3, fit.r2_1, float(np.median(np.abs(I[valid]))) * 1e3


def main() -> None:
    hdr = "  freq[kHz]"
    for lbl, _, _ in DATASETS:
        hdr += f" | {lbl[:24]:>24}"
    print(hdr)

    table = {lbl: [] for lbl, _, _ in DATASETS}
    for f in FREQS:
        row = f"  {f:9d}"
        for lbl, d, is_line in DATASETS:
            p, r2, i = _fit_row(d, f, is_line)
            table[lbl].append((p, r2, i))
            row += f" | {p:7.0f} R2={r2:.2f} I={i:.0f}".rjust(26)
        print(row)

    print()
    for lbl in [x[0] for x in DATASETS]:
        arr = np.array([t[0] for t in table[lbl]])
        ipk = int(np.argmax(arr))
        f_pk = FREQS[ipk]
        print(f"  {lbl:30s}  peak {arr[ipk]:5.0f} kPa @ {f_pk} kHz  "
              f"(R2 {table[lbl][ipk][1]:.2f}, I {table[lbl][ipk][2]:.1f} mA)")


if __name__ == "__main__":
    main()
