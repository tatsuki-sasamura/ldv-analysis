# %%
"""Montage of 1f pressure maps, one panel per scan at its P_1f peak frequency.

A quick diagnostic for a series of scans (e.g. a voltage sweep): for each
scan directory it auto-detects the P_1f peak frequency (via
:func:`ldv_analysis.sweep_fit.sweep_peaks`), grids the 1f pressure at that
frequency in channel-centered coordinates (same geometry as
``pressure_map_2d.py``), and stacks one panel per scan so the mode shape
can be compared across the series at a glance.

The scan series is shared with ``vpp_vs_pressure.py`` (its ``SCANS``,
``DATA_ROOT`` and ``OUT_DIR``), so adding a voltage point in one place
updates both the cascade and this montage.

Color scale:
  * default -- each panel uses its own 5-95 percentile scale, so the mode
    shape is visible at every amplitude (annotated with mode-fit P_1f and
    brightest pixel);
  * ``--shared-scale`` -- one common 0..max scale across panels, so the
    absolute amplitude growth is obvious at the cost of washing out the
    lowest-amplitude panels.

Usage::

    python experiments/2026W21/peak_map_montage.py
    python experiments/2026W21/peak_map_montage.py --shared-scale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, RSSI_THRESHOLD,
)
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.sweep_fit import (  # noqa: E402
    detect_channel_geometry, sweep_peaks,
)
from vpp_vs_pressure import DATA_ROOT, OUT_DIR, SCANS  # noqa: E402  (single source of the scan series)

HW = CHANNEL_WIDTH / 2


def _peak_grid(label: int, dirname: str):
    """Return (cg, grid_kpa, f_khz, peak_p1_kpa) for a scan's peak freq."""
    run_dir = DATA_ROOT / dirname
    cache_dir = OUT_DIR / dirname / "fft_cache"
    sp = sweep_peaks(run_dir, CHANNEL_WIDTH, cache_dir)
    f_khz = int(round(sp.peak_p1_freq_mhz * 1000))
    c = np.load(cache_dir / f"_fft_cache_f{f_khz:04d}000.npz")

    pos_x = np.asarray(c["pos_x"])
    pos_y = np.asarray(c["pos_y"])
    rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
    V = np.asarray(c["voltage_1f"])
    valid = make_valid_mask(V, rssi)
    P1 = np.asarray(c["pressure_1f"]).copy()
    P1[~valid] = np.nan

    a, b = detect_channel_geometry(pos_x, pos_y, rssi, P1, HW)
    pos_x_c = pos_x - (a * pos_y + b)
    inside = np.abs(pos_x_c) <= HW
    cg = make_channel_grid(
        pos_width_c=pos_x_c, pos_length=pos_y,
        n_scan_width=int(c["n_x_meta"]), n_scan_length=int(c["n_y_meta"]),
        channel_width=CHANNEL_WIDTH, raw_width_span=pos_x.max() - pos_x.min(),
        inside=inside, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )
    grid_kpa = cg.to_grid(P1) / 1e3
    return cg, grid_kpa, f_khz, sp.peak_p1_kpa


def main(shared_scale: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = []
    for label, dirname in SCANS:
        try:
            cg, grid_kpa, f_khz, peak_p1 = _peak_grid(label, dirname)
        except (ValueError, FileNotFoundError) as e:
            print(f"{label:>5}  -- skipped ({e})")
            continue
        panels.append((label, cg, grid_kpa, f_khz, peak_p1))

    if not panels:
        print("No scans available.")
        return

    vmax = max(float(np.nanmax(g)) for _, _, g, _, _ in panels) \
        if shared_scale else None

    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 1.25 * nrows + 1),
                             sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, (label, cg, grid_kpa, f_khz, peak_p1) in zip(axes, panels):
        if shared_scale:
            lo, hi = 0.0, vmax
        else:
            lo, hi = np.nanpercentile(grid_kpa, [5, 95])
        im = ax.pcolormesh(cg.length_grid * 1e3, cg.width_grid * 1e3,
                           grid_kpa, shading="nearest", cmap="viridis",
                           vmin=lo, vmax=hi)
        ax.set_aspect("auto")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{label} Vpp  |  peak {f_khz} kHz  |  "
                     f"mode-fit P1f = {peak_p1:.0f} kPa, "
                     f"max pix = {np.nanmax(grid_kpa):.0f} kPa", fontsize=8)
        fig.colorbar(im, ax=ax, label="kPa", fraction=0.025, pad=0.01)

    axes[-1].set_xlabel("Channel length, x [mm]")
    scale_note = "shared color scale" if shared_scale else "per-panel color scale"
    fig.suptitle("1f pressure map at each scan's P_1f peak frequency "
                 f"({scale_note})", fontsize=10)
    fig.tight_layout()
    suffix = "_shared" if shared_scale else ""
    out_path = OUT_DIR / f"peak_map_montage{suffix}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-scale", action="store_true",
                        help="use one common color scale across all panels")
    args = parser.parse_args()
    main(shared_scale=args.shared_scale)
