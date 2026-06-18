# %%
"""Side-by-side visualisation of the two W21 2f-band cavity modes.

The W21 fine-scan + survey resolve two close n=2 transverse modes in
the 2f band at 3.794 MHz (dominant) and ~3.818 MHz (secondary), both
projecting cleanly onto ``cos(2 pi y/W)``.  They differ in their axial
structure (different ``q_y`` axial-order content), which is what makes
them distinct cavity eigenmodes.

This script visualises both side-by-side from the fine scan's f3794000
and f3818000 .h5 files:
  - 2D ``|P_1f|`` maps over the same colour scale
  - axial profile ``p_0(x)`` = best-per-column cos(2 pi y/W) amplitude
  - cross-section at each mode's strongest axial slice (cos fit overlay)

Output:
    output/two_mode_compare/two_mode_compare.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, LDV_DATA_ROOT, RSSI_THRESHOLD,
    figsize_for_layout, get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.mode_fit import fit_columns  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry  # noqa: E402

SCAN_DIR = "sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_20260530_072344"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "two_mode_compare"

PEAKS_HZ = [3_794_000, 3_818_000]
PEAK_LABELS = ["mode 1: 3.794 MHz", "mode 2: 3.818 MHz"]


def load_field(f_hz: int, geom=None):
    """Return (cg, p_grid, p0_y_cos2, valid, geom) for one .h5 file."""
    run_dir = DATA_ROOT / SCAN_DIR
    cache_dir = get_cache_dir(SCAN_DIR, __file__)
    h5_path = run_dir / f"f{f_hz}.h5"
    c = load_or_compute(h5_path, cache_dir, velocity_scale=None)
    V = np.asarray(c["voltage_1f"])
    rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
    valid = make_valid_mask(V, rssi)
    P_abs = np.asarray(c["pressure_1f"])
    pos_x = np.asarray(c["pos_x"])
    pos_y = np.asarray(c["pos_y"])
    n_x = int(c["n_x_meta"])
    n_y = int(c["n_y_meta"])

    hw = CHANNEL_WIDTH / 2
    if geom is None:
        Pg = P_abs.copy()
        Pg[~valid] = np.nan
        geom = detect_channel_geometry(pos_x, pos_y, rssi, Pg, hw)
    a, b = geom
    pos_x_c = pos_x - (a * pos_y + b)
    inside_c = np.abs(pos_x_c) <= hw

    cg = make_channel_grid(
        pos_width_c=pos_x_c, pos_length=pos_y,
        n_scan_width=n_x, n_scan_length=n_y,
        channel_width=CHANNEL_WIDTH, raw_width_span=pos_x.max() - pos_x.min(),
        inside=inside_c, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    p_grid = cg.to_grid(np.where(valid, P_abs, np.nan))   # (n_length, n_width)
    p0_y = fit_columns(p_grid, cg.width_grid, CHANNEL_WIDTH,
                       harmonic=2, sigma_clip=3.0)         # (n_length,)
    return cg, p_grid, p0_y, valid, geom


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load both modes (share geometry detection from mode 1).
    cg1, P1, p0_y1, _, geom = load_field(PEAKS_HZ[0])
    cg2, P2, p0_y2, _, _ = load_field(PEAKS_HZ[1], geom=geom)

    # Best axial slice per mode (for the cross-section)
    ix1 = int(np.nanargmax(p0_y1))
    ix2 = int(np.nanargmax(p0_y2))
    p0_peak1 = float(p0_y1[ix1])
    p0_peak2 = float(p0_y2[ix2])
    print(f"mode 1 ({PEAKS_HZ[0]/1e6:.3f} MHz): "
          f"axial peak at x = {cg1.length_grid[ix1]*1e3:.2f} mm, "
          f"|cos2| = {p0_peak1/1e3:.1f} kPa")
    print(f"mode 2 ({PEAKS_HZ[1]/1e6:.3f} MHz): "
          f"axial peak at x = {cg2.length_grid[ix2]*1e3:.2f} mm, "
          f"|cos2| = {p0_peak2/1e3:.1f} kPa")

    # Common colour scale for the 2D maps
    vmax = float(np.nanmax([np.nanmax(P1), np.nanmax(P2)])) / 1e3

    fig = plt.figure(figsize=figsize_for_layout(3, 2, sharex=False))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)

    for col, (f_hz, label, P, p0_y, ix, p_peak, cg) in enumerate([
        (PEAKS_HZ[0], PEAK_LABELS[0], P1, p0_y1, ix1, p0_peak1, cg1),
        (PEAKS_HZ[1], PEAK_LABELS[1], P2, p0_y2, ix2, p0_peak2, cg2),
    ]):
        # row 0: 2D |P_1f| map
        ax = fig.add_subplot(gs[0, col])
        ext = [
            cg.length_grid.min() * 1e3, cg.length_grid.max() * 1e3,
            cg.width_grid.min() * 1e3, cg.width_grid.max() * 1e3,
        ]
        # grid shape is (n_width, n_length): rows -> y, cols -> x. No transpose.
        im = ax.imshow(P / 1e3, origin="lower", extent=ext, aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.axvline(cg.length_grid[ix] * 1e3, color="white", lw=0.6, ls="--")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{label}\n$|P_{{1f}}|$ map")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("[kPa]", fontsize=7)

        # row 1: axial p_0(x) = best cos(2 pi y/W) amplitude per column
        ax = fig.add_subplot(gs[1, col])
        ax.plot(cg.length_grid * 1e3, p0_y / 1e3, "o-",
                markersize=2, linewidth=0.6, color="C0")
        ax.axvline(cg.length_grid[ix] * 1e3, color="C3", lw=0.5, ls="--",
                   label=f"peak: $x = ${cg.length_grid[ix]*1e3:.2f} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel(r"$p_0(x)$  [kPa]")
        ax.set_title(r"axial $\cos(2\pi y/W)$ amplitude")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)

        # row 2: transverse cross-section at best axial slice + cos fit
        ax = fig.add_subplot(gs[2, col])
        y = cg.width_grid * 1e3                          # mm
        # P shape (n_width, n_length); slice the width column at axial ix.
        col_data = P[:, ix] / 1e3                        # kPa
        y_dense = np.linspace(y.min(), y.max(), 400)
        fit = (p_peak / 1e3) * np.abs(
            np.cos(2 * np.pi * y_dense * 1e-3 / CHANNEL_WIDTH))
        ax.plot(y, col_data, "o", markersize=3, color="0.2", label="data")
        ax.plot(y_dense, fit, "--", color="C2", linewidth=0.8,
                label=fr"fit: $|\cos(2\pi y/W)|\cdot{p_peak/1e3:.0f}$ kPa")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("|P| [kPa]")
        ax.set_title("transverse cross-section at peak axial slice")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Two W21 n=2 cavity modes in the 2f band  "
        f"(both project cleanly on $\\cos(2\\pi y/W)$; "
        f"differ in axial $q_y$ structure)",
        y=1.0,
    )
    out_path = OUT_DIR / "two_mode_compare.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
