"""Transverse mode shapes of every harmonic (1f..5f) at 120 Vpp resonance.

Replicates pressure_map_2d.py's mode-shape panel (kept points = circles,
sigma-clipped = gray x, dashed |sin/cos(n pi y/W)| fit labelled with the
modal amplitude P) for each harmonic n = 1..5 of the 120 Vpp cascade
scan, at the 1f-resonance drive frequency.  Each harmonic is shown at
*its own* best axial slice (max modal amplitude) so its transverse shape
is seen at its cleanest; R^2 of the |mode| fit is annotated so the shape
quality (does 3f really look like cos(3 pi y/W)?) is judgeable by eye.

Cache-only.  Output: harmonic_mode_shapes_120Vpp.png in output/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH,
    RSSI_THRESHOLD,
    get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.mode_fit import _mode_shape, _project, _r2  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry, fit_columns  # noqa: E402

SCAN = "sample_101x21_fsweep_peak_120Vpp_20260525_020136"
HARMONICS = (1, 2, 3, 4, 5)
SIGMA = 3.0
OUT_DIR = ROOT / "experiments" / "2026W21" / "output"


def build_grid(c, valid):
    pos = np.asarray(c["pos_x"])
    pos_y = np.asarray(c["pos_y"])
    rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
    hw = CHANNEL_WIDTH / 2
    p1g = np.where(valid, np.asarray(c["pressure_1f"]), np.nan)
    a_opt, b_opt = detect_channel_geometry(pos, pos_y, rssi, p1g, hw)
    pos_x_c = pos - (a_opt * pos_y + b_opt)
    return make_channel_grid(
        pos_width_c=pos_x_c,
        pos_length=pos_y,
        n_scan_width=int(c["n_x_meta"]),
        n_scan_length=int(c["n_y_meta"]),
        channel_width=CHANNEL_WIDTH,
        raw_width_span=pos.max() - pos.min(),
        inside=np.abs(pos_x_c) <= hw,
        rssi=rssi,
        rssi_threshold=RSSI_THRESHOLD,
    )


def panel(ax, grid_prs, harmonic, cg, label):
    p0_y = fit_columns(grid_prs, cg.width_grid, CHANNEL_WIDTH, harmonic=harmonic, sigma_clip=SIGMA)
    bi = int(np.nanargmax(p0_y))
    p0 = float(p0_y[bi])
    col = grid_prs[:, bi]
    cv = ~np.isnan(col)
    w_mm = cg.width_grid[cv] * 1e3
    p_kpa = col[cv] / 1e3
    mode = _mode_shape(cg.width_grid[cv], CHANNEL_WIDTH, harmonic, use_abs=True)
    _, clip = _project(col[cv], mode, sigma_clip=SIGMA)
    r2 = _r2(col[cv][clip], (p0 * mode)[clip])
    ylim = max(2 * p0 / 1e3, 1.0)

    ax.plot(w_mm[clip], p_kpa[clip], "o", ms=3, color="k", label=f"{label} data")
    ex = ~clip
    if ex.any():
        inr = p_kpa[ex] <= ylim
        if inr.any():
            ax.plot(w_mm[ex][inr], p_kpa[ex][inr], "x", ms=4, color="0.5", label="excluded")
        ab = p_kpa[ex] > ylim
        if ab.any():
            ax.plot(w_mm[ex][ab], np.full(ab.sum(), ylim), "^", ms=4, color="0.5")

    xf = np.linspace(cg.width_grid[0], cg.width_grid[-1], 300)
    ax.plot(
        xf * 1e3,
        p0 / 1e3 * _mode_shape(xf, CHANNEL_WIDTH, harmonic, use_abs=True),
        "--",
        lw=1.0,
        color="C2",
        label=f"$P$ = {p0/1e3:.0f} kPa",
    )
    ax.set_ylim(-0.1 * p0 / 1e3, ylim)
    ax.set_title(rf"{label}  ($x$={cg.length_grid[bi]*1e3:.2f} mm, $R^2$={r2:.2f})", fontsize=9)
    ax.set_xlabel(r"width $y$ [mm]")
    ax.set_ylabel("P [kPa]")
    ax.legend(fontsize=6, frameon=False)
    return p0, r2


def main() -> None:
    from ldv_analysis.config import LDV_DATA_ROOT

    parser = argparse.ArgumentParser(
        description="1f-5f transverse mode shapes at a scan's 1f resonance"
    )
    parser.add_argument(
        "scan", nargs="?", default=SCAN, help="scan dir name under <DATA_ROOT>/output/W21/"
    )
    args = parser.parse_args()
    scan = args.scan
    tag = scan.split("_2026")[0].replace("sample_", "")

    run_dir = LDV_DATA_ROOT / "output" / "W21" / scan
    cache_dir = get_cache_dir(scan, __file__)
    files = sorted(p for p in run_dir.glob("*.h5") if not p.name.endswith(".inprogress"))

    # Pass 1: find the 1f-resonance file (max 1f modal amplitude).
    cg = None
    best = (-1.0, None, None)  # (p1_peak, path, f_drive)
    for p in files:
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        valid = make_valid_mask(
            np.asarray(c["voltage_1f"]), np.asarray(c["rssi"]) if "rssi" in c.files else None
        )
        if int(np.sum(valid)) < 3:
            continue
        if cg is None:
            cg = build_grid(c, valid)
        g1 = cg.to_grid(np.where(valid, np.asarray(c["pressure_1f"]), np.nan))
        peak = float(
            np.nanmax(fit_columns(g1, cg.width_grid, CHANNEL_WIDTH, harmonic=1, sigma_clip=SIGMA))
        )
        if peak > best[0]:
            best = (peak, p, float(c["f_drive"]))

    _, res_path, f_res = best
    print(f"Resonance file: {res_path.name}  (f_drive = {f_res/1e6:.4f} MHz)")

    c = load_or_compute(res_path, cache_dir, velocity_scale=None)
    valid = make_valid_mask(
        np.asarray(c["voltage_1f"]), np.asarray(c["rssi"]) if "rssi" in c.files else None
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    axflat = axes.flat
    for j, n in enumerate(HARMONICS):
        grid = cg.to_grid(np.where(valid, np.asarray(c[f"pressure_{n}f"]), np.nan))
        p0, r2 = panel(axflat[j], grid, n, cg, f"{n}f")
        print(f"  {n}f: P = {p0/1e3:8.1f} kPa,  R^2 = {r2:.3f}")
    axflat[-1].axis("off")

    fig.suptitle(
        f"{tag} --- transverse mode shapes 1f-5f "
        f"at resonance ($f_{{1f}}$ = {f_res/1e6:.3f} MHz, drive harmonics at $nf$)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT_DIR / f"harmonic_mode_shapes_{tag}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
