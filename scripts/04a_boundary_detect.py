# %%
"""Explore channel boundary detection from 2D pressure maps.

Detects channel boundaries by minimising the sum of pressure² outside
a strip of known width (375 µm) with parallel edges.

Parameters: y_left, y_right (centre-line position at left/right edges
of the scan).  Boundary lines:
    centre(x) = y_left + (y_right - y_left) / (x_max - x_min) * (x - x_min)
    upper(x)  = centre(x) + w/2
    lower(x)  = centre(x) - w/2

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CONVERTED_DIR,
    EXCLUDED_FILES,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_output_dir,
)

OUT_DIR = get_output_dir(__file__)

CHANNEL_WIDTH = 0.375  # mm (known physical width)

# %%
# =============================================================================
# Load data (101x101 only)
# =============================================================================

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
npz_files = [f for f in npz_files if f.stem + ".tdms" not in EXCLUDED_FILES]
npz_files = [f for f in npz_files if "101x101" in f.stem]

if not npz_files:
    print("No converted files found.")
    sys.exit(0)

for npz_path in npz_files:
    stem = npz_path.stem
    print(f"--- {stem} ---")

    data = np.load(npz_path)

    # Scan coords → channel coords (swap once, use everywhere below)
    ch_length = data["scan_pos_y"]   # channel length = horizontal axis
    ch_width = data["scan_pos_x"]    # channel width  = vertical axis
    n_length = int(data["meta_n_y"])  # grid points along length
    n_width = int(data["meta_n_x"])   # grid points along width

    wf1 = data["wf_ch1"]
    wf2 = data["wf_ch2"]
    dt = float(data["wf_dt"])
    n_points = wf1.shape[0]
    n_samples = wf1.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    if n_length <= 2:
        continue

    print(f"  {n_points} scan points, grid {n_length} × {n_width}")

    # %%
    # =================================================================
    # FFT → pressure at 1f
    # =================================================================

    fft_v = np.fft.rfft(wf1, axis=1)
    fft_vel = np.fft.rfft(wf2, axis=1)
    peak_idx = np.argmax(np.abs(fft_v[:, 1:]), axis=1) + 1
    pts = np.arange(n_points)
    drive_freqs = freqs[peak_idx]

    velocity_1f = np.abs(fft_vel[pts, peak_idx]) * 2 / n_samples * VELOCITY_SCALE
    pressure_1f = velocity_1f / (2 * np.pi * drive_freqs * SENSITIVITY)

    # %%
    # =================================================================
    # Build 2D grid
    # =================================================================

    length_grid = np.linspace(ch_length.min(), ch_length.max(), n_length)
    width_grid = np.linspace(ch_width.min(), ch_width.max(), n_width)

    l_idx = np.argmin(np.abs(ch_length[:, None] - length_grid[None, :]), axis=1)
    w_idx = np.argmin(np.abs(ch_width[:, None] - width_grid[None, :]), axis=1)

    grid_prs = np.full((n_width, n_length), np.nan)
    grid_prs[w_idx, l_idx] = pressure_1f / 1e3  # kPa

    # %%
    # =================================================================
    # Boundary detection: minimise pressure² outside the strip
    # =================================================================

    hw = CHANNEL_WIDTH / 2
    x_min, x_max = ch_length.min(), ch_length.max()

    # Parameter bounds: centre ± w/2 must stay inside scan range
    w_min, w_max = ch_width.min(), ch_width.max()
    y_lo = w_min + hw
    y_hi = w_max - hw

    prs_sq = pressure_1f**2

    def outside_pressure_sum(params):
        y_left, y_right = params
        # Centre line at each scan point
        centre = y_left + (y_right - y_left) / (x_max - x_min) * (ch_length - x_min)
        outside = np.abs(ch_width - centre) > hw
        return np.nansum(prs_sq[outside])

    # Brute-force grid search over (y_left, y_right)
    N_GRID = 100
    result = brute(outside_pressure_sum,
                   ranges=((y_lo, y_hi), (y_lo, y_hi)),
                   Ns=N_GRID, finish=None)
    y_left_opt, y_right_opt = result

    # Derived parameters
    a_opt = (y_right_opt - y_left_opt) / (x_max - x_min)
    b_opt = y_left_opt - a_opt * x_min
    tilt_deg = np.degrees(np.arctan(a_opt))

    print(f"  y_left={y_left_opt:.4f}, y_right={y_right_opt:.4f}")
    print(f"  Tilt: {tilt_deg:.3f} deg, width: {CHANNEL_WIDTH} mm (fixed)")

    # %%
    # =================================================================
    # Plot: pressure map with boundary overlay
    # =================================================================

    boundary_x = length_grid
    boundary_centre = y_left_opt + (y_right_opt - y_left_opt) / (x_max - x_min) * (boundary_x - x_min)
    boundary_upper = boundary_centre + hw
    boundary_lower = boundary_centre - hw

    length_span = length_grid[-1] - length_grid[0]
    width_span = width_grid[-1] - width_grid[0]
    fig_w = figsize_for_layout(ax_w_scale=2.5)[0]
    fig_h = fig_w * (width_span / length_span) * 3
    fig_h = max(fig_h, 1.5)

    fig, ax = plt.subplots(figsize=(fig_w + 1.2, fig_h + 1.0))

    lo, hi = np.nanpercentile(grid_prs, [5, 95])
    im = ax.pcolormesh(length_grid, width_grid, grid_prs, shading="nearest",
                       cmap="viridis", vmin=lo, vmax=hi)

    ax.plot(boundary_x, boundary_upper, "r-", linewidth=1.0, label="Upper boundary")
    ax.plot(boundary_x, boundary_lower, "r--", linewidth=1.0, label="Lower boundary")
    ax.plot(boundary_x, boundary_centre, "r:", linewidth=0.5, label="Centre line")

    ax.set_xlabel("Channel length, x (mm)")
    ax.set_ylabel("Channel width, y (mm)")
    ax.set_title(f"Boundary detection --- {stem}")
    ax.set_aspect("auto")
    ax.legend(loc="lower right", fontsize=6)
    plt.colorbar(im, ax=ax, label="Acoustic pressure (kPa)")
    plt.tight_layout()

    output_path = OUT_DIR / f"{stem}_boundary.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")

    # %%
    # =================================================================
    # Plot: pressure map in centred coordinates (y' = y - centre(x))
    # =================================================================

    ch_width_c = ch_width - (a_opt * ch_length + b_opt)

    # Grid only within channel boundaries, excluding edge rows
    inside_c = np.abs(ch_width_c) <= hw
    scan_step = width_span / max(n_width - 1, 1)
    n_width_c = max(int(round(CHANNEL_WIDTH / scan_step)), 2)
    half_step = (hw * 2) / n_width_c / 2
    width_c_grid = np.linspace(-hw + half_step, hw - half_step, n_width_c)

    w_c_idx = np.argmin(np.abs(ch_width_c[:, None] - width_c_grid[None, :]), axis=1)

    grid_prs_c = np.full((n_width_c, n_length), np.nan)
    mask = inside_c & ~np.isnan(pressure_1f)
    grid_prs_c[w_c_idx[mask], l_idx[mask]] = pressure_1f[mask] / 1e3  # kPa

    width_c_span = CHANNEL_WIDTH
    fig_h_c = fig_w * (width_c_span / length_span) * 3
    fig_h_c = max(fig_h_c, 1.5)

    fig, ax = plt.subplots(figsize=(fig_w + 1.2, fig_h_c + 1.0))

    lo, hi = np.nanpercentile(grid_prs_c, [5, 95])
    im = ax.pcolormesh(length_grid, width_c_grid, grid_prs_c, shading="nearest",
                       cmap="viridis", vmin=lo, vmax=hi)

    ax.set_xlabel("Channel length, x (mm)")
    ax.set_ylabel("Channel width, y (mm)")
    ax.set_title(f"Centred coordinates --- {stem}")
    ax.set_aspect("auto")
    plt.colorbar(im, ax=ax, label="Acoustic pressure (kPa)")
    plt.tight_layout()

    output_path = OUT_DIR / f"{stem}_centred.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")

print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
