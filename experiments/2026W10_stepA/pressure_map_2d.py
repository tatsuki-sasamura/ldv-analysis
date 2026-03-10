# %%
"""2D spatial maps of acoustic pressure from burst-mode area scan.

Generates pcolormesh heatmaps (velocity, pressure, phase, RSSI) for a
Step A area-scan TDMS file.  Channel boundaries are detected by
minimising pressure² outside a strip of known width (375 µm), then data
is displayed in centred channel coordinates.

Unlike 04_2d_map.py this script loads TDMS directly (no pre-conversion)
and extracts the steady-state portion of burst-mode waveforms for FFT.

Usage:
    python pressure_map_2d.py <path_to_tdms>
    python pressure_map_2d.py <path_to_tdms> --harmonics  # also extract 2f
    python pressure_map_2d.py                              # uses default path
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute

from ldv_analysis.config import (
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute

# LDV range → velocity scale: 1 m/s → 0.5 m/s/V, 2 m/s → 1.0, 5 m/s → 2.5
LDV_RANGE_TO_SCALE = {1: 0.5, 2: 1.0, 5: 2.5}

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = (
    get_data_dir("20260303experimentA")
    / "stepA1967_where_is_the_best_x_position.tdms"
)

# Channel geometry
CHANNEL_WIDTH = 0.375  # mm (known physical width)
RSSI_THRESHOLD = 1.0   # V — exclude poor LDV signal in mode-shape fit

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data (with FFT cache for fast re-runs)
# =============================================================================

parser = argparse.ArgumentParser(description="2D spatial maps of acoustic pressure")
parser.add_argument("tdms_path", nargs="?", default=str(DEFAULT_TDMS),
                    help="Path to TDMS area-scan file")
parser.add_argument("--harmonics", action="store_true",
                    help="Extract 2f harmonic and generate comparison plots")
parser.add_argument("--ldv-range", type=int, choices=[1, 2, 5], default=None,
                    help="LDV velocity range in m/s (auto-detected from filename if not set)")
parser.add_argument("--channel-centre", type=float, default=None,
                    help="Fixed channel centre in mm (skip boundary detection, assume zero tilt)")
args = parser.parse_args()
tdms_path = Path(args.tdms_path)
compute_harmonics = args.harmonics
stem = tdms_path.stem

# Determine actual velocity scale from LDV range
if args.ldv_range is not None:
    actual_vel_scale = LDV_RANGE_TO_SCALE[args.ldv_range]
else:
    m = re.search(r"_(\d+)m_s_max", stem)
    if m:
        actual_vel_scale = LDV_RANGE_TO_SCALE[int(m.group(1))]
    else:
        actual_vel_scale = VELOCITY_SCALE  # default 1.0 m/s/V (2 m/s range)
vel_correction = actual_vel_scale / VELOCITY_SCALE

print(f"Loading: {tdms_path.name}")
if vel_correction != 1.0:
    print(f"  LDV range: {actual_vel_scale * 2:.0f} m/s max "
          f"(VELOCITY_SCALE = {actual_vel_scale} m/s/V, "
          f"correction = {vel_correction:.2f}x)")

cache = load_or_compute(tdms_path, CACHE_DIR)

pos_x = cache["pos_x"]
pos_y = cache["pos_y"]
n_x_meta = int(cache["n_x_meta"])
n_y_meta = int(cache["n_y_meta"])
f_drive = float(cache["f_drive"])
velocity_1f = cache["velocity_1f"] * vel_correction
pressure_1f = cache["pressure_1f"] * vel_correction
phase_1f = cache["phase_1f"]
rssi = cache["rssi"] if "rssi" in cache else None
n_points = len(pos_x)

print(f"  Pressure 1f: mean {np.mean(pressure_1f)/1e3:.1f} kPa, "
      f"max {np.max(pressure_1f)/1e3:.1f} kPa")

# %%
# =============================================================================
# Channel boundary detection
# =============================================================================
# pos_x = channel width direction, pos_y = channel length direction
# Fit tilted channel centre: centre(y) = a*y + b

hw = CHANNEL_WIDTH / 2
x_min, x_max = pos_x.min(), pos_x.max()
y_min, y_max = pos_y.min(), pos_y.max()

if args.channel_centre is not None:
    # Fixed channel centre — zero tilt
    c_left_opt = c_right_opt = args.channel_centre
    a_opt = 0.0
    b_opt = args.channel_centre
    tilt_deg = 0.0
    print(f"  Channel: centre={args.channel_centre:.4f} mm (fixed, zero tilt)")
else:
    # Auto-detect: minimise pressure² outside channel strip
    c_lo = x_min + hw
    c_hi = x_max - hw
    prs_sq = pressure_1f ** 2

    def outside_pressure_sum(params):
        c_left, c_right = params
        centre = c_left + (c_right - c_left) / max(y_max - y_min, 1e-9) * (pos_y - y_min)
        outside = np.abs(pos_x - centre) > hw
        return np.nansum(prs_sq[outside])

    result = brute(outside_pressure_sum,
                   ranges=((c_lo, c_hi), (c_lo, c_hi)),
                   Ns=100, finish=None)
    c_left_opt, c_right_opt = result
    a_opt = (c_right_opt - c_left_opt) / max(y_max - y_min, 1e-9)
    b_opt = c_left_opt - a_opt * y_min
    tilt_deg = np.degrees(np.arctan(a_opt))
    print(f"  Channel: x_left={c_left_opt:.4f}, x_right={c_right_opt:.4f}")
    print(f"  Tilt: {tilt_deg:.3f} deg, width: {CHANNEL_WIDTH} mm (fixed)")

# %%
# =============================================================================
# Build 2D grid in centred channel coordinates
# =============================================================================

# Centred width coordinate
pos_x_c = pos_x - (a_opt * pos_y + b_opt)
inside_c = np.abs(pos_x_c) <= hw

# Grid along channel length (y)
length_grid = np.linspace(y_min, y_max, n_y_meta)
l_idx = np.argmin(np.abs(pos_y[:, None] - length_grid[None, :]), axis=1)

# Grid across channel width (centred)
width_span = pos_x.max() - pos_x.min()
scan_step = width_span / max(n_x_meta - 1, 1)
n_width_c = max(int(round(CHANNEL_WIDTH / scan_step)), 2)
half_step = CHANNEL_WIDTH / n_width_c / 2
width_c_grid = np.linspace(-hw + half_step, hw - half_step, n_width_c)

w_c_idx = np.argmin(np.abs(pos_x_c[:, None] - width_c_grid[None, :]), axis=1)


def to_grid(values):
    grid = np.full((n_width_c, n_y_meta), np.nan)
    mask = inside_c & ~np.isnan(values)
    grid[w_c_idx[mask], l_idx[mask]] = values[mask]
    return grid


grid_vel_1f = to_grid(velocity_1f)
grid_prs_1f = to_grid(pressure_1f / 1e3)  # kPa
grid_phase_1f = to_grid(phase_1f)
grid_rssi = to_grid(rssi) if rssi is not None else None

print(f"  Grid: {n_width_c} width × {n_y_meta} length")

# %%
# =============================================================================
# Plotting helper
# =============================================================================

# figsize: length along horizontal, width along vertical
length_span = length_grid[-1] - length_grid[0]
aspect_ratio = CHANNEL_WIDTH / length_span  # height / width of physical data
ref_w, ref_h = figsize_for_layout(ax_w_scale=2.0)


def map_plot(grid_data, cmap, title, cb_label, output_name,
             vmin=None, vmax=None, pclip=None):
    ax_w = ref_w
    ax_h = ax_w * aspect_ratio
    ax_h = np.clip(ax_h, ref_h * 0.5, ref_h * 2.0)  # keep reasonable
    fw = ax_w + 1.2   # colorbar margin
    fh = ax_h + 1.0   # title/xlabel margin
    fig, ax = plt.subplots(figsize=(fw, fh))
    kwargs = dict(shading="nearest", cmap=cmap)
    if pclip is not None:
        lo, hi = np.nanpercentile(grid_data, [pclip, 100 - pclip])
        kwargs["vmin"] = lo
        kwargs["vmax"] = hi
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    im = ax.pcolormesh(length_grid, width_c_grid, grid_data, **kwargs)
    ax.set_xlabel("Channel length, $x$ (mm)")
    ax.set_ylabel("Channel width, $y$ (mm)")
    ax.set_title(f"{title}\n{stem}")
    ax.set_aspect("auto")
    plt.colorbar(im, ax=ax, label=cb_label)
    plt.tight_layout()
    output_path = OUT_DIR / output_name
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# %%
# =============================================================================
# Plot: 1f maps
# =============================================================================

map_plot(grid_vel_1f, "viridis", "Apparent Velocity at 1f",
         "Apparent velocity (m/s)", f"map2d_velocity_1f_{stem}.png",
         pclip=5)

map_plot(grid_prs_1f, "viridis", "Acoustic Pressure at 1f",
         "Acoustic pressure (kPa)", f"map2d_pressure_1f_{stem}.png",
         pclip=5)

map_plot(grid_phase_1f, "twilight", "Phase at 1f (rel.\\ to Ch1)",
         "Phase (deg)", f"map2d_phase_1f_{stem}.png",
         vmin=-180, vmax=180)

if grid_rssi is not None:
    map_plot(grid_rssi, "viridis", "RSSI",
             "RSSI (V)", f"map2d_rssi_{stem}.png")

# %%
# =============================================================================
# Mode-shape fit: 1f |sin(pi y/W)| or 2f |cos(2pi y/W)|
# =============================================================================
# Auto-detect mode order from drive frequency.

W_m = CHANNEL_WIDTH * 1e-3  # mm -> m
wc_m = width_c_grid * 1e-3  # mm -> m

# Quality mask for mode-shape fit: exclude edge points and low-RSSI points
EDGE_MARGIN = 1  # exclude outermost N width-grid points on each side
quality_mask = np.ones(n_width_c, dtype=bool)
quality_mask[:EDGE_MARGIN] = False
quality_mask[-EDGE_MARGIN:] = False
n_edge_excluded = 2 * EDGE_MARGIN

n_rssi_excluded = 0
if grid_rssi is not None:
    rssi_col_median = np.nanmedian(grid_rssi, axis=1)  # per width position
    rssi_low = rssi_col_median < RSSI_THRESHOLD
    quality_mask &= ~rssi_low
    n_rssi_excluded = int(np.sum(rssi_low & np.ones(n_width_c, dtype=bool)))

print(f"  Mode-shape filter: {n_edge_excluded} edge + {n_rssi_excluded} low-RSSI "
      f"excluded ({quality_mask.sum()}/{n_width_c} width points used)")

is_2f = f_drive > 3e6
if is_2f:
    k = 2 * np.pi / W_m
    mode_profile = np.abs(np.cos(k * wc_m))
    mode_label = "2f"
else:
    k = np.pi / W_m
    mode_profile = np.abs(np.sin(k * wc_m))
    mode_label = "1f"

p0_y = np.full(n_y_meta, np.nan)
for j in range(n_y_meta):
    col = grid_prs_1f[:, j] * 1e3  # kPa -> Pa
    valid = ~np.isnan(col) & quality_mask
    if valid.sum() > 3:
        p0_y[j] = np.sum(col[valid] * mode_profile[valid]) / np.sum(mode_profile[valid] ** 2)

p0_y_kPa = p0_y / 1e3

print(f"  p0(y) range: {np.nanmin(p0_y_kPa):.1f} -- {np.nanmax(p0_y_kPa):.1f} kPa")
best_y_idx = np.nanargmax(p0_y_kPa)
print(f"  Best y position: {length_grid[best_y_idx]:.3f} mm "
      f"(p0 = {p0_y_kPa[best_y_idx]:.1f} kPa)")

# %%
# =============================================================================
# Plot: mode shape at best y-position (data + sinusoidal fit)
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
col_best = grid_prs_1f[:, best_y_idx]
ax.plot(width_c_grid, col_best, "o", markersize=3, label="Data")
x_fine = np.linspace(width_c_grid[0], width_c_grid[-1], 200)
if is_2f:
    fit_fine = np.abs(np.cos(k * x_fine * 1e-3))
else:
    fit_fine = np.abs(np.sin(k * x_fine * 1e-3))
ax.plot(x_fine, p0_y_kPa[best_y_idx] * fit_fine,
        "--", linewidth=1, label=f"$p_0$ = {p0_y_kPa[best_y_idx]:.0f} kPa")
ax.set_xlabel("Channel width, $y$ (mm)")
ax.set_ylabel("Pressure (kPa)")
ax.set_title(f"{mode_label} Mode Shape at $x$ = {length_grid[best_y_idx]:.2f} mm --- {stem}")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / f"map2d_mode_shape_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"  Saved: {output_path.name}")

# %%
# =============================================================================
# Plot: fitted pressure amplitude p0 vs channel length
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(length_grid, p0_y_kPa, "-o", markersize=3, linewidth=0.8)
ax.axvline(length_grid[best_y_idx], color="red", ls=":", alpha=0.5,
           label=f"Best $x$ = {length_grid[best_y_idx]:.2f} mm")
ax.set_xlabel("Channel length, $x$ (mm)")
ax.set_ylabel("Fitted $p_0$ (kPa)")
ax.set_title(f"{mode_label} Pressure Amplitude Along Channel --- {stem}")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / f"map2d_p0_vs_y_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"  Saved: {output_path.name}")

# %%
# =============================================================================
# Optional: 2f harmonic extraction (--harmonics flag, 1f data only)
# =============================================================================

if compute_harmonics and not is_2f:
    from nptdms import TdmsFile

    print("\n--- 2f harmonic extraction ---")
    dt = float(cache["dt"])
    ss_start_idx = int(cache["ss_start"])
    ss_end_idx = int(cache["ss_end"])
    ss_n = ss_end_idx - ss_start_idx

    tone_2f = np.exp(-2j * np.pi * (2 * f_drive) * np.arange(ss_n) * dt)
    pressure_2f = np.empty(n_points)

    print(f"  Reading raw Ch2 waveforms ({n_points} points)...")
    with TdmsFile.open(str(tdms_path)) as tf:
        wf_group = tf["Waveforms"]
        ch2_names = sorted(
            [c.name for c in wf_group.channels()
             if c.name.startswith("WFCh2")])
        for i in range(n_points):
            wf = wf_group[ch2_names[i]][ss_start_idx:ss_end_idx]
            dft = np.dot(wf, tone_2f)
            vel = np.abs(dft) * 2 / ss_n * actual_vel_scale
            pressure_2f[i] = vel / (2 * np.pi * 2 * f_drive * SENSITIVITY)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{n_points}")

    grid_prs_2f = to_grid(pressure_2f / 1e3)  # kPa
    print(f"  Pressure 2f: mean {np.nanmean(pressure_2f)/1e3:.1f} kPa, "
          f"max {np.nanmax(pressure_2f)/1e3:.1f} kPa")

    # 2f pressure map
    map_plot(grid_prs_2f, "viridis", "Acoustic Pressure at 2f",
             "Acoustic pressure (kPa)", f"map2d_pressure_2f_{stem}.png",
             pclip=5)

    # 2f mode-shape fit: |cos(2pi y/W)|
    k_2f = 2 * np.pi / W_m
    mode_2f = np.abs(np.cos(k_2f * wc_m))
    p0_2f_y = np.full(n_y_meta, np.nan)
    for j in range(n_y_meta):
        col = grid_prs_2f[:, j] * 1e3  # kPa -> Pa
        valid = ~np.isnan(col)
        if valid.sum() > 3:
            p0_2f_y[j] = (np.sum(col[valid] * mode_2f[valid])
                          / np.sum(mode_2f[valid] ** 2))

    p0_2f_kPa = p0_2f_y / 1e3
    print(f"  2f p0 range: {np.nanmin(p0_2f_kPa):.1f} -- "
          f"{np.nanmax(p0_2f_kPa):.1f} kPa")

    # 2f mode shape at best 2f y-position
    best_2f_idx = np.nanargmax(p0_2f_kPa)
    fig, ax = plt.subplots(figsize=figsize_for_layout())
    col_2f_best = grid_prs_2f[:, best_2f_idx]
    ax.plot(width_c_grid, col_2f_best, "o", markersize=3, label="2f data")
    x_fine = np.linspace(width_c_grid[0], width_c_grid[-1], 200)
    fit_2f_fine = np.abs(np.cos(k_2f * x_fine * 1e-3))
    ax.plot(x_fine, p0_2f_kPa[best_2f_idx] * fit_2f_fine,
            "--", linewidth=1, color="C3",
            label=f"$p_0$ = {p0_2f_kPa[best_2f_idx]:.0f} kPa")
    ax.set_xlabel("Channel width, $y$ (mm)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title(f"2f Mode Shape at $x$ = {length_grid[best_2f_idx]:.2f} mm --- {stem}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = OUT_DIR / f"map2d_mode_shape_2f_{stem}.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {output_path.name}")

    # Comparison plot: stacked 1f and 2f 2D pressure maps
    ax_w = ref_w
    ax_h = ax_w * aspect_ratio
    ax_h = np.clip(ax_h, ref_h * 0.4, ref_h * 1.5)
    fw = ax_w + 1.2
    fh = ax_h * 2 + 1.4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fw, fh), sharex=True)

    lo1, hi1 = np.nanpercentile(grid_prs_1f, [5, 95])
    im1 = ax1.pcolormesh(length_grid, width_c_grid, grid_prs_1f,
                         shading="nearest", cmap="viridis",
                         vmin=lo1, vmax=hi1)
    ax1.set_ylabel("$y$ (mm)")
    ax1.set_title(f"1f pressure (kPa) --- {stem}")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1)

    lo2, hi2 = np.nanpercentile(grid_prs_2f, [5, 95])
    im2 = ax2.pcolormesh(length_grid, width_c_grid, grid_prs_2f,
                         shading="nearest", cmap="viridis",
                         vmin=lo2, vmax=hi2)
    ax2.set_xlabel("Channel length, $x$ (mm)")
    ax2.set_ylabel("$y$ (mm)")
    ax2.set_title("2f pressure (kPa)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    output_path = OUT_DIR / f"map2d_1f_vs_2f_{stem}.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {output_path.name}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
