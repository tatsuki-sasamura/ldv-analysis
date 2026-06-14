# %%
"""2D spatial maps of acoustic pressure from a burst-mode area scan.

Promoted from ``experiments/2026W10_stepA/`` for use on W21 v2 HDF5
2D scans; the only changes are per-scan-dir output routing and a
v2-HDF5-aware default path.

Generates pcolormesh heatmaps (velocity, pressure, phase, RSSI) for a
single TDMS or HDF5 file.  Channel boundaries are detected by
minimizing pressure^2 outside a strip of known width (375 um), then
data is displayed in centered channel coordinates.

Output layout::

    experiments/2026W21/output/<scan_dir_name>/pressure_map_2d/
        map2d_<file_stem>.png
        ... (additional per-quantity plots if --harmonics)

Default input points at the W21 sample 101x51 coarse 10 Vpp f1908000.h5
(the P_1f peak file).  Pass any other v1 TDMS or v2 HDF5 file to
override.

To process a whole freq-sweep directory, loop the script over its
.h5 files, e.g. (PowerShell)::

    Get-ChildItem path\to\sweep_dir\f*.h5 | ForEach-Object {
        .\.venv\Scripts\python.exe experiments\2026W21\pressure_map_2d.py $_.FullName --harmonics
    }

Usage::

    python pressure_map_2d.py <path_to_h5_or_tdms>
    python pressure_map_2d.py <path_to_h5_or_tdms> --harmonics
    python pressure_map_2d.py                              # uses default path
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute, fmin

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    LDV_DATA_ROOT,
    RSSI_THRESHOLD,
    figsize_for_layout,
    get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask
from ldv_analysis.grid_utils import ChannelGrid, make_channel_grid, make_to_grid
from ldv_analysis.mode_fit import fit_columns

# %%
# =============================================================================
# Configuration
# =============================================================================

# Default to the W21 sample 101x51 coarse 10 Vpp P_1f peak file
DEFAULT_INPUT = (LDV_DATA_ROOT / "output" / "W21"
                 / "sample_101x51_fsweep_coarse_10Vpp_20260524_132528"
                 / "f1908000.h5")

# %%
# =============================================================================
# Load data (with FFT cache for fast re-runs)
# =============================================================================

parser = argparse.ArgumentParser(description="2D spatial maps of acoustic pressure")
parser.add_argument("tdms_path", nargs="?", default=None,
                    help="Path to v2 HDF5 (.h5) or v1 TDMS (.tdms) file")
parser.add_argument("--harmonics", action="store_true",
                    help="Extract 2f harmonic and generate comparison plots")
parser.add_argument("--ldv-range", type=int, choices=[1, 2, 5], default=None,
                    help="LDV velocity range in m/s (auto-detected from filename if not set)")
# Accept the legacy British spelling of the flag as a hidden alias.
_LEGACY_CENTER_FLAG = "--channel-centre"
parser.add_argument("--channel-center", _LEGACY_CENTER_FLAG,
                    dest="channel_center", type=float, default=None,
                    help="Fixed channel center in mm (skip boundary detection, "
                         "assume zero tilt). A legacy-spelling alias is "
                         "accepted for backwards compatibility.")
parser.add_argument("--geometry-file", type=str, default=None,
                    help="Path to channel_geometry JSON (from calibrate_geometry.py)")
parser.add_argument("--animate", choices=["1f", "2f", "composed"], default=None,
                    help="Render an MP4 of the instantaneous pressure over one "
                         "period of f_drive. '1f' shows one cycle of the "
                         "fundamental, '2f' shows two cycles of the second "
                         "harmonic, 'composed' overlays 1f + 2f.")
args = parser.parse_args()
tdms_path = Path(args.tdms_path) if args.tdms_path else DEFAULT_INPUT
compute_harmonics = args.harmonics
stem = tdms_path.stem

# Per-scan-dir OUT_DIR keyed by the input file's parent directory
# name (the scan dir), so multiple scans coexist without overwriting.
OUT_DIR = (ROOT / "experiments" / "2026W21" / "output"
           / tdms_path.parent.name / "pressure_map_2d")
CACHE_DIR = get_cache_dir(tdms_path.parent.name, __file__)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUT_DIR  = {OUT_DIR}")
print(f"INPUT    = {tdms_path}")

# Velocity scale: --ldv-range override, or auto-detected in cache
vel_scale_override = args.ldv_range / 2.0 if args.ldv_range is not None else None

print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR, velocity_scale=vel_scale_override)

pos_x = cache["pos_x"]          # m (width direction)
pos_y = cache["pos_y"]          # m (length direction)
n_x_meta = int(cache["n_x_meta"])
n_y_meta = int(cache["n_y_meta"])
f_drive = float(cache["f_drive"])
velocity_1f = cache["velocity_1f"]
pressure_1f = cache["pressure_1f"]
phase_1f = cache["phase_1f"]
rssi = cache["rssi"] if "rssi" in cache else None
n_points = len(pos_x)

# Quality filters: mask invalid points (missed bursts / shifted timing)
V = cache["voltage_1f"]
valid = make_valid_mask(V, rssi)
if "pt_burst_on_us" in cache and "pt_burst_off_us" in cache:
    valid &= make_burst_timing_mask(cache["pt_burst_on_us"], cache["pt_burst_off_us"])
n_invalid = int(np.sum(~valid))
if n_invalid > 0:
    pressure_1f = pressure_1f.copy()
    pressure_1f[~valid] = np.nan
    print(f"  Masked {n_invalid} invalid points ({n_invalid/n_points*100:.1f}%)")

print(f"  Pressure 1f: mean {np.nanmean(pressure_1f)/1e3:.1f} kPa, "
      f"max {np.nanmax(pressure_1f)/1e3:.1f} kPa")

# %%
# =============================================================================
# Channel boundary detection
# =============================================================================
# pos_x = channel width direction, pos_y = channel length direction
# Fit tilted channel center: center(y) = a*y + b

hw = CHANNEL_WIDTH / 2          # m
x_min, x_max = pos_x.min(), pos_x.max()
y_min, y_max = pos_y.min(), pos_y.max()
y_span = max(y_max - y_min, 1e-12)

# Geometry detection: fallback hierarchy
# 1. --channel-center CLI override (zero tilt)
# 2. --geometry-file CLI arg (explicit JSON path)
# 3. Auto-discover channel_geometry_{dataset}.json in cache directory
# 4. Per-file RSSI-based detection
# 5. Per-file pressure-based detection (last resort)

geom_source = None


# Legacy spelling used in pre-cleanup geometry JSONs.
_LEGACY_CENTER_SPELL = "centre"


def _read_geom_json(path):
    """Read geometry JSON, accepting both _m and legacy _mm keys.

    Also accepts the pre-cleanup British spelling of the channel-center
    keys for backward compatibility with geometry files written before
    the spelling cleanup.
    """
    with open(path) as f:
        gd = json.load(f)
    # Prefer the current spelling, fall back to the legacy spelling;
    # prefer _m keys, fall back to _mm * 1e-3.
    def _get(base):
        candidates = (base, base.replace("center", _LEGACY_CENTER_SPELL, 1))
        for candidate in candidates:
            if f"{candidate}_m" in gd:
                return gd[f"{candidate}_m"]
            if f"{candidate}_mm" in gd:
                return gd[f"{candidate}_mm"] * 1e-3
        legacy = base.replace("center", _LEGACY_CENTER_SPELL, 1)
        raise KeyError(
            f"Geometry file missing both '{base}_m' and legacy "
            f"'{legacy}_m' keys")
    return _get("center_left"), _get("center_right"), gd["tilt_deg"]


if args.channel_center is not None:
    # Level 1: Fixed channel center — zero tilt (CLI accepts mm, convert to m)
    c_left_opt = c_right_opt = args.channel_center * 1e-3
    tilt_deg = 0.0
    geom_source = "CLI --channel-center"

elif args.geometry_file is not None:
    # Level 2: Explicit geometry file
    c_left_opt, c_right_opt, tilt_deg = _read_geom_json(args.geometry_file)
    geom_source = f"geometry file: {Path(args.geometry_file).name}"

else:
    # Level 3: Auto-discover from dataset name
    dataset = tdms_path.parent.name
    geom_path = CACHE_DIR / f"channel_geometry_{dataset}.json"
    if geom_path.exists():
        c_left_opt, c_right_opt, tilt_deg = _read_geom_json(geom_path)
        geom_source = f"saved geometry: {geom_path.name}"

    elif rssi is not None:
        # Level 4: Per-file RSSI-based detection
        c_lo = x_min + hw
        c_hi = x_max - hw

        def neg_mean_rssi(params):
            c_left, c_right = params
            center = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
            inside = np.abs(pos_x - center) <= hw
            n_inside = np.sum(inside)
            if n_inside < 10:
                return 0.0
            return -np.mean(rssi[inside])

        result = brute(neg_mean_rssi,
                       ranges=((c_lo, c_hi), (c_lo, c_hi)),
                       Ns=100, finish=fmin)
        c_left_opt, c_right_opt = float(result[0]), float(result[1])
        tilt_deg = float(np.degrees(np.arctan(
            (c_right_opt - c_left_opt) / y_span)))
        geom_source = "per-file RSSI detection"

    else:
        # Level 5: Per-file pressure-based detection (last resort)
        c_lo = x_min + hw
        c_hi = x_max - hw
        prs_sq = pressure_1f ** 2

        def outside_pressure_sum(params):
            c_left, c_right = params
            center = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
            outside = np.abs(pos_x - center) > hw
            return np.nansum(prs_sq[outside])

        result = brute(outside_pressure_sum,
                       ranges=((c_lo, c_hi), (c_lo, c_hi)),
                       Ns=100, finish=None)
        c_left_opt, c_right_opt = float(result[0]), float(result[1])
        tilt_deg = float(np.degrees(np.arctan(
            (c_right_opt - c_left_opt) / y_span)))
        geom_source = "per-file pressure detection (last resort)"

a_opt = (c_right_opt - c_left_opt) / y_span
b_opt = c_left_opt - a_opt * y_min
print(f"  Channel: x_left={c_left_opt*1e3:.4f}, x_right={c_right_opt*1e3:.4f} mm")
print(f"  Tilt: {tilt_deg:.3f} deg, width: {CHANNEL_WIDTH*1e3:.3f} mm (fixed)")
print(f"  Source: {geom_source}")

# %%
# =============================================================================
# Build 2D grid in centered channel coordinates
# =============================================================================

# Centered width coordinate
pos_x_c = pos_x - (a_opt * pos_y + b_opt)
inside_c = np.abs(pos_x_c) <= hw

raw_width_span = pos_x.max() - pos_x.min()

cg = make_channel_grid(
    pos_width_c=pos_x_c,
    pos_length=pos_y,
    n_scan_width=n_x_meta,
    n_scan_length=n_y_meta,
    channel_width=CHANNEL_WIDTH,
    raw_width_span=raw_width_span,
    inside=inside_c,
    rssi=rssi,
    rssi_threshold=RSSI_THRESHOLD,
)

grid_vel_1f = cg.to_grid(velocity_1f)
grid_prs_1f = cg.to_grid(pressure_1f)  # Pa
grid_phase_1f = cg.to_grid(phase_1f)
# RSSI map: unfiltered for visualization
if rssi is not None:
    # Build an unfiltered grid for RSSI display (no RSSI threshold)
    _w_idx = np.argmin(
        np.abs(pos_x_c[:, None] - cg.width_grid[None, :]), axis=1)
    _l_idx = np.argmin(
        np.abs(pos_y[:, None] - cg.length_grid[None, :]), axis=1)
    grid_rssi = np.full((cg.n_width, cg.n_length), np.nan)
    _m = inside_c & ~np.isnan(rssi)
    grid_rssi[_w_idx[_m], _l_idx[_m]] = rssi[_m]
else:
    grid_rssi = None

print(f"  Grid: {cg.n_width} width x {cg.n_length} length")

# %%
# =============================================================================
# Plotting helper
# =============================================================================

# figsize: length along horizontal, width along vertical
# Convert to mm for display
length_grid_mm = cg.length_grid * 1e3
width_grid_mm = cg.width_grid * 1e3

length_span_mm = length_grid_mm[-1] - length_grid_mm[0]
aspect_ratio = (CHANNEL_WIDTH * 1e3) / length_span_mm  # height / width of physical data
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
    im = ax.pcolormesh(length_grid_mm, width_grid_mm, grid_data, **kwargs)
    ax.set_xlabel("Channel length, $x$ [mm]")
    ax.set_ylabel("Channel width, $y$ [mm]")
    ax.set_title(f"{title}\n{stem}")
    ax.set_aspect("auto")
    plt.colorbar(im, ax=ax, label=cb_label)
    plt.tight_layout()
    output_path = OUT_DIR / output_name
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def mode_shape_plot(grid_prs, harmonic, label, output_name,
                    sigma_clip_val=None):
    """Plot mode shape at best axial position with sigma clipping.

    Shows kept points as circles, excluded as gray x/^.
    ylim set to (-0.1*P, 2*P).
    """
    from ldv_analysis.mode_fit import fit_mode, _mode_shape, _project

    p0_y = fit_columns(grid_prs, cg.width_grid, CHANNEL_WIDTH,
                       harmonic=harmonic, sigma_clip=sigma_clip_val)
    best_idx = np.nanargmax(p0_y)
    p0_best = p0_y[best_idx]
    col = grid_prs[:, best_idx]

    print(f"  {label} p0 range: {np.nanmin(p0_y)/1e3:.1f} -- "
          f"{np.nanmax(p0_y)/1e3:.1f} kPa")

    # Separate kept vs excluded points
    col_valid = ~np.isnan(col)
    w_v = cg.width_grid[col_valid]
    p_v = col[col_valid]
    mode_v = _mode_shape(w_v, CHANNEL_WIDTH, harmonic, use_abs=True)
    _, clip_mask = _project(p_v, mode_v, sigma_clip=sigma_clip_val)

    y_lim = max(2 * p0_best / 1e3, 1)  # kPa, at least 1

    fig, ax = plt.subplots(figsize=figsize_for_layout())
    w_mm = w_v * 1e3
    p_kpa = p_v / 1e3

    # Kept points
    ax.plot(w_mm[clip_mask], p_kpa[clip_mask], "o", markersize=3,
            label=f"{label} data")

    # Excluded points
    excluded = ~clip_mask
    if excluded.any():
        ex_w = w_mm[excluded]
        ex_p = p_kpa[excluded]
        in_range = ex_p <= y_lim
        if in_range.any():
            ax.plot(ex_w[in_range], ex_p[in_range], "x", markersize=4,
                    color="0.4", label="excluded")
        above = ex_p > y_lim
        if above.any():
            ax.plot(ex_w[above], np.full(above.sum(), y_lim), "^",
                    markersize=4, color="0.4")

    # Fit curve
    x_fine = np.linspace(cg.width_grid[0], cg.width_grid[-1], 200)
    fit_fine = _mode_shape(x_fine, CHANNEL_WIDTH, harmonic, use_abs=True)
    ax.plot(x_fine * 1e3, p0_best / 1e3 * fit_fine,
            "--", linewidth=1, color="C3",
            label=f"$P$ = {p0_best/1e3:.0f} kPa")

    ax.set_xlabel("Channel width, $y$ [mm]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_ylim(-0.1 * p0_best / 1e3, y_lim)
    ax.set_title(f"{label} Mode Shape at $x$ = "
                 f"{cg.length_grid[best_idx]*1e3:.2f} mm --- {stem}")
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    output_path = OUT_DIR / output_name
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {output_path.name}")

    return p0_y


# %%
# =============================================================================
# Plot: 1f maps
# =============================================================================

map_plot(grid_vel_1f, "viridis", "Apparent Velocity at 1f",
         "Apparent velocity [m/s]", f"map2d_velocity_1f_{stem}.png",
         pclip=5)

map_plot(grid_prs_1f / 1e3, "viridis", "Acoustic Pressure at 1f",
         "Acoustic pressure [kPa]", f"map2d_pressure_1f_{stem}.png",
         pclip=5)

map_plot(grid_phase_1f, "twilight", "Phase at 1f (rel.\\ to Ch1)",
         "Phase [deg]", f"map2d_phase_1f_{stem}.png",
         vmin=-180, vmax=180)

if grid_rssi is not None:
    map_plot(grid_rssi, "viridis", "RSSI",
             "RSSI [V]", f"map2d_rssi_{stem}.png")

# %%
# =============================================================================
# Mode-shape fit: 1f |sin(pi y/W)| or 2f |cos(2pi y/W)|
# =============================================================================
# Auto-detect mode order from drive frequency.

is_2f = f_drive > 3e6
harmonic = 2 if is_2f else 1
mode_label = "2f" if is_2f else "1f"
k = (2 * np.pi / CHANNEL_WIDTH) if is_2f else (np.pi / CHANNEL_WIDTH)

p0_y = mode_shape_plot(grid_prs_1f, harmonic, mode_label,
                       f"map2d_mode_shape_{stem}.png",
                       sigma_clip_val=3.0)
best_y_idx = np.nanargmax(p0_y)
print(f"  Best y position: {cg.length_grid[best_y_idx]*1e3:.3f} mm "
      f"(p0 = {p0_y[best_y_idx]/1e3:.1f} kPa)")

# %%
# =============================================================================
# Plot: fitted pressure amplitude p0 vs channel length
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(length_grid_mm, p0_y / 1e3, "-o", markersize=3, linewidth=0.8)
ax.axvline(cg.length_grid[best_y_idx] * 1e3, color="red", ls=":", alpha=0.5,
           label=f"Best $x$ = {cg.length_grid[best_y_idx]*1e3:.2f} mm")
ax.set_xlabel("Channel length, $x$ [mm]")
ax.set_ylabel(r"Fitted $P$ [kPa]")
ax.set_title(f"{mode_label} Pressure Amplitude Along Channel --- {stem}")
ax.legend(fontsize=7, frameon=False)
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
    print("\n--- 2f harmonic ---")
    pressure_2f = cache["pressure_2f"].copy()
    pressure_2f[~valid] = np.nan

    grid_prs_2f = cg.to_grid(pressure_2f)  # Pa
    print(f"  Pressure 2f: mean {np.nanmean(pressure_2f)/1e3:.1f} kPa, "
          f"max {np.nanmax(pressure_2f)/1e3:.1f} kPa")

    # 2f pressure map
    map_plot(grid_prs_2f / 1e3, "viridis", "Acoustic Pressure at 2f",
             "Acoustic pressure [kPa]", f"map2d_pressure_2f_{stem}.png",
             pclip=5)

    # 2f mode-shape fit and plot
    mode_shape_plot(grid_prs_2f, 2, "2f",
                    f"map2d_mode_shape_2f_{stem}.png",
                    sigma_clip_val=3.0)

    # Comparison plot: stacked 1f and 2f 2D pressure maps
    ax_w = ref_w
    ax_h = ax_w * aspect_ratio
    ax_h = np.clip(ax_h, ref_h * 0.4, ref_h * 1.5)
    fw = ax_w + 1.2
    fh = ax_h * 2 + 1.4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fw, fh), sharex=True)

    lo1, hi1 = np.nanpercentile(grid_prs_1f / 1e3, [5, 95])
    im1 = ax1.pcolormesh(length_grid_mm, width_grid_mm, grid_prs_1f / 1e3,
                         shading="nearest", cmap="viridis",
                         vmin=lo1, vmax=hi1)
    ax1.set_ylabel("Channel width, $y$ [mm]")
    ax1.set_title(f"1f pressure (kPa) --- {stem}")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1)

    lo2, hi2 = np.nanpercentile(grid_prs_2f / 1e3, [5, 95])
    im2 = ax2.pcolormesh(length_grid_mm, width_grid_mm, grid_prs_2f / 1e3,
                         shading="nearest", cmap="viridis",
                         vmin=lo2, vmax=hi2)
    ax2.set_xlabel("Channel length, $x$ [mm]")
    ax2.set_ylabel("Channel width, $y$ [mm]")
    ax2.set_title("2f pressure (kPa)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    output_path = OUT_DIR / f"map2d_1f_vs_2f_{stem}.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {output_path.name}")

# %%
# =============================================================================
# Optional: 3f harmonic extraction (requires --harmonics and raw waveforms)
# =============================================================================

if compute_harmonics and not is_2f:
    print("\n--- 3f harmonic ---")
    pressure_3f = cache["pressure_3f"].copy()
    pressure_3f[~valid] = np.nan
    grid_prs_3f = cg.to_grid(pressure_3f)
    print(f"  Pressure 3f: mean {np.nanmean(pressure_3f)/1e3:.1f} kPa, "
          f"max {np.nanmax(pressure_3f)/1e3:.1f} kPa")

    # 3f pressure map
    map_plot(grid_prs_3f / 1e3, "viridis", "Acoustic Pressure at 3f",
             "Acoustic pressure [kPa]", f"map2d_pressure_3f_{stem}.png",
             pclip=5)

    # 3f mode-shape fit and plot
    mode_shape_plot(grid_prs_3f, 3, "3f",
                    f"map2d_mode_shape_3f_{stem}.png",
                    sigma_clip_val=3.0)

# %%
# =============================================================================
# Optional: steady-state instantaneous pressure animation (--animate)
# =============================================================================
# p_n(x, y, t) = Re{ A_n(x, y) * exp(j * 2*pi * n * t/T) },
# where A_n = |P_nf| * exp(j * phi_nf) and T = 1/f_drive.

if args.animate is not None:
    print(f"\n--- Animation ({args.animate}) ---")
    import imageio_ffmpeg
    import matplotlib.animation as mpl_anim

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    need_2f = args.animate in ("2f", "composed")
    if need_2f and ("pressure_2f" not in cache or "phase_2f" not in cache):
        raise SystemExit("No 2f data in cache — cannot animate 2f / composed.")

    A1_grid = (grid_prs_1f
               * np.exp(1j * np.deg2rad(grid_phase_1f)))
    if need_2f:
        pressure_2f_arr = cache["pressure_2f"].copy()
        pressure_2f_arr[~valid] = np.nan
        grid_prs_2f_anim = cg.to_grid(pressure_2f_arr)
        grid_phase_2f_anim = cg.to_grid(cache["phase_2f"])
        A2_grid = (grid_prs_2f_anim
                   * np.exp(1j * np.deg2rad(grid_phase_2f_anim)))
    else:
        A2_grid = None

    N_FRAMES = 60
    t_frac = np.linspace(0, 1, N_FRAMES, endpoint=False)

    frames = np.empty((N_FRAMES, *A1_grid.shape))
    for i, tf in enumerate(t_frac):
        if args.animate == "1f":
            frames[i] = np.real(A1_grid * np.exp(1j * 2 * np.pi * tf))
        elif args.animate == "2f":
            frames[i] = np.real(A2_grid * np.exp(1j * 2 * np.pi * 2 * tf))
        else:  # composed
            frames[i] = np.real(
                A1_grid * np.exp(1j * 2 * np.pi * tf)
                + A2_grid * np.exp(1j * 2 * np.pi * 2 * tf))

    vmax = float(np.nanmax(np.abs(frames))) / 1e3  # kPa
    vmin = -vmax
    print(f"  Range: +/- {vmax:.1f} kPa, {N_FRAMES} frames over one 1f period "
          f"(T = {1e6 / f_drive:.3f} us)")

    ax_w = ref_w
    ax_h = ax_w * aspect_ratio
    ax_h = float(np.clip(ax_h, ref_h * 0.5, ref_h * 2.0))
    fw = ax_w + 1.2
    fh = ax_h + 1.0
    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.pcolormesh(length_grid_mm, width_grid_mm, frames[0] / 1e3,
                       shading="nearest", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
    ax.set_xlabel("Channel length [mm]")
    ax.set_ylabel("Channel width [mm]")
    ax.set_aspect("auto")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Instantaneous pressure [kPa]")
    title = ax.set_title(
        f"{args.animate} pressure --- {stem} --- $t/T_1 = 0.000$")
    plt.tight_layout()

    def _update(i):
        im.set_array((frames[i] / 1e3).ravel())
        title.set_text(
            f"{args.animate} pressure --- {stem} --- "
            f"$t/T_1 = {t_frac[i]:.3f}$")
        return im, title

    anim = mpl_anim.FuncAnimation(
        fig, _update, frames=N_FRAMES, interval=50, blit=False)
    output_path = OUT_DIR / f"anim_{args.animate}_{stem}.mp4"
    anim.save(str(output_path), writer="ffmpeg", fps=30,
              dpi=max(FIG_DPI // 2, 100))
    plt.close(fig)
    print(f"  Saved: {output_path.name}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
