# %%
"""2D spatial maps of 1f acoustic pressure from burst-mode area scan.

Generates pcolormesh heatmaps (velocity, pressure, phase, RSSI) for a
Step A area-scan TDMS file.  Channel boundaries are detected by
minimising pressure² outside a strip of known width (375 µm), then data
is displayed in centred channel coordinates.

Unlike 04_2d_map.py this script loads TDMS directly (no pre-conversion)
and extracts the steady-state portion of burst-mode waveforms for FFT.

Usage:
    python map_2d.py <path_to_tdms>
    python map_2d.py                   # uses default path
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute

from ldv_analysis.config import FIG_DPI, figsize_for_layout
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = Path(
    "G:/My Drive/20260303experimentA/"
    "stepA1967_where_is_the_best_x_position.tdms"
)

# Channel geometry
CHANNEL_WIDTH = 0.375  # mm (known physical width)

OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data (with FFT cache for fast re-runs)
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, OUT_DIR)

pos_x = cache["pos_x"]
pos_y = cache["pos_y"]
n_x_meta = int(cache["n_x_meta"])
n_y_meta = int(cache["n_y_meta"])
f_drive = float(cache["f_drive"])
velocity_1f = cache["velocity_1f"]
pressure_1f = cache["pressure_1f"]
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

# Bounds: channel centre must fit within scan range
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
# Sinusoidal mode-shape fit: p(x_c) = p0 * |sin(pi * x_c / W)|
# =============================================================================
# Fit p0(y) at each y-position by least-squares projection onto the expected
# half-wavelength mode shape.

W_m = CHANNEL_WIDTH * 1e-3  # mm -> m
k = np.pi / W_m
wc_m = width_c_grid * 1e-3  # mm -> m
sin_profile = np.abs(np.sin(k * wc_m))

p0_y = np.full(n_y_meta, np.nan)
for j in range(n_y_meta):
    col = grid_prs_1f[:, j] * 1e3  # kPa -> Pa
    valid = ~np.isnan(col)
    if valid.sum() > 3:
        p0_y[j] = np.sum(col[valid] * sin_profile[valid]) / np.sum(sin_profile[valid] ** 2)

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
ax.plot(x_fine, p0_y_kPa[best_y_idx] * np.abs(np.sin(k * x_fine * 1e-3)),
        "--", linewidth=1, label=f"$p_0$ = {p0_y_kPa[best_y_idx]:.0f} kPa")
ax.set_xlabel("Channel width, $y$ (mm)")
ax.set_ylabel("Pressure (kPa)")
ax.set_title(f"Mode Shape at $x$ = {length_grid[best_y_idx]:.2f} mm --- {stem}")
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
ax.set_title(f"1f Pressure Amplitude Along Channel --- {stem}")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / f"map2d_p0_vs_y_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"  Saved: {output_path.name}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
