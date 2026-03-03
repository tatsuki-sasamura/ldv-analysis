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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute

from config import (
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
)
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = Path(
    "G:/My Drive/20260303experimentA/"
    "stepA1967_where_is_the_best_x_position.tdms"
)

# Burst detection
ENVELOPE_CHUNK = 1000
ON_THRESHOLD_FACTOR = 3.0

# Steady-state margins
RING_UP_US = 100.0
RING_DOWN_US = 10.0

# Channel geometry
CHANNEL_WIDTH = 0.375  # mm (known physical width)

OUT_DIR = Path(__file__).parent.parent.parent / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

tdms_file, metadata = load_tdms_file(tdms_path)
n_x_meta = int(metadata.get("n_x", 0))
n_y_meta = int(metadata.get("n_y", 0))
print(f"  Grid: {n_x_meta} x × {n_y_meta} y")

scan = tdms_file["ScanData"]
pos_x = scan["PosX"][:]
pos_y = scan["PosY"][:]
n_points = len(pos_x)

wf_ch1, dt = extract_waveforms(tdms_file, channel=1)
wf_ch2, _ = extract_waveforms(tdms_file, channel=2)
n_samples = wf_ch1.shape[1]

rssi = scan["RSSI"][:n_points] if "RSSI" in [ch.name for ch in scan.channels()] else None

print(f"  {n_points} points, {n_samples} samples, dt = {dt*1e9:.0f} ns")

# %%
# =============================================================================
# Detect burst window from Ch1 (use first scan point)
# =============================================================================

ch1_ref = wf_ch1[0]
n_chunks = n_samples // ENVELOPE_CHUNK
rms_env = np.array([
    np.sqrt(np.mean(ch1_ref[i * ENVELOPE_CHUNK:(i + 1) * ENVELOPE_CHUNK] ** 2))
    for i in range(n_chunks)
])

noise_floor = np.median(rms_env[-max(n_chunks // 10, 1):])
on_mask = rms_env > ON_THRESHOLD_FACTOR * noise_floor
on_indices = np.where(on_mask)[0]
burst_on_us = on_indices[0] * ENVELOPE_CHUNK * dt * 1e6
burst_off_us = (on_indices[-1] + 1) * ENVELOPE_CHUNK * dt * 1e6
print(f"  Burst ON: {burst_on_us:.0f}--{burst_off_us:.0f} us")

ss_start_us = burst_on_us + RING_UP_US
ss_end_us = burst_off_us - RING_DOWN_US
ss_start = int(ss_start_us * 1e-6 / dt)
ss_end = int(ss_end_us * 1e-6 / dt)
ss_n = ss_end - ss_start
df = 1 / (ss_n * dt)
print(f"  FFT window: {ss_start_us:.0f}--{ss_end_us:.0f} us "
      f"({ss_n} samples, df = {df:.0f} Hz)")

# %%
# =============================================================================
# FFT on steady-state window
# =============================================================================

print("  Computing FFT...")
wf_ch1_ss = wf_ch1[:, ss_start:ss_end]
wf_ch2_ss = wf_ch2[:, ss_start:ss_end]
freqs = np.fft.rfftfreq(ss_n, d=dt)

fft_ch1 = np.fft.rfft(wf_ch1_ss, axis=1)
fft_ch2 = np.fft.rfft(wf_ch2_ss, axis=1)

# Drive frequency from Ch1 peak
peak_idx = np.argmax(np.abs(fft_ch1[:, 1:]), axis=1) + 1
pts = np.arange(n_points)
drive_freqs = freqs[peak_idx]
f_drive_mean = np.mean(drive_freqs)
print(f"  Drive: {f_drive_mean / 1e6:.4f} MHz")

# 1f velocity and pressure
velocity_1f = np.abs(fft_ch2[pts, peak_idx]) * 2 / ss_n * VELOCITY_SCALE
pressure_1f = velocity_1f / (2 * np.pi * drive_freqs * SENSITIVITY)

# 1f phase relative to Ch1
diff_1f = np.degrees(
    np.angle(fft_ch2[pts, peak_idx]) - np.angle(fft_ch1[pts, peak_idx])
)
phase_1f = (diff_1f + 180) % 360 - 180

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
