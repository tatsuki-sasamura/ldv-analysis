# %%
"""2D spatial map of apparent velocity and acoustic pressure from test5 (area scan).

test5.tdms is a 2D area scan (101 X × 11 Y = 1111 points) covering
X: 29.99–30.39 mm, Y: 1.6–2.6 mm.  The main dataset files are 1D line
scans at Y ≈ 2.6 mm (top edge of this area).

Note: test5 uses different decoder scaling than the main files:
  Ch2: 0.5 m/s per V  (vs 1.0 default)
  Ch3: 0.25 µm/V      (vs 0.5 default)

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import CONVERTED_DIR, FIG_DPI, SENSITIVITY, figsize_for_layout, get_output_dir

# %%
# =============================================================================
# Configuration
# =============================================================================

# test5-specific decoder scales
VELOCITY_SCALE_TEST5 = 0.5       # m/s per V
DISPLACEMENT_SCALE_TEST5 = 0.25e-6  # m per V

OUT_DIR = get_output_dir(__file__)
NPZ_PATH = CONVERTED_DIR / "test5.npz"

print(f"Loading {NPZ_PATH.name}...")
data = np.load(NPZ_PATH)

pos_x = data["scan_pos_x"]
pos_y = data["scan_pos_y"]
wf1 = data["wf_ch1"]
wf2 = data["wf_ch2"]
wf3 = data["wf_ch3"]
dt = float(data["wf_dt"])
n_points = wf1.shape[0]
n_samples = wf1.shape[1]
freqs = np.fft.rfftfreq(n_samples, d=dt)
rssi = data["scan_rssi"] if "scan_rssi" in data else None

print(f"  {n_points} scan points, {n_samples} samples/waveform")

# %%
# =============================================================================
# Vectorised FFT and extraction at drive frequency
# =============================================================================

fft_v = np.fft.rfft(wf1, axis=1)
fft_vel = np.fft.rfft(wf2, axis=1)
fft_disp = np.fft.rfft(wf3, axis=1)

# Drive frequency per point from Ch1 peak (skip DC)
peak_idx = np.argmax(np.abs(fft_v[:, 1:]), axis=1) + 1
pts = np.arange(n_points)
drive_freqs = freqs[peak_idx]

# Apparent velocity amplitude at 1f
velocity_amp = np.abs(fft_vel[pts, peak_idx]) * 2 / n_samples * VELOCITY_SCALE_TEST5

# Apparent displacement amplitude at 1f
disp_amp = np.abs(fft_disp[pts, peak_idx]) * 2 / n_samples * DISPLACEMENT_SCALE_TEST5

# Pressure from Ch2 (velocity): p = v / (2*pi*f * SENSITIVITY)
pressure_ch2 = velocity_amp / (2 * np.pi * drive_freqs * SENSITIVITY)

# Pressure from Ch3 (displacement): p = d / SENSITIVITY
pressure_ch3 = disp_amp / SENSITIVITY

# Phase relative to Ch1
diff = np.degrees(np.angle(fft_vel[pts, peak_idx]) - np.angle(fft_v[pts, peak_idx]))
phase_rel = (diff + 180) % 360 - 180

print(f"  Drive freq: {drive_freqs.mean()/1e6:.4f} MHz")
print(f"  Velocity:   {velocity_amp.mean():.4e} m/s (mean)")
print(f"  Pressure:   {pressure_ch2.mean()/1e3:.1f} kPa (mean, from Ch2)")

# %%
# =============================================================================
# Build 2D grid
# =============================================================================

# Build grid from metadata (101 X × 11 Y)
n_x = int(data["meta_n_x"])
n_y = int(data["meta_n_y"])
x_unique = np.linspace(pos_x.min(), pos_x.max(), n_x)
y_unique = np.linspace(pos_y.min(), pos_y.max(), n_y)
print(f"  Grid: {n_x} × {n_y}")

# Map each point to nearest grid cell
x_idx = np.argmin(np.abs(pos_x[:, None] - x_unique[None, :]), axis=1)
y_idx = np.argmin(np.abs(pos_y[:, None] - y_unique[None, :]), axis=1)

def to_grid(values):
    grid = np.full((n_y, n_x), np.nan)
    grid[y_idx, x_idx] = values
    return grid

grid_vel = to_grid(velocity_amp)
grid_prs_ch2 = to_grid(pressure_ch2 / 1e3)  # kPa
grid_prs_ch3 = to_grid(pressure_ch3 / 1e3)  # kPa
grid_phase = to_grid(phase_rel)
grid_rssi = to_grid(rssi) if rssi is not None else None

# %%
# =============================================================================
# Plot: apparent velocity heatmap
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout(ax_w_scale=2))
im = ax.pcolormesh(x_unique, y_unique, grid_vel, shading="nearest", cmap="viridis")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_title("Apparent Velocity at 1f --- test5 (2D area scan)")
ax.set_aspect("auto")
plt.colorbar(im, ax=ax, label="Apparent velocity (m/s)")
plt.tight_layout()
output_path = OUT_DIR / "velocity_2d.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: pressure heatmaps (Ch2 and Ch3 side by side)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2, sharex=True, sharey=True), sharex=True, sharey=True,
                         constrained_layout=True)

vmax = max(np.nanmax(grid_prs_ch2), np.nanmax(grid_prs_ch3))
for ax, grid, title in zip(axes, [grid_prs_ch2, grid_prs_ch3],
                            ["From Ch2 (velocity)", "From Ch3 (displacement)"]):
    im = ax.pcolormesh(x_unique, y_unique, grid, shading="nearest",
                       cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xlabel("X position (mm)")
    ax.set_title(title)

axes[0].set_ylabel("Y position (mm)")
fig.suptitle("Acoustic Pressure at 1f --- test5 (2D area scan)", fontsize=13)
fig.colorbar(im, ax=axes, label="Acoustic pressure (kPa)", shrink=0.8)
output_path = OUT_DIR / "pressure_2d.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: phase heatmap
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout(ax_w_scale=2))
im = ax.pcolormesh(x_unique, y_unique, grid_phase, shading="nearest", cmap="twilight",
                   vmin=-180, vmax=180)
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_title("Phase at 1f (rel. to Ch1) --- test5 (2D area scan)")
ax.set_aspect("auto")
plt.colorbar(im, ax=ax, label="Phase (deg)")
plt.tight_layout()
output_path = OUT_DIR / "phase_2d.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: RSSI heatmap
# =============================================================================

if grid_rssi is not None:
    fig, ax = plt.subplots(figsize=figsize_for_layout(ax_w_scale=2))
    im = ax.pcolormesh(x_unique, y_unique, grid_rssi, shading="nearest", cmap="viridis")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("RSSI --- test5 (2D area scan)")
    ax.set_aspect("auto")
    plt.colorbar(im, ax=ax, label="RSSI (V)")
    plt.tight_layout()
    output_path = OUT_DIR / "rssi_2d.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.show()
    print(f"Saved: {output_path}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
