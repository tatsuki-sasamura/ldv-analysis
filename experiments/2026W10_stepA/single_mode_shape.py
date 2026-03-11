# %%
"""Step A single-file analysis: 1f pressure mode shape from burst-mode LDV data.

Loads a single TDMS file from the Step A frequency sweep, extracts the
steady-state portion of burst-mode waveforms, computes the 1f acoustic
pressure via refracto-vibrometry, and visualizes the mode shape across
the channel width.

The scan grid is 101 x-points × 3 y-lines.  x scans across the channel
width; the 3 y-lines at each x serve as repeat measurements for
uncertainty estimation.

Usage:
    python analyze_single.py <path_to_tdms>
    python analyze_single.py                   # uses default path
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import FIG_DPI, SENSITIVITY, VELOCITY_SCALE, figsize_for_layout, get_data_dir, get_output_dir
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = get_data_dir("20260303experimentA") / "stepA1967.tdms"

# Position grouping — snap to nominal grid to absorb stage jitter
X_GRID_STEP = 0.005     # mm (5 µm nominal step)

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data from FFT cache
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR)

pos_x = cache["pos_x"]
pos_y = cache["pos_y"]
f_drive = float(cache["f_drive"])
n_points = len(pos_x)

velocity_1f = cache["velocity_1f"]
phase_1f_deg = cache["phase_1f"]
pressure_1f = cache["pressure_1f"]
rssi = cache["rssi"] if "rssi" in cache else None

print(f"  1f pressure: mean {pressure_1f.mean()/1e3:.2f} kPa, "
      f"max {pressure_1f.max()/1e3:.2f} kPa")

# %%
# =============================================================================
# Group by x-position (3 y-lines → mean ± std)
# =============================================================================

x_snapped = np.round(np.round(pos_x / X_GRID_STEP) * X_GRID_STEP, 4)
x_unique = np.sort(np.unique(x_snapped))
n_x = len(x_unique)

pressure_mean = np.empty(n_x)
pressure_std = np.empty(n_x)
vel_mean = np.empty(n_x)
vel_std = np.empty(n_x)
phase_mean = np.empty(n_x)
phase_std = np.empty(n_x)

for i, xv in enumerate(x_unique):
    mask = x_snapped == xv
    pressure_mean[i] = pressure_1f[mask].mean()
    pressure_std[i] = pressure_1f[mask].std()
    vel_mean[i] = velocity_1f[mask].mean()
    vel_std[i] = velocity_1f[mask].std()
    ph = phase_1f_deg[mask]
    phase_mean[i] = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ph)))))
    phase_std[i] = ph.std()

n_per_x = n_points // n_x
print(f"\n  {n_x} x-positions, {n_per_x} repeats each")
print(f"  Pressure range: {pressure_mean.min()/1e3:.2f} – "
      f"{pressure_mean.max()/1e3:.2f} kPa")
print(f"  Max std: {pressure_std.max()/1e3:.2f} kPa "
      f"({pressure_std.max() / pressure_mean.max() * 100:.1f}% of peak)")

# %%
# =============================================================================
# Plot 1: Mode shape — 1f pressure profile
# =============================================================================

fig, axes = plt.subplots(
    3, 1, figsize=figsize_for_layout(3, 1, sharex=True), sharex=True,
)

axes[0].errorbar(
    x_unique, pressure_mean / 1e3, yerr=pressure_std / 1e3,
    fmt="-o", markersize=2, linewidth=0.8, capsize=1.5, elinewidth=0.5,
)
axes[0].set_ylabel("Pressure (kPa)")
axes[0].set_title(f"1f Acoustic Pressure --- {stem}")
axes[0].grid(True, alpha=0.3)

axes[1].errorbar(
    x_unique, vel_mean * 1e3, yerr=vel_std * 1e3,
    fmt="-o", markersize=2, linewidth=0.8, capsize=1.5, elinewidth=0.5,
    color="C1",
)
axes[1].set_ylabel("Velocity (mm/s)")
axes[1].grid(True, alpha=0.3)

axes[2].errorbar(
    x_unique, phase_mean, yerr=phase_std,
    fmt="-o", markersize=2, linewidth=0.8, capsize=1.5, elinewidth=0.5,
    color="C2",
)
axes[2].set_ylabel("Phase rel.\\ Ch1 (deg)")
axes[2].set_xlabel("$x$ position (mm)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / f"mode_shape_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot 2: Representative waveform + spectrum (highest-pressure point)
# =============================================================================

best_i = int(np.argmax(pressure_1f))
best_x = pos_x[best_i]

# Load raw waveforms for this one point only (memory-efficient)
wfs, dt = load_point_waveforms(tdms_path, best_i, channels=(1, 2))
n_samples = int(cache["n_samples"])
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
ss_n = ss_end - ss_start
ss_start_us = ss_start * dt * 1e6
ss_end_us = ss_end * dt * 1e6

wf_ch2_best = wfs[2]
wf_ch2_ss = wf_ch2_best[ss_start:ss_end]
fft_ch2_best = np.fft.rfft(wf_ch2_ss)
freqs = np.fft.rfftfreq(ss_n, d=dt)

t_full = np.arange(n_samples) * dt

fig, axes = plt.subplots(2, 2, figsize=figsize_for_layout(2, 2))

# Full burst waveform
t_us = t_full * 1e6
axes[0, 0].plot(t_us, wf_ch2_best * VELOCITY_SCALE * 1e3, linewidth=0.3)
axes[0, 0].axvspan(ss_start_us, ss_end_us, alpha=0.1, color="green",
                    label="FFT window")
axes[0, 0].set_xlabel(r"Time (\textmu s)")
axes[0, 0].set_ylabel("Velocity (mm/s)")
axes[0, 0].set_title(f"Ch2 --- point {best_i} ($x$={best_x:.3f} mm)")
axes[0, 0].legend(fontsize=7)
axes[0, 0].grid(True, alpha=0.3)

# Zoom: 5 periods of steady state
n_show = int(5 / f_drive / dt)
t_ss_us = (np.arange(ss_n) + ss_start) * dt * 1e6
axes[0, 1].plot(t_ss_us[:n_show],
                wf_ch2_ss[:n_show] * VELOCITY_SCALE * 1e3,
                linewidth=0.8)
axes[0, 1].set_xlabel(r"Time (\textmu s)")
axes[0, 1].set_ylabel("Velocity (mm/s)")
axes[0, 1].set_title("Steady state (5 cycles)")
axes[0, 1].grid(True, alpha=0.3)

# Velocity spectrum
spec_vel = np.abs(fft_ch2_best) * 2 / ss_n * VELOCITY_SCALE
freq_mask = (freqs >= 0.5e6) & (freqs <= 6e6)
axes[1, 0].plot(freqs[freq_mask] / 1e6, spec_vel[freq_mask] * 1e3,
                linewidth=0.8)
axes[1, 0].axvline(f_drive / 1e6, color="red", ls=":", alpha=0.5, label="1f")
axes[1, 0].axvline(2 * f_drive / 1e6, color="orange", ls=":", alpha=0.5,
                    label="2f")
axes[1, 0].set_xlabel("Frequency (MHz)")
axes[1, 0].set_ylabel("Velocity (mm/s)")
axes[1, 0].set_title("FFT spectrum")
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(True, alpha=0.3)

# Pressure spectrum
spec_prs = np.zeros_like(spec_vel)
spec_prs[1:] = spec_vel[1:] / (2 * np.pi * freqs[1:] * SENSITIVITY)
axes[1, 1].plot(freqs[freq_mask] / 1e6, spec_prs[freq_mask] / 1e3,
                linewidth=0.8, color="C1")
axes[1, 1].axvline(f_drive / 1e6, color="red", ls=":", alpha=0.5, label="1f")
axes[1, 1].axvline(2 * f_drive / 1e6, color="orange", ls=":", alpha=0.5,
                    label="2f")
axes[1, 1].set_xlabel("Frequency (MHz)")
axes[1, 1].set_ylabel("Pressure (kPa)")
axes[1, 1].set_title("Pressure spectrum")
axes[1, 1].legend(fontsize=7)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / f"waveform_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot 3: Repeatability — 3 y-lines overlaid
# =============================================================================

y_snapped = np.round(np.round(pos_y / X_GRID_STEP) * X_GRID_STEP, 4)
y_unique_vals = np.sort(np.unique(y_snapped))

fig, axes = plt.subplots(
    3, 1, figsize=figsize_for_layout(3, 1, sharex=True), sharex=True,
)

for yv in y_unique_vals:
    mask = y_snapped == yv
    x_vals = pos_x[mask]
    order = np.argsort(x_vals)
    axes[0].plot(x_vals[order], pressure_1f[mask][order] / 1e3,
                 marker=".", markersize=1.5, linewidth=0.6,
                 label=f"$y$ = {yv:.3f} mm", alpha=0.8)
    axes[1].plot(x_vals[order], phase_1f_deg[mask][order],
                 marker=".", markersize=1.5, linewidth=0.6,
                 label=f"$y$ = {yv:.3f} mm", alpha=0.8)
    if rssi is not None:
        axes[2].plot(x_vals[order], rssi[mask][order],
                     marker=".", markersize=1.5, linewidth=0.6,
                     label=f"$y$ = {yv:.3f} mm", alpha=0.8)

axes[0].set_ylabel("Pressure (kPa)")
axes[0].set_title(f"Repeatability: 3 y-lines --- {stem}")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("Phase rel.\\ Ch1 (deg)")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("RSSI (V)")
axes[2].set_xlabel("$x$ position (mm)")
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / f"repeatability_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {output_path}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
