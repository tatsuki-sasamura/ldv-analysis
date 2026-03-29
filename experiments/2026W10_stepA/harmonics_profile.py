# %%
"""Visualise pressure harmonic mode shapes (1f–5f) from a single TDMS file.

Computes per-point pressure at each harmonic from raw Ch2 waveforms,
then plots the width-direction mode shape with |sin(h*pi*y/W)| fit.

Usage:
    python harmonics_profile.py <path_to_tdms>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    figsize_for_layout,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.mode_fit import fit_mode
from ldv_analysis.mode_fit import fit_mode_1f

# %%
# =============================================================================
# Configuration
# =============================================================================

MAX_HARMONIC = 5
OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if tdms_path is None:
    print("Usage: python harmonics_profile.py <path_to_tdms>")
    sys.exit(1)

stem = tdms_path.stem

# %%
# =============================================================================
# Load cache and raw waveforms
# =============================================================================

print(f"Loading: {tdms_path.name}")
cache = load_or_compute(tdms_path, CACHE_DIR)

f_drive = float(cache["f_drive"])
dt = float(cache["dt"])
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
ss_n = ss_end - ss_start
vel_scale = float(cache["velocity_scale"])
pos_y = cache["pos_x"]  # width direction
voltage_1f = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache else None
valid = make_valid_mask(voltage_1f, rssi)

# 1f centre from mode-shape fit
p1f_complex = cache["pressure_1f"][valid] * np.exp(
    1j * np.radians(cache["phase_1f"][valid])
)
res_1f = fit_mode_1f(pos_y[valid], p1f_complex, CHANNEL_WIDTH)
centre = res_1f.centre

print(f"  f_drive = {f_drive/1e6:.4f} MHz, vel_scale = {vel_scale}")
print(f"  Valid points: {valid.sum()} / {len(pos_y)}")
print(f"  Channel centre: {centre*1e3:.3f} mm")

# %%
# =============================================================================
# Compute pressure at each harmonic for all points
# =============================================================================

print(f"  Loading pressure harmonics 1f--{MAX_HARMONIC}f from cache...")
pressure_hf = {h: cache[f"pressure_{h}f"] for h in range(1, MAX_HARMONIC + 1)}

# %%
# =============================================================================
# Mode-shape fits
# =============================================================================

y_c_valid = (pos_y[valid] - centre) * 1e3  # mm
y_fine = np.linspace(-CHANNEL_WIDTH / 2, CHANNEL_WIDTH / 2, 200) * 1e3

p0_fits = {}
for h in range(1, MAX_HARMONIC + 1):
    res_h = fit_mode(pos_y[valid], pressure_hf[h][valid], CHANNEL_WIDTH, h,
                     centre=centre)
    p0_fits[h] = abs(res_h.p0)

print(f"\n  {'h':>3} {'f (MHz)':>8} {'p0 (kPa)':>10} {'p0/p0_1f':>10}")
print("  " + "-" * 35)
for h in range(1, MAX_HARMONIC + 1):
    ratio = p0_fits[h] / p0_fits[1] * 100 if p0_fits[1] > 0 else 0
    print(f"  {h:3d} {h*f_drive/1e6:8.3f} {p0_fits[h]/1e3:10.0f} {ratio:9.2f}%")

# %%
# =============================================================================
# Plot: mode shapes
# =============================================================================

n_h = MAX_HARMONIC
fig, axes = plt.subplots(n_h, 1, figsize=figsize_for_layout(n_h, 1, sharex=True),
                         sharex=True)

colors = [f"C{i}" for i in range(n_h)]

for i, h in enumerate(range(1, n_h + 1)):
    ax = axes[i]
    data = pressure_hf[h][valid] / 1e3
    ax.plot(y_c_valid, data, "o", markersize=2, alpha=0.5, color=colors[i])

    k_h = h * np.pi / CHANNEL_WIDTH
    if h % 2 == 1:
        fit_line = p0_fits[h] / 1e3 * np.abs(np.sin(k_h * y_fine * 1e-3))
    else:
        fit_line = p0_fits[h] / 1e3 * np.abs(np.cos(k_h * y_fine * 1e-3))
    ax.plot(y_fine, fit_line, "--", linewidth=1, color="C3",
            label=f"{p0_fits[h]/1e3:.0f} kPa")

    ratio = p0_fits[h] / p0_fits[1] * 100 if p0_fits[1] > 0 else 0
    ax.set_ylabel(f"{h}f ({ratio:.1f}\\%)")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, alpha=0.3)

axes[0].set_title(f"Pressure harmonics --- {stem}")
axes[-1].set_xlabel("Channel width, $y$ [mm]")

plt.tight_layout()
out_path = OUT_DIR / f"harmonics_{stem}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
print("\n=== Done ===")
