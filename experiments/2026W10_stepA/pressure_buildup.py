# %%
"""Visualise how the acoustic pressure field builds up during a burst.

Uses a sliding short-time DFT window to extract the 1f pressure amplitude
at each scan point as a function of time.  Produces:
  1. A pcolormesh (position × time) showing the mode-shape evolving.
  2. Individual mode-shape snapshots at selected times.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CURRENT_SCALE,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = Path("G:/My Drive/20260303experimentA/stepA_sweep_1970.tdms")

# Short-time DFT window
WINDOW_US = 10.0        # µs per window (~20 cycles at 2 MHz)
STEP_US = 5.0           # step between windows (overlap)
T_START_US = 5.0        # first window centre
T_END_US = 500.0        # last window centre

# Mode-shape snapshot times (µs)
SNAPSHOT_TIMES_US = [5, 10, 20, 50, 100, 200, 400]

# Channel geometry
CHANNEL_WIDTH = 0.375e-3  # m
RSSI_THRESHOLD = 1.0      # V

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load cached metadata and raw waveforms
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR)
f_drive = float(cache["f_drive"])
pos_x = cache["pos_x"]
V = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache else None

# Valid-point mask
valid = V > np.median(V) * 0.5
if rssi is not None:
    valid &= rssi > RSSI_THRESHOLD

# Find channel centre (from steady-state fit)
W = CHANNEL_WIDTH
hw = W / 2 * 1e3  # mm
k_mode = np.pi / W

x_trial = np.linspace(pos_x[valid].min() + hw, pos_x[valid].max() - hw, 200)
best_p0, best_xc = 0, x_trial[0]
for xc in x_trial:
    y_c = (pos_x[valid] - xc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    if inside.sum() < 3:
        continue
    sin_prof = np.abs(np.sin(k_mode * y_c[inside]))
    p0_cand = (np.sum(cache["pressure_1f"][valid][inside] * sin_prof)
               / np.sum(sin_prof ** 2))
    if p0_cand > best_p0:
        best_p0 = p0_cand
        best_xc = xc

x_centred = pos_x - best_xc  # mm, centred on channel
print(f"  Channel centre: {best_xc:.4f} mm")
print(f"  Steady-state p0: {best_p0/1e3:.0f} kPa")

# Load all Ch2 (and Ch4 if available) waveforms
print("  Loading raw waveforms...")
tdms_file, _ = load_tdms_file(tdms_path)
wf_ch2, dt = extract_waveforms(tdms_file, channel=2)
n_points, n_samples = wf_ch2.shape

has_ch4 = "current_1f" in cache
wf_ch4 = None
if has_ch4:
    wf_ch4, _ = extract_waveforms(tdms_file, channel=4)
del tdms_file

# %%
# =============================================================================
# Short-time DFT
# =============================================================================

win_n = int(WINDOW_US * 1e-6 / dt)
t_centres_us = np.arange(T_START_US, T_END_US, STEP_US)

print(f"  Computing short-time DFT: {len(t_centres_us)} slices, "
      f"window = {WINDOW_US} µs ({win_n} samples)")

pressure_vs_time = np.zeros((len(t_centres_us), n_points))
current_vs_time = np.zeros((len(t_centres_us), n_points)) if has_ch4 else None

for ti, tc_us in enumerate(t_centres_us):
    tc = int(tc_us * 1e-6 / dt)
    i0 = max(tc - win_n // 2, 0)
    i1 = min(i0 + win_n, n_samples)
    n_win = i1 - i0
    tone = np.exp(-2j * np.pi * f_drive * np.arange(i0, i1) * dt)
    dft = wf_ch2[:, i0:i1] @ tone
    vel = np.abs(dft) * 2 / n_win * VELOCITY_SCALE
    pressure_vs_time[ti] = vel / (2 * np.pi * f_drive * SENSITIVITY)
    if has_ch4:
        dft4 = wf_ch4[:, i0:i1] @ tone
        current_vs_time[ti] = np.abs(dft4) * 2 / n_win * CURRENT_SCALE

del wf_ch2
if wf_ch4 is not None:
    del wf_ch4

# %%
# =============================================================================
# Plot 1: pcolormesh (position × time)
# =============================================================================

# Sort by centred position for clean pcolormesh
sort_idx = np.argsort(x_centred)
x_sorted = x_centred[sort_idx]
p_sorted = pressure_vs_time[:, sort_idx]

fig, ax = plt.subplots(figsize=figsize_for_layout(1, 1, ax_w_scale=1.5))
pcm = ax.pcolormesh(x_sorted, t_centres_us, p_sorted / 1e3,
                     shading="nearest", cmap="inferno", vmin=0)
ax.axvline(-hw, color="w", ls=":", lw=0.5)
ax.axvline(hw, color="w", ls=":", lw=0.5)
ax.set_xlabel("Position (mm)")
ax.set_ylabel(r"Time ($\mu$s)")
ax.set_title(f"Pressure build-up --- {f_drive/1e6:.3f} MHz")
cb = fig.colorbar(pcm, ax=ax)
cb.set_label("Pressure (kPa)")
plt.tight_layout()
out_path = OUT_DIR / f"pressure_buildup_{stem}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 2: mode-shape snapshots
# =============================================================================

inside = np.abs(x_centred * 1e-3) <= W / 2
x_fine = np.linspace(-hw, hw, 200)
sin_fine = np.abs(np.sin(k_mode * x_fine * 1e-3))

n_snaps = len(SNAPSHOT_TIMES_US)
fig, axes = plt.subplots(
    n_snaps, 1,
    figsize=figsize_for_layout(n_snaps, 1, sharex=True, ax_h_scale=0.5),
    sharex=True,
)

for i, t_target in enumerate(SNAPSHOT_TIMES_US):
    ti = int(np.argmin(np.abs(t_centres_us - t_target)))
    ax = axes[i]

    p_slice = pressure_vs_time[ti]

    # Inside channel, valid points
    mask_in = valid & inside
    ax.plot(x_centred[mask_in], p_slice[mask_in] / 1e3,
            ".", markersize=1.5, alpha=0.6, color="C0")

    # Fit p0 at this time slice
    x_in = x_centred[mask_in] * 1e-3
    p_in = p_slice[mask_in]
    sin_prof = np.abs(np.sin(k_mode * x_in))
    denom = np.sum(sin_prof ** 2)
    p0_t = np.sum(p_in * sin_prof) / denom if denom > 0 else 0

    ax.plot(x_fine, p0_t / 1e3 * sin_fine, "--", linewidth=0.6, color="C3")

    ax.annotate(
        rf"{t_centres_us[ti]:.0f} $\mu$s, $p_0$ = {p0_t/1e3:.0f} kPa",
        xy=(0.02, 0.88), xycoords="axes fraction", fontsize=5, va="top",
    )
    ax.set_ylim(0, best_p0 / 1e3 * 1.3)
    ax.set_ylabel("kPa")
    ax.grid(True, alpha=0.3)
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)

axes[-1].set_xlabel("Position (mm)")
axes[0].set_title(f"Mode shape evolution --- {f_drive/1e6:.3f} MHz")
plt.tight_layout()
out_path2 = OUT_DIR / f"pressure_buildup_slices_{stem}.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Plot 3: p0(t) ring-up curve
# =============================================================================

# Fit p0 at each time slice
p0_vs_t = np.zeros(len(t_centres_us))
for ti in range(len(t_centres_us)):
    mask_in = valid & inside
    x_in = x_centred[mask_in] * 1e-3
    p_in = pressure_vs_time[ti, mask_in]
    sin_prof = np.abs(np.sin(k_mode * x_in))
    denom = np.sum(sin_prof ** 2)
    p0_vs_t[ti] = np.sum(p_in * sin_prof) / denom if denom > 0 else 0

fig, ax = plt.subplots(figsize=figsize_for_layout())
ln1 = ax.plot(t_centres_us, p0_vs_t / 1e3, "-", linewidth=0.8, color="C0",
              label=r"$p_0$")
ax.axhline(best_p0 / 1e3, color="C0", ls="--", lw=0.5, alpha=0.5)
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel(r"$p_0$ (kPa)")
ax.set_title(f"Pressure and current ring-up --- {f_drive/1e6:.3f} MHz")
ax.grid(True, alpha=0.3)

# Current on twin y-axis
if has_ch4:
    # Median current across valid points at each time slice
    I_vs_t = np.median(current_vs_time[:, valid], axis=1) * 1e3  # mA
    ax2 = ax.twinx()
    ln2 = ax2.plot(t_centres_us, I_vs_t, "-", linewidth=0.8, color="C1",
                   label="Current")
    I_ss = float(np.median(cache["current_1f"][valid])) * 1e3
    ax2.axhline(I_ss, color="C1", ls="--", lw=0.5, alpha=0.5)
    ax2.set_ylabel("Current (mA)")
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=5)
else:
    ax.legend(fontsize=5)

plt.tight_layout()
out_path3 = OUT_DIR / f"pressure_ringup_{stem}.png"
fig.savefig(out_path3, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path3}")

# %%
print("\n=== Done ===")
