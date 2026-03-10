# %%
"""Frequency sweep at 25 Vpp (test9 data) with 1f and 2f harmonic analysis.

Processes test9_*.tdms files — line scans at x = 9 mm, 25 Vpp drive.
Fits both 1f half-wavelength and 2f full-wavelength mode shapes:

    1f: p(y_c) = p0 * |sin(pi y_c / W)|
    2f: p(y_c) = p0 * |cos(2pi y_c / W)|

Generates:
  1. 5-panel frequency sweep (p0_1f, p0_2f, 2f/1f ratio, current, voltage).
  2. Individual mode-shape plots with both 1f and 2f fits.
  3. Overview grid of mode shapes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile

from ldv_analysis.config import (
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test9_*.tdms"

CHANNEL_WIDTH = 0.375e-3  # m
RSSI_THRESHOLD = 1.0      # V

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Discover and sort sweep files
# =============================================================================

tdms_files = sorted(DATA_DIR.glob(FILE_PATTERN))
tdms_files = [f for f in tdms_files if not f.name.endswith("_index")]
print(f"Found {len(tdms_files)} sweep files in {DATA_DIR}\n")

if not tdms_files:
    print("No files found. Exiting.")
    sys.exit(0)

# %%
# =============================================================================
# Process each file
# =============================================================================

W = CHANNEL_WIDTH
hw = W / 2 * 1e3  # mm
k_1f = np.pi / W
k_2f = 2 * np.pi / W

all_freqs = []
all_p0_1f = []
all_p0_2f = []
all_r2_1f = []
all_r2_2f = []
all_V_med = []
all_I_med = []
all_phase_med = []
mode_shape_data = []

for tdms_path in tdms_files:
    stem = tdms_path.stem
    freq_khz = stem.split("_")[-1]

    cache = load_or_compute(tdms_path, CACHE_DIR)
    f_drive = float(cache["f_drive"])
    dt = float(cache["dt"])
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    ss_n = ss_end - ss_start
    n_samples = int(cache["n_samples"])
    pos_y = cache["pos_x"]   # scan "x" = channel width
    pressure_1f = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None
    n_points = len(pos_y)

    valid = V > np.median(V) * 0.5
    if rssi is not None:
        valid &= rssi > RSSI_THRESHOLD

    all_freqs.append(f_drive / 1e6)
    all_V_med.append(float(np.median(V[valid])))

    has_ch4 = "current_1f" in cache
    if has_ch4:
        I = cache["current_1f"]
        phase_vi = cache["phase_vi"]
        all_I_med.append(float(np.median(I[valid])) * 1e3)
        all_phase_med.append(float(np.median(phase_vi[valid])))
    else:
        all_I_med.append(np.nan)
        all_phase_med.append(np.nan)

    # --- Compute 2f pressure from raw waveforms ---
    tone_2f = np.exp(-2j * np.pi * (2 * f_drive) * np.arange(ss_n) * dt)
    pressure_2f = np.empty(n_points)

    with TdmsFile.open(str(tdms_path)) as tf:
        wf_group = tf["Waveforms"]
        ch2_names = sorted(
            [c.name for c in wf_group.channels()
             if c.name.startswith("WFCh2")])
        for i in range(n_points):
            wf = wf_group[ch2_names[i]][ss_start:ss_end]
            dft = np.dot(wf, tone_2f)
            vel = np.abs(dft) * 2 / ss_n * VELOCITY_SCALE
            pressure_2f[i] = vel / (2 * np.pi * 2 * f_drive * SENSITIVITY)

    # --- 1f mode-shape fit: |sin(pi y/W)| ---
    y_line = pos_y[valid]
    p1_line = pressure_1f[valid]
    p2_line = pressure_2f[valid]

    y_trial = np.linspace(y_line.min() + hw, y_line.max() - hw, 100)
    best_p0_1f, best_yc = 0, y_trial[0] if len(y_trial) > 0 else y_line.mean()
    for yc in y_trial:
        y_c = (y_line - yc) * 1e-3
        inside = np.abs(y_c) <= W / 2
        if inside.sum() < 2:
            continue
        sin_prof = np.abs(np.sin(k_1f * y_c[inside]))
        denom = np.sum(sin_prof ** 2)
        if denom > 0:
            p0_cand = np.sum(p1_line[inside] * sin_prof) / denom
            if p0_cand > best_p0_1f:
                best_p0_1f = p0_cand
                best_yc = yc

    # 1f R²
    y_c = (y_line - best_yc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    if inside.sum() > 2:
        p_pred = best_p0_1f * np.abs(np.sin(k_1f * y_c[inside]))
        ss_res = np.sum((p1_line[inside] - p_pred) ** 2)
        ss_tot = np.sum((p1_line[inside] - p1_line[inside].mean()) ** 2)
        r2_1f = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2_1f = 0

    # --- 2f mode-shape fit: |cos(2pi y/W)| using same centre ---
    cos_prof = np.abs(np.cos(k_2f * y_c[inside]))
    denom_2f = np.sum(cos_prof ** 2)
    best_p0_2f = np.sum(p2_line[inside] * cos_prof) / denom_2f if denom_2f > 0 else 0

    if inside.sum() > 2 and best_p0_2f > 0:
        p_pred_2f = best_p0_2f * cos_prof
        ss_res_2f = np.sum((p2_line[inside] - p_pred_2f) ** 2)
        ss_tot_2f = np.sum((p2_line[inside] - p2_line[inside].mean()) ** 2)
        r2_2f = 1 - ss_res_2f / ss_tot_2f if ss_tot_2f > 0 else 0
    else:
        r2_2f = 0

    all_p0_1f.append(best_p0_1f)
    all_p0_2f.append(best_p0_2f)
    all_r2_1f.append(r2_1f)
    all_r2_2f.append(r2_2f)

    mode_shape_data.append(dict(
        freq_khz=freq_khz, f_mhz=f_drive / 1e6,
        p0_1f=best_p0_1f, p0_2f=best_p0_2f,
        r2_1f=r2_1f, r2_2f=r2_2f,
        y_c=y_line - best_yc,
        p_1f=p1_line, p_2f=p2_line,
    ))

    ratio = best_p0_2f / best_p0_1f * 100 if best_p0_1f > 0 else 0
    print(f"  {stem}: f={f_drive/1e6:.3f} MHz, "
          f"p0_1f={best_p0_1f/1e3:.0f} kPa, "
          f"p0_2f={best_p0_2f/1e3:.0f} kPa, "
          f"2f/1f={ratio:.1f}%")

# %%
# =============================================================================
# Sort by frequency
# =============================================================================

freq_arr = np.array(all_freqs)
sort_f = np.argsort(freq_arr)
freq_arr = freq_arr[sort_f]
p0_1f_arr = np.array(all_p0_1f)[sort_f] / 1e3  # kPa
p0_2f_arr = np.array(all_p0_2f)[sort_f] / 1e3
r2_1f_arr = np.array(all_r2_1f)[sort_f]
r2_2f_arr = np.array(all_r2_2f)[sort_f]
V_arr = np.array(all_V_med)[sort_f]
I_arr = np.array(all_I_med)[sort_f]
phase_arr = np.array(all_phase_med)[sort_f]
ratio_arr = p0_2f_arr / p0_1f_arr * 100

# %%
# =============================================================================
# Plot 1: 5-panel frequency sweep
# =============================================================================

fig, axes = plt.subplots(
    5, 1, figsize=figsize_for_layout(5, 1, sharex=True), sharex=True,
)

axes[0].plot(freq_arr, p0_1f_arr, ".-", markersize=3, linewidth=0.8)
axes[0].set_ylabel(r"$p_0^{1f}$ (kPa)")
axes[0].set_title(r"Frequency sweep --- test9, 25 Vpp, $x = 9$ mm")
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_arr, p0_2f_arr, ".-", markersize=3, linewidth=0.8, color="C4")
axes[1].set_ylabel(r"$p_0^{2f}$ (kPa)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_arr, ratio_arr, ".-", markersize=3, linewidth=0.8, color="C3")
axes[2].set_ylabel(r"$p_0^{2f}/p_0^{1f}$ (\%)")
axes[2].grid(True, alpha=0.3)

axes[3].plot(freq_arr, I_arr, ".-", markersize=3, linewidth=0.8, color="C2")
axes[3].set_ylabel("Current (mA)")
axes[3].grid(True, alpha=0.3)

axes[4].plot(freq_arr, V_arr, ".-", markersize=3, linewidth=0.8, color="C1")
axes[4].set_ylabel("Voltage (V)")
axes[4].set_xlabel("Frequency (MHz)")
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "freq_sweep_test9.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: individual mode-shape plots (1f + 2f on same axes)
# =============================================================================

y_fine = np.linspace(-hw, hw, 200)
sin_fine = np.abs(np.sin(k_1f * y_fine * 1e-3))
cos_fine = np.abs(np.cos(k_2f * y_fine * 1e-3))

mode_data_sorted = [mode_shape_data[i] for i in sort_f]

mode_dir = OUT_DIR / "mode_shapes_test9"
mode_dir.mkdir(parents=True, exist_ok=True)

for md in mode_data_sorted:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_for_layout(2, 1),
                                    sharex=True)
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    # 1f panel
    p0_1f_kpa = md["p0_1f"] / 1e3
    ax1.plot(md["y_c"][~inside], md["p_1f"][~inside] / 1e3,
             "x", markersize=3, alpha=0.3, color="0.6")
    ax1.plot(md["y_c"][inside], md["p_1f"][inside] / 1e3,
             ".", markersize=3, alpha=0.6)
    r2_1f_str = f"{md['r2_1f']:.2f}" if md["r2_1f"] > -10 else "<-10"
    ax1.plot(y_fine, p0_1f_kpa * sin_fine, "--", linewidth=1, color="C3",
             label=rf"$p_0$ = {p0_1f_kpa:.0f} kPa, $R^2$ = {r2_1f_str}")
    ax1.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax1.axvline(hw, color="0.5", ls=":", lw=0.5)
    ax1.set_ylabel("1f Pressure (kPa)")
    ax1.set_title(f"{md['freq_khz']} kHz, 25 Vpp, x = 9 mm")
    ax1.legend(fontsize=5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 2f panel
    p0_2f_kpa = md["p0_2f"] / 1e3
    ax2.plot(md["y_c"][~inside], md["p_2f"][~inside] / 1e3,
             "x", markersize=3, alpha=0.3, color="0.6")
    ax2.plot(md["y_c"][inside], md["p_2f"][inside] / 1e3,
             ".", markersize=3, alpha=0.6, color="C4")
    r2_2f_str = f"{md['r2_2f']:.2f}" if md["r2_2f"] > -10 else "<-10"
    ax2.plot(y_fine, p0_2f_kpa * cos_fine, "--", linewidth=1, color="C3",
             label=rf"$p_0$ = {p0_2f_kpa:.0f} kPa, $R^2$ = {r2_2f_str}")
    ax2.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax2.axvline(hw, color="0.5", ls=":", lw=0.5)
    ax2.set_xlabel("Width position (mm)")
    ax2.set_ylabel("2f Pressure (kPa)")
    ax2.legend(fontsize=5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    fname = f"mode_{md['freq_khz']}kHz.png"
    fig.savefig(mode_dir / fname, dpi=FIG_DPI)
    plt.close()

print(f"Saved {len(mode_data_sorted)} mode-shape plots to {mode_dir}")

# %%
# =============================================================================
# Plot 3: mode-shape overview grid (1f top row, 2f bottom row per freq)
# =============================================================================

n = len(mode_data_sorted)
ncols = min(7, n)
nrows = (n + ncols - 1) // ncols

# 1f overview
fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.8 * nrows),
                         sharex=True, sharey=True)
if nrows == 1:
    axes = axes[np.newaxis, :]
axes_flat = axes.flatten()

for i, md in enumerate(mode_data_sorted):
    ax = axes_flat[i]
    p0_kpa = md["p0_1f"] / 1e3
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    ax.plot(md["y_c"][~inside], md["p_1f"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside], md["p_1f"][inside] / 1e3,
            ".", markersize=1.5, alpha=0.6)
    ax.plot(y_fine, p0_kpa * sin_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa", fontsize=5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("1f mode shapes --- test9, 25 Vpp", fontsize=7)
fig.supxlabel("Width position (mm)", fontsize=6)
fig.supylabel("Pressure (kPa)", fontsize=6)
plt.tight_layout()
out_1f = OUT_DIR / "mode_shapes_test9_1f_overview.png"
fig.savefig(out_1f, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_1f}")

# 2f overview
fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.8 * nrows),
                         sharex=True, sharey=True)
if nrows == 1:
    axes = axes[np.newaxis, :]
axes_flat = axes.flatten()

for i, md in enumerate(mode_data_sorted):
    ax = axes_flat[i]
    p0_kpa = md["p0_2f"] / 1e3
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    ax.plot(md["y_c"][~inside], md["p_2f"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside], md["p_2f"][inside] / 1e3,
            ".", markersize=1.5, alpha=0.6, color="C4")
    ax.plot(y_fine, p0_kpa * cos_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)
    ratio = md["p0_2f"] / md["p0_1f"] * 100 if md["p0_1f"] > 0 else 0
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa ({ratio:.1f}%)",
                 fontsize=5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("2f harmonic mode shapes --- test9, 25 Vpp", fontsize=7)
fig.supxlabel("Width position (mm)", fontsize=6)
fig.supylabel("Pressure (kPa)", fontsize=6)
plt.tight_layout()
out_2f = OUT_DIR / "mode_shapes_test9_2f_overview.png"
fig.savefig(out_2f, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_2f}")

# %%
# =============================================================================
# Summary
# =============================================================================

i_best = np.argmax(p0_1f_arr)
print(f"\n1f peak: p0 = {p0_1f_arr[i_best]:.0f} kPa at {freq_arr[i_best]:.3f} MHz")
print(f"2f at peak: p0 = {p0_2f_arr[i_best]:.0f} kPa "
      f"({ratio_arr[i_best]:.1f}%)")
print(f"Files processed: {n}")

# %%
print("\n=== Done ===")
