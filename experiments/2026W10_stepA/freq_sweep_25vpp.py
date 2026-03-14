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
from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.mode_fit import fit_mode_1f, fit_mode_2f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test9_*.tdms"

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
hw_mm = W / 2 * 1e3  # mm
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
    pos_y = cache["pos_x"]   # scan "x" = channel width
    pressure_1f = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None
    n_points = len(pos_y)

    valid = make_valid_mask(V, rssi)

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

    # --- 2f pressure from cache ---
    pressure_2f = cache["pressure_2f"]

    # --- 1f mode-shape fit ---
    phase_1f = cache["phase_1f"]
    pressure_1f_complex = pressure_1f * np.exp(1j * np.radians(phase_1f))
    res_1f = fit_mode_1f(pos_y[valid], pressure_1f_complex[valid], CHANNEL_WIDTH)
    best_p0_1f, best_yc, r2_1f = abs(res_1f.p0), res_1f.centre, res_1f.r2

    # --- 2f mode-shape fit using same centre ---
    phase_2f = cache["phase_2f"]
    pressure_2f_complex = pressure_2f * np.exp(1j * np.radians(phase_2f))
    res_2f = fit_mode_2f(pos_y[valid], pressure_2f_complex[valid], CHANNEL_WIDTH, best_yc)
    best_p0_2f, r2_2f = abs(res_2f.p0), res_2f.r2

    all_p0_1f.append(best_p0_1f)
    all_p0_2f.append(best_p0_2f)
    all_r2_1f.append(r2_1f)
    all_r2_2f.append(r2_2f)

    mode_shape_data.append(dict(
        freq_khz=freq_khz, f_mhz=f_drive / 1e6,
        p0_1f=best_p0_1f, p0_2f=best_p0_2f,
        p0_1f_complex=res_1f.p0, p0_2f_complex=res_2f.p0,
        r2_1f=r2_1f, r2_2f=r2_2f,
        y_c=pos_y[valid] - best_yc,
        p_1f=pressure_1f[valid], p_2f=pressure_2f[valid],
        phase_1f=phase_1f[valid], phase_2f=phase_2f[valid],
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
axes[0].set_ylabel(r"$P_{1f}$ [kPa]")
axes[0].set_title(r"Frequency sweep --- test9, 25 Vpp, $x = 9$ mm")

axes[1].plot(freq_arr, p0_2f_arr, ".-", markersize=3, linewidth=0.8, color="C4")
axes[1].set_ylabel(r"$P_{2f}$ [kPa]")

axes[2].plot(freq_arr, ratio_arr, ".-", markersize=3, linewidth=0.8, color="C3")
axes[2].set_ylabel(r"$P_{2f}/P_{1f}$ [\%]")

axes[3].plot(freq_arr, I_arr, ".-", markersize=3, linewidth=0.8, color="C2")
axes[3].set_ylabel("Current [mA]")

axes[4].plot(freq_arr, V_arr, ".-", markersize=3, linewidth=0.8, color="C1")
axes[4].set_ylabel("Voltage [V]")
axes[4].set_xlabel("Frequency [MHz]")

plt.tight_layout()
out_path = OUT_DIR / "freq_sweep_test9.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: individual mode-shape plots (1f + 2f on same axes)
# =============================================================================

y_fine_mm = np.linspace(-hw_mm, hw_mm, 200)
sin_fine_signed = np.sin(k_1f * y_fine_mm * 1e-3)
cos_fine_signed = np.cos(k_2f * y_fine_mm * 1e-3)
sin_fine = np.abs(sin_fine_signed)
cos_fine = np.abs(cos_fine_signed)

mode_data_sorted = [mode_shape_data[i] for i in sort_f]

mode_dir = OUT_DIR / "mode_shapes_test9"
mode_dir.mkdir(parents=True, exist_ok=True)

for md in mode_data_sorted:
    fig, axes = plt.subplots(2, 2, figsize=figsize_for_layout(2, 2),
                             sharex=True)
    ax1, ax1p = axes[0]
    ax2, ax2p = axes[1]
    inside = np.abs(md["y_c"]) <= W / 2

    # 1f amplitude panel
    p0_1f_kpa = md["p0_1f"] / 1e3
    ax1.plot(md["y_c"][~inside] * 1e3, md["p_1f"][~inside] / 1e3,
             "x", markersize=3, alpha=0.3, color="0.6")
    ax1.plot(md["y_c"][inside] * 1e3, md["p_1f"][inside] / 1e3,
             ".", markersize=3, alpha=0.6)
    r2_1f_str = f"{md['r2_1f']:.2f}" if md["r2_1f"] > -10 else "<-10"
    ax1.plot(y_fine_mm, p0_1f_kpa * sin_fine, "--", linewidth=1, color="C3",
             label=rf"$p_0$ = {p0_1f_kpa:.0f} kPa, $R^2$ = {r2_1f_str}")
    ax1.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax1.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax1.set_ylabel(r"$P_{1f}$ [kPa]")
    ax1.set_title(f"{md['freq_khz']} kHz, 25 Vpp, x = 9 mm")
    ax1.legend(fontsize=5, frameon=False)
    ax1.set_ylim(bottom=0)

    # 1f phase panel
    ax1p.plot(md["y_c"][~inside] * 1e3, md["phase_1f"][~inside],
              "x", markersize=3, alpha=0.3, color="0.6")
    ax1p.plot(md["y_c"][inside] * 1e3, md["phase_1f"][inside],
              ".", markersize=3, alpha=0.6)
    phase_model_1f = np.degrees(np.angle(md["p0_1f_complex"] * sin_fine_signed))
    ax1p.plot(y_fine_mm, phase_model_1f, "--", linewidth=1, color="C3")
    ax1p.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax1p.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax1p.set_ylabel(r"1f Phase [$^\circ$]")
    ax1p.set_title("Phase (rel. voltage)")
    ax1p.set_ylim(-200, 200)

    # 2f amplitude panel
    p0_2f_kpa = md["p0_2f"] / 1e3
    ax2.plot(md["y_c"][~inside] * 1e3, md["p_2f"][~inside] / 1e3,
             "x", markersize=3, alpha=0.3, color="0.6")
    ax2.plot(md["y_c"][inside] * 1e3, md["p_2f"][inside] / 1e3,
             ".", markersize=3, alpha=0.6, color="C4")
    r2_2f_str = f"{md['r2_2f']:.2f}" if md["r2_2f"] > -10 else "<-10"
    ax2.plot(y_fine_mm, p0_2f_kpa * cos_fine, "--", linewidth=1, color="C3",
             label=rf"$p_0$ = {p0_2f_kpa:.0f} kPa, $R^2$ = {r2_2f_str}")
    ax2.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax2.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax2.set_xlabel("Width position [mm]")
    ax2.set_ylabel(r"$P_{2f}$ [kPa]")
    ax2.legend(fontsize=5, frameon=False)
    ax2.set_ylim(bottom=0)

    # 2f phase panel
    ax2p.plot(md["y_c"][~inside] * 1e3, md["phase_2f"][~inside],
              "x", markersize=3, alpha=0.3, color="0.6")
    ax2p.plot(md["y_c"][inside] * 1e3, md["phase_2f"][inside],
              ".", markersize=3, alpha=0.6, color="C4")
    phase_model_2f = np.degrees(np.angle(md["p0_2f_complex"] * cos_fine_signed))
    ax2p.plot(y_fine_mm, phase_model_2f, "--", linewidth=1, color="C3")
    ax2p.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax2p.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax2p.set_xlabel("Width position [mm]")
    ax2p.set_ylabel(r"2f Phase [$^\circ$]")
    ax2p.set_ylim(-200, 200)

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
    inside = np.abs(md["y_c"]) <= W / 2

    ax.plot(md["y_c"][~inside] * 1e3, md["p_1f"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside] * 1e3, md["p_1f"][inside] / 1e3,
            ".", markersize=1.5, alpha=0.6)
    ax.plot(y_fine_mm, p0_kpa * sin_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa", fontsize=5)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("1f mode shapes --- test9, 25 Vpp", fontsize=7)
fig.supxlabel("Width position [mm]", fontsize=6)
fig.supylabel("Pressure [kPa]", fontsize=6)
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
    inside = np.abs(md["y_c"]) <= W / 2

    ax.plot(md["y_c"][~inside] * 1e3, md["p_2f"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside] * 1e3, md["p_2f"][inside] / 1e3,
            ".", markersize=1.5, alpha=0.6, color="C4")
    ax.plot(y_fine_mm, p0_kpa * cos_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ratio = md["p0_2f"] / md["p0_1f"] * 100 if md["p0_1f"] > 0 else 0
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa ({ratio:.1f}%)",
                 fontsize=5)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("2f harmonic mode shapes --- test9, 25 Vpp", fontsize=7)
fig.supxlabel("Width position [mm]", fontsize=6)
fig.supylabel("Pressure [kPa]", fontsize=6)
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
