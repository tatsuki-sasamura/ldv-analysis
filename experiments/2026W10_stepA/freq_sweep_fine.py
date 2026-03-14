# %%
"""Frequency sweep analysis for test5 data (fine sweep at x = 9 mm).

Processes test5_*.tdms files — single axial position, 101 y-points × 2 lines,
fine frequency steps around the resonance peaks. Generates:
  1. 4-panel frequency sweep (p0, phase, current, voltage).
  2. Overview grid of mode shapes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH, FIG_DPI,
    figsize_for_layout, get_data_dir, get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.mode_fit import fit_mode_1f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test5_*.tdms"

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
k_mode = np.pi / W

all_freqs = []
all_p0 = []
all_r2 = []
all_V_med = []
all_I_med = []
all_phase_med = []
all_rssi_med = []
mode_shape_data = []

for tdms_path in tdms_files:
    stem = tdms_path.stem
    freq_khz = stem.split("_")[-1]

    cache = load_or_compute(tdms_path, CACHE_DIR)
    f_drive = float(cache["f_drive"])
    pos_y = cache["pos_x"]   # scan "x" = channel width
    pressure = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    valid = make_valid_mask(V, rssi)

    all_freqs.append(f_drive / 1e6)
    all_V_med.append(float(np.median(V[valid])))
    all_rssi_med.append(float(np.median(rssi[valid])) if rssi is not None else 0)

    has_ch4 = "current_1f" in cache
    if has_ch4:
        I = cache["current_1f"]
        phase_vi = cache["phase_vi"]
        all_I_med.append(float(np.median(I[valid])) * 1e3)
        all_phase_med.append(float(np.median(phase_vi[valid])))
    else:
        all_I_med.append(np.nan)
        all_phase_med.append(np.nan)

    # Mode-shape fit
    phase_1f = cache["phase_1f"]
    pressure_complex = pressure * np.exp(1j * np.radians(phase_1f))
    result = fit_mode_1f(pos_y[valid], pressure_complex[valid], CHANNEL_WIDTH)
    best_p0, best_yc, r2 = abs(result.p0), result.centre, result.r2

    all_p0.append(best_p0)
    all_r2.append(r2)

    mode_shape_data.append(dict(
        freq_khz=freq_khz, f_mhz=f_drive / 1e6,
        p0=best_p0, p0_complex=result.p0, r2=r2,
        y_c=pos_y[valid] - best_yc, p=pressure[valid],
        phase_1f=phase_1f[valid],
    ))

    print(f"  {stem}: f={f_drive/1e6:.3f} MHz, p0={best_p0/1e3:.0f} kPa, "
          f"R²={r2:.2f}, I={all_I_med[-1]:.1f} mA")

# %%
# =============================================================================
# Sort by frequency
# =============================================================================

freq_arr = np.array(all_freqs)
sort_f = np.argsort(freq_arr)
freq_arr = freq_arr[sort_f]
p0_arr = np.array(all_p0)[sort_f] / 1e3  # kPa
r2_arr = np.array(all_r2)[sort_f]
V_arr = np.array(all_V_med)[sort_f]
I_arr = np.array(all_I_med)[sort_f]
phase_arr = np.array(all_phase_med)[sort_f]
rssi_arr = np.array(all_rssi_med)[sort_f]

# %%
# =============================================================================
# Plot 1: 4-panel frequency sweep
# =============================================================================

fig, axes = plt.subplots(
    4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True,
)

axes[0].plot(freq_arr, p0_arr, ".-", markersize=3, linewidth=0.8)
axes[0].set_ylabel(r"$P$ [kPa]")
axes[0].set_title(r"Frequency sweep --- test5, $x = 9$ mm")

axes[1].plot(freq_arr, phase_arr, ".-", markersize=3, linewidth=0.8, color="C3")
axes[1].set_ylabel(r"V--I phase [deg]")

axes[2].plot(freq_arr, I_arr, ".-", markersize=3, linewidth=0.8, color="C2")
axes[2].set_ylabel("Current [mA]")

axes[3].plot(freq_arr, V_arr, ".-", markersize=3, linewidth=0.8, color="C1")
axes[3].set_ylabel("Voltage [V]")
axes[3].set_xlabel("Frequency [MHz]")

plt.tight_layout()
out_path = OUT_DIR / "freq_sweep_test5.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: individual mode-shape plots (like mode_shapes/)
# =============================================================================

mode_dir = OUT_DIR / "mode_shapes_test5"
mode_dir.mkdir(parents=True, exist_ok=True)

y_fine_mm = np.linspace(-hw_mm, hw_mm, 200)
sin_fine_signed = np.sin(k_mode * y_fine_mm * 1e-3)
sin_fine = np.abs(sin_fine_signed)

mode_data_sorted = [mode_shape_data[i] for i in sort_f]

for md in mode_data_sorted:
    fig, (ax, axp) = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2),
                                   sharex=True)
    p0_kpa = md["p0"] / 1e3
    inside = np.abs(md["y_c"]) <= W / 2

    # Amplitude panel
    ax.plot(md["y_c"][~inside] * 1e3, md["p"][~inside] / 1e3,
            "x", markersize=3, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside] * 1e3, md["p"][inside] / 1e3,
            ".", markersize=3, alpha=0.6)
    r2_str = f"{md['r2']:.2f}" if md["r2"] > -10 else "<-10"
    ax.plot(y_fine_mm, p0_kpa * sin_fine, "--", linewidth=1, color="C3",
            label=rf"$P$ = {p0_kpa:.0f} kPa, $R^2$ = {r2_str}")
    ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax.set_xlabel("Width position [mm]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title(f"{md['freq_khz']} kHz, x = 9 mm")
    ax.legend(fontsize=5, frameon=False)
    ax.set_ylim(bottom=0)

    # Phase panel
    axp.plot(md["y_c"][~inside] * 1e3, md["phase_1f"][~inside],
             "x", markersize=3, alpha=0.3, color="0.6")
    axp.plot(md["y_c"][inside] * 1e3, md["phase_1f"][inside],
             ".", markersize=3, alpha=0.6)
    phase_model = np.degrees(np.angle(md["p0_complex"] * sin_fine_signed))
    axp.plot(y_fine_mm, phase_model, "--", linewidth=1, color="C3")
    axp.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    axp.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    axp.set_xlabel("Width position [mm]")
    axp.set_ylabel(r"Phase [$^\circ$]")
    axp.set_title("Phase (rel. voltage)")
    axp.set_ylim(-200, 200)

    plt.tight_layout()
    fname = f"mode_{md['freq_khz']}kHz.png"
    fig.savefig(mode_dir / fname, dpi=FIG_DPI)
    plt.close()

print(f"Saved {len(mode_data_sorted)} mode-shape plots to {mode_dir}")

# %%
# =============================================================================
# Plot 3: mode-shape overview grid
# =============================================================================

n = len(mode_data_sorted)
ncols = min(5, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 1.8 * nrows),
                         sharex=True, sharey=True)
if nrows == 1:
    axes = axes[np.newaxis, :]
axes_flat = axes.flatten()

for i, md in enumerate(mode_data_sorted):
    ax = axes_flat[i]
    p0_kpa = md["p0"] / 1e3
    inside = np.abs(md["y_c"]) <= W / 2

    ax.plot(md["y_c"][~inside] * 1e3, md["p"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside] * 1e3, md["p"][inside] / 1e3,
            ".", markersize=1.5, alpha=0.6)
    ax.plot(y_fine_mm, p0_kpa * sin_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw_mm, color="0.5", ls=":", lw=0.5)
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa", fontsize=5)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.supxlabel("Width position [mm]", fontsize=6)
fig.supylabel("Pressure [kPa]", fontsize=6)
plt.tight_layout()
out_path2 = OUT_DIR / "mode_shapes_test5_overview.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Summary
# =============================================================================

i_best = np.argmax(p0_arr)
print(f"\nPeak: p0 = {p0_arr[i_best]:.0f} kPa at {freq_arr[i_best]:.3f} MHz")
print(f"Files processed: {n}")

# %%
print("\n=== Done ===")
