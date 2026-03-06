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

from ldv_analysis.config import FIG_DPI, figsize_for_layout, get_output_dir
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("G:/My Drive/20260306experimentA")
FILE_PATTERN = "test5_*.tdms"

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

    valid = V > np.median(V) * 0.5
    if rssi is not None:
        valid &= rssi > RSSI_THRESHOLD

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
    y_line = pos_y[valid]
    p_line = pressure[valid]

    y_trial = np.linspace(y_line.min() + hw, y_line.max() - hw, 100)
    best_p0, best_yc = 0, y_trial[0] if len(y_trial) > 0 else y_line.mean()
    for yc in y_trial:
        y_c = (y_line - yc) * 1e-3
        inside = np.abs(y_c) <= W / 2
        if inside.sum() < 2:
            continue
        sin_prof = np.abs(np.sin(k_mode * y_c[inside]))
        denom = np.sum(sin_prof ** 2)
        if denom > 0:
            p0_cand = np.sum(p_line[inside] * sin_prof) / denom
            if p0_cand > best_p0:
                best_p0 = p0_cand
                best_yc = yc

    # R²
    y_c = (y_line - best_yc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    if inside.sum() > 2:
        p_pred = best_p0 * np.abs(np.sin(k_mode * y_c[inside]))
        ss_res = np.sum((p_line[inside] - p_pred) ** 2)
        ss_tot = np.sum((p_line[inside] - p_line[inside].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2 = 0

    all_p0.append(best_p0)
    all_r2.append(r2)

    mode_shape_data.append(dict(
        freq_khz=freq_khz, f_mhz=f_drive / 1e6,
        p0=best_p0, r2=r2,
        y_c=y_line - best_yc, p=p_line,
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
axes[0].set_ylabel(r"$p_0$ (kPa)")
axes[0].set_title(r"Frequency sweep --- test5, $x = 9$ mm")
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_arr, phase_arr, ".-", markersize=3, linewidth=0.8, color="C3")
axes[1].set_ylabel("V--I phase (deg)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_arr, I_arr, ".-", markersize=3, linewidth=0.8, color="C2")
axes[2].set_ylabel("Current (mA)")
axes[2].grid(True, alpha=0.3)

axes[3].plot(freq_arr, V_arr, ".-", markersize=3, linewidth=0.8, color="C1")
axes[3].set_ylabel("Voltage (V)")
axes[3].set_xlabel("Frequency (MHz)")
axes[3].grid(True, alpha=0.3)

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

y_fine = np.linspace(-hw, hw, 200)
sin_fine = np.abs(np.sin(k_mode * y_fine * 1e-3))

mode_data_sorted = [mode_shape_data[i] for i in sort_f]

for md in mode_data_sorted:
    fig, ax = plt.subplots(figsize=figsize_for_layout())
    p0_kpa = md["p0"] / 1e3
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    ax.plot(md["y_c"][~inside], md["p"][~inside] / 1e3,
            "x", markersize=3, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside], md["p"][inside] / 1e3,
            ".", markersize=3, alpha=0.6)
    r2_str = f"{md['r2']:.2f}" if md["r2"] > -10 else "<-10"
    ax.plot(y_fine, p0_kpa * sin_fine, "--", linewidth=1, color="C3",
            label=rf"$p_0$ = {p0_kpa:.0f} kPa, $R^2$ = {r2_str}")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)

    ax.set_xlabel("Width position (mm)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title(f"{md['freq_khz']} kHz, x = 9 mm")
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

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
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    ax.plot(md["y_c"][~inside], md["p"][~inside] / 1e3,
            "x", markersize=1.5, alpha=0.3, color="0.6")
    ax.plot(md["y_c"][inside], md["p"][inside] / 1e3,
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

fig.supxlabel("Width position (mm)", fontsize=6)
fig.supylabel("Pressure (kPa)", fontsize=6)
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
