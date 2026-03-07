# %%
"""Frequency sweep analysis for test7 data (2f coarse sweep, continuous excitation).

Processes test7_*.tdms files — 21×2 line scans across channel width at
3.7–3.9 MHz.  Fits the full-wavelength (2f) mode shape:

    p(y_c) = p0 * |cos(2π y_c / W)|

where y_c is centred on the channel.  This mode has:
  - antinode at centre (y_c = 0)
  - nodes at y_c = ±W/4
  - antinodes at walls (y_c = ±W/2)

Generates:
  1. 4-panel frequency sweep (p0, phase, current, voltage).
  2. Overview grid of mode shapes with cos fit.
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

DATA_DIR = Path("C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260306experimentA")
FILE_PATTERN = "test7_*.tdms"

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
k_2f = 2 * np.pi / W  # full-wavelength mode: cos(2π y / W)

all_freqs = []
all_p0 = []
all_r2 = []
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
    pressure = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

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

    # Mode-shape fit: p(y_c) = p0 * |cos(2π y_c / W)|
    # Search for channel centre that maximises p0
    y_line = pos_y[valid]
    p_line = pressure[valid]

    y_trial = np.linspace(y_line.min() + hw, y_line.max() - hw, 100)
    best_p0, best_yc = 0, y_trial[0] if len(y_trial) > 0 else y_line.mean()
    for yc in y_trial:
        y_c = (y_line - yc) * 1e-3  # mm → m
        inside = np.abs(y_c) <= W / 2
        if inside.sum() < 2:
            continue
        cos_prof = np.abs(np.cos(k_2f * y_c[inside]))
        denom = np.sum(cos_prof ** 2)
        if denom > 0:
            p0_cand = np.sum(p_line[inside] * cos_prof) / denom
            if p0_cand > best_p0:
                best_p0 = p0_cand
                best_yc = yc

    # R²
    y_c = (y_line - best_yc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    if inside.sum() > 2:
        p_pred = best_p0 * np.abs(np.cos(k_2f * y_c[inside]))
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
          f"R²={r2:.2f}")

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

# %%
# =============================================================================
# Plot 1: 4-panel frequency sweep
# =============================================================================

fig, axes = plt.subplots(
    4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True,
)

axes[0].plot(freq_arr, p0_arr, ".-", markersize=3, linewidth=0.8)
axes[0].set_ylabel(r"$p_0$ (kPa)")
axes[0].set_title("2f frequency sweep --- test7 (continuous)")
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
out_path = OUT_DIR / "freq_sweep_test7.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: individual mode-shape plots
# =============================================================================

y_fine = np.linspace(-hw, hw, 200)
cos_fine = np.abs(np.cos(k_2f * y_fine * 1e-3))

mode_data_sorted = [mode_shape_data[i] for i in sort_f]

mode_dir = OUT_DIR / "mode_shapes_test7"
mode_dir.mkdir(parents=True, exist_ok=True)

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
    ax.plot(y_fine, p0_kpa * cos_fine, "--", linewidth=1, color="C3",
            label=rf"$p_0$ = {p0_kpa:.0f} kPa, $R^2$ = {r2_str}")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)

    ax.set_xlabel("Width position (mm)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title(f"{md['freq_khz']} kHz (2f mode)")
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fname = f"mode_{md['freq_khz']}kHz.png"
    fig.savefig(mode_dir / fname, dpi=FIG_DPI)
    plt.close()

print(f"Saved {len(mode_data_sorted)} mode-shape plots to {mode_dir}")

n = len(mode_data_sorted)
ncols = min(7, n)
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
    ax.plot(y_fine, p0_kpa * cos_fine, "--", linewidth=0.6, color="C3")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)
    ax.set_title(f"{md['f_mhz']:.3f} MHz\n{p0_kpa:.0f} kPa, R²={md['r2']:.2f}",
                 fontsize=5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.supxlabel("Width position (mm)", fontsize=6)
fig.supylabel("Pressure (kPa)", fontsize=6)
plt.tight_layout()
out_path2 = OUT_DIR / "mode_shapes_test7_overview.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Summary
# =============================================================================

i_best = np.argmax(p0_arr)
print(f"\nPeak: p0 = {p0_arr[i_best]:.0f} kPa at {freq_arr[i_best]:.3f} MHz")
print(f"Best R² = {r2_arr[sort_f[np.argmax(r2_arr)]]:.2f} "
      f"at {freq_arr[np.argmax(r2_arr)]:.3f} MHz")
print(f"Files processed: {n}")

# %%
# =============================================================================
# Burst-mode 2f mode shapes (test7appendix files)
# =============================================================================

burst_files = sorted(DATA_DIR.glob("test7appendix_burst_*.tdms"))
burst_files = [f for f in burst_files if not f.name.endswith("_index")]

if burst_files:
    print(f"\n--- Burst-mode files: {len(burst_files)} ---")
    for tdms_path_b in burst_files:
        bstem = tdms_path_b.stem
        freq_tag = bstem.split("_")[-1]

        cache_b = load_or_compute(tdms_path_b, CACHE_DIR)
        pos_y_b = cache_b["pos_x"]
        pressure_b = cache_b["pressure_1f"]
        rssi_b = cache_b["rssi"] if "rssi" in cache_b else None

        valid_b = np.ones(len(pos_y_b), dtype=bool)
        if rssi_b is not None:
            valid_b &= rssi_b > RSSI_THRESHOLD

        y_line_b = pos_y_b[valid_b]
        p_line_b = pressure_b[valid_b]

        y_trial_b = np.linspace(y_line_b.min() + hw, y_line_b.max() - hw, 100)
        best_p0_b = 0
        best_yc_b = y_trial_b[0] if len(y_trial_b) > 0 else y_line_b.mean()
        for yc in y_trial_b:
            y_c = (y_line_b - yc) * 1e-3
            inside = np.abs(y_c) <= W / 2
            if inside.sum() < 2:
                continue
            cos_prof = np.abs(np.cos(k_2f * y_c[inside]))
            denom = np.sum(cos_prof ** 2)
            if denom > 0:
                p0_cand = np.sum(p_line_b[inside] * cos_prof) / denom
                if p0_cand > best_p0_b:
                    best_p0_b = p0_cand
                    best_yc_b = yc

        p0_kpa_b = best_p0_b / 1e3
        y_c_b = (y_line_b - best_yc_b) * 1e-3
        inside_b = np.abs(y_c_b) <= W / 2

        fig, ax = plt.subplots(figsize=figsize_for_layout())
        ax.plot((y_line_b[~inside_b] - best_yc_b), p_line_b[~inside_b] / 1e3,
                "x", markersize=3, alpha=0.3, color="0.6")
        ax.plot((y_line_b[inside_b] - best_yc_b), p_line_b[inside_b] / 1e3,
                ".", markersize=3, alpha=0.6)
        ax.plot(y_fine, p0_kpa_b * cos_fine, "--", linewidth=1, color="C3",
                label=rf"$p_0$ = {p0_kpa_b:.0f} kPa")
        ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
        ax.axvline(hw, color="0.5", ls=":", lw=0.5)
        ax.set_xlabel("Width position (mm)")
        ax.set_ylabel("Pressure (kPa)")
        ax.set_title(f"{freq_tag} kHz (2f burst mode)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        out_burst = OUT_DIR / f"mode_shape_burst_{freq_tag}.png"
        fig.savefig(out_burst, dpi=FIG_DPI)
        plt.close()
        print(f"  {bstem}: p0 = {p0_kpa_b:.0f} kPa -> {out_burst.name}")

# %%
print("\n=== Done ===")
