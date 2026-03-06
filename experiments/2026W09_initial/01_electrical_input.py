# %%
"""Electrical input analysis: drive frequency, voltage Vpp, current App,
and phase difference between current (Ch4) and voltage (Ch1).

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ldv_analysis.config import (
    CONVERTED_DIR,
    CURRENT_SCALE,
    DEFAULT_FIGSIZE,
    EXCLUDED_FILES,
    FIG_DPI,
    VOLTAGE_ATTENUATION,
    figsize_for_layout,
    get_output_dir,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = get_output_dir(__file__)

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

# %%
# =============================================================================
# Analyze each file
# =============================================================================

results = []
per_point_data = {}  # file name -> dict of per-point arrays

for npz_path in npz_files:
    tdms_name = npz_path.stem + ".tdms"
    print(f"\n--- {tdms_name} ---")

    data = np.load(npz_path)
    wf1 = data["wf_ch1"]
    wf4 = data["wf_ch4"]
    dt = float(data["wf_dt"])
    n_points = wf1.shape[0]
    n_samples = wf1.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    # Vectorised FFT: (n_points, n_freq_bins)
    fft_v = np.fft.rfft(wf1, axis=1)
    fft_i = np.fft.rfft(wf4, axis=1)

    amp_v = np.abs(fft_v) * 2 / n_samples
    amp_i = np.abs(fft_i) * 2 / n_samples

    # Drive frequency per point from Ch1 peak (skip DC)
    peak_idx = np.argmax(amp_v[:, 1:], axis=1) + 1
    all_drive_freq = freqs[peak_idx]

    # Gather amplitude at each point's peak index
    pts = np.arange(n_points)
    all_v_pp = amp_v[pts, peak_idx] * VOLTAGE_ATTENUATION * 2
    all_i_pp = amp_i[pts, peak_idx] * CURRENT_SCALE * 2

    # Phase difference: current relative to voltage
    phase_v = np.angle(fft_v[pts, peak_idx])
    phase_i = np.angle(fft_i[pts, peak_idx])
    diff = np.degrees(phase_i - phase_v)
    all_phase_diff = (diff + 180) % 360 - 180

    per_point_data[tdms_name] = {
        "i_pp": all_i_pp,
        "v_pp": all_v_pp,
        "phase_diff": all_phase_diff,
    }

    results.append({
        "file": tdms_name,
        "n_points": n_points,
        "drive_freq_hz": all_drive_freq.mean(),
        "drive_freq_std": all_drive_freq.std(),
        "v_pp_mean": all_v_pp.mean(),
        "v_pp_std": all_v_pp.std(),
        "i_pp_mean": all_i_pp.mean(),
        "i_pp_std": all_i_pp.std(),
        "phase_diff_mean": all_phase_diff.mean(),
        "phase_diff_std": all_phase_diff.std(),
    })

    print(f"  Points:      {n_points}")
    print(f"  Drive freq:  {all_drive_freq.mean()/1e6:.6f} ± {all_drive_freq.std()/1e6:.6f} MHz")
    print(f"  Voltage:     {all_v_pp.mean():.4f} ± {all_v_pp.std():.4f} Vpp")
    print(f"  Current:     {all_i_pp.mean():.6f} ± {all_i_pp.std():.6f} App")
    print(f"  Phase (I-V): {all_phase_diff.mean():.2f} ± {all_phase_diff.std():.2f} deg")

# %%
# =============================================================================
# Summary table
# =============================================================================

df_all = pd.DataFrame(results)
df = df_all[~df_all["file"].isin(EXCLUDED_FILES)].reset_index(drop=True)
df_excluded = df_all[df_all["file"].isin(EXCLUDED_FILES)].reset_index(drop=True)

print("\n=== Main Dataset ===")
print(df.to_string(index=False))

if len(df_excluded) > 0:
    print("\n=== Excluded (exploratory, inconsistent config) ===")
    print(df_excluded.to_string(index=False))

csv_path = OUT_DIR / "electrical_input_summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

if len(df_excluded) > 0:
    csv_excl = OUT_DIR / "electrical_input_excluded.csv"
    df_excluded.to_csv(csv_excl, index=False)
    print(f"Saved: {csv_excl}")

# %%
# =============================================================================
# Plots (main dataset only)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=figsize_for_layout(2, 2))

# Voltage vs file
axes[0, 0].bar(range(len(df)), df["v_pp_mean"], yerr=df["v_pp_std"], tick_label=df["file"],
               capsize=3)
axes[0, 0].set_ylabel("Voltage (Vpp)")
axes[0, 0].set_title("Drive Voltage")
axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7)

# Current vs file
axes[0, 1].bar(range(len(df)), df["i_pp_mean"], yerr=df["i_pp_std"], tick_label=df["file"],
               capsize=3)
axes[0, 1].set_ylabel("Current (App)")
axes[0, 1].set_title("Drive Current")
axes[0, 1].tick_params(axis="x", rotation=45, labelsize=7)

# Phase difference vs file
axes[1, 0].bar(range(len(df)), df["phase_diff_mean"], yerr=df["phase_diff_std"],
               tick_label=df["file"], capsize=3)
axes[1, 0].set_ylabel("Phase I-V (deg)")
axes[1, 0].set_title("Current-Voltage Phase Difference")
axes[1, 0].tick_params(axis="x", rotation=45, labelsize=7)

# Current vs voltage (linearity check)
axes[1, 1].errorbar(df["v_pp_mean"], df["i_pp_mean"],
                    xerr=df["v_pp_std"], yerr=df["i_pp_std"], fmt="o", capsize=3)
for _, row in df.iterrows():
    axes[1, 1].annotate(row["file"].replace(".tdms", ""),
                        (row["v_pp_mean"], row["i_pp_mean"]),
                        fontsize=6, textcoords="offset points", xytext=(4, 4))
axes[1, 1].set_xlabel("Voltage (Vpp)")
axes[1, 1].set_ylabel("Current (App)")
axes[1, 1].set_title("I-V Relationship")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / "electrical_input.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Chronological App change per file (scan point order = time order)
# Drift may indicate resonance frequency shift due to heat accumulation.
# =============================================================================

print("\n=== Chronological Current (App) ===")

main_files = [f for f in per_point_data if f not in EXCLUDED_FILES]

fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
for fname in main_files:
    i_pp = per_point_data[fname]["i_pp"]
    ax.plot(range(len(i_pp)), i_pp, label=fname.replace(".tdms", ""), linewidth=0.8)

ax.set_xlabel("Scan point index (chronological)")
ax.set_ylabel("Current (App)")
ax.set_title("Driving Current vs Scan Point (resonance drift check)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / "current_chronological.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
