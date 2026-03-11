# %%
"""Electro-acoustic energy analysis.

Computes electrical input power P_elec = ½ V I cos(φ) and peak acoustic
energy density E_ac = p₀² / (2 ρ c²) from the fine frequency sweep (test5).
Produces:
  1. Dual-panel plot: P_elec(f) and E_ac(f)
  2. Scatter of E_ac vs P_elec
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import FIG_DPI, figsize_for_layout, get_data_dir, get_output_dir
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.mode_fit import fit_mode_1f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test5_*.tdms"

CHANNEL_WIDTH = 0.375e-3  # m
RSSI_THRESHOLD = 1.0      # V

# Medium properties (water, 25°C)
RHO = 1004.0    # kg/m³
C_SOUND = 1508.0  # m/s

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Process sweep files
# =============================================================================

tdms_files = sorted(DATA_DIR.glob(FILE_PATTERN))
tdms_files = [f for f in tdms_files if not f.name.endswith("_index")]
print(f"Found {len(tdms_files)} sweep files\n")

if not tdms_files:
    print("No files found. Exiting.")
    sys.exit(0)

all_freqs = []   # MHz
all_p0 = []      # Pa
all_V = []       # V (peak)
all_I = []       # A (peak)
all_phase = []   # deg

for tdms_path in tdms_files:
    cache = load_or_compute(tdms_path, CACHE_DIR)
    f_drive = float(cache["f_drive"])
    pos_y = cache["pos_x"]
    pressure = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    valid = V > np.median(V) * 0.5
    if rssi is not None:
        valid &= rssi > RSSI_THRESHOLD

    has_ch4 = "current_1f" in cache
    if not has_ch4:
        print(f"  SKIP (no current data): {tdms_path.name}")
        continue

    I = cache["current_1f"]
    phase_vi = cache["phase_vi"]

    # Mode-shape fit for p0
    result = fit_mode_1f(pos_y[valid], pressure[valid], CHANNEL_WIDTH * 1e3)

    all_freqs.append(f_drive / 1e6)
    all_p0.append(result.p0)
    all_V.append(float(np.median(V[valid])))
    all_I.append(float(np.median(I[valid])))
    all_phase.append(float(np.median(phase_vi[valid])))

    print(f"  {tdms_path.stem}: p0={result.p0/1e3:.0f} kPa, "
          f"V={all_V[-1]:.2f} V, I={all_I[-1]*1e3:.1f} mA")

# %%
# =============================================================================
# Compute derived quantities
# =============================================================================

freq_arr = np.array(all_freqs)
sort_f = np.argsort(freq_arr)
freq_arr = freq_arr[sort_f]

p0_arr = np.array(all_p0)[sort_f]       # Pa
V_arr = np.array(all_V)[sort_f]         # V (peak from DFT)
I_arr = np.array(all_I)[sort_f]         # A (peak from DFT)
phase_arr = np.array(all_phase)[sort_f]  # deg

# Electrical input power: P = ½ V_peak I_peak cos(φ)
P_elec = 0.5 * V_arr * I_arr * np.cos(np.radians(phase_arr))  # W

# Peak acoustic energy density: E_ac = p0² / (2 ρ c²)
E_ac = p0_arr ** 2 / (2 * RHO * C_SOUND ** 2)  # J/m³

print(f"\nPeak P_elec = {np.max(P_elec)*1e3:.1f} mW at "
      f"{freq_arr[np.argmax(P_elec)]:.3f} MHz")
print(f"Peak E_ac = {np.max(E_ac):.1f} J/m³ at "
      f"{freq_arr[np.argmax(E_ac)]:.3f} MHz")

# %%
# =============================================================================
# Plot 1: P_elec(f) and E_ac(f)
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True,
)

ax1.plot(freq_arr, P_elec * 1e3, ".-", markersize=3, linewidth=0.8, color="C1")
ax1.set_ylabel(r"$P_\mathrm{elec}$ (mW)")
ax1.set_title("Electro-acoustic response --- test5")
ax1.grid(True, alpha=0.3)

ax2.plot(freq_arr, E_ac, ".-", markersize=3, linewidth=0.8, color="C0")
ax2.set_ylabel(r"$E_\mathrm{ac}$ (J/m$^3$)")
ax2.set_xlabel("Frequency (MHz)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "electroacoustic_vs_freq.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: E_ac vs P_elec scatter
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
sc = ax.scatter(P_elec * 1e3, E_ac, c=freq_arr, s=10, cmap="viridis",
                edgecolors="none")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Frequency (MHz)")
ax.set_xlabel(r"$P_\mathrm{elec}$ (mW)")
ax.set_ylabel(r"$E_\mathrm{ac}$ (J/m$^3$)")
ax.set_title("Acoustic energy vs electrical power")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = OUT_DIR / "eac_vs_pelec.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'Freq (MHz)':>11s} {'p0 (kPa)':>9s} {'P_elec (mW)':>11s} "
      f"{'E_ac (J/m³)':>11s}")
print("-" * 46)
for i in range(len(freq_arr)):
    print(f"{freq_arr[i]:11.3f} {p0_arr[i]/1e3:9.0f} {P_elec[i]*1e3:11.2f} "
          f"{E_ac[i]:11.1f}")

# %%
print("\n=== Done ===")
