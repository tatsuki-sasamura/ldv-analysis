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

from ldv_analysis.config import (
    FIG_DPI,
    VELOCITY_SCALE,
    channel_centre_func,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.mode_fit import fit_columns, fit_mode_1f, make_quality_mask

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
    phase_1f = cache["phase_1f"]
    pressure_complex = pressure * np.exp(1j * np.radians(phase_1f))
    result = fit_mode_1f(pos_y[valid], pressure_complex[valid], CHANNEL_WIDTH * 1e3)

    all_freqs.append(f_drive / 1e6)
    all_p0.append(abs(result.p0))
    all_V.append(float(np.median(V[valid])))
    all_I.append(float(np.median(I[valid])))
    all_phase.append(float(np.median(phase_vi[valid])))

    print(f"  {tdms_path.stem}: p0={abs(result.p0)/1e3:.0f} kPa, "
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

# Acoustic energy density: E_ac = p0² / (4 ρ c²)  (Bruus 2012 convention)
E_ac = p0_arr ** 2 / (4 * RHO * C_SOUND ** 2)  # J/m³

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
ax1.set_ylabel(r"$P_\mathrm{elec}$ [mW]")
ax1.set_title("Electro-acoustic response --- test5")

ax2.plot(freq_arr, E_ac, ".-", markersize=3, linewidth=0.8, color="C0")
ax2.set_ylabel(r"$E_\mathrm{ac}$ [J/m$^3$]")
ax2.set_xlabel("Frequency [MHz]")

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
cb.set_label("Frequency [MHz]")
ax.set_xlabel(r"$P_\mathrm{elec}$ [mW]")
ax.set_ylabel(r"$E_\mathrm{ac}$ [J/m$^3$]")
ax.set_title("Acoustic energy vs electrical power")

plt.tight_layout()
out_path2 = OUT_DIR / "eac_vs_pelec.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Plot 3: E_ac vs P_elec from voltage sweep (test10, same frequency)
# =============================================================================

VSWEEP_DIR = get_data_dir("20260307experimentB")
VSWEEP_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",  5,  0.5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10, 1.0),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15, 1.0),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20, 1.0),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25, 2.5),
]
_vsweep_geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
_vsweep_centre_fn = channel_centre_func(_vsweep_geom)

hw_v = CHANNEL_WIDTH / 2 * 1e3  # mm

vs_vpp = []
vs_Pelec = []
vs_Eac_peak = []
vs_Eac_avg = []
vs_Eac_2mm = []

for fname, vpp, vel_scale in VSWEEP_FILES:
    tdms_path = VSWEEP_DIR / fname
    if not tdms_path.exists():
        continue
    vc = load_or_compute(tdms_path, CACHE_DIR)
    vel_correction = vel_scale / 1.0  # VELOCITY_SCALE = 1.0

    pos_x = vc["pos_x"]
    pos_y = vc["pos_y"]
    n_x_meta = int(vc["n_x_meta"])
    n_y_meta = int(vc["n_y_meta"])
    V = vc["voltage_1f"]
    I = vc["current_1f"]
    phase_vi = vc["phase_vi"]
    pressure = vc["pressure_1f"] * vel_correction

    # P_elec from median V, I, phase across all valid points
    valid = V > np.median(V) * 0.5
    P_e = 0.5 * float(np.median(V[valid])) * float(np.median(I[valid])) \
        * np.cos(np.radians(float(np.median(phase_vi[valid]))))

    # p0 from column-wise mode-shape fit (same as voltage_sweep.py)
    pos_x_c = pos_x - _vsweep_centre_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw_v

    y_min, y_max = pos_y.min(), pos_y.max()
    length_grid = np.linspace(y_min, y_max, n_y_meta)
    l_idx = np.argmin(np.abs(pos_y[:, None] - length_grid[None, :]), axis=1)

    width_span = pos_x.max() - pos_x.min()
    scan_step = width_span / max(n_x_meta - 1, 1)
    n_width_c = max(int(round(CHANNEL_WIDTH * 1e3 / scan_step)), 2)
    half_step = CHANNEL_WIDTH * 1e3 / n_width_c / 2
    width_c_grid = np.linspace(-hw_v + half_step, hw_v - half_step, n_width_c)
    wc_m = width_c_grid * 1e-3
    w_c_idx = np.argmin(np.abs(pos_x_c[:, None] - width_c_grid[None, :]), axis=1)

    grid = np.full((n_width_c, n_y_meta), np.nan)
    mask = inside & ~np.isnan(pressure)
    grid[w_c_idx[mask], l_idx[mask]] = pressure[mask]

    quality_mask = make_quality_mask(n_width_c)
    p0_y = fit_columns(grid, wc_m, CHANNEL_WIDTH, harmonic=1,
                        quality_mask=quality_mask)
    p0_peak = np.nanmax(p0_y)
    Eac_peak = p0_peak ** 2 / (4 * RHO * C_SOUND ** 2)
    Eac_avg = np.nanmean(p0_y ** 2) / (4 * RHO * C_SOUND ** 2)

    # Average over 2 mm window centred on antinode
    best_j = np.nanargmax(p0_y)
    y_centre = length_grid[best_j]
    win_mask = np.abs(length_grid - y_centre) <= 1.0  # ±1 mm
    Eac_2mm = np.nanmean(p0_y[win_mask] ** 2) / (4 * RHO * C_SOUND ** 2)

    vs_vpp.append(vpp)
    vs_Pelec.append(P_e)
    vs_Eac_peak.append(Eac_peak)
    vs_Eac_avg.append(Eac_avg)
    vs_Eac_2mm.append(Eac_2mm)
    print(f"  {vpp} Vpp: P_elec = {P_e*1e3:.1f} mW, "
          f"E_ac(peak) = {Eac_peak:.1f}, "
          f"<E_ac> = {Eac_avg:.1f}, "
          f"<E_ac>_2mm = {Eac_2mm:.1f} J/m3")

vs_vpp = np.array(vs_vpp)
vs_Pelec = np.array(vs_Pelec)
vs_Eac_peak = np.array(vs_Eac_peak)
vs_Eac_avg = np.array(vs_Eac_avg)
vs_Eac_2mm = np.array(vs_Eac_2mm)

# --- Peak E_ac plot ---
fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(vs_Pelec * 1e3, vs_Eac_peak, "o", markersize=4)
a_fit = np.sum(vs_Pelec * vs_Eac_peak) / np.sum(vs_Pelec ** 2)
P_fine = np.linspace(0, vs_Pelec.max() * 1.1, 100)
ax.plot(P_fine * 1e3, a_fit * P_fine, "--", linewidth=0.8, alpha=0.6,
        label=f"linear: {a_fit/1e3:.1f} kJ/(m$^3$ W)")
ax.set_xlabel(r"$P_\mathrm{elec}$ [mW]")
ax.set_ylabel(r"$E_\mathrm{ac,peak}$ [J/m$^3$]")
ax.set_title(r"Voltage sweep --- 1.907 MHz (peak)")
ax.legend(fontsize=6, frameon=False)
plt.tight_layout()
out_path3 = OUT_DIR / "eac_vs_pelec_voltage_peak.png"
fig.savefig(out_path3, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path3}")

# --- Average E_ac plot ---
fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(vs_Pelec * 1e3, vs_Eac_avg, "o", markersize=4)
a_fit_avg = np.sum(vs_Pelec * vs_Eac_avg) / np.sum(vs_Pelec ** 2)
ax.plot(P_fine * 1e3, a_fit_avg * P_fine, "--", linewidth=0.8, alpha=0.6,
        label=f"linear: {a_fit_avg/1e3:.1f} kJ/(m$^3$ W)")
ax.set_xlabel(r"$P_\mathrm{elec}$ [mW]")
ax.set_ylabel(r"$\langle E_\mathrm{ac} \rangle$ [J/m$^3$]")
ax.set_title(r"Voltage sweep --- 1.907 MHz (average)")
ax.legend(fontsize=6, frameon=False)
plt.tight_layout()
out_path4 = OUT_DIR / "eac_vs_pelec_voltage_avg.png"
fig.savefig(out_path4, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path4}")

# --- Average E_ac over 2 mm window around antinode ---
fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(vs_Pelec * 1e3, vs_Eac_2mm, "o", markersize=4)
a_fit_2mm = np.sum(vs_Pelec * vs_Eac_2mm) / np.sum(vs_Pelec ** 2)
ax.plot(P_fine * 1e3, a_fit_2mm * P_fine, "--", linewidth=0.8, alpha=0.6,
        label=f"linear: {a_fit_2mm/1e3:.1f} kJ/(m$^3$ W)")
ax.set_xlabel(r"$P_\mathrm{elec}$ [mW]")
ax.set_ylabel(r"$\langle E_\mathrm{ac} \rangle_{2\,\mathrm{mm}}$ [J/m$^3$]")
ax.set_title(r"Voltage sweep --- 1.907 MHz (2 mm avg)")
ax.legend(fontsize=6, frameon=False)
plt.tight_layout()
out_path5 = OUT_DIR / "eac_vs_pelec_voltage_2mm.png"
fig.savefig(out_path5, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path5}")

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
