# %%
"""Harmonic pressure amplitudes (1f, 2f, 3f) vs drive voltage.

Fig-8a style plot using test10 2D scan data.  Extracts P_{1f}, P_{2f},
P_{3f} at the axial antinode via mode-shape fit with sigma clipping.

Fits: P_1f = a*V (linear), P_2f = b*V^2 (quadratic), P_3f = c*V^3 (cubic).

Usage:
    python harmonics_vs_voltage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    RSSI_THRESHOLD,
    channel_centre_func,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260307experimentB")
VOLTAGE_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms", 5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MARKER_SIZE = 4
SIGMA_CLIP = 3.0

# %%
# =============================================================================
# Process each voltage level
# =============================================================================

geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
centre_fn = channel_centre_func(geom)
hw = CHANNEL_WIDTH / 2

vpps = []
p0_1f_arr = []
p0_2f_arr = []
p0_3f_arr = []
sig_1f_arr = []
sig_2f_arr = []
sig_3f_arr = []

for fname, vpp in VOLTAGE_FILES:
    tdms_path = DATA_DIR / fname
    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    n_x = int(cache["n_x_meta"])
    n_y = int(cache["n_y_meta"])
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    # Filtering
    valid = make_valid_mask(V, rssi)
    if "pt_burst_on_us" in cache:
        valid &= make_burst_timing_mask(cache["pt_burst_on_us"],
                                        cache["pt_burst_off_us"])

    # Build grid
    pos_x_c = pos_x - centre_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(pos_x_c, pos_y, n_x, n_y,
                           CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
                           rssi=rssi, rssi_threshold=RSSI_THRESHOLD)

    # Mask invalid points
    for h in [1, 2, 3]:
        key = f"pressure_{h}f"
        if key not in cache:
            continue

    # Fit columns with sigma clipping for each harmonic
    results_h = {}
    for h in [1, 2, 3]:
        prs = cache[f"pressure_{h}f"].copy()
        prs[~valid] = np.nan
        grid = cg.to_grid(prs)
        p0_y, sig_y = fit_columns(grid, cg.width_grid, CHANNEL_WIDTH,
                                   harmonic=h, return_sigma=True,
                                   sigma_clip=SIGMA_CLIP)
        results_h[h] = (p0_y, sig_y)

    # Take values at the 1f antinode
    p0_1f_y, sig_1f_y = results_h[1]
    j_anti = int(np.nanargmax(p0_1f_y))

    vpps.append(vpp)
    p0_1f_arr.append(float(p0_1f_y[j_anti]))
    p0_2f_arr.append(float(results_h[2][0][j_anti]))
    p0_3f_arr.append(float(results_h[3][0][j_anti]))
    sig_1f_arr.append(float(sig_1f_y[j_anti]))
    sig_2f_arr.append(float(results_h[2][1][j_anti]))
    sig_3f_arr.append(float(results_h[3][1][j_anti]))

    print(f"  {vpp:2d} Vpp: P_1f={p0_1f_arr[-1]/1e3:.0f}, "
          f"P_2f={p0_2f_arr[-1]/1e3:.0f}, "
          f"P_3f={p0_3f_arr[-1]/1e3:.1f} kPa")

vpps = np.array(vpps)
p0_1f = np.array(p0_1f_arr)
p0_2f = np.array(p0_2f_arr)
p0_3f = np.array(p0_3f_arr)
sig_1f = np.array(sig_1f_arr)
sig_2f = np.array(sig_2f_arr)
sig_3f = np.array(sig_3f_arr)

# %%
# =============================================================================
# Fits through origin
# =============================================================================

a_1f = float(np.sum(vpps * p0_1f) / np.sum(vpps**2))           # Pa/V
b_2f = float(np.sum(vpps**2 * p0_2f) / np.sum(vpps**4))        # Pa/V^2
c_3f = float(np.sum(vpps**3 * p0_3f) / np.sum(vpps**6))        # Pa/V^3

# R^2 for through-origin fits
r2_1f = 1 - np.sum((p0_1f - a_1f * vpps)**2) / np.sum(p0_1f**2)
r2_2f = 1 - np.sum((p0_2f - b_2f * vpps**2)**2) / np.sum(p0_2f**2)
r2_3f = 1 - np.sum((p0_3f - c_3f * vpps**3)**2) / np.sum(p0_3f**2)

print(f"\nFits through origin:")
print(f"  P_1f = {a_1f/1e3:.1f} kPa/V (R2={r2_1f:.4f})")
print(f"  P_2f = {b_2f/1e3:.3f} kPa/V^2 (R2={r2_2f:.4f})")
print(f"  P_3f = {c_3f/1e3:.6f} kPa/V^3 (R2={r2_3f:.4f})")

# %%
# =============================================================================
# Plot
# =============================================================================

V_fine = np.linspace(vpps.min() * 0.8, vpps.max() * 1.15, 100)

fig, ax = plt.subplots(figsize=figsize_for_layout())

# P_1f
ax.errorbar(vpps, p0_1f / 1e6, yerr=sig_1f / 1e6,
            fmt="o", markersize=MARKER_SIZE, color="tab:blue",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(V_fine, a_1f * V_fine / 1e6, ":", linewidth=0.5, color="tab:blue",
        label=r"$P_{1f}=%.1f\,V_\mathrm{drive}$ ($R^2=%.3f$)" % (a_1f / 1e3, r2_1f))

# P_2f
ax.errorbar(vpps, p0_2f / 1e6, yerr=sig_2f / 1e6,
            fmt="s", markersize=MARKER_SIZE, color="tab:red",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(V_fine, b_2f * V_fine**2 / 1e6, ":", linewidth=0.5, color="tab:red",
        label=r"$P_{2f}=%.2f\,V_\mathrm{drive}^2$ ($R^2=%.3f$)" % (b_2f / 1e3, r2_2f))

# P_3f
ax.errorbar(vpps, p0_3f / 1e6, yerr=sig_3f / 1e6,
            fmt="^", markersize=MARKER_SIZE, color="tab:green",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(V_fine, c_3f * V_fine**3 / 1e6, ":", linewidth=0.5, color="tab:green",
        label=r"$P_{3f}=%.5f\,V_\mathrm{drive}^3$ ($R^2=%.3f$)" % (c_3f / 1e3, r2_3f))

ax.set_xlabel(r"Drive voltage $V_\mathrm{drive}$ [$V_\mathrm{pp}$]")
ax.set_ylabel(r"Pressure amplitude [MPa]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(vpps.min() * 0.8, vpps.max() * 1.2)
ax.legend(frameon=False, fontsize=6)

plt.tight_layout()
out_path = OUT_DIR / "harmonics_vs_voltage.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# Ratio plot: P_{2f}/P_{1f} and P_{3f}/P_{1f} vs V_drive
ratio_2f = p0_2f / p0_1f
ratio_3f = p0_3f / p0_1f

# Propagated errors: sigma(r) = r * sqrt((sig_n/P_n)^2 + (sig_1/P_1)^2)
ratio_2f_err = ratio_2f * np.sqrt((sig_2f / p0_2f)**2 + (sig_1f / p0_1f)**2)
ratio_3f_err = ratio_3f * np.sqrt((sig_3f / p0_3f)**2 + (sig_1f / p0_1f)**2)

# Fit slopes: P_2f/P_1f = (b/a)*V, P_3f/P_1f = (c/a)*V^2
slope_2f = b_2f / a_1f
slope_3f = c_3f / a_1f

fig, ax = plt.subplots(figsize=figsize_for_layout())

ax.errorbar(vpps, ratio_2f, yerr=ratio_2f_err,
            fmt="s", markersize=MARKER_SIZE, color="tab:red",
            capsize=3, capthick=0.5, elinewidth=0.5)
r2_ratio_2f = 1 - np.sum((ratio_2f - slope_2f * vpps)**2) / np.sum(ratio_2f**2)
ax.plot(V_fine, slope_2f * V_fine, ":", linewidth=0.5, color="tab:red",
        label=r"$P_{2f}/P_{1f}=%.4f\,V_\mathrm{drive}$ ($R^2=%.3f$)" % (slope_2f, r2_ratio_2f))

ax.errorbar(vpps, ratio_3f, yerr=ratio_3f_err,
            fmt="^", markersize=MARKER_SIZE, color="tab:green",
            capsize=3, capthick=0.5, elinewidth=0.5)
r2_ratio_3f = 1 - np.sum((ratio_3f - slope_3f * vpps**2)**2) / np.sum(ratio_3f**2)
ax.plot(V_fine, slope_3f * V_fine**2, ":", linewidth=0.5, color="tab:green",
        label=r"$P_{3f}/P_{1f}=%.6f\,V_\mathrm{drive}^2$ ($R^2=%.3f$)" % (slope_3f, r2_ratio_3f))

ax.set_xlabel(r"Drive voltage $V_\mathrm{drive}$ [$V_\mathrm{pp}$]")
ax.set_ylabel(r"Harmonic ratio")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(vpps.min() * 0.8, vpps.max() * 1.2)
ax.legend(frameon=False, fontsize=6)
plt.tight_layout()
out_path = OUT_DIR / "harmonics_vs_voltage_ratio.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Mach number variants (x-axis = M = P_1f / (rho * c^2))
# =============================================================================

from ldv_analysis.config import C_SOUND, RHO

M = p0_1f / (RHO * C_SOUND**2)
M_fine = np.linspace(M.min() * 0.8, M.max() * 1.15, 100)

# --- Pressure vs Mach number (log-log) ---
fig, ax = plt.subplots(figsize=figsize_for_layout())

ax.errorbar(M, p0_1f / 1e6, yerr=sig_1f / 1e6,
            fmt="o", markersize=MARKER_SIZE, color="tab:blue",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, M_fine * RHO * C_SOUND**2 / 1e6, ":", linewidth=0.5,
        color="tab:blue", label=r"$P_{1f}$")

ax.errorbar(M, p0_2f / 1e6, yerr=sig_2f / 1e6,
            fmt="s", markersize=MARKER_SIZE, color="tab:red",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, b_2f / a_1f * (M_fine * RHO * C_SOUND**2)**2 / 1e6,
        ":", linewidth=0.5, color="tab:red", label=r"$P_{2f}$")

ax.errorbar(M, p0_3f / 1e6, yerr=sig_3f / 1e6,
            fmt="^", markersize=MARKER_SIZE, color="tab:green",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, c_3f / a_1f * (M_fine * RHO * C_SOUND**2)**3 / 1e6,
        ":", linewidth=0.5, color="tab:green", label=r"$P_{3f}$")

ax.set_xlabel(r"Mach number $M = P_{1f}/(\rho c^2)$")
ax.set_ylabel(r"Pressure amplitude [MPa]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(M.min() * 0.8, M.max() * 1.2)
ax.legend(frameon=False, fontsize=6)
plt.tight_layout()
out_path = OUT_DIR / "harmonics_vs_mach_number.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# --- Ratio vs Mach number (log-log) ---
fig, ax = plt.subplots(figsize=figsize_for_layout())

ax.errorbar(M, ratio_2f, yerr=ratio_2f_err,
            fmt="s", markersize=MARKER_SIZE, color="tab:red",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, slope_2f / a_1f * M_fine * RHO * C_SOUND**2,
        ":", linewidth=0.5, color="tab:red",
        label=r"$P_{2f}/P_{1f} \propto M$ ($R^2=%.3f$)" % r2_ratio_2f)

ax.errorbar(M, ratio_3f, yerr=ratio_3f_err,
            fmt="^", markersize=MARKER_SIZE, color="tab:green",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, slope_3f / a_1f**2 * (M_fine * RHO * C_SOUND**2)**2,
        ":", linewidth=0.5, color="tab:green",
        label=r"$P_{3f}/P_{1f} \propto M^2$ ($R^2=%.3f$)" % r2_ratio_3f)

ax.set_xlabel(r"Mach number $M = P_{1f}/(\rho c^2)$")
ax.set_ylabel(r"Harmonic ratio")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(M.min() * 0.8, M.max() * 1.2)
ax.legend(frameon=False, fontsize=6)
plt.tight_layout()
out_path = OUT_DIR / "harmonics_vs_mach_number_ratio.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

print("\n=== Done ===")
