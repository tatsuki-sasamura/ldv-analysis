# %%
"""Generate manuscript figures (Figs 5–8) for the nonlinear harmonics paper.

Produces publication-ready .eps and .png figures from existing experimental data.

Usage:
    python manuscript_figures.py
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
    channel_centre_func,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.grid_utils import make_to_grid
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms
from ldv_analysis.mode_fit import fit_columns, fit_mode_2f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR_B = get_data_dir("20260307experimentB")
DATA_DIR_A = get_data_dir("20260306experimentA")

# test10 voltage sweep files (Step B)
VOLTAGE_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",  5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

CHANNEL_WIDTH = 0.375e-3   # m

# Fluid properties
RHO = 1004.0    # kg/m³
C0 = 1508.0     # m/s
RSSI_THRESHOLD = 1.0  # V — exclude poor LDV signal

CACHE_DIR = get_output_dir(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("E:/OneDrive - Lund University/Publications"
               "/nonlinearphysics/manuscript/figures")


def save_fig(fig, name):
    """Save figure as both .eps and .png."""
    fig.savefig(FIG_DIR / f"{name}.eps", dpi=FIG_DPI)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=FIG_DPI)
    print(f"Saved: {FIG_DIR / name}.eps/.png")


# %%
# =============================================================================
# Load test10 voltage sweep data (shared by Figs 5, 6, 7)
# =============================================================================

hw = CHANNEL_WIDTH / 2 * 1e3  # mm
cw_mm = CHANNEL_WIDTH * 1e3

# Channel geometry from calibration
_geom_B = load_channel_geometry("20260307experimentB", CACHE_DIR)
_centre_fn_B = channel_centre_func(_geom_B)

results = []

for fname, vpp in VOLTAGE_FILES:
    tdms_path = DATA_DIR_B / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    n_x_meta = int(cache["n_x_meta"])
    n_y_meta = int(cache["n_y_meta"])
    f_drive = float(cache["f_drive"])
    pressure_1f = cache["pressure_1f"]

    # Channel mask
    pos_x_c = pos_x - _centre_fn_B(pos_y)
    inside = np.abs(pos_x_c) <= hw

    # Build grid (same as voltage_sweep.py)
    y_min, y_max = pos_y.min(), pos_y.max()
    length_grid = np.linspace(y_min, y_max, n_y_meta)
    l_idx = np.argmin(np.abs(pos_y[:, None] - length_grid[None, :]), axis=1)

    width_span = pos_x.max() - pos_x.min()
    scan_step = width_span / max(n_x_meta - 1, 1)
    n_width_c = max(int(round(CHANNEL_WIDTH * 1e3 / scan_step)), 2)
    half_step = CHANNEL_WIDTH * 1e3 / n_width_c / 2
    width_c_grid = np.linspace(-hw + half_step, hw - half_step, n_width_c)
    wc_m = width_c_grid * 1e-3

    w_c_idx = np.argmin(np.abs(pos_x_c[:, None] - width_c_grid[None, :]), axis=1)

    rssi = cache["rssi"] if "rssi" in cache else None
    to_grid = make_to_grid(w_c_idx, l_idx, inside, n_width_c, n_y_meta,
                           rssi=rssi, rssi_threshold=RSSI_THRESHOLD)

    # 1f fit — axial profile of p0(x)
    grid_prs_1f = to_grid(pressure_1f)
    p0_1f_y = fit_columns(grid_prs_1f,
                          wc_m, CHANNEL_WIDTH, harmonic=1)
    # Whole-channel average E_ac (Baasch et al. 2024): ⟨E_ac⟩ = ⟨p0²⟩/(4ρc²)
    Eac_1f = np.nanmean(p0_1f_y**2) / (4 * RHO * C0**2)

    # 2f fit
    pressure_2f = cache["pressure_2f"]
    grid_prs_2f = to_grid(pressure_2f)
    p0_2f_y = fit_columns(grid_prs_2f,
                          wc_m, CHANNEL_WIDTH, harmonic=2)
    Eac_2f = np.nanmean(p0_2f_y**2) / (4 * RHO * C0**2)

    # Electrical power: P_in = ½ V I cos(φ)
    V = cache["voltage_1f"]
    I = cache["current_1f"]
    phase_vi = cache["phase_vi"]
    valid_e = V > np.median(V) * 0.5
    P_in = 0.5 * float(np.median(V[valid_e])) * float(np.median(I[valid_e])) \
        * np.cos(np.radians(float(np.median(phase_vi[valid_e]))))

    # Peak p0 (for Fig 6 harmonic panels)
    p0_1f_peak = np.nanmax(p0_1f_y)  # Pa
    p0_2f_peak = np.nanmax(p0_2f_y)  # Pa

    # Ch1 drive voltage harmonics: 2f/1f ratio (for Fig 6c)
    # Use median point for Ch1 spectrum (signal is spatially uniform)
    n_points = len(pos_x)
    mid_pt = n_points // 2
    wf_ch1, dt = load_point_waveforms(tdms_path, mid_pt, channels=(1,))
    ch1 = wf_ch1[1]
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    ss_n = ss_end - ss_start
    ch1_ss = ch1[ss_start:ss_end]
    tone_1f = np.exp(-2j * np.pi * f_drive * np.arange(ss_n) * dt)
    tone_2f = np.exp(-2j * np.pi * 2 * f_drive * np.arange(ss_n) * dt)
    ch1_1f_amp = np.abs(ch1_ss @ tone_1f) * 2 / ss_n
    ch1_2f_amp = np.abs(ch1_ss @ tone_2f) * 2 / ss_n
    ch1_ratio = ch1_2f_amp / ch1_1f_amp * 100  # %

    results.append(dict(
        vpp=vpp, Eac_1f=Eac_1f, Eac_2f=Eac_2f, P_in=P_in,
        p0_1f_peak=p0_1f_peak, p0_2f_peak=p0_2f_peak,
        p0_1f_y=p0_1f_y, p0_2f_y=p0_2f_y,
        ch1_ratio=ch1_ratio,
        f_drive=f_drive, tdms_path=tdms_path,
        cache=cache,
    ))
    print(f"  {fname}: p0_1f = {p0_1f_peak/1e3:.0f} kPa, "
          f"p0_2f = {p0_2f_peak/1e3:.0f} kPa, "
          f"Ch1 2f/1f = {ch1_ratio:.2f}%, "
          f"P_in = {P_in*1e3:.1f} mW")

Vpp = np.array([r["vpp"] for r in results])
Eac_1f_arr = np.array([r["Eac_1f"] for r in results])       # J/m³
Eac_2f_arr = np.array([r["Eac_2f"] for r in results])       # J/m³
P_in_arr = np.array([r["P_in"] for r in results])            # W
p0_1f_peak_arr = np.array([r["p0_1f_peak"] for r in results])  # Pa
p0_2f_peak_arr = np.array([r["p0_2f_peak"] for r in results])  # Pa
ch1_ratio_arr = np.array([r["ch1_ratio"] for r in results])    # %
f_drive = results[0]["f_drive"]

# %%
# =============================================================================
# Fig 5: E_ac,avg vs P_in  (Baasch et al. 2024 convention)
# =============================================================================

P_in_mW = P_in_arr * 1e3  # mW

# Linear fit through origin: E_ac = a * P_in
a_eac = np.sum(P_in_arr * Eac_1f_arr) / np.sum(P_in_arr**2)
P_fine = np.linspace(0, P_in_arr.max() * 1.1, 100)

fig, ax = plt.subplots(figsize=(3.375, 2.5))
ax.plot(P_in_mW, Eac_1f_arr, "ko", markersize=4, zorder=3)
ax.plot(P_fine * 1e3, a_eac * P_fine, "k:", linewidth=0.5,
        label=f"Linear fit")
ax.set_xlabel(r"$P_\mathrm{in}$ [mW]")
ax.set_ylabel(r"$\langle E_\mathrm{ac} \rangle$ [J/m$^3$]")
ax.legend(frameon=False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
save_fig(fig, "Fig5")
plt.close()

# Print E_ac values
print("\n--- Fig 5: <E_ac> vs P_in ---")
for i, r in enumerate(results):
    print(f"  {r['vpp']:2d} Vpp: <E_ac> = {Eac_1f_arr[i]:.1f} J/m³, "
          f"P_in = {P_in_mW[i]:.1f} mW")
print(f"  Linear slope: {a_eac/1e3:.1f} kJ/(m³ W)")

# %%
print("\n=== Fig 5 Done ===")

# %%
# =============================================================================
# Fig 6: Drive-resolved harmonics
# =============================================================================

# Include origin for fits
Vpp_0 = np.array([0] + list(Vpp))
p0_1f_0 = np.array([0] + list(p0_1f_peak_arr))  # Pa
p0_2f_0 = np.array([0] + list(p0_2f_peak_arr))  # Pa

# Fits through origin: P_1f = a*V, P_2f = b*V²
a_1f = np.sum(Vpp_0 * p0_1f_0) / np.sum(Vpp_0**2)
b_2f = np.sum(Vpp_0**2 * p0_2f_0) / np.sum(Vpp_0**4)
V_fine = np.linspace(0, Vpp.max() * 1.15, 100)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.375, 6.0))

# (a) P_1f and P_2f vs Vpp
ax1.plot(Vpp, p0_1f_peak_arr / 1e6, "o", markersize=4, color="tab:blue",
         label=r"$P_{1f}$")
ax1.plot(V_fine, a_1f * V_fine / 1e6, ":", linewidth=0.5, color="tab:blue")
ax1.plot(Vpp, p0_2f_peak_arr / 1e6, "s", markersize=3, color="tab:red",
         label=r"$P_{2f}$")
ax1.plot(V_fine, b_2f * V_fine**2 / 1e6, ":", linewidth=0.5, color="tab:red")
ax1.set_ylabel(r"Pressure amplitude [MPa]")
ax1.legend(frameon=False)
ax1.set_ylim(bottom=0)
ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, va="top",
         fontweight="bold")

# (b) P_2f / P_1f ratio vs Vpp
ratio = p0_2f_peak_arr / p0_1f_peak_arr
ax2.plot(Vpp, ratio, "D", markersize=4, color="C2")
# Linear fit through origin: ratio = (b_2f/a_1f)*V
ratio_slope = b_2f / a_1f  # 1/V
ax2.plot(V_fine[1:], ratio_slope * V_fine[1:], ":", linewidth=0.5,
         color="C2")
ax2.set_ylabel(r"$P_{2f}/P_{1f}$")
ax2.set_ylim(bottom=0)
ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, va="top",
         fontweight="bold")

# (c) Ch1 drive voltage 2f/1f harmonic ratio
ax3.plot(Vpp, ch1_ratio_arr / 100, "^", markersize=4, color="C3")
ax3.set_ylabel(r"PZT $V_{2f}/V_{1f}$")
ax3.set_xlabel(r"Drive voltage $V_\mathrm{pp}$ [V]")
ax3.set_ylim(bottom=0)
ax3.text(0.02, 0.95, "(c)", transform=ax3.transAxes, va="top",
         fontweight="bold")

plt.tight_layout()
save_fig(fig, "Fig6")
plt.close()

# Print summary
print("\n--- Fig 6: Harmonics vs Vpp ---")
print(f"  Fit: P_1f = {a_1f/1e3:.1f} kPa/V")
print(f"  Fit: P_2f = {b_2f/1e3:.3f} kPa/V²")
print(f"  Ratio slope: {ratio_slope:.4f} /V")
for i, r in enumerate(results):
    print(f"  {r['vpp']:2d} Vpp: P_1f = {p0_1f_peak_arr[i]/1e3:.0f} kPa, "
          f"P_2f = {p0_2f_peak_arr[i]/1e3:.0f} kPa, "
          f"ratio = {ratio[i]:.3f}, "
          f"Ch1 2f/1f = {ch1_ratio_arr[i]:.2f}%")

# %%
print("\n=== Fig 6 Done ===")
