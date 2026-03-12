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
    CHANNEL_WIDTH,
    C_SOUND,
    CURRENT_SCALE,
    FIG_DPI,
    RHO,
    RSSI_THRESHOLD,
    SENSITIVITY,
    channel_centre_func,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import (
    detect_velocity_scale, load_or_compute, load_point_waveforms,
)
from ldv_analysis.grid_utils import make_channel_grid
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

hw = CHANNEL_WIDTH / 2  # m

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

    # Build grid
    rssi = cache["rssi"] if "rssi" in cache else None
    cg = make_channel_grid(
        pos_x_c, pos_y, n_x_meta, n_y_meta,
        CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    # 1f fit — axial profile of p0(x)
    grid_prs_1f = cg.to_grid(pressure_1f)
    p0_1f_y = fit_columns(grid_prs_1f,
                          cg.width_grid, CHANNEL_WIDTH, harmonic=1)
    # Whole-channel average E_ac (Baasch et al. 2024): ⟨E_ac⟩ = ⟨p0²⟩/(4ρc²)
    Eac_1f = np.nanmean(p0_1f_y**2) / (4 * RHO * C_SOUND**2)

    # 2f fit
    pressure_2f = cache["pressure_2f"]
    grid_prs_2f = cg.to_grid(pressure_2f)
    p0_2f_y = fit_columns(grid_prs_2f,
                          cg.width_grid, CHANNEL_WIDTH, harmonic=2)
    Eac_2f = np.nanmean(p0_2f_y**2) / (4 * RHO * C_SOUND**2)

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

# %%
# =============================================================================
# Fig 7: Time-domain waveform distortion (10 Vpp vs 25 Vpp)
# =============================================================================
#
# 2 rows (drive levels) × 5 columns (positions across channel width).
# Each panel: raw velocity waveform (grey) with reconstructed 1f (blue)
# and 1f+2f (red dashed) overlays.  Shows visible harmonic distortion
# at high drive and the spatial structure of the second harmonic.

# --- Configuration ---
FIG7_FILES = [
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]
FIG7_TARGET_XC = [-hw, -hw / 2, 0, hw / 2, hw]  # -W/2, -W/4, 0, +W/4, +W/2
FIG7_POS_LABELS = [
    r"$y = -W/2$", r"$y = -W/4$", r"$y = 0$", r"$y = +W/4$", r"$y = +W/2$",
]
FIG7_WINDOW_US = 1.0  # display window length (µs)


def _find_best_shared_row(file_list):
    """Find the axial (y) row that passes RSSI and voltage checks at all 5
    target positions for every file, with the best wall symmetry."""
    W = CHANNEL_WIDTH
    targets = FIG7_TARGET_XC

    # Collect caches
    caches = []
    for fname, _ in file_list:
        caches.append(load_or_compute(DATA_DIR_B / fname, CACHE_DIR))

    # Common y-values (rounded to 10 µm)
    y_sets = [set(np.round(c["pos_y"], 5)) for c in caches]
    common_y = sorted(y_sets[0].intersection(*y_sets[1:]))

    best = None
    for yval in common_y:
        sym_sum = 0.0
        ok = True
        for c in caches:
            pos_x_c = c["pos_x"] - _centre_fn_B(c["pos_y"])
            rssi = c["rssi"] if "rssi" in c else None
            V = c["voltage_1f"]
            V_med = np.median(V)
            y_mask = np.abs(c["pos_y"] - yval) < 1e-5
            row_idx = np.where(y_mask)[0]
            if len(row_idx) < 5:
                ok = False
                break
            p_walls = []
            for target in targets:
                dists = np.abs(pos_x_c[row_idx] - target)
                bi = np.argmin(dists)
                idx = row_idx[bi]
                if dists[bi] > 10e-6:
                    ok = False
                    break
                if rssi is not None and rssi[idx] < RSSI_THRESHOLD:
                    ok = False
                    break
                if V[idx] < V_med * 0.5:
                    ok = False
                    break
                if abs(target) == W / 2:
                    p_walls.append(c["pressure_1f"][idx])
            if not ok:
                break
            if len(p_walls) == 2:
                sym_sum += abs(p_walls[0] - p_walls[1]) / max(p_walls)
        if not ok:
            continue
        if best is None or sym_sum < best[1]:
            best = (yval, sym_sum)

    return best[0]


fig7_y = _find_best_shared_row(FIG7_FILES)
print(f"\n--- Fig 7: y = {fig7_y*1e3:.3f} mm ---")

fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.0))

for row, (fname, vpp) in enumerate(FIG7_FILES):
    tdms_path = DATA_DIR_B / fname
    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x_c = cache["pos_x"] - _centre_fn_B(cache["pos_y"])
    f_dr = float(cache["f_drive"])
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    ss_n = ss_end - ss_start
    vel_scale = detect_velocity_scale(tdms_path)

    y_mask = np.abs(cache["pos_y"] - fig7_y) < 1e-5
    row_idx = np.where(y_mask)[0]

    for col, (target, plabel) in enumerate(
            zip(FIG7_TARGET_XC, FIG7_POS_LABELS)):
        dists = np.abs(pos_x_c[row_idx] - target)
        pt_idx = row_idx[np.argmin(dists)]

        # Raw waveform (Ch2 velocity) → pressure
        wf, dt = load_point_waveforms(tdms_path, pt_idx, channels=(2,))
        vel = wf[2] * vel_scale  # m/s
        prs = vel / (2 * np.pi * f_dr * SENSITIVITY)  # Pa

        # Display window: fixed 1 µs centred in steady state
        n_show = int(FIG7_WINDOW_US * 1e-6 / dt)
        t0 = (ss_start + ss_end) // 2 - n_show // 2
        t1 = t0 + n_show
        t_local = np.arange(t1 - t0) * dt
        t_us = t_local * 1e6

        # Local DFT at 1f and 2f over display window
        raw_seg = prs[t0:t1]
        n_loc = t1 - t0
        c1f = (raw_seg @ np.exp(-2j * np.pi * f_dr * t_local)) * 2 / n_loc
        c2f = (raw_seg @ np.exp(-2j * np.pi * 2 * f_dr * t_local)) * 2 / n_loc

        raw_mpa = raw_seg / 1e6  # MPa
        recon_1f = np.real(c1f * np.exp(2j * np.pi * f_dr * t_local)) / 1e6
        recon_12f = recon_1f + np.real(
            c2f * np.exp(2j * np.pi * 2 * f_dr * t_local)) / 1e6

        ax = axes[row, col]
        ax.plot(t_us, raw_mpa, linewidth=0.5, color="C0")
        ax.plot(t_us, recon_12f, linewidth=0.5, color="C3", ls="--")
        ax.axhline(0, color="0.85", lw=0.25)
        ax.tick_params(labelsize=4)
        ylim = 1 if vpp == 10 else 3
        ax.set_ylim(-ylim, ylim)

        if row == 0:
            ax.set_title(plabel, fontsize=7)
        if col == 0:
            ax.set_ylabel(f"{vpp} Vpp\nPressure [MPa]", fontsize=5)

        a1f = abs(c1f) / 1e3  # kPa for print
        a2f = abs(c2f) / 1e3
        ratio_pct = a2f / a1f * 100 if a1f > 0 else 0
        print(f"  {vpp:2d} Vpp {plabel:>6s}: p1f={a1f:.0f}, "
              f"p2f={a2f:.0f} kPa ({ratio_pct:.1f}%)")

# Shared x-label
for ax in axes[-1]:
    ax.set_xlabel(r"Time [\textmu s]", fontsize=5)

# Legend on first panel
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color="C0", lw=0.7, label="Raw"),
    Line2D([], [], color="C3", lw=0.7, ls="--", label="$1f+2f$"),
]
axes[0, 0].legend(handles=handles, fontsize=3.5, loc="lower left",
                  frameon=True, fancybox=False, edgecolor="0.8")

plt.tight_layout()
save_fig(fig, "Fig7")
plt.close()

# %%
print("\n=== Fig 7 Done ===")
