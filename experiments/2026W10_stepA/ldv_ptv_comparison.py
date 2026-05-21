# %%
"""LDV–PTV pressure comparison plot.

Compares LDV refracto-vibrometry and particle tracking velocimetry
pressure estimates at matched drive voltages (5, 10, 15 Vpp).

Usage:
    python ldv_ptv_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    RSSI_THRESHOLD,
    channel_center_func,
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
# Uncertainty budget — multiplicative on p_0.
#
# The figure plots a single symmetric ±1σ envelope per method.  The
# authoritative numerical reference for the LDV/PTV gap analysis is
# ``reports/2026-05-21_ldv_ptv_uncertainty_budget.md``.  That report
# separates the LDV side into:
#   (a) a *known one-sided bias* for glass photoelastic only
#       (F_bias = 1.145; central ratio compares against 1.145, not 1.0),
#   (b) a symmetric uncertainty σ_L = 0.103 covering dn/dp, channel
#       height, and the velocity scale.
#
# Air-null structural residual: previously included in the LDV envelope.
# Dropped 2026-05-21 (third revision) after empirical R² = −4.56 on the
# W21 ``sample_wide_20V_AIR`` mode-fit confirmed that the air-filled
# signal lacks the |sin(πy/W)| mode shape and is therefore filtered out
# by the mode-fit p_0 extraction.  Any residual is captured by
# noise_rms_pressure in the LDV stat error, not as a multiplicative
# systematic.  See report §2.2 and §8 limit 3.
#
# This script uses a *simpler* "everything bundled into one symmetric
# multiplicative band" view that is easier to read in a single figure
# but does not separate biases from symmetric uncertainties.  Numerical
# values differ slightly from the report; the report is authoritative
# for the σ-equivalent of the gap.
#
# Per-source σ_i:
#   glass photoelastic (14.6 % central)  reports/2026-05-21_glass_pressure_self_verification.md
#   dn/dp_water (1.48e-10, ±5 %)         src/ldv_analysis/config.py + IAPWS-95
#   channel_height (150 µm, ±7.5 %)      Si wet-etch manufacturer spec
#   velocity scale (±5 %)                Polytec decoder vendor spec
#   κ_PS (2.49e-10 → 3.30e-10, ±14 %)    Settnes-Bruus 2012 vs Barnkob 2010
#                                        (DISCRETE literature choice, not Gaussian;
#                                        treated as σ here for envelope purposes only)
#   particle radius (2.5 µm, ±10 %)      manufacturer + sensitivity-sweep
#                                        verification (σ_radius corrected
#                                        from 0.15 to 0.10 on 2026-05-21
#                                        after sensitivity sweep showed
#                                        p_0 ∝ 1/R, not R³)
# =============================================================================

LDV_SYS_CONTRIB = {
    "dn_dp_water":          0.05,
    "channel_height":       0.075,
    "glass_photoelastic":   0.145,
    "velocity_scale":       0.05,
}
LDV_SYS_FRAC = float(np.sqrt(sum(v * v for v in LDV_SYS_CONTRIB.values())))
# ≈ 0.179 → ±18 % combined LDV envelope (everything bundled — figure-only
# view; report's bias + σ_L decomposition gives a different breakdown)

PTV_SYS_CONTRIB = {
    "kappa_p":              0.14,
    "radius":               0.10,   # corrected 2026-05-21: p_0 ∝ 1/R (see report §3.2)
    "fluid_water":          0.01,
    "wall_streaming":       0.075,
}
PTV_SYS_FRAC = float(np.sqrt(sum(v * v for v in PTV_SYS_CONTRIB.values())))
# ≈ 0.188 → ±19 % combined PTV envelope


def _ldv_stat_err(cache, valid_mask, p0_peak):
    """Random uncertainty on the LDV peak p_0 (Pa).

    Two contributions combined in quadrature:
    (1) per-spatial-point ``noise_rms_pressure`` (already in the FFT cache)
        averaged across the channel, divided by sqrt(N_valid) — the standard
        error of the mode amplitude from random measurement noise;
    (2) mode-fit residual: 5 % of p_0_peak (conservative — ``ModeFitResult``
        carries no covariance, and the per-scan R² is consistently >0.9).
    Falls back to a 2 % floor of p_0_peak when the cache is continuous-mode
    (noise_rms is NaN).
    """
    noise_pa = cache["noise_rms_pressure"]
    if np.all(np.isnan(noise_pa)):
        stat_noise = 0.02 * p0_peak
    else:
        stat_noise = float(np.nanmean(noise_pa)) / np.sqrt(int(valid_mask.sum()))
    fit_residual = 0.05 * p0_peak
    return float(np.sqrt(stat_noise ** 2 + fit_residual ** 2))


def _ptv_stat_err(csv_df):
    """Random uncertainty on the PTV peak p_0 (Pa) — ``p0_err`` of the bin
    with peak p_0 in ``fitting_A_per_x.csv`` (curve_fit covariance)."""
    i_peak = csv_df["p0"].idxmax()
    return float(csv_df.loc[i_peak, "p0_err"])


# %%
# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LDV_DATA_DIR = get_data_dir("20260307experimentB")
PTV_OUTPUT_DIR = Path("C:/Users/tatsu/Documents/particle-tracking/output")

VOLTAGE_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms", 5, "5Vpp"),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10, "10Vpp"),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15, "15Vpp"),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20, None),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25, None),
]

MARKER_SIZE = 4

# %%
# =============================================================================
# Load LDV data
# =============================================================================

geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
center_fn = channel_center_func(geom)
hw = CHANNEL_WIDTH / 2

vpps = []
ldv_p0 = []
ldv_stat_err = []

for fname, vpp, _ in VOLTAGE_FILES:
    cache = load_or_compute(LDV_DATA_DIR / fname, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    valid = make_valid_mask(V, rssi)
    if "pt_burst_on_us" in cache:
        valid &= make_burst_timing_mask(cache["pt_burst_on_us"],
                                        cache["pt_burst_off_us"])

    pos_x_c = pos_x - center_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(
        pos_x_c, pos_y, int(cache["n_x_meta"]), int(cache["n_y_meta"]),
        CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    prs = cache["pressure_1f"].copy()
    prs[~valid] = np.nan
    p0_y = fit_columns(cg.to_grid(prs), cg.width_grid, CHANNEL_WIDTH)
    p0_peak = float(np.nanmax(p0_y))
    stat_err = _ldv_stat_err(cache, valid, p0_peak)

    vpps.append(vpp)
    ldv_p0.append(p0_peak)
    ldv_stat_err.append(stat_err)
    print(f"  LDV {vpp:2d} Vpp: p0 = {p0_peak/1e3:.0f} kPa "
          f"(stat ±{stat_err/1e3:.1f}, sys ±{LDV_SYS_FRAC*p0_peak/1e3:.0f})")

vpps = np.array(vpps)
ldv_p0 = np.array(ldv_p0)
ldv_stat_err = np.array(ldv_stat_err)

# %%
# =============================================================================
# Load PTV data
# =============================================================================

ptv_p0 = []
ptv_stat_err = []

for _, vpp, ptv_dir in VOLTAGE_FILES:
    if ptv_dir is None:
        ptv_p0.append(np.nan)
        ptv_stat_err.append(np.nan)
        continue
    csv_path = PTV_OUTPUT_DIR / ptv_dir / "06_fitting" / "fitting_A_per_x.csv"
    if not csv_path.exists():
        print(f"  PTV {vpp:2d} Vpp: CSV not found ({csv_path})")
        ptv_p0.append(np.nan)
        ptv_stat_err.append(np.nan)
        continue
    df = pd.read_csv(csv_path)
    p0_peak = float(df["p0"].max())
    stat_err = _ptv_stat_err(df)
    ptv_p0.append(p0_peak)
    ptv_stat_err.append(stat_err)
    print(f"  PTV {vpp:2d} Vpp: p0 = {p0_peak/1e3:.0f} kPa "
          f"(stat ±{stat_err/1e3:.1f}, sys ±{PTV_SYS_FRAC*p0_peak/1e3:.0f})")

ptv_p0 = np.array(ptv_p0)
ptv_stat_err = np.array(ptv_stat_err)

# %%
# =============================================================================
# PTV → physical coordinate mapping (calibrated from PZT edge positions)
# =============================================================================
# See calibration/ptv_stage_alignment/README.md for derivation.
#
# Calibration points:
#   PZT inlet edge:  physical x = 5.6 mm,  PTV pixel 1289, stage = 2.409 mm
#   PZT outlet edge: physical x = 11.6 mm, PTV pixel 1070, stage = -3.494 mm
#
# Model: physical_x_mm = -stage_x_mm - pixel * 0.00065 + offset
# Offset: 8.847 (inlet), 8.802 (outlet), average 8.825 mm
#
# PTV data capture: stage_x = -0.7699 mm
# → physical_x_mm = 9.595 - X_um / 1000

_PTV_STAGE_X = -0.7699    # mm (from nd2 metadata)
_PTV_PIXEL_SIZE = 0.65e-3  # mm/px
_PTV_CAL_OFFSET = 8.825    # mm (average of inlet/outlet calibration)

ptv_files = [(f, v, d) for f, v, d in VOLTAGE_FILES if d is not None]


def ptv_x_to_mm(ptv_x_um):
    """Convert PTV X column (um, image centroid) to physical channel-length coordinate (mm)."""
    return -_PTV_STAGE_X - ptv_x_um / 1e3 + _PTV_CAL_OFFSET

print(f"\n  PTV coordinate mapping: physical_x = {-_PTV_STAGE_X + _PTV_CAL_OFFSET:.3f} - X_um/1000")
print(f"  PTV X range -> physical x: "
      f"X=0 -> {ptv_x_to_mm(0):.2f} mm, X=1800 -> {ptv_x_to_mm(1800):.2f} mm")

# %%
# =============================================================================
# Comparison table
# =============================================================================

ratio = ldv_p0 / ptv_p0
ptv_valid = ~np.isnan(ptv_p0)
print(f"\n{'Vpp':>5} {'LDV (kPa)':>10} {'PTV (kPa)':>10} {'Ratio':>8}")
print("-" * 38)
for i in range(len(vpps)):
    ptv_str = f"{ptv_p0[i]/1e3:10.0f}" if ptv_valid[i] else "       N/A"
    r_str = f"{ratio[i]:8.2f}" if ptv_valid[i] else "     N/A"
    print(f"{vpps[i]:5d} {ldv_p0[i]/1e3:10.0f} {ptv_str} {r_str}")
print(f"{'Mean':>5} {'':>10} {'':>10} {np.nanmean(ratio):8.2f}")

# %%
# =============================================================================
# Plot 1: p0 vs voltage (both methods)
# =============================================================================

V_fine = np.linspace(0, vpps.max() * 1.15, 100)

# Linear fits through origin (math unchanged)
a_ldv = float(np.sum(vpps * ldv_p0) / np.sum(vpps**2))
ptv_vpps = vpps[ptv_valid]
ptv_p0_valid = ptv_p0[ptv_valid]
ptv_stat_err_valid = ptv_stat_err[ptv_valid]
a_ptv = float(np.sum(ptv_vpps * ptv_p0_valid) / np.sum(ptv_vpps**2))

# Combined (stat + sys in quadrature) error bar per point
ldv_total_err = np.sqrt(ldv_stat_err ** 2 + (LDV_SYS_FRAC * ldv_p0) ** 2)
ptv_total_err = np.sqrt(ptv_stat_err_valid ** 2 + (PTV_SYS_FRAC * ptv_p0_valid) ** 2)

fig, ax = plt.subplots(figsize=figsize_for_layout())

# Systematic envelope cone around each linear fit
ax.fill_between(V_fine,
                a_ldv * V_fine * (1 - LDV_SYS_FRAC) / 1e3,
                a_ldv * V_fine * (1 + LDV_SYS_FRAC) / 1e3,
                alpha=0.12, color="tab:blue", linewidth=0)
ax.fill_between(V_fine,
                a_ptv * V_fine * (1 - PTV_SYS_FRAC) / 1e3,
                a_ptv * V_fine * (1 + PTV_SYS_FRAC) / 1e3,
                alpha=0.12, color="tab:red", linewidth=0)

# Linear-fit lines
ax.plot(V_fine, a_ldv * V_fine / 1e3, ":", linewidth=0.7, color="tab:blue")
ax.plot(V_fine, a_ptv * V_fine / 1e3, ":", linewidth=0.7, color="tab:red")

# Outer (thin, no caps): total uncertainty = stat ⊕ sys
ax.errorbar(vpps, ldv_p0 / 1e3, yerr=ldv_total_err / 1e3,
            fmt="none", elinewidth=0.5, color="tab:blue",
            alpha=0.5, capsize=0)
ax.errorbar(ptv_vpps, ptv_p0_valid / 1e3, yerr=ptv_total_err / 1e3,
            fmt="none", elinewidth=0.5, color="tab:red",
            alpha=0.5, capsize=0)
# Inner (thick, capped): random / stat only
ax.errorbar(vpps, ldv_p0 / 1e3, yerr=ldv_stat_err / 1e3,
            fmt="o", markersize=MARKER_SIZE, color="tab:blue",
            capsize=3, capthick=0.8, elinewidth=1.0,
            label=rf"LDV {a_ldv/1e3:.1f} kPa/V (±{LDV_SYS_FRAC*100:.0f}% sys)")
ax.errorbar(ptv_vpps, ptv_p0_valid / 1e3, yerr=ptv_stat_err_valid / 1e3,
            fmt="s", markersize=MARKER_SIZE, color="tab:red",
            capsize=3, capthick=0.8, elinewidth=1.0,
            label=rf"PTV {a_ptv/1e3:.1f} kPa/V (±{PTV_SYS_FRAC*100:.0f}% sys)")

ax.set_xlabel(r"Drive voltage $V_\mathrm{drive}$ [$V_\mathrm{pp}$]")
ax.set_ylabel(r"$P_{1f}$ [kPa]")
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
out_path = OUT_DIR / "ldv_ptv_p0_vs_voltage.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 1b: LDV/PTV ratio with combined ±1σ band
# =============================================================================
# Visualises whether the observed ~1.7-1.9× gap survives the combined LDV+PTV
# systematic budget — if the observed point sits OUTSIDE the shaded band, the
# gap is statistically significant under the quoted uncertainties.

band_lo = 1.0 / ((1.0 + LDV_SYS_FRAC) * (1.0 + PTV_SYS_FRAC))
band_hi = (1.0 + LDV_SYS_FRAC) * (1.0 + PTV_SYS_FRAC)

ratio_obs = ldv_p0[ptv_valid] / ptv_p0_valid
# Propagate stat uncertainty to the ratio:  σ_r/r = √((σ_L/L)² + (σ_P/P)²)
ratio_stat_err = ratio_obs * np.sqrt(
    (ldv_stat_err[ptv_valid] / ldv_p0[ptv_valid]) ** 2
    + (ptv_stat_err_valid / ptv_p0_valid) ** 2
)

fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.axhspan(band_lo, band_hi, alpha=0.15, color="0.5",
           label=rf"combined 1$\sigma$ band [{band_lo:.2f}, {band_hi:.2f}]")
ax.axhline(1.0, color="k", linestyle="--", linewidth=0.5,
           label="1:1 agreement")
ax.errorbar(vpps[ptv_valid], ratio_obs, yerr=ratio_stat_err,
            fmt="o", markersize=MARKER_SIZE + 1, color="black",
            capsize=3, capthick=0.8, elinewidth=1.0, label="observed")
ax.set_xlabel(r"Drive voltage $V_\mathrm{drive}$ [$V_\mathrm{pp}$]")
ax.set_ylabel(r"LDV $P_{1f}$ / PTV $P_{1f}$")
ax.set_xlim(left=0)
ax.set_ylim(0.5, 2.5)
ax.legend(frameon=False, fontsize=6, loc="lower right")
plt.tight_layout()
out_path = OUT_DIR / "ldv_ptv_ratio.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 2: LDV vs PTV scatter (1:1 comparison)
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
p_max = max(ldv_p0.max(), ptv_p0.max()) / 1e3 * 1.1

ax.plot([0, p_max], [0, p_max], "k:", linewidth=0.5, label="1:1")
ax.plot([0, p_max], [0, p_max / np.mean(ratio)], ":", linewidth=0.5,
        color="tab:gray",
        label=f"LDV/PTV = {np.mean(ratio):.2f}")

for i, vpp in enumerate(vpps):
    ax.plot(ptv_p0[i] / 1e3, ldv_p0[i] / 1e3, "o", markersize=MARKER_SIZE + 2,
            color=f"C{i}")
    ax.annotate(f"{vpp} Vpp", (ptv_p0[i] / 1e3, ldv_p0[i] / 1e3),
                fontsize=6, textcoords="offset points", xytext=(5, -5))

ax.set_xlabel(r"PTV $P_{1f}$ [kPa]")
ax.set_ylabel(r"LDV $P_{1f}$ [kPa]")
ax.legend(frameon=False, fontsize=6)
ax.set_xlim(0, p_max)
ax.set_ylim(0, p_max)
ax.set_aspect("equal")
plt.tight_layout()
out_path = OUT_DIR / "ldv_ptv_scatter.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 3: Axial profile comparison (p0 vs x for each voltage)
# =============================================================================

fig, axes = plt.subplots(1, len(ptv_files), figsize=figsize_for_layout(1, len(ptv_files)),
                          sharey=True)
if len(ptv_files) == 1:
    axes = [axes]

for i, (fname, vpp, ptv_dir) in enumerate(ptv_files):
    ax = axes[i]

    # LDV axial profile
    cache = load_or_compute(LDV_DATA_DIR / fname, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None
    valid = make_valid_mask(V, rssi)
    if "pt_burst_on_us" in cache:
        valid &= make_burst_timing_mask(cache["pt_burst_on_us"],
                                        cache["pt_burst_off_us"])
    pos_x_c = pos_x - center_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(
        pos_x_c, pos_y, int(cache["n_x_meta"]), int(cache["n_y_meta"]),
        CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )
    prs = cache["pressure_1f"].copy()
    prs[~valid] = np.nan
    p0_y = fit_columns(cg.to_grid(prs), cg.width_grid, CHANNEL_WIDTH)
    ldv_x_mm = cg.length_grid * 1e3
    ldv_p0_kpa = p0_y / 1e3

    # PTV axial profile — use shared x alignment
    csv_path = PTV_OUTPUT_DIR / ptv_dir / "06_fitting" / "fitting_A_per_x.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        ax.errorbar(ptv_x_to_mm(df["x"].values), df["p0_kPa"].values,
                     yerr=df["p0_err"].values / 1e3,
                     fmt="s", markersize=2, color="tab:red",
                     capsize=2, capthick=0.3, elinewidth=0.3, label="PTV")

    ax.plot(ldv_x_mm, ldv_p0_kpa, "-", linewidth=0.75,
            color="tab:blue", label="LDV")

    # Clip x range to PTV extent with margin
    if csv_path.exists():
        ptv_x_mm = ptv_x_to_mm(df["x"].values)
        ptv_span = ptv_x_mm.max() - ptv_x_mm.min()
        margin = ptv_span * 0.1
        ax.set_xlim(ptv_x_mm.min() - margin, ptv_x_mm.max() + margin)

    ax.set_xlabel("$x$ [mm]")
    ax.set_title(f"{vpp} Vpp")
    if i == 0:
        ax.set_ylabel(r"$P_{1f}$ [kPa]")

axes[0].legend(frameon=False, fontsize=5)
plt.tight_layout()
out_path = OUT_DIR / "ldv_ptv_axial_profile.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 4: E_ac 2D maps — LDV (sum of harmonics) vs PTV (Gor'kov)
# =============================================================================

from ldv_analysis.config import C_SOUND, RHO

# PTV material parameters (must match 06_fitting.py)
_PTV_RHO_F = 1.003586e3
_PTV_C_F = 1.50757e3
_PTV_W = 375e-6
_PTV_N_X_BINS = 20
_PTV_N_Y_BINS = 20

for fname, vpp, ptv_dir in ptv_files:
    # --- LDV E_ac map ---
    cache = load_or_compute(LDV_DATA_DIR / fname, CACHE_DIR)
    pos_x = cache["pos_x"]; pos_y = cache["pos_y"]
    V = cache["voltage_1f"]; rssi = cache["rssi"] if "rssi" in cache else None
    valid = make_valid_mask(V, rssi)
    if "pt_burst_on_us" in cache:
        valid &= make_burst_timing_mask(cache["pt_burst_on_us"],
                                        cache["pt_burst_off_us"])
    pos_x_c = pos_x - center_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(pos_x_c, pos_y, int(cache["n_x_meta"]),
        int(cache["n_y_meta"]), CHANNEL_WIDTH,
        pos_x.max() - pos_x.min(), inside,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD)

    eac_grid = np.zeros_like(cg.to_grid(cache["pressure_1f"]))
    for h in [1, 2, 3]:
        prs = cache[f"pressure_{h}f"].copy()
        prs[~valid] = np.nan
        g = cg.to_grid(prs)
        eac_grid += g**2
    eac_grid /= (4 * RHO * C_SOUND**2)

    ldv_w_mm = cg.width_grid * 1e3
    ldv_l_mm = cg.length_grid * 1e3
    # --- PTV E_pot map with per-x-bin zero-crossing y0 ---
    csv_path = PTV_OUTPUT_DIR / ptv_dir / "06_fitting" / "fitting_A_per_x.csv"
    df_ptv = pd.read_csv(csv_path)
    ptv_x_um = df_ptv["x"].values
    ptv_p0 = df_ptv["p0"].values  # Pa

    ptv_x_mm = ptv_x_to_mm(ptv_x_um)

    # Load raw tracking data
    raw_csv = PTV_OUTPUT_DIR / ptv_dir / "summary" / "data_combine_filter.csv"
    df_raw = pd.read_csv(raw_csv, usecols=["X", "Y", "DY"])
    _L = 375.0  # channel width in um

    # X bins matching the fitting script
    x_edges = np.linspace(df_raw["X"].min(), df_raw["X"].max(), _PTV_N_X_BINS + 1)

    # Y grid for the 2D map
    ptv_y_edges = np.linspace(df_raw["Y"].min(), df_raw["Y"].max(), _PTV_N_Y_BINS + 1)
    ptv_y_centers = (ptv_y_edges[:-1] + ptv_y_edges[1:]) / 2

    # Per-x-bin: estimate y0 from DY zero-crossing, compute E_pot column
    k_ptv = np.pi / _PTV_W
    ptv_eac = np.full((_PTV_N_Y_BINS, _PTV_N_X_BINS), np.nan)

    for xi in range(_PTV_N_X_BINS):
        x_mask = (df_raw["X"] >= x_edges[xi]) & (df_raw["X"] < x_edges[xi + 1])
        subset = df_raw[x_mask]
        if len(subset) < 20:
            continue

        # Zero-crossing of mean DY vs Y for this x bin
        y_bin_edges = np.linspace(subset["Y"].min(), subset["Y"].max(), 21)
        y_bin_centers = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2
        mean_dy = np.array([
            subset.loc[(subset["Y"] >= y_bin_edges[j]) &
                        (subset["Y"] < y_bin_edges[j + 1]), "DY"].mean()
            for j in range(len(y_bin_edges) - 1)
        ])
        valid_dy = ~np.isnan(mean_dy)
        if valid_dy.sum() < 3:
            continue

        # Find zero crossing
        dy_v = mean_dy[valid_dy]
        y_v = y_bin_centers[valid_dy]
        sign_changes = np.where(np.diff(np.sign(dy_v)))[0]
        if len(sign_changes) > 0:
            sc = sign_changes[0]
            y0_xi = y_v[sc] - dy_v[sc] * (y_v[sc + 1] - y_v[sc]) / (dy_v[sc + 1] - dy_v[sc])
        else:
            y0_xi = _L / 2  # fallback

        # E_pot at each y bin
        y_c_m = (ptv_y_centers - y0_xi) * 1e-6
        sin2 = np.sin(k_ptv * y_c_m)**2
        ptv_eac[:, xi] = ptv_p0[xi]**2 * sin2 / (4 * _PTV_RHO_F * _PTV_C_F**2)

    # PTV y in mm, centered on midpoint of Y range
    y0_display = (df_raw["Y"].min() + df_raw["Y"].max()) / 2
    ptv_y_mm = (ptv_y_centers - y0_display) / 1e3

    # LDV: full range; PTV overlaid on top
    eac_clipped = eac_grid
    ldv_l_clipped = ldv_l_mm

    # --- Independent color scales from mode-fit peak ---
    # Peak E_ac from mode-fit p0 (not per-pixel max)
    p0_y_ldv = fit_columns(cg.to_grid(cache["pressure_1f"].copy()), cg.width_grid, CHANNEL_WIDTH)
    ldv_peak = float(np.nanmax(p0_y_ldv))**2 / (4 * RHO * C_SOUND**2)
    ptv_peak = float(np.max(ptv_p0))**2 / (4 * _PTV_RHO_F * _PTV_C_F**2)

    # --- Plot side by side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2))

    im1 = ax1.pcolormesh(ldv_l_clipped, ldv_w_mm, eac_clipped,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ldv_peak)
    ax1.set_xlabel("$x$ [mm]")
    ax1.set_ylabel("$y$ [mm]")
    ax1.set_title(f"LDV $E_{{ac}}$ (peak {ldv_peak:.0f} J/m$^3$)")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1, label=r"$E_\mathrm{ac}$ [J/m$^3$]")

    im2 = ax2.pcolormesh(ptv_x_mm, ptv_y_mm, ptv_eac,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ptv_peak)
    ax2.set_ylim(ldv_w_mm[0], ldv_w_mm[-1])
    ax2.set_xlabel("$x$ [mm]")
    ax2.set_ylabel("$y$ [mm]")
    ax2.set_title(f"PTV $E_{{pot}}$ (peak {ptv_peak:.0f} J/m$^3$)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2, label=r"$E_\mathrm{pot}$ [J/m$^3$]")

    fig.suptitle(f"{vpp} Vpp, 1.907 MHz", fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / f"ldv_ptv_eac_map_{vpp}Vpp.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

    # --- Reconstructed LDV E_pot map (mode-fit p0 × sin²) vs PTV ---
    # LDV: E_pot(y, x) = p0_1f(x)² × sin²(πy/W) / (4ρc²)
    k_ldv = np.pi / CHANNEL_WIDTH
    sin2_ldv = np.sin(k_ldv * cg.width_grid)**2  # (n_width,)
    ldv_epot_recon = p0_y_ldv[None, :]**2 * sin2_ldv[:, None] / (4 * RHO * C_SOUND**2)
    ldv_epot_clipped = ldv_epot_recon

    ldv_recon_vmax = ldv_peak  # p0_max² / (4ρc²)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2))

    im1 = ax1.pcolormesh(ldv_l_clipped, ldv_w_mm, ldv_epot_clipped,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ldv_recon_vmax)
    ax1.set_xlabel("$x$ [mm]")
    ax1.set_ylabel("$y$ [mm]")
    ax1.set_title(f"LDV $E_{{pot}}$ (peak {ldv_peak:.0f} J/m$^3$)")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1, label=r"$E_\mathrm{pot}$ [J/m$^3$]")

    im2 = ax2.pcolormesh(ptv_x_mm, ptv_y_mm, ptv_eac,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ptv_peak)
    ax2.set_ylim(ldv_w_mm[0], ldv_w_mm[-1])
    ax2.set_xlabel("$x$ [mm]")
    ax2.set_ylabel("$y$ [mm]")
    ax2.set_title(f"PTV $E_{{pot}}$ (peak {ptv_peak:.0f} J/m$^3$)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2, label=r"$E_\mathrm{pot}$ [J/m$^3$]")

    fig.suptitle(f"{vpp} Vpp, 1.907 MHz (reconstructed)", fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / f"ldv_ptv_epot_recon_{vpp}Vpp.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

    # --- Pressure: PTV |p| derived from ptv_eac (reuses per-x-bin y0) ---
    ptv_p_grid = np.sqrt(ptv_eac * 4 * _PTV_RHO_F * _PTV_C_F**2)
    ldv_p_peak = float(np.nanmax(p0_y_ldv)) / 1e3  # kPa, mode-fit peak
    ptv_p_peak = float(np.max(ptv_p0)) / 1e3  # kPa, per-x-bin fit peak

    # --- Raw pressure map: LDV |p_1f(x,y)| from cache vs PTV |p| ---
    ldv_p1f_grid = np.abs(cg.to_grid(cache["pressure_1f"].copy()))
    ldv_p1f_grid[np.isnan(eac_grid)] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2))

    im1 = ax1.pcolormesh(ldv_l_clipped, ldv_w_mm, ldv_p1f_grid / 1e3,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ldv_p_peak)
    ax1.set_xlabel("$x$ [mm]")
    ax1.set_ylabel("$y$ [mm]")
    ax1.set_title(f"LDV $|p_{{1f}}|$ (peak {ldv_p_peak:.0f} kPa)")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1, label=r"$|p_{1f}|$ [kPa]")

    im2 = ax2.pcolormesh(ptv_x_mm, ptv_y_mm, ptv_p_grid / 1e3,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ptv_p_peak)
    ax2.set_ylim(ldv_w_mm[0], ldv_w_mm[-1])
    ax2.set_xlabel("$x$ [mm]")
    ax2.set_ylabel("$y$ [mm]")
    ax2.set_title(f"PTV $|p_0|$ (peak {ptv_p_peak:.0f} kPa)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2, label=r"$|p_0|$ [kPa]")

    fig.suptitle(f"{vpp} Vpp, 1.907 MHz (pressure)", fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / f"ldv_ptv_pressure_map_{vpp}Vpp.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

    # --- Reconstructed pressure map: LDV p0(x) × |sin(πy/W)| vs PTV ---
    ldv_p_recon = np.abs(p0_y_ldv[None, :]) * np.abs(np.sin(k_ldv * cg.width_grid)[:, None])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_for_layout(1, 2))

    im1 = ax1.pcolormesh(ldv_l_clipped, ldv_w_mm, ldv_p_recon / 1e3,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ldv_p_peak)
    ax1.set_xlabel("$x$ [mm]")
    ax1.set_ylabel("$y$ [mm]")
    ax1.set_title(f"LDV $|p_{{1f}}|$ (peak {ldv_p_peak:.0f} kPa)")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1, label=r"$|p_{1f}|$ [kPa]")

    im2 = ax2.pcolormesh(ptv_x_mm, ptv_y_mm, ptv_p_grid / 1e3,
                          shading="nearest", cmap="viridis", vmin=0, vmax=ptv_p_peak)
    ax2.set_ylim(ldv_w_mm[0], ldv_w_mm[-1])
    ax2.set_xlabel("$x$ [mm]")
    ax2.set_ylabel("$y$ [mm]")
    ax2.set_title(f"PTV $|p_0|$ (peak {ptv_p_peak:.0f} kPa)")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2, label=r"$|p_0|$ [kPa]")

    fig.suptitle(f"{vpp} Vpp, 1.907 MHz (pressure, reconstructed)", fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / f"ldv_ptv_pressure_recon_{vpp}Vpp.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

print("\n=== Done ===")
