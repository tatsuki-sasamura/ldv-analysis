# %%
"""Harmonic pressure amplitudes (1f, 2f, 3f) vs drive voltage.

Extracts P_{1f}, P_{2f}, P_{3f} at the axial antinode via mode-shape
fit with sigma clipping from 2D scan data at multiple voltages.

Fits: P_1f = a*V (linear), P_2f = b*V^2 (quadratic), P_3f = c*V^3 (cubic).

Usage:
    python harmonics_vs_voltage.py <tdms_1> <vpp_1> [<tdms_2> <vpp_2> ...]
    python harmonics_vs_voltage.py --data-dir <dir> --pattern "W16test3_{vpp}Vpp*.tdms" --vpps 10 20 30 40

Examples:
    python harmonics_vs_voltage.py G:/data/10Vpp.tdms 10 G:/data/20Vpp.tdms 20
    python harmonics_vs_voltage.py --data-dir "G:/My Drive/260413_ldv" \\
        --pattern "W16test3_{vpp}Vpp1904kHz_5m_s_max.tdms" --vpps 10 20 30 40
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    C_SOUND,
    FIG_DPI,
    RHO,
    RSSI_THRESHOLD,
    channel_center_func,
    figsize_for_layout,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns, fit_mode

# %%
# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("pairs", nargs="*", default=[],
                   help="Alternating: tdms_path vpp tdms_path vpp ...")
group.add_argument("--data-dir", type=Path,
                   help="Directory containing TDMS files")
parser.add_argument("--pattern", type=str,
                    help="Filename pattern with {vpp} placeholder (used with --data-dir)")
parser.add_argument("--vpps", type=int, nargs="+",
                    help="Voltage values in Vpp (used with --data-dir)")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for geometry lookup (auto-detected if omitted)")
parser.add_argument("--label", type=str, default=None,
                    help="Label for output filenames")
args = parser.parse_args()

# Build file list
files = []  # list of (Path, vpp)
if args.data_dir:
    if not args.pattern or not args.vpps:
        parser.error("--data-dir requires --pattern and --vpps")
    for vpp in sorted(args.vpps):
        fname = args.pattern.replace("{vpp}", str(vpp))
        files.append((args.data_dir / fname, vpp))
else:
    pairs = args.pairs
    if len(pairs) % 2 != 0:
        parser.error("Positional args must be pairs: tdms_path vpp tdms_path vpp ...")
    for i in range(0, len(pairs), 2):
        files.append((Path(pairs[i]), int(pairs[i + 1])))

# Infer dataset from first file
dataset = args.dataset or files[0][0].parent.name
label = args.label or dataset

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MARKER_SIZE = 4
SIGMA_CLIP = 3.0

print(f"Dataset: {dataset}")
print(f"Files: {len(files)}")
for p, v in files:
    print(f"  {v:3d} Vpp: {p.name}")

# %%
# =============================================================================
# Process each voltage level
# =============================================================================

geom = load_channel_geometry(dataset, CACHE_DIR)
center_fn = channel_center_func(geom)
hw = CHANNEL_WIDTH / 2

vpps = []
p0_1f_arr = []
p0_2f_arr = []
p0_3f_arr = []
sig_1f_arr = []
sig_2f_arr = []
sig_3f_arr = []
phase_1f_arr = []
phase_2f_arr = []
phase_3f_arr = []
phase_vi_arr = []
mode_data = []  # for mode-shape plots

for tdms_path, vpp in files:
    if not tdms_path.exists():
        print(f"  SKIP (not found): {tdms_path.name}")
        continue

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
    pos_x_c = pos_x - center_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(pos_x_c, pos_y, n_x, n_y,
                           CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
                           rssi=rssi, rssi_threshold=RSSI_THRESHOLD)

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

    # Complex mode-shape fit at antinode column for phase extraction
    col_w = cg.width_grid  # centered width positions
    md = {"vpp": vpp, "width_grid": col_w}
    for h in [1, 2, 3]:
        prs_amp = cache[f"pressure_{h}f"].copy()
        prs_phase = cache[f"phase_{h}f"].copy()
        prs_amp[~valid] = np.nan
        prs_phase[~valid] = np.nan
        grid_amp = cg.to_grid(prs_amp)
        grid_ph = cg.to_grid(prs_phase)
        col_amp = grid_amp[:, j_anti]
        col_ph = grid_ph[:, j_anti]
        ok = ~np.isnan(col_amp) & ~np.isnan(col_ph)
        if ok.sum() > 3:
            p_complex = col_amp[ok] * np.exp(1j * np.radians(col_ph[ok]))
            res = fit_mode(col_w[ok], p_complex, CHANNEL_WIDTH, h,
                           center=0.0, sigma_clip=SIGMA_CLIP)
            md[f"p0_{h}f"] = res.p0
            md[f"r2_{h}f"] = res.r2
            md[f"data_{h}f"] = (col_w[ok], col_amp[ok], col_ph[ok])
        else:
            md[f"p0_{h}f"] = 0j
            md[f"r2_{h}f"] = 0.0
            md[f"data_{h}f"] = (np.array([]), np.array([]), np.array([]))

    phase_1f_arr.append(np.degrees(np.angle(md["p0_1f"])))
    phase_2f_arr.append(np.degrees(np.angle(md["p0_2f"])))
    phase_3f_arr.append(np.degrees(np.angle(md["p0_3f"])))
    if "phase_vi" in cache:
        phase_vi_arr.append(float(np.median(cache["phase_vi"][valid])))
    else:
        phase_vi_arr.append(np.nan)
    mode_data.append(md)

    print(f"  {vpp:2d} Vpp: P_1f={p0_1f_arr[-1]/1e3:.0f}, "
          f"P_2f={p0_2f_arr[-1]/1e3:.0f}, "
          f"P_3f={p0_3f_arr[-1]/1e3:.1f} kPa  "
          f"ph_1f={phase_1f_arr[-1]:+.1f} ph_2f={phase_2f_arr[-1]:+.1f} "
          f"ph_vi={phase_vi_arr[-1]:+.1f} deg")

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
# Plot: Pressure vs voltage (log-log)
# =============================================================================

plt.style.use(["science", "ieee"])
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
out_path = OUT_DIR / f"harmonics_vs_voltage_{label}.png"
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
out_path = OUT_DIR / f"harmonics_vs_voltage_ratio_{label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Mach number variants (x-axis = M = P_1f / (rho * c^2))
# =============================================================================

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
p1f_fine = M_fine * RHO * C_SOUND**2  # Pa
ax.plot(M_fine, b_2f / a_1f**2 * p1f_fine**2 / 1e6,
        ":", linewidth=0.5, color="tab:red", label=r"$P_{2f}$")

ax.errorbar(M, p0_3f / 1e6, yerr=sig_3f / 1e6,
            fmt="^", markersize=MARKER_SIZE, color="tab:green",
            capsize=3, capthick=0.5, elinewidth=0.5)
ax.plot(M_fine, c_3f / a_1f**3 * p1f_fine**3 / 1e6,
        ":", linewidth=0.5, color="tab:green", label=r"$P_{3f}$")

ax.set_xlabel(r"Mach number $M = P_{1f}/(\rho c^2)$")
ax.set_ylabel(r"Pressure amplitude [MPa]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(M.min() * 0.8, M.max() * 1.2)
ax.legend(frameon=False, fontsize=6)
plt.tight_layout()
out_path = OUT_DIR / f"harmonics_vs_mach_{label}.png"
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
out_path = OUT_DIR / f"harmonics_vs_mach_ratio_{label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Phase vs voltage (from complex mode-shape fit)
# =============================================================================

phase_1f = np.unwrap(np.radians(phase_1f_arr))
phase_2f = np.unwrap(np.radians(phase_2f_arr))
phase_3f = np.unwrap(np.radians(phase_3f_arr))
phase_1f = np.degrees(phase_1f)
phase_2f = np.degrees(phase_2f)
phase_3f = np.degrees(phase_3f)
phase_vi = np.array(phase_vi_arr)

fig, ax = plt.subplots(figsize=figsize_for_layout())
ax.plot(vpps, phase_1f, "o-", markersize=MARKER_SIZE, color="tab:blue", label=r"Acoustic $1f$")
ax.plot(vpps, phase_2f, "s-", markersize=MARKER_SIZE, color="tab:red", label=r"Acoustic $2f$")
ax.plot(vpps, phase_3f, "^-", markersize=MARKER_SIZE, color="tab:green", label=r"Acoustic $3f$")
if not np.all(np.isnan(phase_vi)):
    ax.plot(vpps, phase_vi, "d-", markersize=MARKER_SIZE, color="tab:gray", label=r"V--I phase")
ax.set_xlabel(r"Drive voltage [$V_\mathrm{pp}$]")
ax.set_ylabel(r"Phase [deg]")
ax.legend(frameon=False, fontsize=6)
plt.tight_layout()
out_path = OUT_DIR / f"phase_vs_voltage_{label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Mode-shape fit plots: amplitude + phase for 1f, 2f, 3f at each voltage
# =============================================================================

from ldv_analysis.mode_fit import _mode_shape, _project

n_vpp = len(mode_data)
# Layout: columns = 1f, 2f, 3f; each harmonic has 2 sub-rows (amp, phase)
fig, axes = plt.subplots(n_vpp * 2, 3, figsize=(7.0, 1.2 * n_vpp * 2),
                         squeeze=False)

w_fine = np.linspace(-hw, hw, 200)
hw_mm = hw * 1e3

for vi, md in enumerate(mode_data):
    vpp = md["vpp"]
    row_amp = vi * 2
    row_ph = vi * 2 + 1

    for col, h in enumerate([1, 2, 3]):
        ax_a = axes[row_amp, col]
        ax_p = axes[row_ph, col]
        w_data, amp_data, ph_data = md[f"data_{h}f"]
        p0_c = md[f"p0_{h}f"]
        r2 = md[f"r2_{h}f"]

        if len(w_data) == 0:
            ax_a.set_visible(False)
            ax_p.set_visible(False)
            continue

        w_mm = w_data * 1e3

        # Sigma clipping to identify kept/excluded points
        mode_v = _mode_shape(w_data, CHANNEL_WIDTH, h, use_abs=True)
        _, clip_mask = _project(amp_data, mode_v, sigma_clip=SIGMA_CLIP)
        kept = clip_mask
        excluded = ~clip_mask

        # Fit curves
        mode_fine_abs = _mode_shape(w_fine, CHANNEL_WIDTH, h, use_abs=True)
        mode_fine_signed = _mode_shape(w_fine, CHANNEL_WIDTH, h, use_abs=False)

        # --- Amplitude panel ---
        p0_amp = abs(p0_c)
        y_lim = max(2 * p0_amp / 1e3, 1)

        ax_a.plot(w_mm[kept], amp_data[kept] / 1e3, "o", markersize=2,
                  alpha=0.6, color=f"C{col}")
        if excluded.any():
            ex_p = amp_data[excluded] / 1e3
            in_range = ex_p <= y_lim
            if in_range.any():
                ax_a.plot(w_mm[excluded][in_range], ex_p[in_range], "x",
                          markersize=3, color="0.4")
            above = ex_p > y_lim
            if above.any():
                ax_a.plot(w_mm[excluded][above],
                          np.full(above.sum(), y_lim), "^",
                          markersize=3, color="0.4")

        ax_a.plot(w_fine * 1e3, p0_amp * mode_fine_abs / 1e3,
                  "--", linewidth=0.6, color="C3")
        ax_a.set_ylim(-0.1 * p0_amp / 1e3, y_lim)
        ax_a.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
        ax_a.axvline(hw_mm, color="0.5", ls=":", lw=0.5)

        # Annotation
        txt = f"{p0_amp/1e3:.0f} kPa, $R^2$={r2:.3f}"
        ax_a.annotate(txt, xy=(0.02, 0.92), xycoords="axes fraction",
                      fontsize=5, va="top")

        if col == 0:
            ax_a.set_ylabel(r"$P$ [kPa]")

        # --- Phase panel ---
        phase_model = np.degrees(np.angle(p0_c * mode_fine_signed))
        ax_p.plot(w_mm[kept], ph_data[kept], "o", markersize=2,
                  alpha=0.6, color=f"C{col}")
        if excluded.any():
            ax_p.plot(w_mm[excluded], ph_data[excluded], "x",
                      markersize=3, color="0.4")
        ax_p.plot(w_fine * 1e3, phase_model, "--", linewidth=0.6, color="C3")
        ax_p.set_ylim(-200, 200)
        ax_p.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
        ax_p.axvline(hw_mm, color="0.5", ls=":", lw=0.5)

        if col == 0:
            ax_p.set_ylabel(r"Phase [$^\circ$]")

        # Column title (top row only)
        if vi == 0:
            ax_a.set_title(f"${h}f$")

        # x-label (bottom row only)
        if vi == n_vpp - 1:
            ax_p.set_xlabel("Width [mm]")

    # Row label
    axes[row_amp, 0].text(-0.35, 0.0, f"{vpp} Vpp",
                          transform=axes[row_amp, 0].transAxes,
                          fontsize=7, ha="right", va="center", rotation=90)

plt.tight_layout()
out_path = OUT_DIR / f"mode_shapes_{label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

print("\n=== Done ===")
