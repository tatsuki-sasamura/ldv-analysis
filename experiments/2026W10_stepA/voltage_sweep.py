# %%
"""Voltage sweep analysis: p0(1f) and p0(2f) vs drive voltage.

Processes test10 2D map files at multiple Vpp levels (same frequency,
same chip position).  For each file, loads the FFT cache and extracts
the fitted p0 at the best axial position.  Generates:

  1. 3-panel plot: p0(1f), p0(2f), and 2f/1f ratio vs Vpp
  2. Summary table printed to stdout

Usage:
    python voltage_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    FIG_DPI,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.mode_fit import fit_columns, make_quality_mask

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260307experimentB")

# (filename, Vpp, LDV velocity scale in m/s/V)
FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",  5,  0.5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10, 1.0),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15, 1.0),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20, 1.0),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25, 2.5),
]

CHANNEL_WIDTH = 0.375e-3  # m
CHANNEL_CENTRE = 27.087   # mm (fixed from test6_1907 reference)

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Process each file
# =============================================================================

hw = CHANNEL_WIDTH / 2 * 1e3  # mm
k_1f = np.pi / CHANNEL_WIDTH
k_2f = 2 * np.pi / CHANNEL_WIDTH

results = []

for fname, vpp, vel_scale in FILES:
    tdms_path = DATA_DIR / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    vel_correction = vel_scale / VELOCITY_SCALE

    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    n_x_meta = int(cache["n_x_meta"])
    n_y_meta = int(cache["n_y_meta"])
    f_drive = float(cache["f_drive"])
    pressure_1f = cache["pressure_1f"] * vel_correction

    # Channel mask (centred coordinates)
    pos_x_c = pos_x - CHANNEL_CENTRE
    inside = np.abs(pos_x_c) <= hw

    # Build grid
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

    def to_grid(values):
        grid = np.full((n_width_c, n_y_meta), np.nan)
        mask = inside & ~np.isnan(values)
        grid[w_c_idx[mask], l_idx[mask]] = values[mask]
        return grid

    grid_prs_1f = to_grid(pressure_1f / 1e3)  # kPa

    # Quality mask
    quality_mask = make_quality_mask(n_width_c)

    # 1f mode-shape fit
    p0_1f_y = fit_columns(grid_prs_1f * 1e3, wc_m, CHANNEL_WIDTH,
                          harmonic=1, quality_mask=quality_mask)
    best_idx = np.nanargmax(p0_1f_y)
    p0_1f_kPa = p0_1f_y[best_idx] / 1e3

    # 2f pressure from cache
    pressure_2f = cache["pressure_2f"] * vel_correction
    grid_prs_2f = to_grid(pressure_2f / 1e3)  # kPa

    # 2f mode-shape fit
    p0_2f_y = fit_columns(grid_prs_2f * 1e3, wc_m, CHANNEL_WIDTH,
                          harmonic=2, quality_mask=quality_mask)
    best_2f_idx = np.nanargmax(p0_2f_y)
    p0_2f_kPa = p0_2f_y[best_2f_idx] / 1e3

    results.append(dict(
        vpp=vpp, p0_1f=p0_1f_kPa, p0_2f=p0_2f_kPa,
        best_x=length_grid[best_idx], best_x_2f=length_grid[best_2f_idx],
    ))
    print(f"  {fname}: p0_1f = {p0_1f_kPa:.0f} kPa, "
          f"p0_2f = {p0_2f_kPa:.0f} kPa, "
          f"ratio = {p0_2f_kPa/p0_1f_kPa*100:.1f}%")

# %%
# =============================================================================
# Add origin
# =============================================================================

Vpp = np.array([0] + [r["vpp"] for r in results])
p0_1f = np.array([0] + [r["p0_1f"] for r in results])
p0_2f = np.array([0] + [r["p0_2f"] for r in results])

# %%
# =============================================================================
# Fits through origin
# =============================================================================

# p0_1f = a * V
a_1f = np.sum(Vpp * p0_1f) / np.sum(Vpp ** 2)
# p0_2f = b * V^2
b_2f = np.sum((Vpp ** 2) * p0_2f) / np.sum(Vpp ** 4)

V_fine = np.linspace(0, Vpp.max() * 1.35, 100)
fit_1f = a_1f * V_fine
fit_2f = b_2f * V_fine ** 2
fit_ratio = np.where(fit_1f > 0, fit_2f / fit_1f * 100, np.nan)

print(f"\nFit: p0_1f = {a_1f:.1f} * V  kPa")
print(f"Fit: p0_2f = {b_2f:.3f} * V^2  kPa")
print(f"Ratio slope: {b_2f/a_1f*100:.3f} %/V")

# %%
# =============================================================================
# Plot
# =============================================================================

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=figsize_for_layout(3, 1, sharex=True), sharex=True,
)

ax1.plot(Vpp, p0_1f, "o", markersize=4)
ax1.plot(V_fine, fit_1f, "--", linewidth=0.8, alpha=0.6,
         label=f"linear: {a_1f:.0f} kPa/V")
ax1.set_ylabel(r"$p_0^{1f}$ (kPa)")
ax1.legend(fontsize=6)
ax1.grid(True, alpha=0.3)
ax1.set_title("Voltage sweep at 1.907 MHz (test10)")

ax2.plot(Vpp, p0_2f, "s", markersize=4, color="C1")
ax2.plot(V_fine, fit_2f, "--", linewidth=0.8, alpha=0.6, color="C1",
         label=f"quadratic: {b_2f:.2f} kPa/V$^2$")
ax2.set_ylabel(r"$p_0^{2f}$ (kPa)")
ax2.legend(fontsize=6)
ax2.grid(True, alpha=0.3)

ax3.plot(Vpp[1:], p0_2f[1:] / p0_1f[1:] * 100, "D", markersize=4, color="C2")
ax3.plot(V_fine[1:], fit_ratio[1:], "--", linewidth=0.8, alpha=0.6, color="C2",
         label=f"{b_2f/a_1f*100:.2f}" + r" \%/V")
ax3.set_ylabel(r"$p_0^{2f}/p_0^{1f}$ (\%)")
ax3.set_xlabel(r"Drive voltage (V$_{pp}$)")
ax3.legend(fontsize=6)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / "voltage_sweep_p0.png"
fig.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")

# %%
# =============================================================================
# Summary table
# =============================================================================

print("\n| Vpp | p0_1f (kPa) | p0_2f (kPa) | 2f/1f (%) |")
print("|-----|-------------|-------------|-----------|")
for r in results:
    print(f"| {r['vpp']:3d} | {r['p0_1f']:11.0f} | {r['p0_2f']:11.0f} | "
          f"{r['p0_2f']/r['p0_1f']*100:9.1f} |")

# %%
print("\n=== Done ===")
