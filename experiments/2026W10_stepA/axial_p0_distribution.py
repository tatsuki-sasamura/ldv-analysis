# %%
"""Axial distribution of fitted p₀ along the channel length.

For each 2D scan file, fits the 1f mode shape p(x_c) = p₀|sin(πx_c/W)| at
every axial (y) position and plots p₀(y).  Overlays all voltage levels
on one figure.

Usage:
    python axial_p0_distribution.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
from ldv_analysis.filters import make_voltage_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260307experimentB")
FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",   5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
centre_fn = channel_centre_func(geom)
hw = CHANNEL_WIDTH / 2

# %%
# =============================================================================
# Compute p0(y) for each file
# =============================================================================

results = []

for fname, vpp in FILES:
    tdms_path = DATA_DIR / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    n_x_meta = int(cache["n_x_meta"])
    n_y_meta = int(cache["n_y_meta"])
    pressure = cache["pressure_1f"]

    pos_x_c = pos_x - centre_fn(pos_y)
    inside = np.abs(pos_x_c) <= hw

    rssi = cache["rssi"] if "rssi" in cache else None
    width_span = pos_x.max() - pos_x.min()
    cg = make_channel_grid(pos_x_c, pos_y, n_x_meta, n_y_meta,
                           CHANNEL_WIDTH, width_span, inside,
                           rssi=rssi, rssi_threshold=RSSI_THRESHOLD)
    grid = cg.to_grid(pressure)

    p0_y, sigma_y = fit_columns(grid, cg.width_grid, CHANNEL_WIDTH,
                                harmonic=1, return_sigma=True)
    y_mm = cg.length_grid * 1e3

    peak_idx = np.nanargmax(p0_y)
    print(f"  {vpp:2d} Vpp: p0_peak = {p0_y[peak_idx]/1e3:.0f} kPa "
          f"at y = {y_mm[peak_idx]:.1f} mm")

    results.append({
        "vpp": vpp,
        "y_mm": y_mm,
        "p0_kPa": p0_y / 1e3,
        "sigma_kPa": sigma_y / 1e3,
        "fname": fname,
    })

# %%
# =============================================================================
# Plot: all voltages overlaid
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
for r in results:
    ax.plot(r["y_mm"], r["p0_kPa"], "-", linewidth=0.8,
            label=f"{r['vpp']} Vpp")
    ax.fill_between(r["y_mm"],
                    r["p0_kPa"] - r["sigma_kPa"],
                    r["p0_kPa"] + r["sigma_kPa"],
                    alpha=0.15)

ax.set_xlabel(r"Axial position $y$ [mm]")
ax.set_ylabel(r"$p_{0,\mathrm{1f}}$ [kPa]")
ax.set_title(r"Axial distribution of $p_{0,\mathrm{1f}}$ --- 1.907 MHz")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = OUT_DIR / "axial_p0_1f_test10.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot: individual panels per voltage
# =============================================================================

n = len(results)
fig, axes = plt.subplots(n, 1, figsize=figsize_for_layout(n, 1), sharex=True)
if n == 1:
    axes = [axes]

for ax, r in zip(axes, results):
    ax.plot(r["y_mm"], r["p0_kPa"], "-", linewidth=0.8, color="C0")
    ax.fill_between(r["y_mm"],
                    r["p0_kPa"] - r["sigma_kPa"],
                    r["p0_kPa"] + r["sigma_kPa"],
                    alpha=0.15, color="C0")
    peak = np.nanmax(r["p0_kPa"])
    ax.set_ylabel(r"$p_0$ [kPa]")
    ax.set_title(f"{r['vpp']} Vpp (peak {peak:.0f} kPa)", fontsize=7)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(r"Axial position $y$ [mm]")
fig.suptitle(r"$p_{0,\mathrm{1f}}(y)$ --- 1.907 MHz", fontsize=9)
plt.tight_layout()

out_path2 = OUT_DIR / "axial_p0_1f_test10_panels.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
print("\n=== Done ===")
