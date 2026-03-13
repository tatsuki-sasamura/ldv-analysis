# %%
"""Debug script: evaluate mode-shape fitting error for Fig 8.

Compares two approaches for the fit amplitude:
  1. nanmax (currently used in manuscript_figures.py)
  2. Least-squares projection (fit_columns method)

Reports per-point residuals, RMS error, and R² for each.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    RSSI_THRESHOLD,
    channel_centre_func,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns

# %%
DATA_DIR_B = get_data_dir("20260307experimentB")
CACHE_DIR = get_output_dir(__file__).parent / "cache"
hw = CHANNEL_WIDTH / 2

VOLTAGE_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms", 5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

_geom_B = load_channel_geometry("20260307experimentB", CACHE_DIR)
_centre_fn_B = channel_centre_func(_geom_B)

# %%
# Build results (same as manuscript_figures.py)
results = []
for fname, vpp in VOLTAGE_FILES:
    tdms_path = DATA_DIR_B / fname
    if not tdms_path.exists():
        print(f"  SKIP: {fname}")
        continue
    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    n_x_meta = int(cache["n_x_meta"])
    n_y_meta = int(cache["n_y_meta"])
    pos_x_c = pos_x - _centre_fn_B(pos_y)
    inside = np.abs(pos_x_c) <= hw
    rssi = cache["rssi"] if "rssi" in cache else None

    cg = make_channel_grid(
        pos_x_c, pos_y, n_x_meta, n_y_meta,
        CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )
    results.append(dict(vpp=vpp, cache=cache, cg=cg))

# Pick antinode from 25 Vpp mode-fit profile
r_peak = results[-1]
cg_peak = r_peak["cg"]
grid_1f_peak = cg_peak.to_grid(r_peak["cache"]["pressure_1f"])
p0_1f_y = fit_columns(grid_1f_peak, cg_peak.width_grid, CHANNEL_WIDTH, harmonic=1)
j_best = int(np.nanargmax(p0_1f_y))
y_best = cg_peak.length_grid[j_best]
print(f"Axial antinode: y = {y_best*1e3:.3f} mm (col {j_best})")
print()

# %%
# Evaluate fitting for each voltage
print(f"{'Vpp':>4s}  {'Harm':>4s}  {'Method':<12s}  {'p0 [kPa]':>10s}  "
      f"{'RMS [kPa]':>10s}  {'RMS/peak%':>10s}  {'R²':>8s}")
print("-" * 72)

for r in results:
    cg_i = r["cg"]
    w = cg_i.width_grid  # m, centred

    grid_1f = cg_i.to_grid(r["cache"]["pressure_1f"])
    grid_2f = cg_i.to_grid(r["cache"]["pressure_2f"])

    for harmonic, grid, label in [(1, grid_1f, "1f"), (2, grid_2f, "2f")]:
        col = grid[:, j_best]  # Pa
        valid = ~np.isnan(col)
        data = col[valid]
        w_valid = w[valid]

        if harmonic == 1:
            mode = np.abs(np.sin(np.pi * w_valid / CHANNEL_WIDTH))
        else:
            mode = np.abs(np.cos(2 * np.pi * w_valid / CHANNEL_WIDTH))

        # Method 1: nanmax
        p0_max = np.max(data)
        fit_max = p0_max * mode
        res_max = data - fit_max
        rms_max = np.sqrt(np.mean(res_max**2))
        ss_res_max = np.sum(res_max**2)
        ss_tot = np.sum((data - data.mean())**2)
        r2_max = 1 - ss_res_max / ss_tot if ss_tot > 0 else 0

        # Method 2: least-squares projection
        p0_lsq = np.sum(data * mode) / np.sum(mode**2)
        fit_lsq = p0_lsq * mode
        res_lsq = data - fit_lsq
        rms_lsq = np.sqrt(np.mean(res_lsq**2))
        ss_res_lsq = np.sum(res_lsq**2)
        r2_lsq = 1 - ss_res_lsq / ss_tot if ss_tot > 0 else 0

        peak = np.max(data)
        print(f"{r['vpp']:4d}  {label:>4s}  {'nanmax':<12s}  {p0_max/1e3:10.1f}  "
              f"{rms_max/1e3:10.1f}  {rms_max/peak*100:9.1f}%  {r2_max:8.4f}")
        print(f"{'':4s}  {'':>4s}  {'lsq':<12s}  {p0_lsq/1e3:10.1f}  "
              f"{rms_lsq/1e3:10.1f}  {rms_lsq/peak*100:9.1f}%  {r2_lsq:8.4f}")

    print()
