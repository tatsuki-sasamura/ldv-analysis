# %%
"""Plot RSSI and pressure cross-section at a given channel length position.

Useful for inspecting channel boundary quality and edge artifacts.

Usage:
    python cross_section_profile.py <tdms_path> [--y-pos 8.0] [--dataset 260413_ldv]
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
    FIG_DPI,
    channel_centre_func,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tdms_path", type=Path)
parser.add_argument("--y-pos", type=float, default=8.0,
                    help="Channel length position in mm (default: 8.0)")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for geometry lookup (auto-detected if omitted)")
args = parser.parse_args()

tdms_path = args.tdms_path
stem = tdms_path.stem
y_target_mm = args.y_pos

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data
# =============================================================================

print(f"Loading: {tdms_path.name}")
cache = load_or_compute(tdms_path, CACHE_DIR)

pos_x = cache["pos_x"]       # width (stage x) [m]
pos_y = cache["pos_y"]       # length (stage y) [m]
rssi = cache["rssi"]
prs_1f = cache["pressure_1f"]
n_x = int(cache["n_x_meta"])
n_y = int(cache["n_y_meta"])

# %%
# =============================================================================
# Load geometry
# =============================================================================

if args.dataset:
    dataset = args.dataset
else:
    # Auto-detect from path: parent folder name
    dataset = tdms_path.parent.name
    print(f"  Auto-detected dataset: {dataset}")

geom = load_channel_geometry(dataset, CACHE_DIR)
centre_fn = channel_centre_func(geom)
hw = CHANNEL_WIDTH / 2

# %%
# =============================================================================
# Extract cross-section at target y
# =============================================================================

y_grid = np.linspace(pos_y.min(), pos_y.max(), n_y)
j = np.argmin(np.abs(y_grid - y_target_mm * 1e-3))
y_actual = y_grid[j]

l_idx = np.argmin(np.abs(pos_y[:, None] - y_grid[None, :]), axis=1)
mask = l_idx == j

x_col = pos_x[mask] * 1e3        # mm
rssi_col = rssi[mask]
prs_col = prs_1f[mask] / 1e3     # kPa

centre_mm = centre_fn(y_actual) * 1e3
lo_mm = centre_mm - hw * 1e3
hi_mm = centre_mm + hw * 1e3

print(f"  y = {y_actual*1e3:.3f} mm ({np.sum(mask)} points)")
print(f"  Centre: {centre_mm:.4f} mm")
print(f"  Boundaries: {lo_mm:.4f} -- {hi_mm:.4f} mm")

# %%
# =============================================================================
# Plot
# =============================================================================

plt.style.use(["science", "ieee"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)

# RSSI
ax1.plot(x_col, rssi_col, ".", markersize=3, color="C0")
ax1.axvline(centre_mm, color="red", linewidth=0.8, linestyle="--", label="Centre")
ax1.axvline(lo_mm, color="red", linewidth=0.8, label="Boundary")
ax1.axvline(hi_mm, color="red", linewidth=0.8)
ax1.axhline(1.0, color="gray", linewidth=0.5, linestyle=":", label="RSSI threshold")
ax1.set_ylabel("RSSI [V]")
ax1.set_title(r"Cross-section at $y = %.1f$ mm (%s)" % (y_actual * 1e3, stem.replace("_", r"\_")))
ax1.legend(fontsize=5, frameon=False)

# Pressure
ax2.plot(x_col, prs_col, ".", markersize=3, color="C1")
ax2.axvline(centre_mm, color="red", linewidth=0.8, linestyle="--")
ax2.axvline(lo_mm, color="red", linewidth=0.8)
ax2.axvline(hi_mm, color="red", linewidth=0.8)
ax2.set_xlabel("Stage position [mm]")
ax2.set_ylabel(r"Pressure $p_{1f}$ [kPa]")

plt.tight_layout()
out_path = OUT_DIR / f"cross_section_y{y_target_mm:.0f}mm_{stem}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")
