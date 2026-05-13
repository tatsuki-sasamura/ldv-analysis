# %%
"""Calibrate channel geometry from multiple FFT cache files.

Jointly estimates channel center and tilt from RSSI data across all files
sharing the same physical geometry (same chip, same mounting, same scan
region).  The result is saved as a JSON file that pressure_map_2d.py can
load, avoiding per-file pressure-based detection which fails at low SNR.

Usage:
    python calibrate_geometry.py <tdms_path_1> [tdms_path_2 ...]
    python calibrate_geometry.py <glob_pattern>

Examples:
    python calibrate_geometry.py D:/.../20260307experimentB/test10_*.tdms
    python calibrate_geometry.py D:/.../20260306experimentA/test6_1907.tdms D:/.../test8_3845.tdms
"""

import argparse
import glob
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute, fmin

from ldv_analysis.config import CHANNEL_WIDTH, FIG_DPI, figsize_for_layout
from ldv_analysis.fft_cache import load_or_compute

CACHE_DIR = Path(__file__).resolve().parent / "output" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Calibrate channel geometry from multiple scan files")
parser.add_argument("paths", nargs="+",
                    help="TDMS file paths or glob patterns")
parser.add_argument("--dataset", default=None,
                    help="Dataset name override (default: infer from first file's parent dir)")
parser.add_argument("--width", type=float, default=CHANNEL_WIDTH,
                    help=f"Channel width in m (default: {CHANNEL_WIDTH})")
parser.add_argument("--method", choices=["mean", "clipped", "binary"],
                    default="clipped",
                    help="Objective function: mean (old), clipped (clipped+symmetric), "
                         "binary (binarised+symmetric). Default: clipped")
args = parser.parse_args()

# Expand glob patterns
tdms_paths = []
for p in args.paths:
    expanded = glob.glob(p)
    if expanded:
        tdms_paths.extend(Path(f) for f in sorted(expanded))
    else:
        tdms_paths.append(Path(p))

if not tdms_paths:
    print("Error: no files found")
    sys.exit(1)

# Infer dataset name from first file's parent directory
dataset = args.dataset or tdms_paths[0].parent.name
channel_width = args.width
hw = channel_width / 2

print(f"Dataset: {dataset}")
print(f"Channel width: {channel_width * 1e3:.3f} mm ({channel_width} m)")
print(f"Files: {len(tdms_paths)}")

# %%
# =============================================================================
# Load and stack RSSI + positions from all caches
# =============================================================================

all_pos_x = []
all_pos_y = []
all_rssi = []
n_x_meta = n_y_meta = None

for tdms_path in tdms_paths:
    print(f"\n  Loading: {tdms_path.name}")
    cache = load_or_compute(tdms_path, CACHE_DIR)

    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    if n_x_meta is None:
        n_x_meta = int(cache["n_x_meta"])
        n_y_meta = int(cache["n_y_meta"])

    if "rssi" not in cache:
        print(f"    WARNING: no RSSI in cache, skipping")
        continue

    rssi = cache["rssi"]
    n = len(pos_x)
    print(f"    {n} points, RSSI range: {rssi.min():.2f} -- {rssi.max():.2f} V")

    all_pos_x.append(pos_x)
    all_pos_y.append(pos_y)
    all_rssi.append(rssi)

if not all_rssi:
    print("\nError: no files with RSSI data")
    sys.exit(1)

pos_x = np.concatenate(all_pos_x)
pos_y = np.concatenate(all_pos_y)
rssi = np.concatenate(all_rssi)
n_total = len(pos_x)

print(f"\nCombined: {n_total} points from {len(all_rssi)} files")
print(f"  x range: {pos_x.min()*1e3:.3f} -- {pos_x.max()*1e3:.3f} mm")
print(f"  y range: {pos_y.min()*1e3:.3f} -- {pos_y.max()*1e3:.3f} mm")

# %%
# =============================================================================
# RSSI-based geometry optimization
# =============================================================================
# Three methods available:
#   mean    — original: maximize mean(rssi[inside])
#   clipped — clamp RSSI floor + asymmetry penalty
#   binary  — binarise RSSI (good/bad) + asymmetry penalty

x_min, x_max = pos_x.min(), pos_x.max()
y_min, y_max = pos_y.min(), pos_y.max()
y_span = max(y_max - y_min, 1e-12)

c_lo = x_min + hw
c_hi = x_max - hw

method = args.method
LAMBDA_ASYM = 1.0  # asymmetry penalty weight

if method == "mean":
    rssi_obj = rssi.copy()
    print(f"\n  Method: mean RSSI (no clipping, no symmetry)")

elif method == "clipped":
    RSSI_FLOOR = float(np.percentile(rssi, 25))
    rssi_obj = np.clip(rssi, RSSI_FLOOR, None)
    print(f"\n  Method: clipped + symmetric")
    print(f"  RSSI floor (P25): {RSSI_FLOOR:.3f} V")

elif method == "binary":
    # Binarise using Otsu's method: find threshold that maximizes
    # between-class variance of the bimodal RSSI distribution
    # (outside/glass ~1.5 V vs inside/water ~2.3 V).
    n_bins = 256
    hist_counts, bin_edges = np.histogram(rssi, bins=n_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = hist_counts.sum()
    best_sigma, best_t = 0.0, bin_centres[0]
    w0, sum0 = 0, 0.0
    sum_total = (hist_counts * bin_centres).sum()
    for i in range(n_bins):
        w0 += hist_counts[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist_counts[i] * bin_centres[i]
        mu0 = sum0 / w0
        mu1 = (sum_total - sum0) / w1
        sigma = w0 * w1 * (mu0 - mu1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            best_t = bin_centres[i]
    RSSI_THRESH = float(best_t)
    rssi_obj = (rssi >= RSSI_THRESH).astype(float)
    print(f"\n  Method: binary + symmetric (Otsu)")
    print(f"  Otsu threshold: {RSSI_THRESH:.3f} V")
    print(f"  Good points: {np.sum(rssi_obj > 0)} / {len(rssi_obj)}")


y_mid = (y_min + y_max) / 2


def objective(params):
    c_left, c_right = params
    center = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
    dist = pos_x - center
    inside = np.abs(dist) <= hw
    n_inside = np.sum(inside)
    if n_inside < 10:
        return 0.0  # penalise degenerate solutions

    rssi_in = rssi_obj[inside]
    mean_all = np.mean(rssi_in)

    if method == "mean":
        return -mean_all

    # Four-quadrant asymmetry penalty:
    #   pos_x = channel width direction (vertical in RSSI plot)
    #   pos_y = channel length direction (horizontal in RSSI plot)
    #
    # Quadrants (matching visual layout of RSSI heatmap):
    #   upper-left  upper-right     (upper = pos_x > center)
    #   lower-left  lower-right     (lower = pos_x < center)
    #                               (left = pos_y < y_mid, right = pos_y >= y_mid)
    #
    # Three penalty terms:
    #   |mean_upper - mean_lower|         — constrains center (width)
    #   |mean_UL - mean_LL|               — constrains tilt at left end
    #   |mean_UR - mean_LR|               — constrains tilt at right end
    dist_in = dist[inside]
    y_in = pos_y[inside]
    upper = dist_in >= 0
    lower = dist_in < 0
    left = y_in < y_mid
    right = y_in >= y_mid

    asym = 0.0
    # Global upper-lower balance (center constraint)
    if upper.any() and lower.any():
        asym += abs(np.mean(rssi_in[upper]) - np.mean(rssi_in[lower]))
    # Per-end upper-lower balance (tilt constraint)
    for end in [left, right]:
        eu = end & upper
        el = end & lower
        if eu.any() and el.any():
            asym += abs(np.mean(rssi_in[eu]) - np.mean(rssi_in[el]))

    return -mean_all + LAMBDA_ASYM * asym


print(f"\nOptimising: brute(Ns=100) + bounded refinement...")
print(f"  Search range: c = [{c_lo*1e3:.3f}, {c_hi*1e3:.3f}] mm")

from scipy.optimize import minimize

# Brute search (no fmin finish — it's unconstrained and can escape)
result_brute = brute(objective,
                     ranges=((c_lo, c_hi), (c_lo, c_hi)),
                     Ns=100, finish=None)

# Bounded refinement
bounds = [(c_lo, c_hi), (c_lo, c_hi)]
result_refined = minimize(objective, result_brute, method="L-BFGS-B", bounds=bounds)
result = result_refined.x

c_left_opt = float(result[0])
c_right_opt = float(result[1])
tilt_slope = (c_right_opt - c_left_opt) / y_span
tilt_deg = float(np.degrees(np.arctan(tilt_slope)))

# Verify: compute stats for the final geometry
centre_final = c_left_opt + tilt_slope * (pos_y - y_min)
inside_final = np.abs(pos_x - centre_final) <= hw
mean_rssi_inside = float(np.mean(rssi[inside_final]))
n_inside = int(np.sum(inside_final))

print(f"\nResult:")
print(f"  Center left  (y={y_min*1e3:.2f} mm): {c_left_opt*1e3:.4f} mm")
print(f"  Center right (y={y_max*1e3:.2f} mm): {c_right_opt*1e3:.4f} mm")
print(f"  Tilt: {tilt_deg:.3f} deg")
print(f"  Points inside: {n_inside}/{n_total}")
print(f"  Mean RSSI inside: {mean_rssi_inside:.3f} V")

# %%
# =============================================================================
# Save geometry JSON
# =============================================================================

geom = {
    "channel_width_m": channel_width,
    "centre_left_m": round(float(c_left_opt), 7),
    "centre_right_m": round(float(c_right_opt), 7),
    "y_min_m": round(float(y_min), 6),
    "y_max_m": round(float(y_max), 6),
    "tilt_deg": round(tilt_deg, 4),
    "calibrated_from": [p.name for p in tdms_paths],
    "method": f"rssi_{method}",
    "n_points_total": n_total,
    "mean_rssi_inside": round(mean_rssi_inside, 4),
    "created": str(date.today()),
}

geom_path = CACHE_DIR / f"channel_geometry_{dataset}.json"
geom_path.write_text(json.dumps(geom, indent=4) + "\n")
print(f"\nSaved: {geom_path}")

# %%
# =============================================================================
# Visualise: RSSI map with detected channel boundaries
# =============================================================================

OUT_DIR = CACHE_DIR.parent / "calibrate_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Snap to nominal grid using linspace (handles stage jitter)
x_grid = np.linspace(x_min, x_max, n_x_meta)
y_grid = np.linspace(y_min, y_max, n_y_meta)
ix = np.argmin(np.abs(pos_x[:, None] - x_grid[None, :]), axis=1)
iy = np.argmin(np.abs(pos_y[:, None] - y_grid[None, :]), axis=1)
grid_rssi = np.full((n_x_meta, n_y_meta), np.nan)
grid_rssi[ix, iy] = rssi

# Boundary lines in raw coordinates
y_line = np.array([y_min, y_max])
centre_line = c_left_opt + tilt_slope * (y_line - y_min)
left_edge = centre_line - hw
right_edge = centre_line + hw

fig, ax = plt.subplots(figsize=figsize_for_layout(ax_w_scale=2.0))
im = ax.pcolormesh(y_grid * 1e3, x_grid * 1e3, grid_rssi, shading="nearest", cmap="viridis")
ax.plot(y_line * 1e3, centre_line * 1e3, "r--", linewidth=1, label="Center")
ax.plot(y_line * 1e3, left_edge * 1e3, "r-", linewidth=0.8, label="Boundary")
ax.plot(y_line * 1e3, right_edge * 1e3, "r-", linewidth=0.8)
ax.set_xlabel("Channel length [mm]")
ax.set_ylabel("Stage position [mm]")
ax.set_title(f"RSSI with detected boundary — {dataset}")
ax.legend(fontsize=6, frameon=False)
plt.colorbar(im, ax=ax, label="RSSI [V]")
plt.tight_layout()
# Include source file stems in output name to avoid overwriting
src_stems = "_".join(p.stem for p in tdms_paths)
if len(src_stems) > 100:
    src_stems = src_stems[:100]
out_path = OUT_DIR / f"geometry_rssi_{src_stems}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")
