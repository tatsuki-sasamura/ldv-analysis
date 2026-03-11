# %%
"""Calibrate channel geometry from multiple FFT cache files.

Jointly estimates channel centre and tilt from RSSI data across all files
sharing the same physical geometry (same chip, same mounting, same scan
region).  The result is saved as a JSON file that pressure_map_2d.py can
load, avoiding per-file pressure-based detection which fails at low SNR.

Usage:
    python calibrate_geometry.py <tdms_path_1> [tdms_path_2 ...]
    python calibrate_geometry.py <glob_pattern>

Examples:
    python calibrate_geometry.py E:/.../20260307experimentB/test10_*.tdms
    python calibrate_geometry.py E:/.../20260306experimentA/test6_1907.tdms E:/.../test8_3845.tdms
"""

import argparse
import glob
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from scipy.optimize import brute, fmin

from ldv_analysis.fft_cache import load_or_compute

# Channel geometry
CHANNEL_WIDTH = 0.375  # mm (known physical width)

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
                    help=f"Channel width in mm (default: {CHANNEL_WIDTH})")
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
print(f"Channel width: {channel_width} mm")
print(f"Files: {len(tdms_paths)}")

# %%
# =============================================================================
# Load and stack RSSI + positions from all caches
# =============================================================================

all_pos_x = []
all_pos_y = []
all_rssi = []

for tdms_path in tdms_paths:
    print(f"\n  Loading: {tdms_path.name}")
    cache = load_or_compute(tdms_path, CACHE_DIR)

    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]

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
print(f"  x range: {pos_x.min():.3f} -- {pos_x.max():.3f} mm")
print(f"  y range: {pos_y.min():.3f} -- {pos_y.max():.3f} mm")

# %%
# =============================================================================
# RSSI-based geometry optimisation
# =============================================================================
# Maximise mean(rssi[inside tilted strip]) over (c_left, c_right).
# c_left = centre at y_min, c_right = centre at y_max.

x_min, x_max = pos_x.min(), pos_x.max()
y_min, y_max = pos_y.min(), pos_y.max()
y_span = max(y_max - y_min, 1e-9)

c_lo = x_min + hw
c_hi = x_max - hw


def objective(params):
    c_left, c_right = params
    centre = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
    inside = np.abs(pos_x - centre) <= hw
    n_inside = np.sum(inside)
    if n_inside < 10:
        return 0.0  # penalise degenerate solutions
    return -np.mean(rssi[inside])


print(f"\nOptimising: brute(Ns=100) + fmin refinement...")
print(f"  Search range: c = [{c_lo:.3f}, {c_hi:.3f}] mm")

result = brute(objective,
               ranges=((c_lo, c_hi), (c_lo, c_hi)),
               Ns=100, finish=fmin)

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
print(f"  Centre left  (y={y_min:.2f} mm): {c_left_opt:.4f} mm")
print(f"  Centre right (y={y_max:.2f} mm): {c_right_opt:.4f} mm")
print(f"  Tilt: {tilt_deg:.3f} deg")
print(f"  Points inside: {n_inside}/{n_total}")
print(f"  Mean RSSI inside: {mean_rssi_inside:.3f} V")

# %%
# =============================================================================
# Save geometry JSON
# =============================================================================

geom = {
    "channel_width_mm": channel_width,
    "centre_left_mm": round(float(c_left_opt), 4),
    "centre_right_mm": round(float(c_right_opt), 4),
    "y_min_mm": round(float(y_min), 3),
    "y_max_mm": round(float(y_max), 3),
    "tilt_deg": round(tilt_deg, 4),
    "calibrated_from": [p.name for p in tdms_paths],
    "method": "rssi",
    "n_points_total": n_total,
    "mean_rssi_inside": round(mean_rssi_inside, 4),
    "created": str(date.today()),
}

geom_path = CACHE_DIR / f"channel_geometry_{dataset}.json"
geom_path.write_text(json.dumps(geom, indent=4) + "\n")
print(f"\nSaved: {geom_path}")
