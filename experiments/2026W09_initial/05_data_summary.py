# %%
"""Print summary info for all converted .npz data files.

Reports per file: number of scan points, waveform samples, sample rate,
scan range (X, Y), and whether optional fields (RSSI, Z) are present.

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ldv_analysis.config import CONVERTED_DIR, EXCLUDED_FILES

# %%
# =============================================================================
# Scan all .npz files
# =============================================================================

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} converted files in {CONVERTED_DIR}\n")

total_points = 0

for npz_path in npz_files:
    tdms_name = npz_path.stem + ".tdms"
    excluded = tdms_name in EXCLUDED_FILES
    tag = " [EXCLUDED]" if excluded else ""

    data = np.load(npz_path)

    n_points = data["wf_ch1"].shape[0]
    n_samples = data["wf_ch1"].shape[1]
    dt = float(data["wf_dt"])
    sample_rate = 1.0 / dt

    pos_x = data["scan_pos_x"]
    pos_y = data["scan_pos_y"]
    x_min, x_max = pos_x.min(), pos_x.max()
    y_min, y_max = pos_y.min(), pos_y.max()

    has_rssi = "scan_rssi" in data
    has_z = "scan_z_actual" in data

    # Grid dimensions from metadata (if available)
    n_x = int(data["meta_n_x"]) if "meta_n_x" in data else None
    n_y = int(data["meta_n_y"]) if "meta_n_y" in data else None

    print(f"--- {npz_path.stem}{tag} ---")
    print(f"  Points:      {n_points}")
    print(f"  Samples/wf:  {n_samples}")
    print(f"  Sample rate: {sample_rate/1e6:.1f} MHz  (dt = {dt*1e9:.1f} ns)")
    print(f"  Duration:    {n_samples * dt * 1e6:.1f} us per waveform")
    if n_x is not None and n_y is not None:
        print(f"  Grid:        {n_x} x {n_y} = {n_x * n_y}")
    print(f"  X range:     {x_min:.4f} -- {x_max:.4f} mm  (span {x_max - x_min:.4f} mm)")
    print(f"  Y range:     {y_min:.4f} -- {y_max:.4f} mm  (span {y_max - y_min:.4f} mm)")
    print(f"  RSSI: {'yes' if has_rssi else 'no'},  Z actual: {'yes' if has_z else 'no'}")
    print(f"  File size:   {npz_path.stat().st_size / 1e6:.1f} MB")
    print()

    total_points += n_points

# %%
# =============================================================================
# Summary
# =============================================================================

n_main = sum(1 for f in npz_files if f.stem + ".tdms" not in EXCLUDED_FILES)
n_excl = len(npz_files) - n_main

print("=== Summary ===")
print(f"  Total files:    {len(npz_files)}  ({n_main} main + {n_excl} excluded)")
print(f"  Total points:   {total_points}")
