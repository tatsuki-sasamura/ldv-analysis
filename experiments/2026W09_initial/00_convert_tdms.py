# %%
"""Convert TDMS files to .npz for fast loading.

Reads each TDMS file once and saves all channels' waveform arrays,
ScanData fields, and metadata into a single .npz file.

Output structure (per file):
    wf_ch1, wf_ch2, wf_ch3, wf_ch4  : (n_points, n_samples) waveform arrays
    wf_dt                            : sample time increment (scalar)
    scan_pos_x, scan_pos_y           : (n_points,) position arrays
    scan_ch{1-4}_freq/amp/phase      : (n_points,) ScanData per channel
    scan_rssi, scan_z_actual         : (n_points,) optional fields
    meta_n_x, meta_n_y, ...          : scalar metadata
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ldv_analysis import list_tdms_files, load_tdms_file
from ldv_analysis.io_utils import _extract_metadata

from ldv_analysis.config import CONVERTED_DIR, DATA_DIR, N_CHANNELS, get_output_dir

# %%
# =============================================================================
# Configuration
# =============================================================================

CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

files = list_tdms_files(DATA_DIR)
print(f"Found {len(files)} TDMS files")
print(f"Output: {CONVERTED_DIR}")

# %%
# =============================================================================
# Convert each file
# =============================================================================

for fp in files:
    out_path = CONVERTED_DIR / (fp.stem + ".npz")

    if out_path.exists():
        print(f"  SKIP (exists): {out_path.name}")
        continue

    print(f"\n--- {fp.name} ---")
    t0 = time.perf_counter()

    f, metadata = load_tdms_file(fp)

    arrays = {}

    # Metadata as scalars
    for k, v in metadata.items():
        arrays[f"meta_{k}"] = np.array(v)

    # ScanData
    scan = f["ScanData"]
    n_points = len(scan["PosX"])
    arrays["scan_pos_x"] = scan["PosX"][:n_points]
    arrays["scan_pos_y"] = scan["PosY"][:n_points]

    for ch in range(1, N_CHANNELS + 1):
        arrays[f"scan_ch{ch}_freq"] = scan[f"Ch{ch}Freq"][:n_points]
        arrays[f"scan_ch{ch}_amp"] = scan[f"Ch{ch}Amp"][:n_points]
        arrays[f"scan_ch{ch}_phase"] = scan[f"Ch{ch}Phase"][:n_points]

    for name, key in [("scan_rssi", "RSSI"), ("scan_z_actual", "ZPosActual")]:
        try:
            arrays[name] = scan[key][:n_points]
        except KeyError:
            pass

    # Waveforms
    wf_group = f["Waveforms"]
    for ch in range(1, N_CHANNELS + 1):
        prefix = f"WFCh{ch}"
        channels = [c for c in wf_group.channels() if c.name.startswith(prefix)]
        if not channels:
            continue

        if ch == 1:
            dt = channels[0].properties.get("wf_increment", 8e-9)
            arrays["wf_dt"] = np.array(dt)

        n_samples = len(channels[0])
        wf = np.empty((len(channels), n_samples), dtype=np.float64)
        for i, c in enumerate(channels):
            wf[i] = c[:]
        arrays[f"wf_ch{ch}"] = wf

    np.savez_compressed(out_path, **arrays)
    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

# %%
# =============================================================================
# Verify load speed
# =============================================================================

print("\n=== Load speed comparison ===")
sample_npz = CONVERTED_DIR / (files[0].stem + ".npz")

t0 = time.perf_counter()
load_tdms_file(files[0])
t_tdms = time.perf_counter() - t0

t0 = time.perf_counter()
np.load(sample_npz)
t_npz = time.perf_counter() - t0

print(f"  TDMS: {t_tdms:.3f}s")
print(f"  .npz: {t_npz:.3f}s")
print(f"  Speedup: {t_tdms / t_npz:.1f}x")

# %%
print("\n=== Done ===")
print(f"Converted files: {CONVERTED_DIR}")
