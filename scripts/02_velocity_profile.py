# %%
"""Apparent velocity (Ch2) amplitude and phase profile along the scan line.

The LDV velocity decoder output represents *apparent* velocity from
refracto-vibrometry: the acoustic pressure field in the water-filled
microchannel modulates the refractive index, and the LDV interprets the
resulting optical path length change as velocity.

Computes amplitude and phase (relative to Ch1 drive voltage) via FFT of
raw waveforms at the actual drive frequency. Also plots RSSI for signal
quality assessment.

The ScanData amplitudes are NOT used because they were measured at the
vibrometer's system frequency (1.877 MHz), not the actual drive frequency
(1.970 MHz).

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    EXCLUDED_FILES,
    FIG_DPI,
    VELOCITY_SCALE,
    get_output_dir,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = get_output_dir(__file__)
CONVERTED_DIR = DATA_DIR / "converted"

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

# %%
# =============================================================================
# Compute velocity amplitude and phase from FFT at drive frequency
# =============================================================================

file_results = {}  # stem -> dict of arrays

for npz_path in npz_files:
    tdms_name = npz_path.stem + ".tdms"
    if tdms_name in EXCLUDED_FILES:
        continue

    print(f"Processing {npz_path.stem}...")
    data = np.load(npz_path)
    pos_x = data["scan_pos_x"]
    wf1 = data["wf_ch1"]  # voltage (to find drive freq & reference phase)
    wf2 = data["wf_ch2"]  # velocity
    dt = float(data["wf_dt"])
    n_points = wf1.shape[0]
    n_samples = wf1.shape[1]

    # Vectorised FFT: (n_points, n_freq_bins)
    fft_v = np.fft.rfft(wf1, axis=1)
    fft_vel = np.fft.rfft(wf2, axis=1)

    # Drive frequency per point from Ch1 peak (skip DC)
    peak_idx = np.argmax(np.abs(fft_v[:, 1:]), axis=1) + 1

    # Gather values at each point's peak index
    pts = np.arange(n_points)
    velocity_amp = np.abs(fft_vel[pts, peak_idx]) * 2 / n_samples * VELOCITY_SCALE

    # Phase relative to Ch1 voltage
    diff = np.degrees(np.angle(fft_vel[pts, peak_idx]) - np.angle(fft_v[pts, peak_idx]))
    phase_rel = (diff + 180) % 360 - 180

    # RSSI from ScanData
    rssi = data["scan_rssi"] if "scan_rssi" in data else None

    file_results[npz_path.stem] = {
        "pos_x": pos_x,
        "velocity_amp": velocity_amp,
        "phase_rel": phase_rel,
        "rssi": rssi,
    }

# %%
# =============================================================================
# Plot: velocity amplitude
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
for stem, r in file_results.items():
    order = np.argsort(r["pos_x"])
    ax.plot(r["pos_x"][order], r["velocity_amp"][order], label=stem,
            marker=".", markersize=2, linewidth=0.8)
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Apparent velocity amplitude (m/s)")
ax.set_title("Refracto-vibrometry Profile (Ch2, FFT at drive freq)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / "velocity_profile.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: phase relative to Ch1
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
for stem, r in file_results.items():
    order = np.argsort(r["pos_x"])
    ax.plot(r["pos_x"][order], r["phase_rel"][order], label=stem,
            marker=".", markersize=2, linewidth=0.8)
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Phase (deg, relative to Ch1 voltage)")
ax.set_title("Apparent Velocity Phase Profile (Ch2 - Ch1)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / "phase_profile.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: RSSI (signal quality)
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))
for stem, r in file_results.items():
    if r["rssi"] is not None:
        order = np.argsort(r["pos_x"])
        ax.plot(r["pos_x"][order], r["rssi"][order], label=stem,
                marker=".", markersize=2, linewidth=0.8)
ax.set_xlabel("X position (mm)")
ax.set_ylabel("RSSI (V)")
ax.set_title("Signal Quality (RSSI) Along Scan Line")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUT_DIR / "rssi_profile.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
