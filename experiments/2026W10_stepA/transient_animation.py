# %%
"""Animate the transient pressure field during burst-mode excitation.

Loads raw Ch2 waveforms from a 2D scan, converts velocity to pressure
via FFT-based integration with 6 MHz LPF, and produces an MP4 animation
of the instantaneous pressure p(x, y, t).

Usage:
    python transient_animation.py [tdms_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    RSSI_THRESHOLD,
    SENSITIVITY,
    channel_centre_func,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute, detect_velocity_scale
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.io_utils import extract_waveforms, load_tdms_file

# %%
# =============================================================================
# Configuration
# =============================================================================

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tdms", nargs="?", default=None, help="TDMS file path")
parser.add_argument("--test", action="store_true", help="Quick test (50-55 us)")
parser.add_argument("--t-start", type=float, default=0, help="Start time (us)")
parser.add_argument("--t-end", type=float, default=None, help="End time (us)")
parser.add_argument("--fps", type=int, default=None, help="Playback frame rate")
_args = parser.parse_args()

DEFAULT_TDMS = (get_data_dir("20260307experimentB")
                / "test10_1907_25Vpp_5m_s_max.tdms")

if _args.test:
    T_START_US = 50.0
    T_END_US = 55.0
else:
    T_START_US = _args.t_start
    T_END_US = _args.t_end if _args.t_end else 100.0
SUBSAMPLE = 1           # take every Nth sample (66 frames/cycle at 1.9 MHz)
F_LPF = 6e6            # low-pass cutoff (Hz)
F_HPF = 0.5e6          # high-pass cutoff (Hz) — removes drift
FPS = _args.fps if _args.fps else (60 if not _args.test else 30)

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load metadata and build grid
# =============================================================================

tdms_path = Path(_args.tdms) if _args.tdms else DEFAULT_TDMS
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR)
f_drive = float(cache["f_drive"])
dt = float(cache["dt"])
n_samples = int(cache["n_samples"])
pos_x = cache["pos_x"]
pos_y = cache["pos_y"]
n_x_meta = int(cache["n_x_meta"])
n_y_meta = int(cache["n_y_meta"])
V = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache else None

# Channel geometry
geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
centre_fn = channel_centre_func(geom)
hw = CHANNEL_WIDTH / 2

pos_x_c = pos_x - centre_fn(pos_y)
inside = np.abs(pos_x_c) <= hw

cg = make_channel_grid(
    pos_x_c, pos_y, n_x_meta, n_y_meta,
    CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
    rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
)

print(f"  Grid: {cg.n_width} × {cg.n_length}")
print(f"  Drive: {f_drive/1e6:.3f} MHz, dt = {dt*1e9:.0f} ns")

# %%
# =============================================================================
# Load waveforms and convert velocity → pressure
# =============================================================================

n_load = min(int(T_END_US * 1e-6 / dt), n_samples)
i_start = int(T_START_US * 1e-6 / dt)
# Subsample indices within [i_start, n_load)
frame_indices = np.arange(i_start, n_load, SUBSAMPLE)
n_frames = len(frame_indices)
vel_scale = detect_velocity_scale(tdms_path)

print(f"  Loading Ch2 waveforms ({len(pos_x)} points x {n_load} samples)...")
tdms_file, _ = load_tdms_file(tdms_path)
wf_ch2, _ = extract_waveforms(tdms_file, channel=2)
del tdms_file
wf_ch2 = wf_ch2[:, :n_load] * vel_scale  # trim and scale to m/s

# FFT-based integration + bandpass: P(f) = -V(f) / (j2pif x SENSITIVITY)
print(f"  Converting velocity -> pressure (FFT integration + "
      f"{F_HPF/1e6:.1f}-{F_LPF/1e6:.0f} MHz bandpass)...")
n_rfft = wf_ch2.shape[1]
freqs = np.fft.rfftfreq(n_rfft, d=dt)

# Bandpass mask
bp_mask = (freqs >= F_HPF) & (freqs <= F_LPF)

# Process in chunks to limit memory
CHUNK = 1000
prs_sub = np.zeros((len(pos_x), n_frames), dtype=np.float32)

for i0 in range(0, len(pos_x), CHUNK):
    i1 = min(i0 + CHUNK, len(pos_x))
    V_freq = np.fft.rfft(wf_ch2[i0:i1], axis=1)

    # Integration: divide by j2pif, negate for sign convention
    P_freq = np.zeros_like(V_freq)
    P_freq[:, bp_mask] = -V_freq[:, bp_mask] / (
        1j * 2 * np.pi * freqs[bp_mask] * SENSITIVITY)

    prs_chunk = np.fft.irfft(P_freq, n=n_rfft, axis=1)
    prs_sub[i0:i1] = prs_chunk[:, frame_indices].astype(np.float32)

    if (i0 // CHUNK) % 3 == 0:
        print(f"    chunk {i0}-{i1} / {len(pos_x)}")

del wf_ch2

# Mask invalid points
valid = make_valid_mask(V, rssi)
if "pt_burst_on_us" in cache and "pt_burst_off_us" in cache:
    valid &= make_burst_timing_mask(cache["pt_burst_on_us"], cache["pt_burst_off_us"])
n_invalid = int(np.sum(~valid))
if n_invalid > 0:
    prs_sub[~valid, :] = np.nan
    print(f"  Masked {n_invalid} invalid points ({n_invalid/len(V)*100:.1f}%)")

print(f"  Result: {prs_sub.shape} ({prs_sub.nbytes / 1e6:.0f} MB)")

# %%
# =============================================================================
# Precompute 2D grids for each frame
# =============================================================================

print("  Mapping to 2D grids...")
# Determine color scale from 99th percentile
ss_grids = np.concatenate([
    cg.to_grid(prs_sub[:, fi]).ravel()
    for fi in range(0, n_frames, max(1, n_frames // 20))
])
vmax = float(np.nanpercentile(np.abs(ss_grids), 99)) / 1e3  # kPa

t_us = frame_indices * dt * 1e6

# %%
# =============================================================================
# Animate
# =============================================================================

print(f"  Animating {n_frames} frames at {FPS} fps ({n_frames/FPS:.1f} s)...")

fig, ax = plt.subplots(figsize=(6, 3))
w_um = cg.width_grid * 1e6
l_mm = cg.length_grid * 1e3

# Initial frame
grid0 = cg.to_grid(prs_sub[:, 0]) / 1e3  # kPa
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("black")
mesh = ax.pcolormesh(l_mm, w_um, grid0, shading="nearest",
                     cmap=cmap, vmin=-vmax, vmax=vmax)
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label("Pressure [kPa]")
ax.set_xlabel("Length, $x$ [mm]")
ax.set_ylabel("Width, $y$ [µm]")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
fig.tight_layout()


def update(frame):
    grid = cg.to_grid(prs_sub[:, frame]) / 1e3
    mesh.set_array(grid.ravel())
    time_text.set_text(f"$t$ = {t_us[frame]:.1f} µs")
    return mesh, time_text


ani = anim.FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // FPS, blit=True)

suffix = "_test" if _args.test else ""
out_path = OUT_DIR / f"transient_{tdms_path.stem}{suffix}.mp4"
ani.save(str(out_path), writer="ffmpeg", fps=FPS,
         dpi=150, extra_args=["-pix_fmt", "yuv420p"])
plt.close()

print(f"\nSaved: {out_path}")
print(f"  Duration: {n_frames/FPS:.1f} s, {n_frames} frames")
print("\n=== Done ===")
