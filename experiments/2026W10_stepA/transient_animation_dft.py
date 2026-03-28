# %%
"""Animate the transient pressure envelope during burst-mode excitation.

Uses a sliding single-frequency DFT to extract the pressure amplitude
at harmonic n×f_drive for each scan point as a function of time.
Produces an MP4 animation of p_{nf}(x, y, t).

Usage:
    python transient_animation_dft.py                  # 1f, full
    python transient_animation_dft.py --harmonic 2     # 2f
    python transient_animation_dft.py --test            # quick test
"""

import argparse
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
    VELOCITY_SCALE,
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
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tdms", nargs="?", default=None, help="TDMS file path")
parser.add_argument("--harmonic", "-n", type=int, default=1, help="Harmonic number (default: 1)")
parser.add_argument("--test", action="store_true", help="Quick test (0-10 us, 2 us window)")
parser.add_argument("--t-start", type=float, default=0, help="Start time (us)")
parser.add_argument("--t-end", type=float, default=None, help="End time (us)")
parser.add_argument("--window", type=float, default=None, help="DFT window (us)")
parser.add_argument("--step", type=float, default=None, help="Time step between frames (us)")
parser.add_argument("--fps", type=int, default=30, help="Playback frame rate (default: 30)")
_args = parser.parse_args()

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = (get_data_dir("20260307experimentB")
                / "test10_1907_25Vpp_5m_s_max.tdms")

HARMONIC = _args.harmonic
T_START_US = _args.t_start
T_END_US = _args.t_end if _args.t_end else (10.0 if _args.test else 100.0)
WIN_US = _args.window if _args.window else (2.0 if _args.test else 5.0)
STEP_US = _args.step if _args.step else (0.5 if _args.test else 1.0)
FPS = _args.fps

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load metadata and build grid
# =============================================================================

tdms_path = Path(_args.tdms) if _args.tdms else DEFAULT_TDMS
print(f"Loading: {tdms_path.name}")
print(f"  Harmonic: {HARMONIC}f, window: {WIN_US} us, step: {STEP_US} us")

cache = load_or_compute(tdms_path, CACHE_DIR)
f_drive = float(cache["f_drive"])
f_harm = HARMONIC * f_drive
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

vel_scale = detect_velocity_scale(tdms_path)
prs_scale = vel_scale / (2 * np.pi * f_harm * SENSITIVITY)

print(f"  Grid: {cg.n_width} x {cg.n_length}")
print(f"  Drive: {f_drive/1e6:.3f} MHz, {HARMONIC}f = {f_harm/1e6:.3f} MHz")

# %%
# =============================================================================
# Load waveforms and compute sliding DFT
# =============================================================================

n_load = min(int((T_END_US + WIN_US) * 1e-6 / dt), n_samples)
win_n = int(WIN_US * 1e-6 / dt)
t_centres_us = np.arange(T_START_US, T_END_US, STEP_US)
n_frames = len(t_centres_us)

print(f"  Loading Ch2 waveforms ({len(pos_x)} points x {n_load} samples)...")
tdms_file, _ = load_tdms_file(tdms_path)
wf_ch2, _ = extract_waveforms(tdms_file, channel=2)
del tdms_file
wf_ch2 = wf_ch2[:, :n_load]  # trim (keep raw volts, scale later)

print(f"  Computing sliding DFT ({n_frames} frames, {win_n} samples/window)...")
pressure_vs_time = np.zeros((n_frames, len(pos_x)), dtype=np.float32)

for ti, tc_us in enumerate(t_centres_us):
    tc = int(tc_us * 1e-6 / dt)
    i0 = max(tc - win_n // 2, 0)
    i1 = min(i0 + win_n, n_load)
    n_win = i1 - i0

    # DFT at harmonic frequency, vectorized over all points
    tone = np.exp(-2j * np.pi * f_harm * np.arange(i0, i1) * dt)
    dft = wf_ch2[:, i0:i1] @ tone  # (n_points,) complex

    # Amplitude -> pressure (unsigned)
    pressure_vs_time[ti] = (np.abs(dft) * 2 / n_win * prs_scale).astype(np.float32)

    if ti % max(1, n_frames // 10) == 0:
        print(f"    frame {ti}/{n_frames} (t = {tc_us:.1f} us)")

del wf_ch2

# Mask invalid points (missed bursts / low RSSI / shifted burst timing)
valid = make_valid_mask(V, rssi)
if "pt_burst_on_us" in cache and "pt_burst_off_us" in cache:
    burst_valid = make_burst_timing_mask(cache["pt_burst_on_us"], cache["pt_burst_off_us"])
    valid &= burst_valid
    n_burst_bad = int(np.sum(~burst_valid))
    print(f"  Burst timing outliers: {n_burst_bad}")
n_invalid = int(np.sum(~valid))
if n_invalid > 0:
    pressure_vs_time[:, ~valid] = np.nan
    print(f"  Masked {n_invalid} invalid points total ({n_invalid/len(V)*100:.1f}%)")

print(f"  Result: {pressure_vs_time.shape} ({pressure_vs_time.nbytes / 1e6:.0f} MB)")

# %%
# =============================================================================
# Color scale and animate
# =============================================================================

# Color scale from steady-state 99th percentile (avoids outlier domination)
ss_start = int(n_frames * 0.7)
ss_grids = np.concatenate([
    cg.to_grid(pressure_vs_time[fi]).ravel()
    for fi in range(ss_start, n_frames, max(1, (n_frames - ss_start) // 10))
])
vmax = float(np.nanpercentile(ss_grids, 99)) / 1e3  # kPa

print(f"  vmax = {vmax:.0f} kPa")
print(f"  Animating {n_frames} frames at {FPS} fps ({n_frames/FPS:.1f} s)...")

fig, ax = plt.subplots(figsize=(6, 3))
w_um = cg.width_grid * 1e6
l_mm = cg.length_grid * 1e3

grid0 = cg.to_grid(pressure_vs_time[0]) / 1e3
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("red")
mesh = ax.pcolormesh(l_mm, w_um, grid0, shading="nearest",
                     cmap=cmap, vmin=0, vmax=vmax)
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label(f"$|p_{{{HARMONIC}f}}|$ [kPa]")
ax.set_xlabel("Length, $x$ [mm]")
ax.set_ylabel("Width, $y$ [um]")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
fig.tight_layout()


def update(frame):
    grid = cg.to_grid(pressure_vs_time[frame]) / 1e3
    mesh.set_array(grid.ravel())
    time_text.set_text(f"$t$ = {t_centres_us[frame]:.1f} us")
    return mesh, time_text


ani = anim.FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // FPS, blit=True)

suffix = "_test" if _args.test else ""
out_path = OUT_DIR / f"transient_{HARMONIC}f_{tdms_path.stem}{suffix}.mp4"
ani.save(str(out_path), writer="ffmpeg", fps=FPS,
         dpi=150, extra_args=["-pix_fmt", "yuv420p"])
plt.close()

print(f"\nSaved: {out_path}")
print(f"  Duration: {n_frames/FPS:.1f} s, {n_frames} frames")
print("\n=== Done ===")
