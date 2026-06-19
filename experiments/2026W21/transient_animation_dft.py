# %%
"""Animate the transient pressure envelope p_{nf}(x, y, t) on W21 data.

Sliding single-frequency DFT at n*f_drive across every scan point gives
the time-resolved amplitude of the nf component as the burst rings up
(and rings down).  Renders an MP4.

Promoted from ``experiments/2026W10_stepA/transient_animation_dft.py``:
  - reads W21 v2 HDF5 via ``load_scan`` (format-agnostic);
  - detects channel geometry via ``sweep_fit.detect_channel_geometry``
    (no W10 geometry-JSON dependency);
  - routes the FFT cache through ``get_cache_dir`` so the shared
    OneDrive cache works.

Usage::

    python transient_animation_dft.py [path_to_h5]
    python transient_animation_dft.py --harmonic 2
    python transient_animation_dft.py --test         # quick 0-10 us

Default input: ``sample_101x21_fsweep_peak_30Vpp_…/f1904000.h5`` (the
30 Vpp cascade near the cavity peak, perturbative regime, clean SNR).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, LDV_DATA_ROOT, RSSI_THRESHOLD, SENSITIVITY,
    get_cache_dir,
)
from ldv_analysis.fft_cache import detect_velocity_scale, load_or_compute  # noqa: E402
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.io_utils import ROLE_LDV_OUTPUT, load_scan  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry  # noqa: E402


# =============================================================================
# Arguments
# =============================================================================

DEFAULT_INPUT = (
    LDV_DATA_ROOT / "output" / "W21"
    / "sample_101x21_fsweep_peak_30Vpp_20260524_210338"
    / "f1904000.h5"
)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("input", nargs="?", default=None,
                    help="Path to v2 HDF5 (.h5) or v1 TDMS file")
parser.add_argument("--harmonic", "-n", type=int, default=1,
                    help="Harmonic to follow (default: 1 = fundamental)")
parser.add_argument("--test", action="store_true",
                    help="Quick test: 0-10 us, 2 us DFT window, 0.5 us step")
parser.add_argument("--t-start", type=float, default=0.0,
                    help="Start time in us (default: 0)")
parser.add_argument("--t-end", type=float, default=None,
                    help="End time in us; default: full burst window")
parser.add_argument("--window", type=float, default=None,
                    help="DFT window in us (default: 2 in test, 5 otherwise)")
parser.add_argument("--step", type=float, default=None,
                    help="Time between frames in us (default: 0.5 / 1.0)")
parser.add_argument("--fps", type=int, default=30,
                    help="Playback frame rate (default: 30)")
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip if output MP4 already exists")
_args = parser.parse_args()

HARMONIC = _args.harmonic
T_START_US = _args.t_start
T_END_US = (_args.t_end if _args.t_end is not None
            else (10.0 if _args.test else 600.0))
WIN_US = _args.window if _args.window else (2.0 if _args.test else 5.0)
STEP_US = _args.step if _args.step else (0.5 if _args.test else 1.0)
FPS = _args.fps

h5_path = Path(_args.input) if _args.input else DEFAULT_INPUT
print(f"Loading: {h5_path.name}")
print(f"  Harmonic: {HARMONIC}f, window: {WIN_US} us, step: {STEP_US} us, "
      f"t = {T_START_US:.0f}-{T_END_US:.0f} us")

# Per-dataset OUT_DIR so multiple input files coexist.
OUT_DIR = (ROOT / "experiments" / "2026W21" / "output"
           / h5_path.parent.name / "transient_animation_dft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

suffix = "_test" if _args.test else ""
out_path = OUT_DIR / f"transient_{HARMONIC}f_{h5_path.stem}{suffix}.mp4"
if _args.skip_existing and out_path.exists():
    print(f"  Output exists, skipping: {out_path.name}")
    sys.exit(0)

# =============================================================================
# FFT cache + geometry
# =============================================================================

fft_cache_dir = get_cache_dir(h5_path.parent.name, __file__)
cache = load_or_compute(h5_path, fft_cache_dir)
f_drive = float(cache["f_drive"])
f_harm = HARMONIC * f_drive
dt = float(cache["dt"])
n_samples = int(cache["n_samples"])
pos_x = cache["pos_x"]
pos_y = cache["pos_y"]
n_x_meta = int(cache["n_x_meta"])
n_y_meta = int(cache["n_y_meta"])
V = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache.files else None
P1_abs = cache["pressure_1f"]

# Tilted-channel geometry detection (same approach as pressure_map_2d.py).
hw = CHANNEL_WIDTH / 2
Pg = np.asarray(P1_abs, dtype=float).copy()
valid_for_geom = make_valid_mask(V, rssi)
Pg[~valid_for_geom] = np.nan
a, b = detect_channel_geometry(pos_x, pos_y, rssi, Pg, hw)
pos_x_c = pos_x - (a * pos_y + b)
inside = np.abs(pos_x_c) <= hw

cg = make_channel_grid(
    pos_width_c=pos_x_c, pos_length=pos_y,
    n_scan_width=n_x_meta, n_scan_length=n_y_meta,
    channel_width=CHANNEL_WIDTH, raw_width_span=pos_x.max() - pos_x.min(),
    inside=inside, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
)
print(f"  Grid: {cg.n_width} (width) x {cg.n_length} (length)")
print(f"  Drive: {f_drive/1e6:.3f} MHz, {HARMONIC}f = {f_harm/1e6:.3f} MHz")
print(f"  Channel center tilt: a = {a*1e3:.3f} um/mm")

# Velocity -> apparent pressure scale at the harmonic frequency.
# velocity_scale takes priority from the scan metadata (v2 HDF5) but
# detect_velocity_scale handles legacy TDMS filenames.
vel_scale_from_meta = cache.get("velocity_scale")
if vel_scale_from_meta is not None:
    vel_scale = float(vel_scale_from_meta)
else:
    vel_scale = detect_velocity_scale(h5_path)
prs_scale = vel_scale / (2 * np.pi * f_harm * SENSITIVITY)

# =============================================================================
# Load waveforms in the time range we need + sliding DFT
# =============================================================================

win_n = int(WIN_US * 1e-6 / dt)
t_centers_us = np.arange(T_START_US, T_END_US, STEP_US)
n_frames = len(t_centers_us)

t_load_start = max(T_START_US - WIN_US / 2, 0.0) * 1e-6
t_load_end = min((T_END_US + WIN_US / 2) * 1e-6, n_samples * dt)
i_load_start = int(t_load_start / dt)
i_load_end = int(t_load_end / dt)
n_load = i_load_end - i_load_start
print(f"  Loading LDV waveforms: {len(pos_x)} pts x {n_load} samples "
      f"({t_load_start*1e6:.0f}-{t_load_end*1e6:.0f} us)...")

scan = load_scan(h5_path)
wf_ch2 = np.empty((scan.n_points, n_load), dtype=np.float64)
CHUNK = 200
for i0 in range(0, scan.n_points, CHUNK):
    i1 = min(i0 + CHUNK, scan.n_points)
    block = scan.load_waveforms(ROLE_LDV_OUTPUT, slice(i0, i1))
    wf_ch2[i0:i1] = block[:, i_load_start:i_load_end]
del scan

print(f"  Computing sliding DFT ({n_frames} frames, {win_n} samp/window)...")
pressure_vs_time = np.zeros((n_frames, len(pos_x)), dtype=np.float32)
for ti, tc_us in enumerate(t_centers_us):
    tc = int(tc_us * 1e-6 / dt) - i_load_start
    i0 = max(tc - win_n // 2, 0)
    i1 = min(i0 + win_n, wf_ch2.shape[1])
    n_win = i1 - i0
    # Use absolute time so the tone has the right phase.
    t_abs_start = (i0 + i_load_start) * dt
    tone = np.exp(-2j * np.pi * f_harm
                  * (t_abs_start + np.arange(n_win) * dt))
    dft = wf_ch2[:, i0:i1] @ tone
    pressure_vs_time[ti] = (
        np.abs(dft) * 2 / n_win * prs_scale
    ).astype(np.float32)
    if ti % max(1, n_frames // 10) == 0:
        print(f"    frame {ti}/{n_frames} (t = {tc_us:.1f} us)")
del wf_ch2

# Mask invalid points (RSSI / burst timing).
valid = make_valid_mask(V, rssi)
if "pt_burst_on_us" in cache.files and "pt_burst_off_us" in cache.files:
    burst_valid = make_burst_timing_mask(
        cache["pt_burst_on_us"], cache["pt_burst_off_us"]
    )
    valid &= burst_valid
    print(f"  Burst timing outliers: {int(np.sum(~burst_valid))}")
n_invalid = int(np.sum(~valid))
if n_invalid > 0:
    pressure_vs_time[:, ~valid] = np.nan
    print(f"  Masked {n_invalid} invalid points "
          f"({n_invalid/len(V)*100:.1f}%)")

print(f"  Result: {pressure_vs_time.shape} "
      f"({pressure_vs_time.nbytes / 1e6:.0f} MB)")

# =============================================================================
# Color scale and animate
# =============================================================================

ss_start_us = float(cache["ss_start"]) * dt * 1e6
ss_end_us = float(cache["ss_end"]) * dt * 1e6
ss_frame_start = max(0, int(np.searchsorted(t_centers_us, ss_start_us)))
ss_frame_end = min(n_frames, int(np.searchsorted(t_centers_us, ss_end_us)))
if ss_frame_end <= ss_frame_start:
    ss_frame_start = int(n_frames * 0.3)
    ss_frame_end = int(n_frames * 0.7)
ss_grids = np.concatenate([
    cg.to_grid(pressure_vs_time[fi]).ravel()
    for fi in range(ss_frame_start, ss_frame_end,
                    max(1, (ss_frame_end - ss_frame_start) // 10))
])
vmax = float(np.nanpercentile(ss_grids, 99)) / 1e3   # kPa
print(f"  vmax = {vmax:.0f} kPa")
print(f"  Animating {n_frames} frames at {FPS} fps "
      f"({n_frames/FPS:.1f} s)...")

fig, ax = plt.subplots(figsize=(6, 3))
w_um = cg.width_grid * 1e6
l_mm = cg.length_grid * 1e3
grid0 = cg.to_grid(pressure_vs_time[0]) / 1e3
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("black")
mesh = ax.pcolormesh(l_mm, w_um, grid0, shading="nearest",
                     cmap=cmap, vmin=0, vmax=vmax)
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label(f"$|p_{{{HARMONIC}f}}|$ [kPa]")
ax.set_xlabel(r"Length, $x$ [mm]")
ax.set_ylabel(r"Width, $y$ [$\mu$m]")
time_text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left",
    fontsize=9, color="white",
    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
)
fig.tight_layout()


def update(frame):
    grid = cg.to_grid(pressure_vs_time[frame]) / 1e3
    mesh.set_array(grid.ravel())
    time_text.set_text(rf"$t = {t_centers_us[frame]:.1f}$ $\mu$s")
    return mesh, time_text


ani = anim.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 // FPS, blit=True,
)
ani.save(str(out_path), writer="ffmpeg", fps=FPS, dpi=150,
         extra_args=["-pix_fmt", "yuv420p"])
plt.close()

print(f"\nSaved: {out_path}")
print(f"  Duration: {n_frames/FPS:.1f} s, {n_frames} frames")
