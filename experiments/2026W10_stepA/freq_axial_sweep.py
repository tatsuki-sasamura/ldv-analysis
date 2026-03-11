# %%
"""Frequency × axial-position sweep analysis.

Processes test3_*.tdms files (coarse width scan at multiple y-positions)
to find the resonance frequency and axial antinode. Generates:
  1. Individual mode-shape plots for every (frequency, x-position) pair.
  2. A summary heatmap of p0(f, x).

Coordinate convention (matching experiment):
  - y = channel width  (21 scan points, mode shape fitted here)
  - x = channel length (11 scan positions, axial structure)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import FIG_DPI, figsize_for_layout, get_data_dir, get_output_dir
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.mode_fit import fit_mode_1f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test3_*.tdms"

# Channel geometry
CHANNEL_WIDTH = 0.375e-3  # m
RSSI_THRESHOLD = 1.0      # V

# Position snapping
X_SNAP_STEP = 1.0         # mm (axial positions are integer mm)

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Discover and sort sweep files
# =============================================================================

tdms_files = sorted(DATA_DIR.glob(FILE_PATTERN))
tdms_files = [f for f in tdms_files if not f.name.endswith("_index")]
print(f"Found {len(tdms_files)} sweep files in {DATA_DIR}\n")

if not tdms_files:
    print("No files found. Exiting.")
    sys.exit(0)

# %%
# =============================================================================
# Process each file
# =============================================================================

W = CHANNEL_WIDTH
hw = W / 2 * 1e3  # mm
k_mode = np.pi / W

# Collect results for summary
all_freqs = []        # MHz
all_x_positions = []  # mm (axial)
all_p0 = []           # p0[freq_idx][x_idx]
all_r2 = []
all_rssi_med = []     # median RSSI per (freq, x)
all_V_med = []        # median voltage per file
all_I_med = []        # median current per file
all_phase_med = []    # median V-I phase per file
mode_shape_data = []  # for individual plots

for tdms_path in tdms_files:
    stem = tdms_path.stem
    freq_khz = stem.split("_")[-1]
    print(f"--- {tdms_path.name} ---")

    cache = load_or_compute(tdms_path, CACHE_DIR)
    f_drive = float(cache["f_drive"])
    pos_y = cache["pos_x"]   # scan "x" = channel width (y in our convention)
    pos_x = cache["pos_y"]   # scan "y" = channel length (x in our convention)
    pressure = cache["pressure_1f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    # Valid mask
    valid = V > np.median(V) * 0.5
    if rssi is not None:
        valid &= rssi > RSSI_THRESHOLD

    # Snap axial positions
    x_snap = np.round(pos_x / X_SNAP_STEP) * X_SNAP_STEP
    x_positions = np.sort(np.unique(x_snap))

    all_freqs.append(f_drive / 1e6)
    all_V_med.append(float(np.median(V[valid])))
    has_ch4 = "current_1f" in cache
    if has_ch4:
        I = cache["current_1f"]
        phase_vi = cache["phase_vi"]
        all_I_med.append(float(np.median(I[valid])) * 1e3)  # mA
        all_phase_med.append(float(np.median(phase_vi[valid])))
    else:
        all_I_med.append(np.nan)
        all_phase_med.append(np.nan)
    p0_at_x = []
    r2_at_x = []
    rssi_at_x = []

    for xv in x_positions:
        mask = valid & (np.abs(x_snap - xv) < X_SNAP_STEP / 2)
        if mask.sum() < 3:
            p0_at_x.append(0)
            r2_at_x.append(0)
            rssi_at_x.append(0)
            mode_shape_data.append(None)
            continue

        # RSSI median at this (freq, x) — use all points (not just valid)
        all_mask = np.abs(x_snap - xv) < X_SNAP_STEP / 2
        rssi_at_x.append(float(np.median(rssi[all_mask])) if rssi is not None else 0)

        result = fit_mode_1f(pos_y[mask], pressure[mask], CHANNEL_WIDTH * 1e3)
        best_p0, best_yc, r2 = result.p0, result.centre, result.r2

        p0_at_x.append(best_p0)
        r2_at_x.append(r2)

        # Store for individual plot
        y_centred = pos_y[mask] - best_yc
        mode_shape_data.append(dict(
            freq_khz=freq_khz, f_mhz=f_drive / 1e6,
            x_pos=xv, p0=best_p0, r2=r2,
            y_c=y_centred, p=pressure[mask],
        ))

    all_p0.append(p0_at_x)
    all_r2.append(r2_at_x)
    all_rssi_med.append(rssi_at_x)
    if len(all_x_positions) == 0:
        all_x_positions = x_positions

    print(f"  f = {f_drive/1e6:.3f} MHz, "
          f"max p0 = {max(p0_at_x)/1e3:.0f} kPa at "
          f"x = {x_positions[np.argmax(p0_at_x)]:.0f} mm\n")

# %%
# =============================================================================
# Plot 1: Individual mode-shape plots
# =============================================================================

mode_dir = OUT_DIR / "mode_shapes_test3"
mode_dir.mkdir(parents=True, exist_ok=True)

y_fine = np.linspace(-hw, hw, 200)
sin_fine = np.abs(np.sin(k_mode * y_fine * 1e-3))

for md in mode_shape_data:
    if md is None:
        continue

    fig, ax = plt.subplots(figsize=figsize_for_layout())
    p0_kpa = md["p0"] / 1e3
    y_c_m = md["y_c"] * 1e-3
    inside = np.abs(y_c_m) <= W / 2

    # Outside channel
    ax.plot(md["y_c"][~inside], md["p"][~inside] / 1e3,
            "x", markersize=3, alpha=0.3, color="0.6")
    # Inside channel
    ax.plot(md["y_c"][inside], md["p"][inside] / 1e3,
            ".", markersize=3, alpha=0.6)
    # Fit
    r2_str = f"{md['r2']:.2f}" if md["r2"] > -10 else "<-10"
    ax.plot(y_fine, p0_kpa * sin_fine, "--", linewidth=1, color="C3",
            label=rf"$p_0$ = {p0_kpa:.0f} kPa, $R^2$ = {r2_str}")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)

    ax.set_xlabel("Width position (mm)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title(f"{md['freq_khz']} kHz, x = {md['x_pos']:.0f} mm")
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fname = f"mode_{md['freq_khz']}kHz_x{md['x_pos']:.0f}mm.png"
    fig.savefig(mode_dir / fname, dpi=FIG_DPI)
    plt.close()

print(f"Saved mode-shape plots to {mode_dir}")

# %%
# =============================================================================
# Plot 2: summary heatmaps (p0, RSSI, R²)
# =============================================================================

p0_grid = np.array(all_p0) / 1e3  # (n_freq, n_x) in kPa
r2_grid = np.array(all_r2)
rssi_grid = np.array(all_rssi_med)
freq_arr = np.array(all_freqs)
x_arr = np.array(all_x_positions)

# Sort by frequency
sort_f = np.argsort(freq_arr)
freq_arr = freq_arr[sort_f]
p0_grid = p0_grid[sort_f]
r2_grid = r2_grid[sort_f]
rssi_grid = rssi_grid[sort_f]

heatmaps = [
    (p0_grid, r"$p_0$ (kPa)", "inferno", 0, None, "freq_x_p0_heatmap.png"),
    (rssi_grid, "RSSI median (V)", "viridis", 0, None, "freq_x_rssi_heatmap.png"),
    (r2_grid, r"$R^2$", "RdYlGn", -1, 1, "freq_x_r2_heatmap.png"),
]

for grid, label, cmap, vmin, vmax, fname in heatmaps:
    fig, ax = plt.subplots(figsize=figsize_for_layout(1, 1, ax_w_scale=1.5))
    pcm = ax.pcolormesh(x_arr, freq_arr, grid,
                         shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Axial position $x$ (mm)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(f"{label} --- frequency $\\times$ axial position")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(label)
    plt.tight_layout()
    out_path = OUT_DIR / fname
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 4: freq_sweep-style 4-panel at best axial position
# =============================================================================

# Pick x with highest peak p0
x_best_idx = int(np.argmax(np.max(p0_grid, axis=0)))
x_repr = x_arr[x_best_idx]
p0_at_repr = p0_grid[:, x_best_idx]  # kPa

V_arr = np.array(all_V_med)[sort_f]
I_arr = np.array(all_I_med)[sort_f]
phase_arr = np.array(all_phase_med)[sort_f]

fig, axes = plt.subplots(
    4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True,
)

axes[0].plot(freq_arr, p0_at_repr, ".-", markersize=3, linewidth=0.8)
axes[0].set_ylabel(r"$p_0$ (kPa)")
axes[0].set_title(f"Frequency sweep --- test3, $x = {x_repr:.0f}$ mm")
axes[0].grid(True, alpha=0.3)

axes[1].plot(freq_arr, phase_arr, ".-", markersize=3, linewidth=0.8, color="C3")
axes[1].set_ylabel("V--I phase (deg)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(freq_arr, I_arr, ".-", markersize=3, linewidth=0.8, color="C2")
axes[2].set_ylabel("Current (mA)")
axes[2].grid(True, alpha=0.3)

axes[3].plot(freq_arr, V_arr, ".-", markersize=3, linewidth=0.8, color="C1")
axes[3].set_ylabel("Voltage (V)")
axes[3].set_xlabel("Frequency (MHz)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "freq_sweep_test3.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'Freq (MHz)':>11s}", end="")
for xv in x_arr:
    print(f"  x={xv:.0f}", end="")
print()
print("-" * (11 + 6 * len(x_arr)))
for i, f in enumerate(freq_arr):
    print(f"{f:11.3f}", end="")
    for j in range(len(x_arr)):
        print(f"  {p0_grid[i, j]:4.0f}", end="")
    print()

# Best overall
i_best, j_best = np.unravel_index(np.argmax(p0_grid), p0_grid.shape)
print(f"\nPeak: p0 = {p0_grid[i_best, j_best]:.0f} kPa "
      f"at {freq_arr[i_best]:.3f} MHz, x = {x_arr[j_best]:.0f} mm")

# %%
print("\n=== Done ===")
