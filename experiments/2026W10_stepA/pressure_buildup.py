# %%
"""Visualise how the acoustic pressure field builds up during a burst.

Uses a sliding short-time DFT window to extract the 1f pressure amplitude
at each scan point as a function of time.  Produces:
  1. A pcolormesh (position × time) showing the mode-shape evolving.
  2. Individual mode-shape snapshots at selected times.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    CURRENT_SCALE,
    FIG_DPI,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms
from ldv_analysis.mode_fit import fit_mode_1f

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = get_data_dir("20260303experimentA") / "stepA_sweep_1970.tdms"

# Short-time DFT window
WINDOW_US = 10.0        # µs per window (~20 cycles at 2 MHz)
STEP_US = 5.0           # step between windows (overlap)
T_START_US = 5.0        # first window centre
T_END_US = 500.0        # last window centre

# Mode-shape snapshot times (µs)
SNAPSHOT_TIMES_US = [5, 10, 20, 50, 100, 200, 400]

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load cached metadata and raw waveforms
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR)
f_drive = float(cache["f_drive"])
pos_x = cache["pos_x"]
V = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache else None

# Valid-point mask
valid = make_valid_mask(V, rssi)

# Find channel centre (from steady-state fit)
W = CHANNEL_WIDTH
hw_mm = W / 2 * 1e3  # mm (for plot boundaries)
k_mode = np.pi / W

phase_1f = cache["phase_1f"]
pressure_1f_complex = cache["pressure_1f"] * np.exp(1j * np.radians(phase_1f))
result = fit_mode_1f(pos_x[valid], pressure_1f_complex[valid], CHANNEL_WIDTH)
best_p0, best_xc = abs(result.p0), result.centre

x_centred = pos_x - best_xc  # m, centred on channel
print(f"  Channel centre: {best_xc*1e3:.4f} mm")
print(f"  Steady-state p0: {best_p0/1e3:.0f} kPa")

# Load Ch2 (and Ch4 if available) waveforms for the needed time range
dt = float(cache["dt"])
t_load_end = (T_END_US + WINDOW_US / 2) * 1e-6
print(f"  Loading raw waveforms (0-{t_load_end*1e6:.0f} us)...")
tdms_file, _ = load_tdms_file(tdms_path)
wf_ch2, dt = extract_waveforms(tdms_file, channel=2, t_range_s=(0, t_load_end))
n_points, n_samples = wf_ch2.shape

has_ch4 = "current_1f" in cache
wf_ch4 = None
if has_ch4:
    wf_ch4, _ = extract_waveforms(tdms_file, channel=4, t_range_s=(0, t_load_end))
del tdms_file

# %%
# =============================================================================
# Short-time DFT
# =============================================================================

win_n = int(WINDOW_US * 1e-6 / dt)
t_centres_us = np.arange(T_START_US, T_END_US, STEP_US)

print(f"  Computing short-time DFT: {len(t_centres_us)} slices, "
      f"window = {WINDOW_US} µs ({win_n} samples)")

pressure_vs_time = np.zeros((len(t_centres_us), n_points))
phase_vs_time = np.zeros((len(t_centres_us), n_points))
current_vs_time = np.zeros((len(t_centres_us), n_points)) if has_ch4 else None

for ti, tc_us in enumerate(t_centres_us):
    tc = int(tc_us * 1e-6 / dt)
    i0 = max(tc - win_n // 2, 0)
    i1 = min(i0 + win_n, n_samples)
    n_win = i1 - i0
    tone = np.exp(-2j * np.pi * f_drive * np.arange(i0, i1) * dt)
    dft = wf_ch2[:, i0:i1] @ tone
    vel = np.abs(dft) * 2 / n_win * VELOCITY_SCALE
    pressure_vs_time[ti] = vel * abs(velocity_to_pressure(f_drive))
    phase_vs_time[ti] = np.degrees(np.angle(dft))
    if has_ch4:
        dft4 = wf_ch4[:, i0:i1] @ tone
        current_vs_time[ti] = np.abs(dft4) * 2 / n_win * CURRENT_SCALE

del wf_ch2
if wf_ch4 is not None:
    del wf_ch4

# %%
# =============================================================================
# Plot 1: pcolormesh (position × time)
# =============================================================================

# Sort by centred position for clean pcolormesh
sort_idx = np.argsort(x_centred)
x_sorted = x_centred[sort_idx]
p_sorted = pressure_vs_time[:, sort_idx]

fig, ax = plt.subplots(figsize=figsize_for_layout(1, 1, ax_w_scale=1.5))
pcm = ax.pcolormesh(x_sorted * 1e3, t_centres_us, p_sorted / 1e3,
                     shading="nearest", cmap="inferno", vmin=0)
ax.axvline(-hw_mm, color="w", ls=":", lw=0.5)
ax.axvline(hw_mm, color="w", ls=":", lw=0.5)
ax.set_xlabel("Position [mm]")
ax.set_ylabel(r"Time [\textmu s]")
ax.set_title(f"Pressure build-up --- {f_drive/1e6:.3f} MHz")
cb = fig.colorbar(pcm, ax=ax)
cb.set_label("Pressure [kPa]")
plt.tight_layout()
out_path = OUT_DIR / f"pressure_buildup_{stem}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 2: mode-shape snapshots
# =============================================================================

inside = np.abs(x_centred) <= W / 2
x_fine_mm = np.linspace(-hw_mm, hw_mm, 200)
sin_fine_signed = np.sin(k_mode * x_fine_mm * 1e-3)
sin_fine = np.abs(sin_fine_signed)

n_snaps = len(SNAPSHOT_TIMES_US)
fig, axes = plt.subplots(
    n_snaps, 2,
    figsize=figsize_for_layout(n_snaps, 2, sharex=True, ax_h_scale=0.5),
    sharex=True,
)

for i, t_target in enumerate(SNAPSHOT_TIMES_US):
    ti = int(np.argmin(np.abs(t_centres_us - t_target)))
    ax = axes[i, 0]
    axp = axes[i, 1]

    p_slice = pressure_vs_time[ti]
    ph_slice = phase_vs_time[ti]

    # Inside channel, valid points
    mask_in = valid & inside

    # Complex fit for p0 at this time slice
    x_in = x_centred[mask_in]
    p_in = p_slice[mask_in]
    ph_in = ph_slice[mask_in]
    p_complex = p_in * np.exp(1j * np.radians(ph_in))
    sin_prof = np.sin(k_mode * x_in)
    denom = np.sum(sin_prof ** 2)
    p0_t_complex = np.sum(p_complex * sin_prof) / denom if denom > 0 else 0j
    p0_t = abs(p0_t_complex)

    # Amplitude panel
    ax.plot(x_centred[mask_in] * 1e3, p_slice[mask_in] / 1e3,
            ".", markersize=1.5, alpha=0.6, color="C0")
    ax.plot(x_fine_mm, p0_t / 1e3 * sin_fine, "--", linewidth=0.6, color="C3")
    ax.annotate(
        rf"{t_centres_us[ti]:.0f} \textmu s, $P$ = {p0_t/1e3:.0f} kPa",
        xy=(0.02, 0.88), xycoords="axes fraction", fontsize=5, va="top",
    )
    ax.set_ylim(0, abs(best_p0) / 1e3 * 1.3)
    ax.set_ylabel(r"$P_{1f}$ [kPa]")
    ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw_mm, color="0.5", ls=":", lw=0.5)

    # Phase panel
    axp.plot(x_centred[mask_in] * 1e3, ph_in,
             ".", markersize=1.5, alpha=0.6, color="C0")
    phase_model = np.degrees(np.angle(p0_t_complex * sin_fine_signed))
    axp.plot(x_fine_mm, phase_model, "--", linewidth=0.6, color="C3")
    axp.set_ylim(-200, 200)
    axp.set_ylabel(r"Phase [$^\circ$]")
    axp.axvline(-hw_mm, color="0.5", ls=":", lw=0.5)
    axp.axvline(hw_mm, color="0.5", ls=":", lw=0.5)

axes[-1, 0].set_xlabel("Position [mm]")
axes[-1, 1].set_xlabel("Position [mm]")
axes[0, 0].set_title(f"Mode shape evolution --- {f_drive/1e6:.3f} MHz")
axes[0, 1].set_title("Phase")
plt.tight_layout()
out_path2 = OUT_DIR / f"pressure_buildup_slices_{stem}.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Plot 3: p0(t) ring-up curve
# =============================================================================

# Fit p0 at each time slice (complex)
mask_in = valid & inside
x_in = x_centred[mask_in]
sin_prof = np.sin(k_mode * x_in)
denom = np.sum(sin_prof ** 2)
p0_vs_t = np.zeros(len(t_centres_us))
for ti in range(len(t_centres_us)):
    p_complex = (pressure_vs_time[ti, mask_in]
                 * np.exp(1j * np.radians(phase_vs_time[ti, mask_in])))
    p0_vs_t[ti] = abs(np.sum(p_complex * sin_prof) / denom) if denom > 0 else 0

fig, ax = plt.subplots(figsize=figsize_for_layout())
ln1 = ax.plot(t_centres_us, p0_vs_t / 1e3, "-", linewidth=0.8, color="C0",
              label=r"$P$")
ax.axhline(best_p0 / 1e3, color="C0", ls="--", lw=0.5, alpha=0.5)
ax.set_xlabel(r"Time [\textmu s]")
ax.set_ylabel(r"$P$ [kPa]")
ax.set_title(f"Pressure and current ring-up --- {f_drive/1e6:.3f} MHz")

# Current on twin y-axis
if has_ch4:
    # Median current across valid points at each time slice
    I_vs_t = np.median(current_vs_time[:, valid], axis=1) * 1e3  # mA
    ax2 = ax.twinx()
    ln2 = ax2.plot(t_centres_us, I_vs_t, "-", linewidth=0.8, color="C1",
                   label="Current")
    I_ss = float(np.median(cache["current_1f"][valid])) * 1e3
    ax2.axhline(I_ss, color="C1", ls="--", lw=0.5, alpha=0.5)
    ax2.set_ylabel("Current [mA]")
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=5, frameon=False)
else:
    ax.legend(fontsize=5, frameon=False)

plt.tight_layout()
out_path3 = OUT_DIR / f"pressure_ringup_{stem}.png"
fig.savefig(out_path3, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path3}")

# %%
print("\n=== Done ===")
