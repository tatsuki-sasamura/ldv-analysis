# %%
"""Human-readable sanity check of the 1f extraction pipeline.

Picks one scan point from a low-Vpp test10 dataset (high 1f purity),
plots the raw Ch2 voltage waveform, overlays a pure sine at the same
frequency with amplitude reconstructed from the pipeline's P_1f, and
prints the back-of-envelope conversion so a human can verify:

    P_1f_peak  ≈  V_ch2_peak  ×  |velocity_to_pressure(f_drive)|

Output: experiments/2026W10_stepA/output/sanity/fft_sanity_check.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    FIG_DPI,
    SENSITIVITY,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import (
    detect_velocity_scale, load_or_compute, load_point_waveforms,
)
from ldv_analysis.filters import make_burst_timing_mask, make_valid_mask


# %%
# =============================================================================
# Configuration: lowest Vpp, lowest 2f content → cleanest 1f waveform
# =============================================================================
DATA_DIR = get_data_dir("20260307experimentB")
TDMS_NAME = "test10_1907_5Vpp_1m_s_max.tdms"

OUT_DIR = get_output_dir(__file__) / "sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR.parent.parent / "cache"

tdms_path = DATA_DIR / TDMS_NAME

# %%
# =============================================================================
# Load cache, pick the cleanest valid point
# =============================================================================
cache = load_or_compute(tdms_path, CACHE_DIR)
f_drive = float(cache["f_drive"])
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
dt = float(cache["dt_s"]) if "dt_s" in cache else 8e-9

V = cache["voltage_1f"]
rssi = cache["rssi"] if "rssi" in cache else None
valid = make_valid_mask(V, rssi)
if "pt_burst_on_us" in cache:
    valid &= make_burst_timing_mask(cache["pt_burst_on_us"],
                                    cache["pt_burst_off_us"])

p1f = cache["pressure_1f"]
p2f = cache["pressure_2f"]
ratio = np.where(p1f > 0, p2f / p1f, np.inf)

# "Best" point: high P_1f and lowest P_2f/P_1f among valid points.
# Filter to top 20 % P_1f, then pick the one with smallest harmonic ratio.
score = np.full(len(p1f), np.inf)
high_amp = valid & (p1f > np.nanpercentile(p1f[valid], 80))
score[high_amp] = ratio[high_amp]
best_idx = int(np.nanargmin(score))

print(f"Drive frequency: f = {f_drive/1e6:.4f} MHz")
print(f"Sample interval: dt = {dt*1e9:.1f} ns")
print(f"Selected point: index {best_idx}")
print(f"  P_1f = {p1f[best_idx]/1e3:.1f} kPa")
print(f"  P_2f = {p2f[best_idx]/1e3:.2f} kPa  ({ratio[best_idx]*100:.2f} % of P_1f)")
print(f"  V_ch1 (drive) = {V[best_idx]:.3f} V")
print()

# %%
# =============================================================================
# Load the raw Ch2 waveform for this point
# =============================================================================
wfs, dt = load_point_waveforms(tdms_path, best_idx, channels=(1, 2))
ch2 = wfs[2]
n = len(ch2)
t = np.arange(n) * dt

# Steady-state window (set by the pipeline's burst detector)
ss_t = np.arange(ss_start, ss_end) * dt
ss_ch2 = ch2[ss_start:ss_end]

# %%
# =============================================================================
# Reconstruct the expected sine from the pipeline's P_1f
# =============================================================================
# Pipeline: P_1f = |V_ch2_amplitude_1f| * velocity_scale * |1/(2*pi*f*S)|
# Inverse:  expected V_ch2 amplitude = P_1f / |velocity_to_pressure(f)|
vel_scale = detect_velocity_scale(tdms_path)
factor = velocity_to_pressure(f_drive, velocity_scale=vel_scale)  # signed Pa/V
expected_v_amp = p1f[best_idx] / abs(factor)

# Measured 1f-component amplitude via DFT over the steady-state window
omega = 2 * np.pi * f_drive
ss_n = ss_end - ss_start
ss_t = np.arange(ss_n) * dt
tone = np.exp(-1j * omega * ss_t)
dft = ss_ch2 @ tone
v_amp_meas_dft = float(abs(dft) * 2 / ss_n)
v_peak_max = float(np.max(np.abs(ss_ch2)))   # raw peak (incl. noise)

# Hand calculation, printed for the reader
print("Hand-check arithmetic:")
print(f"  velocity_scale         = {vel_scale:.3f} m/s per V "
      f"(detected from filename '_1m_s_max')")
print(f"  SENSITIVITY (H*dn/dp)  = {SENSITIVITY:.3e} m/Pa")
print(f"  |conversion factor|    = velocity_scale / (2*pi * f * SENSITIVITY)")
print(f"                         = {abs(factor):.3e} Pa/V")
print(f"  expected V_ch2 amp     = P_1f / |conv factor|")
print(f"                         = {p1f[best_idx]:.0f} Pa / {abs(factor):.3e} Pa/V")
print(f"                         = {expected_v_amp*1e3:.1f} mV")
print()
print(f"Measured V_ch2 1f amplitude (DFT)   = {v_amp_meas_dft*1e3:.1f} mV")
print(f"  ratio (measured / expected)       = {v_amp_meas_dft/expected_v_amp:.4f}")
print(f"Raw V_ch2 max (noise-inflated peak) = {v_peak_max*1e3:.1f} mV")
print()
print("  For a clean 1f signal, the DFT amplitude should match the expected")
print("  expected V_ch2 amp to ~1.0; the raw peak max is biased upward by")
print("  noise spikes and is not the right number to compare.")

# %%
# =============================================================================
# Plot: raw waveform with reconstructed sine overlay
# =============================================================================
n_cycles_show = 5
samples_per_cycle = int(round(1 / (f_drive * dt)))
n_show = n_cycles_show * samples_per_cycle

# Pick a window inside the steady state
i0_show = ss_start + (ss_end - ss_start) // 2 - n_show // 2
i1_show = i0_show + n_show
t_show = (np.arange(i0_show, i1_show) - i0_show) * dt * 1e6  # µs from window start
v_show = ch2[i0_show:i1_show]

# Reconstructed pure sine: same frequency, amplitude from pipeline's P_1f.
# Phase is fitted to the measured waveform so the overlay aligns visually.
omega = 2 * np.pi * f_drive
sin_t = np.sin(omega * t_show * 1e-6)
cos_t = np.cos(omega * t_show * 1e-6)
a = float(np.dot(v_show, sin_t)) / np.dot(sin_t, sin_t)
b = float(np.dot(v_show, cos_t)) / np.dot(cos_t, cos_t)
phi_meas = np.arctan2(b, a)
v_recon = expected_v_amp * np.sin(omega * t_show * 1e-6 + phi_meas)

fig, axes = plt.subplots(2, 1, figsize=figsize_for_layout(2, 1, sharex=True),
                          sharex=True)

# Top: raw waveform + reconstruction
ax = axes[0]
ax.plot(t_show, v_show, ".-", markersize=2, lw=0.5, color="C0",
        label="Measured Ch2 voltage")
ax.plot(t_show, v_recon, "--", lw=0.8, color="C3",
        label=fr"Pure sine, amp = $P_{{1f}} / |$factor$| = {expected_v_amp:.3f}$ V")
ax.axhline(0, color="0.85", lw=0.3)
ax.set_ylabel("Ch2 voltage [V]")
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.set_title(
    f"point {best_idx}, $f$ = {f_drive/1e6:.4f} MHz, "
    f"$P_{{1f}}$ = {p1f[best_idx]/1e3:.0f} kPa",
    fontsize=8,
)
fig.suptitle(f"FFT sanity check --- {TDMS_NAME}", fontsize=8, y=1.0)

# Bottom: residual
ax = axes[1]
ax.plot(t_show, v_show - v_recon, "-", lw=0.5, color="0.4")
ax.axhline(0, color="0.85", lw=0.3)
ax.set_ylabel("Residual [V]")
ax.set_xlabel(r"Time within window [$\mu$s]")
ax.set_title(
    f"Residual rms = {np.std(v_show - v_recon)*1e3:.1f} mV "
    f"({np.std(v_show - v_recon)/expected_v_amp*100:.2f}\\% of expected amp)",
    fontsize=8,
)

plt.tight_layout()
out_path = OUT_DIR / "fft_sanity_check.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
print("\n=== Done ===")
