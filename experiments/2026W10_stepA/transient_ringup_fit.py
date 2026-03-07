# %%
"""Transient analysis: ring-up/ring-down time constants from burst-mode data.

Fits envelopes of Ch2 (acoustic, sliding DFT at f_drive) and
Ch4 (current, Hilbert) at the strongest scan point.

Ch4 rise uses the Butterworth-Van Dyke (BVD) model:
    I(t) = I_C0 + I_mot * (1 - exp(-t/tau))
where I_C0 is the instantaneous current through the static capacitance C0,
and I_mot is the motional branch contribution with time constant tau.

Ch2 rise uses a simple exponential (acoustic cavity filling).

Ch4 fall uses the BVD model:
    I(t) = I_C0 * exp(-t/tau_C0) + I_mot * exp(-t/tau_mot)
where tau_C0 is the fast C0 discharge and tau_mot is the slow motional
ring-down.

Ch2 fall uses single and double exponential fits.

Usage:
    python transient_fit.py <path_to_tdms>
    python transient_fit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert

from ldv_analysis.config import (
    CURRENT_SCALE,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = Path("C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260303experimentA/stepA1967.tdms")
ENVELOPE_SMOOTH_WIN = 63    # samples (~0.5 us) — tight enough to resolve C0 step
SLIDING_DFT_WIN_US = 5.0    # µs — window for Ch2 sliding DFT (~10 cycles at 2 MHz)
FIT_SKIP_US = 5.0           # µs — exclude initial flat region from Ch2 fits
RISE_FIT_WINDOW_US = 100.0
FALL_FIT_WINDOW_US = 100.0

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, CACHE_DIR)

# Select valid points: good RSSI and pressure above median
vel = cache["velocity_1f"]
pressure = cache["pressure_1f"]
rssi = cache["rssi"] if "rssi" in cache else np.ones_like(vel)
valid = (rssi > 1.0) & (pressure > np.median(pressure[rssi > 1.0]) if (rssi > 1.0).any() else True)
n_valid = valid.sum()
best_i = int(np.where(valid)[0][np.argmax(vel[valid])]) if valid.any() else int(np.argmax(vel))
print(f"  Valid points: {n_valid} / {len(vel)}")
print(f"  Strongest valid point: {best_i} (RSSI = {rssi[best_i]:.2f} V)")

f1 = float(cache["f_drive"])
n_samples = int(cache["n_samples"])
t_us = np.arange(n_samples) * float(cache["dt"]) * 1e6
print(f"  Drive frequency: {f1 / 1e6:.4f} MHz")

# %%
# =============================================================================
# Compute envelopes
# =============================================================================


def smooth_envelope(wf, win=ENVELOPE_SMOOTH_WIN):
    env = np.abs(hilbert(wf))
    return np.convolve(env, np.ones(win) / win, mode="same")


def sliding_dft_envelope(wf, dt, f_target, win_us=SLIDING_DFT_WIN_US):
    """Sliding single-frequency DFT: lock-in detection at f_target."""
    n = len(wf)
    win_n = int(win_us * 1e-6 / dt)
    if win_n % 2 == 0:
        win_n += 1
    tone = np.exp(-2j * np.pi * f_target * np.arange(n) * dt)
    baseband = wf * tone
    window = np.hanning(win_n)
    window /= window.sum()
    filtered = np.convolve(baseband, window, mode="same")
    return np.abs(filtered) * 2


# --- Ch2: average normalised envelope over all valid points ---
print(f"  Loading all Ch2 waveforms...")
tdms_file, _ = load_tdms_file(tdms_path)
wf_ch2_all, dt = extract_waveforms(tdms_file, channel=2)

ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
to_kPa = VELOCITY_SCALE / (2 * np.pi * f1 * SENSITIVITY) / 1e3

env_ch2_norm_wsum = np.zeros(n_samples)
weight_sum = 0.0
n_used = 0
for idx in np.where(valid)[0]:
    env = sliding_dft_envelope(wf_ch2_all[idx], dt, f1)
    env_kPa = env * to_kPa
    p_ss_i = np.mean(env_kPa[ss_start:ss_end])
    if p_ss_i > 0:
        w = p_ss_i  # weight by p_ss (higher signal → more weight)
        env_ch2_norm_wsum += w * (env_kPa / p_ss_i)
        weight_sum += w
        n_used += 1
env_ch2_norm = env_ch2_norm_wsum / weight_sum
print(f"  Averaged {n_used} normalised Ch2 envelopes (p_ss-weighted)")
del wf_ch2_all

# Best-point Ch2 envelope (for display in kPa)
wfs_best, _ = load_point_waveforms(tdms_path, best_i, channels=(1, 2, 4))
env_ch2_best = sliding_dft_envelope(wfs_best[2], dt, f1)
env_ch2_kPa = env_ch2_best * to_kPa
p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))

# Ch1: Hilbert (for burst boundary detection)
env_ch1 = smooth_envelope(wfs_best[1])
# Ch4: Hilbert (clean electrical signal, need sub-µs resolution for C0 step)
env_ch4 = smooth_envelope(wfs_best[4])
del tdms_file

# Burst boundaries from Ch1
on_mask = env_ch1 > 0.5 * np.max(env_ch1)
on_idx = np.where(on_mask)[0]
burst_on = on_idx[0]
burst_off = on_idx[-1]
print(f"  Burst ON: {burst_on * dt * 1e6:.1f}"
      f"--{burst_off * dt * 1e6:.1f} us")

# Convert Ch4 envelope to physical units
env_ch4_mA = env_ch4 * CURRENT_SCALE * 1e3

# %%
# =============================================================================
# Fit models
# =============================================================================


def rise_simple(t, tau):
    """Simple exponential rise to 1 (for Ch2 acoustic, scaled by p_ss)."""
    return 1 - np.exp(-t / tau)


def rise_bvd(t, I0, I_mot, tau):
    """BVD model: instant C0 step + motional exponential rise (for Ch4)."""
    return I0 + I_mot * (1 - np.exp(-t / tau))


def fall_simple(t, tau):
    return np.exp(-t / tau)


def fall_bvd(t, I_C0, I_mot, tau_C0, tau_mot):
    """BVD fall: fast C0 discharge + slow motional ring-down."""
    return I_C0 * np.exp(-t / tau_C0) + I_mot * np.exp(-t / tau_mot)


rise_n = int(RISE_FIT_WINDOW_US * 1e-6 / dt)
fall_n = int(FALL_FIT_WINDOW_US * 1e-6 / dt)

# Steady-state pressure: mean envelope during FFT window
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))

# %%
# =============================================================================
# Fit Ch2 (acoustic): exponential rise with fixed p_ss
# =============================================================================

# Use averaged normalised envelope (≈1 at steady state)
# Start from t=0 at skip point; rise starts from ~0, fall from ~1
skip_n = int(FIT_SKIP_US * 1e-6 / dt)
rise_start = burst_on + skip_n
rise_end = burst_on + int(RISE_FIT_WINDOW_US * 1e-6 / dt)
ch2_rise = {
    "t": np.arange(rise_end - rise_start) * dt * 1e6,
    "e": env_ch2_norm[rise_start:rise_end],
}
ch2_rise["po"], ch2_rise["pcov"] = curve_fit(
    rise_simple, ch2_rise["t"], ch2_rise["e"],
    p0=[10], bounds=([0.1], [500]))

# Ch2 fall
fall_start = burst_off + skip_n
end_f = min(burst_off + int(FALL_FIT_WINDOW_US * 1e-6 / dt), n_samples)
ch2_fall = {
    "t": np.arange(end_f - fall_start) * dt * 1e6,
    "e": env_ch2_norm[fall_start:end_f],
}
ch2_fall["po"], ch2_fall["pcov"] = curve_fit(
    fall_simple, ch2_fall["t"], ch2_fall["e"],
    p0=[10], bounds=([0.1], [500]))

# %%
# =============================================================================
# Fit Ch4 (current): BVD model rise
# =============================================================================

ch4_rise = {
    "t": np.arange(rise_n) * dt * 1e6,
    "e": env_ch4_mA[burst_on:burst_on + rise_n],
}
I0_guess = np.mean(ch4_rise["e"][:int(1e-6 / dt)])
I_ss_guess = np.max(ch4_rise["e"])
ch4_rise["po"], ch4_rise["pcov"] = curve_fit(
    rise_bvd, ch4_rise["t"], ch4_rise["e"],
    p0=[I0_guess, I_ss_guess - I0_guess, 5.0],
    bounds=([0, 0, 0.1], [I_ss_guess * 3, I_ss_guess * 3, 200]),
)

# Ch4 fall — BVD model: fast C0 discharge + slow motional ring-down
ch4_fall_n = end_f - burst_off
ch4_fall = {
    "t": np.arange(ch4_fall_n) * dt * 1e6,
    "e": env_ch4_mA[burst_off:end_f],
}
# Use rise-fit fractions as initial guesses
I0_rise, I_mot_rise, tau_rise = ch4_rise["po"]
frac_C0 = I0_rise / (I0_rise + I_mot_rise)
I_fall_0 = ch4_fall["e"][0]
ch4_fall["po"], ch4_fall["pcov"] = curve_fit(
    fall_bvd, ch4_fall["t"], ch4_fall["e"],
    p0=[I_fall_0 * frac_C0, I_fall_0 * (1 - frac_C0), 1.0, tau_rise],
    bounds=([0, 0, 0.01, 0.1], [I_fall_0 * 5, I_fall_0 * 5, 50, 500]),
    maxfev=50000,
)

# %%
# =============================================================================
# Print results
# =============================================================================

I0, I_mot, tau_ch4 = ch4_rise["po"]
perr = np.sqrt(np.diag(ch4_rise["pcov"]))
tau_ch2 = ch2_rise["po"][0]
tau_ch2_err = np.sqrt(ch2_rise["pcov"][0, 0])

print(f"\n--- Ch2 (acoustic) ---")
print(f"  p_ss (mean in FFT window) = {p_ss:.0f} kPa")
print(f"  Rise:  tau = {tau_ch2:.2f} +/- {tau_ch2_err:.2f} us  ->  Q = {np.pi * f1 * tau_ch2 * 1e-6:.0f}")
tau_ch2_f = ch2_fall["po"][0]
print(f"  Fall:  tau = {tau_ch2_f:.2f} us  ->  Q = {np.pi * f1 * tau_ch2_f * 1e-6:.0f}")

print(f"\n--- Ch4 (current, BVD model) ---")
print(f"  I_C0       = {I0:.2f} +/- {perr[0]:.2f} mA  "
      f"({I0 / (I0 + I_mot) * 100:.1f}% of total)")
print(f"  I_mot      = {I_mot:.2f} +/- {perr[1]:.2f} mA  "
      f"({I_mot / (I0 + I_mot) * 100:.1f}% of total)")
print(f"  tau_mot    = {tau_ch4:.2f} +/- {perr[2]:.2f} us")
print(f"  Q_mot      = {np.pi * f1 * tau_ch4 * 1e-6:.0f}")

# Estimate C0
# I_C0 (absolute) / (2*pi*f * V_peak) = C0
V_peak = cache["voltage_1f"][best_i]
I_total = cache["current_1f"][best_i]
I_C0_abs = I0 / (I0 + I_mot) * I_total
C0_est = I_C0_abs / (2 * np.pi * f1 * V_peak)
print(f"  C0         = {C0_est * 1e12:.1f} pF")

I_C0_f, I_mot_f, tau_C0_f, tau_mot_f = ch4_fall["po"]
perr_f = np.sqrt(np.diag(ch4_fall["pcov"]))
print(f"  Fall (BVD):")
print(f"    I_C0     = {I_C0_f:.2f} +/- {perr_f[0]:.2f} mA  "
      f"({I_C0_f / (I_C0_f + I_mot_f) * 100:.1f}% of total)")
print(f"    I_mot    = {I_mot_f:.2f} +/- {perr_f[1]:.2f} mA  "
      f"({I_mot_f / (I_C0_f + I_mot_f) * 100:.1f}% of total)")
print(f"    tau_C0   = {tau_C0_f:.2f} +/- {perr_f[2]:.2f} us")
print(f"    tau_mot  = {tau_mot_f:.2f} +/- {perr_f[3]:.2f} us  "
      f"->  Q = {np.pi * f1 * tau_mot_f * 1e-6:.0f}")

# %%
# =============================================================================
# Plot: 3 rows × 2 cols (Ch2 left, Ch4 right)
# =============================================================================

fig, axes = plt.subplots(3, 2, figsize=figsize_for_layout(3, 2))
titles = {"Ch2": "Ch2 (acoustic)", "Ch4": "Ch4 (current)"}

# --- Ch2 column (col 0) ---

# FFT window from cache (in µs)
ss_start_us = int(cache["ss_start"]) * dt * 1e6
ss_end_us = int(cache["ss_end"]) * dt * 1e6

# Row 0: full envelope (best point in kPa + averaged normalised)
ax = axes[0, 0]
ax.plot(t_us, env_ch2_kPa, linewidth=0.4, color="C0", alpha=0.5, label="best point")
ax.plot(t_us, env_ch2_norm * p_ss, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
ax.axvline(burst_on * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvline(burst_off * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvspan(burst_on * dt * 1e6, burst_on * dt * 1e6 + RISE_FIT_WINDOW_US,
           alpha=0.08, color="green")
ax.axvspan(burst_off * dt * 1e6, burst_off * dt * 1e6 + FALL_FIT_WINDOW_US,
           alpha=0.08, color="red")
ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue", label="FFT window")
ax.set_ylabel("Pressure (kPa)")
ax.set_xlabel(r"Time ($\mathrm{\mu s}$)")
ax.set_title("Ch2 (acoustic) envelope")
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# Row 1: Ch2 rise — normalised averaged envelope (t=0 at skip point)
ax = axes[1, 0]
ax.plot(ch2_rise["t"], ch2_rise["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
t_fine = np.linspace(0, ch2_rise["t"][-1], 500)
ax.plot(t_fine, rise_simple(t_fine, *ch2_rise["po"]), "--", color="C3", linewidth=1.2,
        label=r"$\tau$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
        % (tau_ch2, np.pi * f1 * tau_ch2 * 1e-6))
ax.set_ylabel(r"Normalised $p / p_{ss}$")
ax.set_xlabel(r"Time from burst ON + %.0f $\mathrm{\mu s}$" % FIT_SKIP_US)
ax.set_title(f"Ch2 ring-up (avg {n_used} pts)")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Row 2: Ch2 fall
ax = axes[2, 0]
ax.plot(ch2_fall["t"], ch2_fall["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
t_fine_f = np.linspace(0, ch2_fall["t"][-1], 500)
ax.plot(t_fine_f, fall_simple(t_fine_f, *ch2_fall["po"]), "--", color="C3",
        linewidth=1.2,
        label=r"$\tau$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
        % (tau_ch2_f, np.pi * f1 * tau_ch2_f * 1e-6))
ax.set_ylabel(r"Normalised $p / p_{ss}$")
ax.set_xlabel(r"Time from burst OFF + %.0f $\mathrm{\mu s}$" % FIT_SKIP_US)
ax.set_title(f"Ch2 ring-down (avg {n_used} pts)")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# --- Ch4 column (col 1) ---

# Row 0: full envelope
ax = axes[0, 1]
ax.plot(t_us, env_ch4_mA, linewidth=0.4, color="C1")
ax.axvline(burst_on * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvline(burst_off * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvspan(burst_on * dt * 1e6, burst_on * dt * 1e6 + RISE_FIT_WINDOW_US,
           alpha=0.08, color="green")
ax.axvspan(burst_off * dt * 1e6, burst_off * dt * 1e6 + FALL_FIT_WINDOW_US,
           alpha=0.08, color="red")
ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue", label="FFT window")
ax.set_ylabel("Current (mA)")
ax.set_xlabel(r"Time ($\mathrm{\mu s}$)")
ax.set_title("Ch4 (current) envelope")
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# Row 1: Ch4 rise — BVD model
ax = axes[1, 1]
ax.plot(ch4_rise["t"], ch4_rise["e"], "-", linewidth=0.5, color="C1", alpha=0.7)
t_fine = np.linspace(0, ch4_rise["t"][-1], 500)
ax.plot(t_fine, rise_bvd(t_fine, *ch4_rise["po"]), "--", color="C3", linewidth=1.2,
        label=(r"BVD: $\tau_\mathrm{mot}$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
               % (tau_ch4, np.pi * f1 * tau_ch4 * 1e-6)))
ax.axhline(I0, color="C2", ls=":", lw=0.8,
           label=r"$I_{C_0}$ = %.1f mA (%.0f\%%)" % (I0, I0 / (I0 + I_mot) * 100))
ax.set_ylabel("Current (mA)")
ax.set_xlabel(r"Time from burst ON ($\mathrm{\mu s}$)")
ax.set_title("Ch4 ring-up (BVD)")
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# Row 2: Ch4 fall — BVD model
ax = axes[2, 1]
ax.plot(ch4_fall["t"], ch4_fall["e"], "-", linewidth=0.5, color="C1", alpha=0.7)
t_fine_f = np.linspace(0, ch4_fall["t"][-1], 500)
ax.plot(t_fine_f, fall_bvd(t_fine_f, *ch4_fall["po"]), "--", color="C3", linewidth=1.2,
        label=(r"BVD: $\tau_{C_0}$ = %.1f, $\tau_\mathrm{mot}$ = %.1f $\mathrm{\mu s}$"
               % (tau_C0_f, tau_mot_f)))
# Show individual components
ax.plot(t_fine_f, I_C0_f * np.exp(-t_fine_f / tau_C0_f),
        color="C2", ls=":", lw=0.8, label=r"$C_0$ discharge")
ax.plot(t_fine_f, I_mot_f * np.exp(-t_fine_f / tau_mot_f),
        color="C3", ls=":", lw=0.8, label="Motional decay")
ax.set_ylabel("Current (mA)")
ax.set_xlabel(r"Time from burst OFF ($\mathrm{\mu s}$)")
ax.set_title("Ch4 ring-down (BVD)")
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / f"transient_fit_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")

# %%
print(f"\n=== Done ===")
