# %%
"""Transient analysis: ring-up/ring-down time constants from burst-mode data.

Fits Hilbert envelope of Ch2 (acoustic) and Ch4 (current) at the
strongest scan point.

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

from ldv_analysis.config import FIG_DPI, figsize_for_layout
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TDMS = Path("G:/My Drive/20260303experimentA/stepA1967.tdms")
ENVELOPE_SMOOTH_WIN = 63    # samples (~0.5 us) — tight enough to resolve C0 step
RISE_FIT_WINDOW_US = 30.0
FALL_FIT_WINDOW_US = 100.0

OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load data
# =============================================================================

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, OUT_DIR)

# Strongest point from cached pressure (avoids loading all waveforms)
best_i = int(np.argmax(cache["velocity_1f"]))
print(f"  Strongest point: {best_i}")

# Load waveforms for this one point only (memory-efficient)
wfs, dt = load_point_waveforms(tdms_path, best_i, channels=(1, 2, 4))
n_samples = int(cache["n_samples"])
t_us = np.arange(n_samples) * dt * 1e6

# %%
# =============================================================================
# Compute envelopes
# =============================================================================


def smooth_envelope(wf, win=ENVELOPE_SMOOTH_WIN):
    env = np.abs(hilbert(wf))
    return np.convolve(env, np.ones(win) / win, mode="same")


env_ch1 = smooth_envelope(wfs[1])
env_ch2 = smooth_envelope(wfs[2])
env_ch4 = smooth_envelope(wfs[4])

# Burst boundaries from Ch1
on_mask = env_ch1 > 0.5 * np.max(env_ch1)
on_idx = np.where(on_mask)[0]
burst_on = on_idx[0]
burst_off = on_idx[-1]
print(f"  Burst ON: {burst_on * dt * 1e6:.1f}"
      f"--{burst_off * dt * 1e6:.1f} us")

# Normalise to steady-state max
ss = slice(int(50e-6 / dt), int(400e-6 / dt))
env_ch2_n = env_ch2 / np.max(env_ch2[ss])
env_ch4_n = env_ch4 / np.max(env_ch4[ss])

# %%
# =============================================================================
# Drive frequency (from cache)
# =============================================================================

f1 = float(cache["f_drive"])
print(f"  Drive frequency: {f1 / 1e6:.4f} MHz")

# %%
# =============================================================================
# Fit models
# =============================================================================


def rise_simple(t, A, tau):
    """Simple exponential rise from zero (for Ch2 acoustic)."""
    return A * (1 - np.exp(-t / tau))


def rise_bvd(t, I0, I_mot, tau):
    """BVD model: instant C0 step + motional exponential rise (for Ch4)."""
    return I0 + I_mot * (1 - np.exp(-t / tau))


def fall_simple(t, A, tau):
    return A * np.exp(-t / tau)


def fall_bvd(t, I_C0, I_mot, tau_C0, tau_mot):
    """BVD fall: fast C0 discharge + slow motional ring-down."""
    return I_C0 * np.exp(-t / tau_C0) + I_mot * np.exp(-t / tau_mot)


rise_n = int(RISE_FIT_WINDOW_US * 1e-6 / dt)
fall_n = int(FALL_FIT_WINDOW_US * 1e-6 / dt)

# %%
# =============================================================================
# Fit Ch2 (acoustic): simple exponential rise
# =============================================================================

ch2_rise = {
    "t": np.arange(rise_n) * dt * 1e6,
    "e": env_ch2_n[burst_on:burst_on + rise_n],
}
ch2_rise["po"], _ = curve_fit(rise_simple, ch2_rise["t"], ch2_rise["e"], p0=[1, 5])

# Ch2 fall
end_f = min(burst_off + fall_n, n_samples)
actual_fall_n = end_f - burst_off
ch2_fall = {
    "t": np.arange(actual_fall_n) * dt * 1e6,
    "e": env_ch2_n[burst_off:end_f],
}
ch2_fall["po"], _ = curve_fit(fall_simple, ch2_fall["t"], ch2_fall["e"], p0=[1, 10])

# %%
# =============================================================================
# Fit Ch4 (current): BVD model rise
# =============================================================================

ch4_rise = {
    "t": np.arange(rise_n) * dt * 1e6,
    "e": env_ch4_n[burst_on:burst_on + rise_n],
}
I0_guess = np.mean(ch4_rise["e"][:int(1e-6 / dt)])
ch4_rise["po"], ch4_rise["pcov"] = curve_fit(
    rise_bvd, ch4_rise["t"], ch4_rise["e"],
    p0=[I0_guess, 1.0 - I0_guess, 5.0],
    bounds=([0, 0, 0.1], [1.5, 1.5, 200]),
)

# Ch4 fall — BVD model: fast C0 discharge + slow motional ring-down
ch4_fall = {
    "t": np.arange(actual_fall_n) * dt * 1e6,
    "e": env_ch4_n[burst_off:end_f],
}
# Use rise-fit fractions as initial guesses
I0_rise, I_mot_rise, tau_rise = ch4_rise["po"]
frac_C0 = I0_rise / (I0_rise + I_mot_rise)
ch4_fall["po"], ch4_fall["pcov"] = curve_fit(
    fall_bvd, ch4_fall["t"], ch4_fall["e"],
    p0=[frac_C0, 1.0 - frac_C0, 1.0, tau_rise],
    bounds=([0, 0, 0.01, 0.1], [2, 2, 50, 500]),
    maxfev=50000,
)

# %%
# =============================================================================
# Print results
# =============================================================================

I0, I_mot, tau_ch4 = ch4_rise["po"]
perr = np.sqrt(np.diag(ch4_rise["pcov"]))
tau_ch2 = ch2_rise["po"][1]

print(f"\n--- Ch2 (acoustic) ---")
print(f"  Rise:  tau = {tau_ch2:.2f} us  ->  Q = {np.pi * f1 * tau_ch2 * 1e-6:.0f}")
tau_ch2_f = ch2_fall["po"][1]
print(f"  Fall:  tau = {tau_ch2_f:.2f} us  ->  Q = {np.pi * f1 * tau_ch2_f * 1e-6:.0f}")

print(f"\n--- Ch4 (current, BVD model) ---")
print(f"  I_C0       = {I0:.4f} +/- {perr[0]:.4f}  "
      f"({I0 / (I0 + I_mot) * 100:.1f}% of total)")
print(f"  I_mot      = {I_mot:.4f} +/- {perr[1]:.4f}  "
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
print(f"    I_C0     = {I_C0_f:.4f} +/- {perr_f[0]:.4f}  "
      f"({I_C0_f / (I_C0_f + I_mot_f) * 100:.1f}% of total)")
print(f"    I_mot    = {I_mot_f:.4f} +/- {perr_f[1]:.4f}  "
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

# Row 0: full envelope
ax = axes[0, 0]
ax.plot(t_us, env_ch2_n, linewidth=0.4, color="C0")
ax.axvline(burst_on * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvline(burst_off * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvspan(burst_on * dt * 1e6, burst_on * dt * 1e6 + RISE_FIT_WINDOW_US,
           alpha=0.08, color="green")
ax.axvspan(burst_off * dt * 1e6, burst_off * dt * 1e6 + FALL_FIT_WINDOW_US,
           alpha=0.08, color="red")
ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue", label="FFT window")
ax.set_ylabel("Normalised envelope")
ax.set_xlabel(r"Time ($\mathrm{\mu s}$)")
ax.set_title("Ch2 (acoustic) envelope")
ax.legend(fontsize=5)
ax.set_ylim(-0.05, 1.4)
ax.grid(True, alpha=0.3)

# Row 1: Ch2 rise — simple exponential
ax = axes[1, 0]
ax.plot(ch2_rise["t"], ch2_rise["e"], ".", markersize=0.3, color="C0", alpha=0.4)
t_fine = np.linspace(0, ch2_rise["t"][-1], 500)
ax.plot(t_fine, rise_simple(t_fine, *ch2_rise["po"]), color="k", linewidth=1.2,
        label=r"$\tau$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
        % (tau_ch2, np.pi * f1 * tau_ch2 * 1e-6))
ax.set_ylabel("Normalised envelope")
ax.set_xlabel(r"Time from burst ON ($\mathrm{\mu s}$)")
ax.set_title("Ch2 ring-up")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Row 2: Ch2 fall
ax = axes[2, 0]
ax.plot(ch2_fall["t"], ch2_fall["e"], ".", markersize=0.3, color="C0", alpha=0.4)
t_fine_f = np.linspace(0, ch2_fall["t"][-1], 500)
ax.plot(t_fine_f, fall_simple(t_fine_f, *ch2_fall["po"]), color="k",
        linewidth=1.2,
        label=r"$\tau$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
        % (tau_ch2_f, np.pi * f1 * tau_ch2_f * 1e-6))
ax.set_ylabel("Normalised envelope")
ax.set_xlabel(r"Time from burst OFF ($\mathrm{\mu s}$)")
ax.set_title("Ch2 ring-down")
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# --- Ch4 column (col 1) ---

# Row 0: full envelope
ax = axes[0, 1]
ax.plot(t_us, env_ch4_n, linewidth=0.4, color="C1")
ax.axvline(burst_on * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvline(burst_off * dt * 1e6, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.axvspan(burst_on * dt * 1e6, burst_on * dt * 1e6 + RISE_FIT_WINDOW_US,
           alpha=0.08, color="green")
ax.axvspan(burst_off * dt * 1e6, burst_off * dt * 1e6 + FALL_FIT_WINDOW_US,
           alpha=0.08, color="red")
ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue", label="FFT window")
ax.set_ylabel("Normalised envelope")
ax.set_xlabel(r"Time ($\mathrm{\mu s}$)")
ax.set_title("Ch4 (current) envelope")
ax.legend(fontsize=5)
ax.set_ylim(-0.05, 1.4)
ax.grid(True, alpha=0.3)

# Row 1: Ch4 rise — BVD model
ax = axes[1, 1]
ax.plot(ch4_rise["t"], ch4_rise["e"], ".", markersize=0.3, color="C1", alpha=0.4)
t_fine = np.linspace(0, ch4_rise["t"][-1], 500)
ax.plot(t_fine, rise_bvd(t_fine, *ch4_rise["po"]), color="k", linewidth=1.2,
        label=(r"BVD: $\tau_\mathrm{mot}$ = %.1f $\mathrm{\mu s}$ (Q = %d)"
               % (tau_ch4, np.pi * f1 * tau_ch4 * 1e-6)))
ax.axhline(I0, color="C2", ls=":", lw=0.8,
           label=r"$I_{C_0}$ = %.0f\%% of total" % (I0 / (I0 + I_mot) * 100))
ax.set_ylabel("Normalised envelope")
ax.set_xlabel(r"Time from burst ON ($\mathrm{\mu s}$)")
ax.set_title("Ch4 ring-up (BVD)")
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# Row 2: Ch4 fall — BVD model
ax = axes[2, 1]
ax.plot(ch4_fall["t"], ch4_fall["e"], ".", markersize=0.3, color="C1", alpha=0.4)
t_fine_f = np.linspace(0, ch4_fall["t"][-1], 500)
ax.plot(t_fine_f, fall_bvd(t_fine_f, *ch4_fall["po"]), color="k", linewidth=1.2,
        label=(r"BVD: $\tau_{C_0}$ = %.1f, $\tau_\mathrm{mot}$ = %.1f $\mathrm{\mu s}$"
               % (tau_C0_f, tau_mot_f)))
# Show individual components
ax.plot(t_fine_f, I_C0_f * np.exp(-t_fine_f / tau_C0_f),
        color="C2", ls=":", lw=0.8, label=r"$C_0$ discharge")
ax.plot(t_fine_f, I_mot_f * np.exp(-t_fine_f / tau_mot_f),
        color="C3", ls=":", lw=0.8, label="Motional decay")
ax.set_ylabel("Normalised envelope")
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
