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

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import hilbert

from ldv_analysis.config import (
    CURRENT_SCALE,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms

# %%
# =============================================================================
# Configuration
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("tdms", nargs="?", default=None)
parser.add_argument("--fresh", action="store_true", help="Recompute envelopes")
_args = parser.parse_args()

DEFAULT_TDMS = get_data_dir("20260303experimentA") / "stepA1967.tdms"
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

tdms_path = Path(_args.tdms) if _args.tdms else DEFAULT_TDMS
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


def sliding_dft_envelope_complex(wf, dt, f_target, win_us=SLIDING_DFT_WIN_US):
    """Sliding single-frequency DFT returning complex envelope."""
    n = len(wf)
    win_n = int(win_us * 1e-6 / dt)
    if win_n % 2 == 0:
        win_n += 1
    tone = np.exp(-2j * np.pi * f_target * np.arange(n) * dt)
    baseband = wf * tone
    window = np.hanning(win_n)
    window /= window.sum()
    filtered = np.convolve(baseband, window, mode="same")
    return filtered * 2


def estimate_beat_freq(env_complex, dt_us):
    """Estimate beat frequency from FFT of complex transient residual.

    dt_us: sample spacing in microseconds. Returns frequency in Hz.
    """
    n = len(env_complex)
    if n < 8:
        return 0.0
    n_pad = max(n * 4, 2**16)
    spec = np.abs(np.fft.fft(env_complex * np.hanning(n), n=n_pad))
    freqs = np.fft.fftfreq(n_pad, d=dt_us * 1e-6)
    pos = freqs > 0
    peak = np.argmax(spec[pos])
    return float(freqs[pos][peak])


# --- Ch2: average normalised envelope over all valid points ---
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
to_kPa = VELOCITY_SCALE / (2 * np.pi * f1 * SENSITIVITY) / 1e3
to_kPa_2f = VELOCITY_SCALE / (2 * np.pi * 2 * f1 * SENSITIVITY) / 1e3

env_cache_path = CACHE_DIR / f"_transient_env_{stem}.npz"
if env_cache_path.exists() and not _args.fresh:
    print(f"  Loading envelope cache: {env_cache_path.name}")
    _ec = np.load(env_cache_path)
    env_ch2_norm_complex = _ec["env_1f_complex"]
    env_ch2_norm = _ec["env_1f_mag"]
    env_ch2_2f_norm_complex = _ec["env_2f_complex"]
    env_ch2_2f_norm = _ec["env_2f_mag"]
    env_ch2_kPa = _ec["best_1f_kPa"]
    env_ch2_2f_best_kPa = _ec["best_2f_kPa"]
    env_ch4_mA = _ec["best_ch4_mA"]
    env_ch4_complex = _ec["best_ch4_complex"] if "best_ch4_complex" in _ec else None
    env_ch1 = _ec["best_ch1"]
    n_used = int(_ec["n_used"])
    p_ss = float(_ec["p_ss"])
    p_ss_2f = float(_ec["p_ss_2f"])
    dt = float(_ec["dt"])
    del _ec
else:
    print(f"  Computing averaged Ch2 envelopes (per-point)...")
    tdms_file, _ = load_tdms_file(tdms_path)
    wf_group = tdms_file["Waveforms"]
    wf_channels = [ch for ch in wf_group.channels()
                   if ch.name.startswith("WFCh2")]
    dt = wf_channels[0].properties.get("wf_increment", 8e-9)

    env_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_mag_wsum = np.zeros(n_samples)
    env_2f_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_2f_mag_wsum = np.zeros(n_samples)
    weight_sum = 0.0
    n_used = 0
    for idx in np.where(valid)[0]:
        wf = wf_channels[idx][:]
        env_c = sliding_dft_envelope_complex(wf, dt, f1)
        env_c *= to_kPa
        env_2f_c = sliding_dft_envelope_complex(wf, dt, 2 * f1)
        env_2f_c *= to_kPa_2f
        p_ss_c = np.mean(env_c[ss_start:ss_end])
        p_ss_mag = np.abs(p_ss_c)
        if p_ss_mag > 0:
            w = p_ss_mag
            env_complex_wsum += w * (env_c / p_ss_c)
            env_mag_wsum += w * (np.abs(env_c) / p_ss_mag)
            p_ss_2f_c = np.mean(env_2f_c[ss_start:ss_end])
            p_ss_2f_mag = np.abs(p_ss_2f_c)
            if p_ss_2f_mag > 0:
                env_2f_complex_wsum += w * (env_2f_c / p_ss_2f_c)
                env_2f_mag_wsum += w * (np.abs(env_2f_c) / p_ss_2f_mag)
            weight_sum += w
            n_used += 1
    env_ch2_norm_complex = env_complex_wsum / weight_sum
    env_ch2_norm = env_mag_wsum / weight_sum
    env_ch2_2f_norm_complex = env_2f_complex_wsum / weight_sum
    env_ch2_2f_norm = env_2f_mag_wsum / weight_sum
    print(f"  Averaged {n_used} normalised Ch2 envelopes (p_ss-weighted)")

    # Best-point Ch2 envelope (for display in kPa)
    wfs_best, _ = load_point_waveforms(tdms_path, best_i, channels=(1, 2, 4))
    env_ch2_kPa = sliding_dft_envelope(wfs_best[2], dt, f1) * to_kPa
    p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))
    env_ch2_2f_best_kPa = sliding_dft_envelope(wfs_best[2], dt, 2 * f1) * to_kPa_2f
    p_ss_2f = float(np.mean(env_ch2_2f_best_kPa[ss_start:ss_end]))

    # Ch1: Hilbert (for burst boundary detection)
    env_ch1 = smooth_envelope(wfs_best[1])
    # Ch4: Hilbert (clean electrical signal, need sub-µs resolution for C0 step)
    env_ch4_mA = smooth_envelope(wfs_best[4]) * CURRENT_SCALE * 1e3
    # Ch4: sliding DFT at f_drive (complex, for beat fit)
    env_ch4_complex = sliding_dft_envelope_complex(
        wfs_best[4], dt, f1) * CURRENT_SCALE * 1e3  # mA
    del tdms_file

    # Save envelope cache
    np.savez(env_cache_path,
             env_1f_complex=env_ch2_norm_complex, env_1f_mag=env_ch2_norm,
             env_2f_complex=env_ch2_2f_norm_complex, env_2f_mag=env_ch2_2f_norm,
             best_1f_kPa=env_ch2_kPa, best_2f_kPa=env_ch2_2f_best_kPa,
             best_ch4_mA=env_ch4_mA, best_ch4_complex=env_ch4_complex,
             best_ch1=env_ch1,
             n_used=n_used, p_ss=p_ss, p_ss_2f=p_ss_2f, dt=dt)
    print(f"  Saved envelope cache: {env_cache_path.name}")

# Burst boundaries from Ch1
on_mask = env_ch1 > 0.5 * np.max(env_ch1)
on_idx = np.where(on_mask)[0]
burst_on = on_idx[0]
burst_off = on_idx[-1]
print(f"  Burst ON: {burst_on * dt * 1e6:.1f}"
      f"--{burst_off * dt * 1e6:.1f} us")

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


def rise_2f(t, tau2, tau1):
    """2f rise: first-order resonator driven by (1 - exp(-t/tau1))^2.

    Analytic solution of  tau2 * dp/dt + p = (1 - exp(-t/tau1))^2
    with p(0) = 0.  Returns normalised envelope (→ 1 at steady state).
    """
    a = tau2 / tau1
    # Guard against singularities at alpha=1 and alpha=0.5
    if abs(1 - a) < 1e-6 or abs(1 - 2 * a) < 1e-6:
        return np.full_like(t, np.nan)
    c_tau2 = -2 * a**2 / ((1 - a) * (1 - 2 * a))
    c_tau1 = -2 / (1 - a)
    c_2tau1 = 1 / (1 - 2 * a)
    return 1 + c_tau2 * np.exp(-t / tau2) + c_tau1 * np.exp(-t / tau1) \
        + c_2tau1 * np.exp(-2 * t / tau1)


def fall_simple(t, tau):
    return np.exp(-t / tau)


def fall_bvd(t, I_C0, I_mot, tau_C0, tau_mot):
    """BVD fall: fast C0 discharge + slow motional ring-down."""
    return I_C0 * np.exp(-t / tau_C0) + I_mot * np.exp(-t / tau_mot)


def rise_bvd_beat(t, I0, I_mot, tau, df):
    """BVD rise with damped-cosine ringing (for Hilbert envelope).

    I(t) = I_C0 + I_mot * (1 - exp(-t/tau) * cos(2*pi*df*t*1e-6))
    df in Hz, t in µs.  Phase is fixed at 0 so that I(0) = I_C0
    (only capacitive current flows at burst onset).
    The cosine term captures the beat between the drive frequency
    and the PZT electrical resonance.
    """
    return I0 + I_mot * (1 - np.exp(-t / tau)
                         * np.cos(2 * np.pi * df * t * 1e-6))


def rise_beat_residual(params, t_us, E_re, E_im):
    """Damped-beat rise residual (t_us in µs, df in Hz)."""
    tau, df, phi = params
    model = 1.0 - np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


def fall_beat_residual(params, t_us, E_re, E_im):
    """Damped-beat fall residual (t_us in µs, df in Hz)."""
    tau, df, phi = params
    model = np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


DF_THRESHOLD = 500  # Hz — below this, beat is not resolvable

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
    "ec": env_ch2_norm_complex[rise_start:rise_end],
}

# Simple exponential fit (magnitude)
ch2_rise["po"], ch2_rise["pcov"] = curve_fit(
    rise_simple, ch2_rise["t"], ch2_rise["e"],
    p0=[10], bounds=([0.1], [500]))

# Damped-beat fit (complex)
_rise_res = ch2_rise["ec"] - 1.0
_df_guess_r = estimate_beat_freq(_rise_res, ch2_rise["t"][1] - ch2_rise["t"][0])
_tau_guess_r = ch2_rise["po"][0]
ch2_rise["beat"] = least_squares(
    rise_beat_residual,
    x0=[_tau_guess_r, _df_guess_r, 0.0],
    args=(ch2_rise["t"], ch2_rise["ec"].real, ch2_rise["ec"].imag),
    bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
    method="trf",
)

# Ch2 fall
fall_start = burst_off + skip_n
end_f = min(burst_off + int(FALL_FIT_WINDOW_US * 1e-6 / dt), n_samples)
has_fall = fall_start < end_f - 10  # need at least ~10 samples

if has_fall:
    ch2_fall = {
        "t": np.arange(end_f - fall_start) * dt * 1e6,
        "e": env_ch2_norm[fall_start:end_f],
        "ec": env_ch2_norm_complex[fall_start:end_f],
    }

    # Simple exponential fit (magnitude)
    ch2_fall["po"], ch2_fall["pcov"] = curve_fit(
        fall_simple, ch2_fall["t"], ch2_fall["e"],
        p0=[10], bounds=([0.1], [500]))

    # Damped-beat fit (complex)
    _df_guess_f = estimate_beat_freq(ch2_fall["ec"], ch2_fall["t"][1] - ch2_fall["t"][0])
    _tau_guess_f = ch2_fall["po"][0]
    ch2_fall["beat"] = least_squares(
        fall_beat_residual,
        x0=[_tau_guess_f, _df_guess_f, 0.0],
        args=(ch2_fall["t"], ch2_fall["ec"].real, ch2_fall["ec"].imag),
        bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
        method="trf",
    )
else:
    ch2_fall = None
    print("  WARNING: no fall data (burst extends to end of recording)")

# %%
# =============================================================================
# Fit Ch2 2f: driven resonator model (source ∝ p1²)
# =============================================================================

tau1_fixed = ch2_rise["po"][0]  # µs, from 1f simple fit
ch2_2f_rise = {
    "t": ch2_rise["t"],
    "e": env_ch2_2f_norm[rise_start:rise_end],
}
# Estimate noise floor from pre-burst region
_pre_burst = env_ch2_2f_norm[max(0, burst_on - 100):burst_on]
noise_floor_2f = float(np.mean(_pre_burst)) if len(_pre_burst) > 0 else 0.0

ch2_2f_rise["po"], ch2_2f_rise["pcov"] = curve_fit(
    lambda t, tau2, A, b: b + A * rise_2f(t, tau2, tau1_fixed),
    ch2_2f_rise["t"], ch2_2f_rise["e"],
    p0=[tau1_fixed * 0.3, 1.0 - noise_floor_2f, noise_floor_2f],
    bounds=([0.1, 0, 0], [500, 5, 1]))
tau_2f, A_2f, baseline_2f = ch2_2f_rise["po"]
tau_2f_err = np.sqrt(ch2_2f_rise["pcov"][0, 0])

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
if has_fall:
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
else:
    ch4_fall = None

# Ch4 beat fit: damped-cosine on Hilbert envelope (captures acoustic back-reaction)
# Fix I_C0 from simple BVD to avoid the Hilbert smoothing confusing the
# C0/motional split.  Only fit: I_mot, tau, df.
I0, I_mot, tau_ch4_simple = ch4_rise["po"]
I_ss_ch4 = I0 + I_mot
_I0_fixed = I0

ch4_rise["po_beat"], ch4_rise["pcov_beat"] = curve_fit(
    lambda t, I_mot_b, tau_b, df_b: rise_bvd_beat(t, _I0_fixed, I_mot_b, tau_b, df_b),
    ch4_rise["t"], ch4_rise["e"],
    p0=[I_mot, tau_ch4_simple, 60e3],
    bounds=([0, 0.1, 1e3],
            [I_ss_ch4 * 3, 200, 200e3]),
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

# Beat fit results
tau_beat_r, df_beat_r, phi_beat_r = ch2_rise["beat"].x
beat_sig_r = abs(df_beat_r) > DF_THRESHOLD
if has_fall:
    tau_beat_f, df_beat_f, phi_beat_f = ch2_fall["beat"].x
    beat_sig_f = abs(df_beat_f) > DF_THRESHOLD
    tau_ch2_f = ch2_fall["po"][0]

print(f"\n--- Ch2 (acoustic) ---")
print(f"  p_ss_1f (mean in FFT window) = {p_ss:.0f} kPa")
print(f"  p_ss_2f (mean in FFT window) = {p_ss_2f:.0f} kPa")
print(f"  Rise (simple):  tau = {tau_ch2:.2f} +/- {tau_ch2_err:.2f} us"
      f"  ->  Q = {np.pi * f1 * tau_ch2 * 1e-6:.0f}")
print(f"  Rise (beat):    tau = {tau_beat_r:.2f} us"
      f"  ->  Q = {np.pi * f1 * tau_beat_r * 1e-6:.0f}"
      f"  df = {df_beat_r:.0f} Hz" + ("" if beat_sig_r else "  (not significant)"))
if has_fall:
    print(f"  Fall (simple):  tau = {tau_ch2_f:.2f} us"
          f"  ->  Q = {np.pi * f1 * tau_ch2_f * 1e-6:.0f}")
    print(f"  Fall (beat):    tau = {tau_beat_f:.2f} us"
          f"  ->  Q = {np.pi * f1 * tau_beat_f * 1e-6:.0f}"
          f"  df = {df_beat_f:.0f} Hz" + ("" if beat_sig_f else "  (not significant)"))
    if beat_sig_r or beat_sig_f:
        df_avg = np.mean([abs(df_beat_r), abs(df_beat_f)])
        print(f"  Mode splitting: {df_avg/1e3:.1f} kHz")
else:
    print(f"  Fall: no data")

print(f"\n--- Ch2 2f (driven resonator, source ~ p1^2) ---")
print(f"  tau_1 (fixed from 1f) = {tau1_fixed:.2f} us")
print(f"  tau_2 = {tau_2f:.2f} +/- {tau_2f_err:.2f} us"
      f"  ->  Q_2f = {np.pi * 2 * f1 * tau_2f * 1e-6:.0f}")
print(f"  baseline = {baseline_2f:.3f}, A = {A_2f:.3f}")

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

if has_fall:
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

I_mot_b, tau_ch4b, df_ch4b = ch4_rise["po_beat"]
perr_b = np.sqrt(np.diag(ch4_rise["pcov_beat"]))
ch4_beat_sig = abs(df_ch4b) > DF_THRESHOLD
print(f"  Rise (beat):  I_C0 = {_I0_fixed:.2f} (fixed), I_mot = {I_mot_b:.2f} mA, "
      f"tau = {tau_ch4b:.2f} +/- {perr_b[0]:.2f} us"
      f"  ->  Q = {np.pi * f1 * tau_ch4b * 1e-6:.0f}"
      f"  df = {df_ch4b:.0f} Hz"
      + ("" if ch4_beat_sig else "  (not significant)"))

# %%
# =============================================================================
# Plot: 3 rows × 2 cols (Ch2 left, Ch4 right)
# =============================================================================

fig, axes = plt.subplots(3, 5, figsize=figsize_for_layout(3, 5))

# FFT window from cache (in µs)
ss_start_us = int(cache["ss_start"]) * dt * 1e6
ss_end_us = int(cache["ss_end"]) * dt * 1e6
burst_on_us = burst_on * dt * 1e6
burst_off_us = burst_off * dt * 1e6


def _add_burst_markers(ax):
    ax.axvline(burst_on_us, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(burst_off_us, color="gray", ls="--", lw=0.7, alpha=0.6)


def _add_burst_spans(ax):
    _add_burst_markers(ax)
    ax.axvspan(burst_on_us, burst_on_us + RISE_FIT_WINDOW_US,
               alpha=0.08, color="green")
    ax.axvspan(burst_off_us, burst_off_us + FALL_FIT_WINDOW_US,
               alpha=0.08, color="red")
    ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue",
               label="FFT window")


# --- Col 0: Ch2 1f amplitude ---

# Row 0: full envelope
ax = axes[0, 0]
ax.plot(t_us, env_ch2_kPa, linewidth=0.4, color="C0", alpha=0.5, label="best point")
ax.plot(t_us, env_ch2_norm * p_ss, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
_add_burst_spans(ax)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 1f envelope")
ax.legend(fontsize=5, frameon=False)

# Row 1: rise fit
ax = axes[1, 0]
ax.plot(ch2_rise["t"], ch2_rise["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
t_fine = np.linspace(0, ch2_rise["t"][-1], 500)
ax.plot(t_fine, rise_simple(t_fine, *ch2_rise["po"]), "--", color="0.6",
        linewidth=0.8, alpha=0.7,
        label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
        % (tau_ch2, np.pi * f1 * tau_ch2 * 1e-6))
beat_rise_model = 1.0 - np.exp(-t_fine / tau_beat_r) * np.exp(
    1j * (2 * np.pi * df_beat_r * t_fine * 1e-6 + phi_beat_r))
ax.plot(t_fine, np.abs(beat_rise_model), "--", color="C3", linewidth=1.2,
        label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
        % (tau_beat_r, np.pi * f1 * tau_beat_r * 1e-6, df_beat_r / 1e3))
ax.set_ylabel(r"Normalised $P / P_{ss}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title(f"Ch2 1f ring-up (avg {n_used} pts)")
ax.legend(fontsize=5, frameon=False)

# Row 2: fall fit
ax = axes[2, 0]
if has_fall:
    ax.plot(ch2_fall["t"], ch2_fall["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
    t_fine_f = np.linspace(0, ch2_fall["t"][-1], 500)
    ax.plot(t_fine_f, fall_simple(t_fine_f, *ch2_fall["po"]), "--", color="0.6",
            linewidth=0.8, alpha=0.7,
            label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
            % (tau_ch2_f, np.pi * f1 * tau_ch2_f * 1e-6))
    beat_fall_model = np.exp(-t_fine_f / tau_beat_f) * np.exp(
        1j * (2 * np.pi * df_beat_f * t_fine_f * 1e-6 + phi_beat_f))
    ax.plot(t_fine_f, np.abs(beat_fall_model), "--", color="C3", linewidth=1.2,
            label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
            % (tau_beat_f, np.pi * f1 * tau_beat_f * 1e-6, df_beat_f / 1e3))
    ax.legend(fontsize=5, frameon=False)
else:
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")
ax.set_ylabel(r"Normalised $P / P_{ss}$")
ax.set_title(f"Ch2 1f ring-down")

# --- Col 1: Ch2 1f phase ---

ax = axes[0, 1]
ax.plot(t_us, np.degrees(np.angle(env_ch2_norm_complex)),
        linewidth=0.4, color="C0", alpha=0.7)
_add_burst_markers(ax)
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 1f phase (avg)")
ax.set_ylim(-200, 200)

ax = axes[1, 1]
ax.plot(ch2_rise["t"], np.degrees(np.angle(ch2_rise["ec"])),
        "-", linewidth=0.5, color="C0", alpha=0.7)
ax.plot(t_fine, np.degrees(np.angle(beat_rise_model)),
        "--", color="C3", linewidth=1.2)
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title("Ch2 1f rise phase")
ax.set_ylim(-200, 200)

ax = axes[2, 1]
if has_fall:
    ax.plot(ch2_fall["t"], np.degrees(np.angle(ch2_fall["ec"])),
            "-", linewidth=0.5, color="C0", alpha=0.7)
    ax.plot(t_fine_f, np.degrees(np.angle(beat_fall_model)),
            "--", color="C3", linewidth=1.2)
else:
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_title("Ch2 1f fall phase")
ax.set_ylim(-200, 200)

# --- Col 2: Ch2 2f amplitude ---

ax = axes[0, 2]
ax.plot(t_us, env_ch2_2f_best_kPa, linewidth=0.4, color="C0", alpha=0.5,
        label="best point")
ax.plot(t_us, env_ch2_2f_norm * p_ss_2f, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
_add_burst_spans(ax)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 2f envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 2]
ax.plot(ch2_rise["t"], env_ch2_2f_norm[rise_start:rise_end],
        "-", linewidth=0.5, color="C0", alpha=0.7)
t_fine_2f = np.linspace(0, ch2_rise["t"][-1], 500)
ax.plot(t_fine_2f, baseline_2f + A_2f * rise_2f(t_fine_2f, tau_2f, tau1_fixed),
        "--", color="C3", linewidth=1.2,
        label=r"$\tau_2$ = %.1f \textmu s ($Q_{2f}$ = %d)"
        % (tau_2f, np.pi * 2 * f1 * tau_2f * 1e-6))
ax.plot(t_fine_2f, A_2f * rise_simple(t_fine_2f, tau1_fixed)**2,
        ":", color="0.6", linewidth=0.8, alpha=0.7,
        label=r"source $(1-e^{-t/\tau_1})^2$")
ax.set_ylabel(r"Normalised $P_{2f} / P_{ss,2f}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title("Ch2 2f ring-up")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 2]
if has_fall:
    ax.plot(ch2_fall["t"], env_ch2_2f_norm[fall_start:end_f],
            "-", linewidth=0.5, color="C0", alpha=0.7)
else:
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")
ax.set_ylabel(r"Normalised $P_{2f} / P_{ss,2f}$")
ax.set_title("Ch2 2f ring-down")

# --- Col 3: Ch2 2f phase ---

ax = axes[0, 3]
ax.plot(t_us, np.degrees(np.angle(env_ch2_2f_norm_complex)),
        linewidth=0.4, color="C0", alpha=0.7)
_add_burst_markers(ax)
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 2f phase (avg)")
ax.set_ylim(-200, 200)

ax = axes[1, 3]
ax.plot(ch2_rise["t"],
        np.degrees(np.angle(env_ch2_2f_norm_complex[rise_start:rise_end])),
        "-", linewidth=0.5, color="C0", alpha=0.7)
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title("Ch2 2f rise phase")
ax.set_ylim(-200, 200)

ax = axes[2, 3]
if has_fall:
    ax.plot(ch2_fall["t"],
            np.degrees(np.angle(env_ch2_2f_norm_complex[fall_start:end_f])),
            "-", linewidth=0.5, color="C0", alpha=0.7)
else:
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")
ax.set_ylabel(r"Phase [$^\circ$]")
ax.set_title("Ch2 2f fall phase")
ax.set_ylim(-200, 200)

# --- Col 4: Ch4 current ---

ax = axes[0, 4]
ax.plot(t_us, env_ch4_mA, linewidth=0.4, color="C1")
_add_burst_spans(ax)
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch4 (current) envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 4]
ax.plot(ch4_rise["t"], ch4_rise["e"], "-", linewidth=0.5, color="C1", alpha=0.7)
t_fine = np.linspace(0, ch4_rise["t"][-1], 500)
ax.plot(t_fine, rise_bvd(t_fine, *ch4_rise["po"]), "--", color="0.6", linewidth=0.8,
        alpha=0.7,
        label=(r"simple: $\tau_\mathrm{mot}$ = %.1f \textmu s (Q = %d)"
               % (tau_ch4, np.pi * f1 * tau_ch4 * 1e-6)))
ax.plot(t_fine, rise_bvd_beat(t_fine, _I0_fixed, *ch4_rise["po_beat"]), "--", color="C3",
        linewidth=1.2,
        label=(r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
               % (tau_ch4b, np.pi * f1 * tau_ch4b * 1e-6, df_ch4b / 1e3)))
ax.axhline(I0, color="C2", ls=":", lw=0.8,
           label=r"$I_{C_0}$ = %.1f mA (%.0f\%%)" % (I0, I0 / (I0 + I_mot) * 100))
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time from burst ON [\textmu s]")
ax.set_title("Ch4 ring-up (BVD)")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 4]
if has_fall:
    ax.plot(ch4_fall["t"], ch4_fall["e"], "-", linewidth=0.5, color="C1", alpha=0.7)
    t_fine_f = np.linspace(0, ch4_fall["t"][-1], 500)
    ax.plot(t_fine_f, fall_bvd(t_fine_f, *ch4_fall["po"]), "--", color="C3",
            linewidth=1.2,
            label=(r"BVD: $\tau_{C_0}$ = %.1f, $\tau_\mathrm{mot}$ = %.1f \textmu s"
                   % (tau_C0_f, tau_mot_f)))
    ax.plot(t_fine_f, I_C0_f * np.exp(-t_fine_f / tau_C0_f),
            color="C2", ls=":", lw=0.8, label=r"$C_0$ discharge")
    ax.plot(t_fine_f, I_mot_f * np.exp(-t_fine_f / tau_mot_f),
            color="C3", ls=":", lw=0.8, label="Motional decay")
    ax.legend(fontsize=5, frameon=False)
else:
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")
ax.set_ylabel("Current [mA]")
ax.set_title("Ch4 ring-down (BVD)")

plt.tight_layout()
output_path = OUT_DIR / f"transient_fit_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")

# %%
print(f"\n=== Done ===")
