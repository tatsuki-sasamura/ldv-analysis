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
    python transient_ringup_fit.py <path_to_tdms>
    python transient_ringup_fit.py
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
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.io_utils import load_tdms_file
from ldv_analysis.plotting import save_fig

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
DF_THRESHOLD = 500          # Hz — below this, beat is not resolvable

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# Signal processing functions
# =============================================================================


def smooth_envelope(wf, win=ENVELOPE_SMOOTH_WIN):
    env = np.abs(hilbert(wf))
    return np.convolve(env, np.ones(win) / win, mode="same")


def sliding_dft_envelope(wf, dt, f_target, win_us=SLIDING_DFT_WIN_US,
                         *, return_complex=False):
    """Sliding single-frequency DFT (lock-in detection at f_target).

    Parameters
    ----------
    wf : 1-D array
        Time-domain waveform.
    dt : float
        Sample interval (s).
    f_target : float
        Target frequency (Hz).
    win_us : float
        Sliding window width (µs).
    return_complex : bool
        If True, return the complex envelope; otherwise return magnitude.
    """
    n = len(wf)
    win_n = int(win_us * 1e-6 / dt)
    if win_n % 2 == 0:
        win_n += 1
    tone = np.exp(-2j * np.pi * f_target * np.arange(n) * dt)
    window = np.hanning(win_n)
    window /= window.sum()
    filtered = np.convolve(wf * tone, window, mode="same") * 2
    return filtered if return_complex else np.abs(filtered)


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


def tau_to_Q(f_drive, tau_us, harmonic=1):
    """Convert time constant (µs) to quality factor."""
    return np.pi * harmonic * f_drive * tau_us * 1e-6


# %%
# =============================================================================
# Transient fit model functions
# =============================================================================


def rise_simple(t, tau):
    """Simple exponential rise to 1 (for Ch2 acoustic, scaled by p_ss)."""
    return 1 - np.exp(-t / tau)


def rise_2f(t, tau2, tau1):
    """2f rise: first-order resonator driven by (1 - exp(-t/tau1))^2.

    Analytic solution of  tau2 * dp/dt + p = (1 - exp(-t/tau1))^2
    with p(0) = 0.  Returns normalised envelope (-> 1 at steady state).
    """
    a = tau2 / tau1
    if abs(1 - a) < 1e-6 or abs(1 - 2 * a) < 1e-6:
        return np.full_like(t, np.nan)
    c_tau2 = -2 * a**2 / ((1 - a) * (1 - 2 * a))
    c_tau1 = -2 / (1 - a)
    c_2tau1 = 1 / (1 - 2 * a)
    return 1 + c_tau2 * np.exp(-t / tau2) + c_tau1 * np.exp(-t / tau1) \
        + c_2tau1 * np.exp(-2 * t / tau1)


def fall_simple(t, tau):
    return np.exp(-t / tau)


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


def decay_beat_residual(params, t_us, E_re, E_im):
    """Decaying complex sinusoid: A*exp(-t/tau)*exp(j*(2pi*df*t + phi))."""
    A, tau, df, phi = params
    model = A * np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


def beat_model_complex(t_fine, tau, df, phi):
    """Complex beat model: exp(-t/tau) * exp(j*(2pi*df*t + phi))."""
    return np.exp(-t_fine / tau) * np.exp(
        1j * (2 * np.pi * df * t_fine * 1e-6 + phi))


# %%
# =============================================================================
# Plotting helper functions
# =============================================================================


def add_burst_markers(ax, burst_on_us, burst_off_us):
    """Draw vertical dashed lines at burst ON and OFF."""
    ax.axvline(burst_on_us, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(burst_off_us, color="gray", ls="--", lw=0.7, alpha=0.6)


def add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us):
    """Draw burst markers plus shaded rise/fall/FFT windows."""
    add_burst_markers(ax, burst_on_us, burst_off_us)
    ax.axvspan(burst_on_us, burst_on_us + RISE_FIT_WINDOW_US,
               alpha=0.08, color="green")
    ax.axvspan(burst_off_us, burst_off_us + FALL_FIT_WINDOW_US,
               alpha=0.08, color="red")
    ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue",
               label="FFT window")


def plot_no_data(ax):
    """Place 'No fall data' placeholder text in an axes."""
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")


def plot_phase_column(axes_col, t_us, burst_on_us, burst_off_us,
                      env_full, rise_t, rise_ec, fall_t, fall_ec,
                      title_prefix, *,
                      beat_model_rise=None, beat_model_fall=None):
    """Plot full / rise / fall phase for one harmonic (1f or 2f).

    Parameters
    ----------
    axes_col : array of 3 Axes
        The three row-axes for this column.
    t_us : 1-D array
        Full time axis (µs).
    burst_on_us, burst_off_us : float
        Burst boundaries (µs).
    env_full : 1-D complex array
        Full complex envelope for the overview panel.
    rise_t : 1-D array
        Time axis for rise window (µs, relative to skip).
    rise_ec : 1-D complex array
        Complex envelope in rise window.
    fall_t : 1-D array or None
        Time axis for fall window (µs).
    fall_ec : 1-D complex array or None
        Complex envelope in fall window.
    title_prefix : str
        E.g. "Ch2 1f" or "Ch2 2f".
    beat_model_rise, beat_model_fall : 1-D complex or None
        Beat model evaluated on linspace for overlay.
    """
    # Row 0: full phase
    ax = axes_col[0]
    ax.plot(t_us, np.degrees(np.angle(env_full)),
            linewidth=0.4, color="C0", alpha=0.7)
    add_burst_markers(ax, burst_on_us, burst_off_us)
    ax.set_ylabel(r"Phase [$^\circ$]")
    ax.set_xlabel(r"Time [\textmu s]")
    ax.set_title(f"{title_prefix} phase (avg)")
    ax.set_ylim(-200, 200)

    # Row 1: rise phase
    ax = axes_col[1]
    ax.plot(rise_t, np.degrees(np.angle(rise_ec)),
            "-", linewidth=0.5, color="C0", alpha=0.7)
    if beat_model_rise is not None:
        t_fine = np.linspace(0, rise_t[-1], 500)
        ax.plot(t_fine, np.degrees(np.angle(beat_model_rise)),
                "--", color="C3", linewidth=1.2)
    ax.set_ylabel(r"Phase [$^\circ$]")
    ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
    ax.set_title(f"{title_prefix} rise phase")
    ax.set_ylim(-200, 200)

    # Row 2: fall phase
    ax = axes_col[2]
    if fall_t is not None and fall_ec is not None:
        ax.plot(fall_t, np.degrees(np.angle(fall_ec)),
                "-", linewidth=0.5, color="C0", alpha=0.7)
        if beat_model_fall is not None:
            t_fine_f = np.linspace(0, fall_t[-1], 500)
            ax.plot(t_fine_f, np.degrees(np.angle(beat_model_fall)),
                    "--", color="C3", linewidth=1.2)
    else:
        plot_no_data(ax)
    ax.set_ylabel(r"Phase [$^\circ$]")
    ax.set_title(f"{title_prefix} fall phase")
    ax.set_ylim(-200, 200)


# %%
# #############################################################################
#                              DATA LOADING
# #############################################################################

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
# #############################################################################
#                          ENVELOPE COMPUTATION
# #############################################################################

ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
to_kPa = VELOCITY_SCALE / (2 * np.pi * f1 * SENSITIVITY) / 1e3
to_kPa_2f = VELOCITY_SCALE / (2 * np.pi * 2 * f1 * SENSITIVITY) / 1e3

env_cache_path = CACHE_DIR / f"_transient_env_{stem}.npz"
if env_cache_path.exists() and not _args.fresh:
    print(f"  Loading envelope cache: {env_cache_path.name}")
    _ec = np.load(env_cache_path)
    env_ch2_norm_complex = _ec["env_1f_complex"]
    env_ch2_2f_norm_complex = _ec["env_2f_complex"]
    env_ch2_kPa = _ec["best_1f_kPa"]
    env_ch2_2f_best_kPa = _ec["best_2f_kPa"]
    env_ch4_norm_complex = _ec["env_ch4_complex"] if "env_ch4_complex" in _ec else None
    i_ss_ch4 = float(_ec["i_ss_ch4"]) if "i_ss_ch4" in _ec else None
    env_ch1 = _ec["best_ch1"]
    n_used = int(_ec["n_used"])
    p_ss = float(_ec["p_ss"])
    p_ss_2f = float(_ec["p_ss_2f"])
    dt = float(_ec["dt"])
    del _ec
else:
    print(f"  Computing averaged Ch2+Ch4 envelopes (per-point)...")
    tdms_file, _ = load_tdms_file(tdms_path)
    wf_group = tdms_file["Waveforms"]
    ch2_channels = [ch for ch in wf_group.channels()
                    if ch.name.startswith("WFCh2")]
    ch4_channels = [ch for ch in wf_group.channels()
                    if ch.name.startswith("WFCh4")]
    dt = ch2_channels[0].properties.get("wf_increment", 8e-9)

    env_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_2f_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_ch4_complex_wsum = np.zeros(n_samples, dtype=complex)
    weight_sum = 0.0
    n_used = 0
    for idx in np.where(valid)[0]:
        wf = ch2_channels[idx][:]
        env_c = sliding_dft_envelope(wf, dt, f1, return_complex=True) * to_kPa
        env_2f_c = sliding_dft_envelope(wf, dt, 2 * f1, return_complex=True) * to_kPa_2f
        p_ss_c = np.mean(env_c[ss_start:ss_end])
        p_ss_mag = np.abs(p_ss_c)
        if p_ss_mag > 0:
            w = p_ss_mag
            env_complex_wsum += w * (env_c / p_ss_c)
            p_ss_2f_c = np.mean(env_2f_c[ss_start:ss_end])
            p_ss_2f_mag = np.abs(p_ss_2f_c)
            if p_ss_2f_mag > 0:
                env_2f_complex_wsum += w * (env_2f_c / p_ss_2f_c)
            # Ch4: same weighting by Ch2 p_ss
            wf4 = ch4_channels[idx][:]
            env_ch4_c = sliding_dft_envelope(
                wf4, dt, f1, return_complex=True) * CURRENT_SCALE * 1e3
            env_ch4_complex_wsum += env_ch4_c
            weight_sum += w
            n_used += 1
    env_ch2_norm_complex = env_complex_wsum / weight_sum
    env_ch2_2f_norm_complex = env_2f_complex_wsum / weight_sum
    # Ch4: plain average (current is spatially uniform)
    env_ch4_avg = env_ch4_complex_wsum / n_used  # mA, complex
    _ch4_ss = np.mean(env_ch4_avg[ss_start:ss_end])
    env_ch4_norm_complex = env_ch4_avg / _ch4_ss
    i_ss_ch4 = float(np.abs(_ch4_ss))
    print(f"  Averaged {n_used} normalised Ch2+Ch4 envelopes (p_ss-weighted)")

    # Best-point envelopes (for display in kPa)
    wfs_best, _ = load_point_waveforms(tdms_path, best_i, channels=(1, 2))
    env_ch2_kPa = sliding_dft_envelope(wfs_best[2], dt, f1) * to_kPa
    p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))
    env_ch2_2f_best_kPa = sliding_dft_envelope(wfs_best[2], dt, 2 * f1) * to_kPa_2f
    p_ss_2f = float(np.mean(env_ch2_2f_best_kPa[ss_start:ss_end]))
    env_ch1 = smooth_envelope(wfs_best[1])
    del tdms_file

    np.savez(env_cache_path,
             env_1f_complex=env_ch2_norm_complex,
             env_2f_complex=env_ch2_2f_norm_complex,
             env_ch4_complex=env_ch4_norm_complex,
             i_ss_ch4=i_ss_ch4,
             best_1f_kPa=env_ch2_kPa, best_2f_kPa=env_ch2_2f_best_kPa,
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
# #############################################################################
#                          TRANSIENT FITTING
# #############################################################################

rise_n = int(RISE_FIT_WINDOW_US * 1e-6 / dt)
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))

skip_n = int(FIT_SKIP_US * 1e-6 / dt)
rise_start = burst_on + skip_n
rise_end = burst_on + int(RISE_FIT_WINDOW_US * 1e-6 / dt)
fall_start = burst_off + skip_n
end_f = min(burst_off + int(FALL_FIT_WINDOW_US * 1e-6 / dt), n_samples)
has_fall = fall_start < end_f - 10

# --- Ch2 1f: exponential + beat ---

ch2_rise = {
    "t": np.arange(rise_end - rise_start) * dt * 1e6,
    "e": np.abs(env_ch2_norm_complex)[rise_start:rise_end],
    "ec": env_ch2_norm_complex[rise_start:rise_end],
}

ch2_rise["po"], ch2_rise["pcov"] = curve_fit(
    rise_simple, ch2_rise["t"], ch2_rise["e"],
    p0=[10], bounds=([0.1], [500]))

_rise_res = ch2_rise["ec"] - 1.0
_df_guess_r = estimate_beat_freq(_rise_res, ch2_rise["t"][1] - ch2_rise["t"][0])
ch2_rise["beat"] = least_squares(
    rise_beat_residual,
    x0=[ch2_rise["po"][0], _df_guess_r, 0.0],
    args=(ch2_rise["t"], ch2_rise["ec"].real, ch2_rise["ec"].imag),
    bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
    method="trf",
)

if has_fall:
    ch2_fall = {
        "t": np.arange(end_f - fall_start) * dt * 1e6,
        "e": np.abs(env_ch2_norm_complex)[fall_start:end_f],
        "ec": env_ch2_norm_complex[fall_start:end_f],
    }
    ch2_fall["po"], ch2_fall["pcov"] = curve_fit(
        fall_simple, ch2_fall["t"], ch2_fall["e"],
        p0=[10], bounds=([0.1], [500]))

    _df_guess_f = estimate_beat_freq(ch2_fall["ec"], ch2_fall["t"][1] - ch2_fall["t"][0])
    ch2_fall["beat"] = least_squares(
        fall_beat_residual,
        x0=[ch2_fall["po"][0], _df_guess_f, 0.0],
        args=(ch2_fall["t"], ch2_fall["ec"].real, ch2_fall["ec"].imag),
        bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
        method="trf",
    )
else:
    ch2_fall = None
    print("  WARNING: no fall data (burst extends to end of recording)")

# --- Ch2 2f: driven resonator (source ~ p1^2) ---

tau1_fixed = ch2_rise["po"][0]  # µs, from 1f simple fit
ch2_2f_rise = {
    "t": ch2_rise["t"],
    "e": np.abs(env_ch2_2f_norm_complex)[rise_start:rise_end],
}
_pre_burst = np.abs(env_ch2_2f_norm_complex)[max(0, burst_on - 100):burst_on]
noise_floor_2f = float(np.mean(_pre_burst)) if len(_pre_burst) > 0 else 0.0

ch2_2f_rise["po"], ch2_2f_rise["pcov"] = curve_fit(
    lambda t, tau2, A, b: b + A * rise_2f(t, tau2, tau1_fixed),
    ch2_2f_rise["t"], ch2_2f_rise["e"],
    p0=[tau1_fixed * 0.3, 1.0 - noise_floor_2f, noise_floor_2f],
    bounds=([0.1, 0, 0], [500, 5, 1]))
tau_2f, A_2f, baseline_2f = ch2_2f_rise["po"]
tau_2f_err = np.sqrt(ch2_2f_rise["pcov"][0, 0])

# --- Ch4: complex beat fit (averaged sliding DFT, skip first 5 µs) ---

if env_ch4_norm_complex is not None and i_ss_ch4 is not None:
    _ch4_norm_c = env_ch4_norm_complex
    I_ss_ch4 = i_ss_ch4

    ch4_rise_beat = {
        "t": np.arange(rise_end - rise_start) * dt * 1e6,
        "ec": _ch4_norm_c[rise_start:rise_end],
    }
    _dev_r = ch4_rise_beat["ec"] - 1.0
    _A0_r = float(np.abs(_dev_r[0]))
    _df_guess_r = estimate_beat_freq(
        _dev_r, ch4_rise_beat["t"][1] - ch4_rise_beat["t"][0])
    ch4_rise_beat["beat"] = least_squares(
        decay_beat_residual,
        x0=[_A0_r, 10.0, _df_guess_r, 0.0],
        args=(ch4_rise_beat["t"], _dev_r.real, _dev_r.imag),
        bounds=([0, 0.1, -1e6, -np.pi], [10, 500, 1e6, np.pi]),
        method="trf",
    )

    if has_fall:
        ch4_fall_beat = {
            "t": np.arange(end_f - fall_start) * dt * 1e6,
            "ec": _ch4_norm_c[fall_start:end_f],
        }
        _A0_f = float(np.abs(ch4_fall_beat["ec"][0]))
        _df_guess_f = estimate_beat_freq(
            ch4_fall_beat["ec"],
            ch4_fall_beat["t"][1] - ch4_fall_beat["t"][0])
        ch4_fall_beat["beat"] = least_squares(
            decay_beat_residual,
            x0=[_A0_f, 10.0, _df_guess_f, 0.0],
            args=(ch4_fall_beat["t"],
                  ch4_fall_beat["ec"].real, ch4_fall_beat["ec"].imag),
            bounds=([0, 0.1, -1e6, -np.pi], [10, 500, 1e6, np.pi]),
            method="trf",
        )
    else:
        ch4_fall_beat = None
else:
    ch4_rise_beat = None
    ch4_fall_beat = None
    print("  NOTE: no Ch4 complex envelope in cache; run with --fresh")

# --- Extract key fit parameters for reporting and plotting ---

tau_ch2 = ch2_rise["po"][0]
tau_ch2_err = np.sqrt(ch2_rise["pcov"][0, 0])

tau_beat_r, df_beat_r, phi_beat_r = ch2_rise["beat"].x
beat_sig_r = abs(df_beat_r) > DF_THRESHOLD
if has_fall:
    tau_beat_f, df_beat_f, phi_beat_f = ch2_fall["beat"].x
    beat_sig_f = abs(df_beat_f) > DF_THRESHOLD
    tau_ch2_f = ch2_fall["po"][0]

if ch4_rise_beat is not None:
    A_ch4b_r, tau_ch4b_r, df_ch4b_r, phi_ch4b_r = ch4_rise_beat["beat"].x
    ch4_beat_sig_r = abs(df_ch4b_r) > DF_THRESHOLD
if ch4_fall_beat is not None:
    A_ch4b_f, tau_ch4b_f, df_ch4b_f, phi_ch4b_f = ch4_fall_beat["beat"].x
    ch4_beat_sig_f = abs(df_ch4b_f) > DF_THRESHOLD


# %%
# #############################################################################
#                          RESULTS SUMMARY
# #############################################################################

print(f"\n--- Ch2 (acoustic) ---")
print(f"  p_ss_1f (mean in FFT window) = {p_ss:.0f} kPa")
print(f"  p_ss_2f (mean in FFT window) = {p_ss_2f:.0f} kPa")
print(f"  Rise (simple):  tau = {tau_ch2:.2f} +/- {tau_ch2_err:.2f} us"
      f"  ->  Q = {tau_to_Q(f1, tau_ch2):.0f}")
print(f"  Rise (beat):    tau = {tau_beat_r:.2f} us"
      f"  ->  Q = {tau_to_Q(f1, tau_beat_r):.0f}"
      f"  df = {df_beat_r:.0f} Hz" + ("" if beat_sig_r else "  (not significant)"))
if has_fall:
    print(f"  Fall (simple):  tau = {tau_ch2_f:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_ch2_f):.0f}")
    print(f"  Fall (beat):    tau = {tau_beat_f:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_beat_f):.0f}"
          f"  df = {df_beat_f:.0f} Hz" + ("" if beat_sig_f else "  (not significant)"))
    if beat_sig_r or beat_sig_f:
        df_avg = np.mean([abs(df_beat_r), abs(df_beat_f)])
        print(f"  Mode splitting: {df_avg/1e3:.1f} kHz")
else:
    print(f"  Fall: no data")

print(f"\n--- Ch2 2f (driven resonator, source ~ p1^2) ---")
print(f"  tau_1 (fixed from 1f) = {tau1_fixed:.2f} us")
print(f"  tau_2 = {tau_2f:.2f} +/- {tau_2f_err:.2f} us"
      f"  ->  Q_2f = {tau_to_Q(f1, tau_2f, harmonic=2):.0f}")
print(f"  baseline = {baseline_2f:.3f}, A = {A_2f:.3f}")

print(f"\n--- Ch4 (current, complex beat) ---")
print(f"  I_ss = {I_ss_ch4:.2f} mA" if i_ss_ch4 is not None else "  (no data)")
if ch4_rise_beat is not None:
    print(f"  Rise (beat):  tau = {tau_ch4b_r:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_ch4b_r):.0f}"
          f"  df = {df_ch4b_r:.0f} Hz, A = {A_ch4b_r:.4f}"
          + ("" if ch4_beat_sig_r else "  (not significant)"))
    if ch4_fall_beat is not None:
        print(f"  Fall (beat):  tau = {tau_ch4b_f:.2f} us"
              f"  ->  Q = {tau_to_Q(f1, tau_ch4b_f):.0f}"
              f"  df = {df_ch4b_f:.0f} Hz, A = {A_ch4b_f:.4f}"
              + ("" if ch4_beat_sig_f else "  (not significant)"))


# %%
# #############################################################################
#                     PRECOMPUTE MODEL CURVES FOR PLOTTING
# #############################################################################

# Ch2 1f rise
t_fine_rise = np.linspace(0, ch2_rise["t"][-1], 500)
curve_ch2_rise_simple = rise_simple(t_fine_rise, *ch2_rise["po"])
curve_ch2_rise_beat = 1.0 - beat_model_complex(
    t_fine_rise, tau_beat_r, df_beat_r, phi_beat_r)

# Ch2 1f fall
if has_fall:
    t_fine_fall = np.linspace(0, ch2_fall["t"][-1], 500)
    curve_ch2_fall_simple = fall_simple(t_fine_fall, *ch2_fall["po"])
    curve_ch2_fall_beat = beat_model_complex(
        t_fine_fall, tau_beat_f, df_beat_f, phi_beat_f)

# Ch2 2f rise
t_fine_2f = np.linspace(0, ch2_rise["t"][-1], 500)
curve_2f_rise_model = baseline_2f + A_2f * rise_2f(t_fine_2f, tau_2f, tau1_fixed)
curve_2f_rise_source = A_2f * rise_simple(t_fine_2f, tau1_fixed)**2

# Ch4 beat curves (amplitude + phase)
curve_ch4_rise_beat_c = None
curve_ch4_fall_beat_c = None
if ch4_rise_beat is not None:
    t_fine_ch4_rise_beat = np.linspace(0, ch4_rise_beat["t"][-1], 500)
    curve_ch4_rise_beat_c = (
        1.0 + A_ch4b_r * beat_model_complex(
            t_fine_ch4_rise_beat, tau_ch4b_r, df_ch4b_r, phi_ch4b_r))
    curve_ch4_rise_beat = np.real(curve_ch4_rise_beat_c) * I_ss_ch4
if has_fall and ch4_fall_beat is not None:
    t_fine_ch4_fall_beat = np.linspace(0, ch4_fall_beat["t"][-1], 500)
    curve_ch4_fall_beat_c = (
        A_ch4b_f * beat_model_complex(
            t_fine_ch4_fall_beat, tau_ch4b_f, df_ch4b_f, phi_ch4b_f))
    curve_ch4_fall_beat = np.real(curve_ch4_fall_beat_c) * I_ss_ch4


# %%
# #############################################################################
#                            VISUALIZATION
# #############################################################################

fig, axes = plt.subplots(3, 6, figsize=figsize_for_layout(3, 6))

ss_start_us = int(cache["ss_start"]) * dt * 1e6
ss_end_us = int(cache["ss_end"]) * dt * 1e6
burst_on_us = burst_on * dt * 1e6
burst_off_us = burst_off * dt * 1e6

# --- Col 0: Ch2 1f amplitude ---

ax = axes[0, 0]
ax.plot(t_us, env_ch2_kPa, linewidth=0.4, color="C0", alpha=0.5, label="best point")
ax.plot(t_us, np.abs(env_ch2_norm_complex) * p_ss, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 1f envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 0]
ax.plot(ch2_rise["t"], ch2_rise["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
ax.plot(t_fine_rise, curve_ch2_rise_simple, "--", color="0.6",
        linewidth=0.8, alpha=0.7,
        label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
        % (tau_ch2, tau_to_Q(f1, tau_ch2)))
ax.plot(t_fine_rise, np.real(curve_ch2_rise_beat), "--", color="C3", linewidth=1.2,
        label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
        % (tau_beat_r, tau_to_Q(f1, tau_beat_r), df_beat_r / 1e3))
ax.set_ylabel(r"Normalised $P / P_{ss}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title(f"Ch2 1f ring-up (avg {n_used} pts)")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 0]
if has_fall:
    ax.plot(ch2_fall["t"], ch2_fall["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
    ax.plot(t_fine_fall, curve_ch2_fall_simple, "--", color="0.6",
            linewidth=0.8, alpha=0.7,
            label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
            % (tau_ch2_f, tau_to_Q(f1, tau_ch2_f)))
    ax.plot(t_fine_fall, np.real(curve_ch2_fall_beat), "--", color="C3", linewidth=1.2,
            label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
            % (tau_beat_f, tau_to_Q(f1, tau_beat_f), df_beat_f / 1e3))
    ax.legend(fontsize=5, frameon=False)
else:
    plot_no_data(ax)
ax.set_ylabel(r"Normalised $P / P_{ss}$")
ax.set_title("Ch2 1f ring-down")

# --- Col 1: Ch2 1f phase ---

plot_phase_column(
    axes[:, 1], t_us, burst_on_us, burst_off_us,
    env_ch2_norm_complex,
    ch2_rise["t"], ch2_rise["ec"],
    ch2_fall["t"] if has_fall else None,
    ch2_fall["ec"] if has_fall else None,
    "Ch2 1f",
    beat_model_rise=curve_ch2_rise_beat,
    beat_model_fall=curve_ch2_fall_beat if has_fall else None)

# --- Col 2: Ch2 2f amplitude ---

ax = axes[0, 2]
ax.plot(t_us, env_ch2_2f_best_kPa, linewidth=0.4, color="C0", alpha=0.5,
        label="best point")
ax.plot(t_us, np.abs(env_ch2_2f_norm_complex) * p_ss_2f, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 2f envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 2]
ax.plot(ch2_rise["t"], np.abs(env_ch2_2f_norm_complex)[rise_start:rise_end],
        "-", linewidth=0.5, color="C0", alpha=0.7)
ax.plot(t_fine_2f, curve_2f_rise_model, "--", color="C3", linewidth=1.2,
        label=r"$\tau_2$ = %.1f \textmu s ($Q_{2f}$ = %d)"
        % (tau_2f, tau_to_Q(f1, tau_2f, harmonic=2)))
ax.plot(t_fine_2f, curve_2f_rise_source, ":", color="0.6", linewidth=0.8, alpha=0.7,
        label=r"source $(1-e^{-t/\tau_1})^2$")
ax.set_ylabel(r"Normalised $P_{2f} / P_{ss,2f}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title("Ch2 2f ring-up")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 2]
if has_fall:
    ax.plot(ch2_fall["t"], np.abs(env_ch2_2f_norm_complex)[fall_start:end_f],
            "-", linewidth=0.5, color="C0", alpha=0.7)
else:
    plot_no_data(ax)
ax.set_ylabel(r"Normalised $P_{2f} / P_{ss,2f}$")
ax.set_title("Ch2 2f ring-down")

# --- Col 3: Ch2 2f phase ---

plot_phase_column(
    axes[:, 3], t_us, burst_on_us, burst_off_us,
    env_ch2_2f_norm_complex,
    ch2_rise["t"], env_ch2_2f_norm_complex[rise_start:rise_end],
    ch2_fall["t"] if has_fall else None,
    env_ch2_2f_norm_complex[fall_start:end_f] if has_fall else None,
    "Ch2 2f")

# --- Col 4: Ch4 current ---

ax = axes[0, 4]
if env_ch4_norm_complex is not None and i_ss_ch4 is not None:
    ax.plot(t_us, np.abs(env_ch4_norm_complex) * i_ss_ch4, linewidth=0.6, color="C1",
            label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch4 (current) envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 4]
if ch4_rise_beat is not None:
    ax.plot(ch4_rise_beat["t"] + FIT_SKIP_US,
            np.abs(ch4_rise_beat["ec"]) * I_ss_ch4,
            "-", linewidth=0.5, color="C1", alpha=0.7)
    ax.plot(t_fine_ch4_rise_beat + FIT_SKIP_US, curve_ch4_rise_beat,
            "--", color="C3", linewidth=1.2,
            label=(r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
                   % (tau_ch4b_r, tau_to_Q(f1, tau_ch4b_r), df_ch4b_r / 1e3)))
    ax.legend(fontsize=5, frameon=False)
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time from burst ON [\textmu s]")
ax.set_title("Ch4 ring-up (beat)")

ax = axes[2, 4]
if has_fall and ch4_fall_beat is not None:
    ax.plot(ch4_fall_beat["t"] + FIT_SKIP_US,
            np.abs(ch4_fall_beat["ec"]) * I_ss_ch4,
            "-", linewidth=0.5, color="C1", alpha=0.7)
    ax.plot(t_fine_ch4_fall_beat + FIT_SKIP_US, curve_ch4_fall_beat,
            "--", color="C3", linewidth=1.2,
            label=(r"beat: $\tau$ = %.1f \textmu s, $\Delta f$ = %.1f kHz"
                   % (tau_ch4b_f, df_ch4b_f / 1e3)))
    ax.legend(fontsize=5, frameon=False)
elif not has_fall:
    plot_no_data(ax)
ax.set_ylabel("Current [mA]")
ax.set_title("Ch4 ring-down (beat)")

# --- Col 5: Ch4 phase ---

if env_ch4_norm_complex is not None:
    plot_phase_column(
        axes[:, 5], t_us, burst_on_us, burst_off_us,
        _ch4_norm_c,
        ch4_rise_beat["t"] if ch4_rise_beat is not None else ch2_rise["t"],
        ch4_rise_beat["ec"] if ch4_rise_beat is not None else None,
        ch4_fall_beat["t"] if ch4_fall_beat is not None else None,
        ch4_fall_beat["ec"] if ch4_fall_beat is not None else None,
        "Ch4",
        beat_model_rise=curve_ch4_rise_beat_c,
        beat_model_fall=curve_ch4_fall_beat_c)
else:
    for row in range(3):
        plot_no_data(axes[row, 5])
        axes[row, 5].set_title("Ch4 phase" if row == 0 else "")

plt.tight_layout()
save_fig(fig, f"transient_fit_{stem}", OUT_DIR)

# %%
print(f"\n=== Done ===")
