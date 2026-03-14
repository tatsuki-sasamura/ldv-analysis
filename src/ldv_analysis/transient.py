"""Shared utilities for transient ring-up/ring-down analysis.

Constants, signal processing, fit models, plot helpers, and data loading
used by both transient_ch2_acoustic.py and transient_ch4_current.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from ldv_analysis.config import get_data_dir
from ldv_analysis.fft_cache import load_or_compute

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TDMS = get_data_dir("20260303experimentA") / "stepA1967.tdms"
ENVELOPE_SMOOTH_WIN = 63
SLIDING_DFT_WIN_US = 5.0
FIT_SKIP_US = 5.0
RISE_FIT_WINDOW_US = 100.0
FALL_FIT_WINDOW_US = 100.0
DF_THRESHOLD = 500


# =============================================================================
# Signal processing
# =============================================================================

def smooth_envelope(wf, win=ENVELOPE_SMOOTH_WIN):
    env = np.abs(hilbert(wf))
    return np.convolve(env, np.ones(win) / win, mode="same")


def sliding_dft_envelope(wf, dt, f_target, win_us=SLIDING_DFT_WIN_US,
                         *, return_complex=False):
    """Sliding single-frequency DFT (lock-in detection at f_target)."""
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
    """Estimate beat frequency from FFT of complex transient residual."""
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


# =============================================================================
# Fit models
# =============================================================================

def rise_simple(t, tau):
    return 1 - np.exp(-t / tau)


def rise_2f(t, tau2, tau1):
    """2f rise: first-order resonator driven by (1 - exp(-t/tau1))^2."""
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
    tau, df, phi = params
    model = 1.0 - np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


def fall_beat_residual(params, t_us, E_re, E_im):
    tau, df, phi = params
    model = np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


def decay_beat_residual(params, t_us, E_re, E_im):
    """A*exp(-t/tau)*exp(j*(2pi*df*t + phi))."""
    A, tau, df, phi = params
    model = A * np.exp(-t_us / tau) * np.exp(
        1j * (2 * np.pi * df * t_us * 1e-6 + phi))
    return np.concatenate([model.real - E_re, model.imag - E_im])


def beat_model_complex(t_fine, tau, df, phi):
    return np.exp(-t_fine / tau) * np.exp(
        1j * (2 * np.pi * df * t_fine * 1e-6 + phi))


# =============================================================================
# Plot helpers
# =============================================================================

def add_burst_markers(ax, burst_on_us, burst_off_us):
    ax.axvline(burst_on_us, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.axvline(burst_off_us, color="gray", ls="--", lw=0.7, alpha=0.6)


def add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us):
    add_burst_markers(ax, burst_on_us, burst_off_us)
    ax.axvspan(burst_on_us, burst_on_us + RISE_FIT_WINDOW_US,
               alpha=0.08, color="green")
    ax.axvspan(burst_off_us, burst_off_us + FALL_FIT_WINDOW_US,
               alpha=0.08, color="red")
    ax.axvspan(ss_start_us, ss_end_us, alpha=0.10, color="blue",
               label="FFT window")


def plot_no_data(ax):
    ax.text(0.5, 0.5, "No fall data", transform=ax.transAxes, ha="center")


def plot_phase_column(axes_col, t_us, burst_on_us, burst_off_us,
                      env_full, rise_t, rise_ec, fall_t, fall_ec,
                      title_prefix, *,
                      beat_model_rise=None, beat_model_fall=None):
    """Plot full / rise / fall phase for one harmonic."""
    ax = axes_col[0]
    ax.plot(t_us, np.degrees(np.angle(env_full)),
            linewidth=0.4, color="C0", alpha=0.7)
    add_burst_markers(ax, burst_on_us, burst_off_us)
    ax.set_ylabel(r"Phase [$^\circ$]")
    ax.set_xlabel(r"Time [\textmu s]")
    ax.set_title(f"{title_prefix} phase (avg)")
    ax.set_ylim(-200, 200)

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


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class TransientData:
    """Common data loaded from FFT cache for transient analysis."""
    tdms_path: Path
    stem: str
    cache: dict
    f1: float
    n_samples: int
    t_us: np.ndarray
    dt: float
    ss_start: int
    ss_end: int
    valid: np.ndarray
    best_i: int
    n_valid: int


def load_transient_data(tdms_path, cache_dir):
    """Load FFT cache and compute valid mask."""
    cache = load_or_compute(tdms_path, cache_dir)
    vel = cache["velocity_1f"]
    pressure = cache["pressure_1f"]
    rssi = cache["rssi"] if "rssi" in cache else np.ones_like(vel)
    valid = ((rssi > 1.0)
             & (pressure > np.median(pressure[rssi > 1.0])
                if (rssi > 1.0).any() else True))
    best_i = (int(np.where(valid)[0][np.argmax(vel[valid])])
              if valid.any() else int(np.argmax(vel)))

    f1 = float(cache["f_drive"])
    n_samples = int(cache["n_samples"])
    dt_val = float(cache["dt"])

    td = TransientData(
        tdms_path=tdms_path, stem=tdms_path.stem, cache=cache,
        f1=f1, n_samples=n_samples,
        t_us=np.arange(n_samples) * dt_val * 1e6,
        dt=dt_val,
        ss_start=int(cache["ss_start"]), ss_end=int(cache["ss_end"]),
        valid=valid, best_i=best_i, n_valid=int(valid.sum()),
    )
    print(f"Loading: {tdms_path.name}")
    print(f"  Valid points: {td.n_valid} / {len(vel)}")
    print(f"  Strongest valid point: {best_i} (RSSI = {rssi[best_i]:.2f} V)")
    print(f"  Drive frequency: {f1 / 1e6:.4f} MHz")
    return td


def detect_burst(env_ch1, dt):
    """Detect burst ON/OFF from Ch1 Hilbert envelope. Returns (burst_on, burst_off) indices."""
    on_mask = env_ch1 > 0.5 * np.max(env_ch1)
    on_idx = np.where(on_mask)[0]
    return int(on_idx[0]), int(on_idx[-1])


def compute_fit_windows(burst_on, burst_off, dt, n_samples):
    """Compute rise/fall sample windows. Returns dict."""
    skip_n = int(FIT_SKIP_US * 1e-6 / dt)
    rise_start = burst_on + skip_n
    rise_end = burst_on + int(RISE_FIT_WINDOW_US * 1e-6 / dt)
    fall_start = burst_off + skip_n
    end_f = min(burst_off + int(FALL_FIT_WINDOW_US * 1e-6 / dt), n_samples)
    has_fall = fall_start < end_f - 10
    return dict(
        skip_n=skip_n, rise_start=rise_start, rise_end=rise_end,
        fall_start=fall_start, end_f=end_f, has_fall=has_fall,
    )
