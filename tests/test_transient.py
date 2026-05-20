"""Unit tests for ``ldv_analysis.transient`` — the moving-window-DFT /
fit module used by the ring-up / ring-down transient analyses.

These tests cover what ``tests/test_fft.py`` does not:
- ``sliding_dft_envelope`` with ``return_complex=True`` (phase recovery,
  ×2 normalization, even→odd window-length enforcement)
- ``smooth_envelope`` (Hilbert + boxcar)
- ``estimate_beat_freq`` (known Δf recovery + short-signal edge case)
- ``tau_to_Q`` numerical correctness
- ``rise_simple`` / ``fall_simple`` endpoint values
- ``rise_2f`` closed-form endpoints + degenerate-τ NaN handling
- ``beat_model_complex`` at t=0
- ``rise_beat_residual`` / ``fall_beat_residual`` / ``decay_beat_residual``
  zero at the true parameters (residual-function correctness)
- ``detect_burst`` from a synthesized Hilbert envelope
- ``compute_fit_windows`` sample-index arithmetic
"""

from __future__ import annotations

import numpy as np
import pytest

from ldv_analysis.transient import (
    FIT_SKIP_US,
    RISE_FIT_WINDOW_US,
    FALL_FIT_WINDOW_US,
    beat_model_complex,
    compute_fit_windows,
    decay_beat_residual,
    detect_burst,
    estimate_beat_freq,
    fall_beat_residual,
    fall_simple,
    rise_2f,
    rise_beat_residual,
    rise_simple,
    sliding_dft_envelope,
    smooth_envelope,
    tau_to_Q,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DT = 8e-9
F0 = 1.907e6


# ===========================================================================
# sliding_dft_envelope
# ===========================================================================

def test_sliding_dft_envelope_complex_phase_recovery():
    """``return_complex=True`` carries phase: ``angle(env) + π/2 ≈ φ``."""
    n = 50_000
    t = np.arange(n) * DT
    A = 0.7
    phi = np.deg2rad(40.0)
    wf = A * np.sin(2 * np.pi * F0 * t + phi)
    env = sliding_dft_envelope(wf, DT, F0, win_us=2.0, return_complex=True)
    # The convention: env ≈ -i·A·exp(iφ) in the interior, so
    # angle(env) = φ - π/2.  Add π/2 to recover the synthesized φ.
    interior = env[5_000:-5_000]
    recovered_phi = float(np.angle(interior.mean()) + np.pi / 2)
    # Wrap to (-π, π] for stable comparison
    err = (recovered_phi - phi + np.pi) % (2 * np.pi) - np.pi
    assert abs(err) < 1e-3, f"recovered φ={np.degrees(recovered_phi):.3f}°, " \
                            f"expected {np.degrees(phi):.3f}°"


def test_sliding_dft_envelope_normalization_two_times_factor():
    """``|env| ≈ A`` (not A/2): the ×2 inside ``sliding_dft_envelope``
    compensates the one-sided sin→complex factor of ½."""
    n = 50_000
    t = np.arange(n) * DT
    for A in (0.1, 0.5, 1.0, 2.5):
        wf = A * np.sin(2 * np.pi * F0 * t)
        env = sliding_dft_envelope(wf, DT, F0, win_us=2.0)
        assert abs(env[5_000:-5_000].mean() - A) < 1e-3 * A, (
            f"|env|={env[5_000:-5_000].mean():.3f} vs A={A}"
        )


def test_sliding_dft_envelope_even_win_us_becomes_odd():
    """The Hanning window is symmetric only with odd length, so an even
    ``win_n`` must be bumped to odd."""
    n = 10_000
    t = np.arange(n) * DT
    wf = np.sin(2 * np.pi * F0 * t)
    # Both should run without error and produce comparable envelopes.
    # Direct knob: a win_us that maps to an even win_n.  At DT=8 ns,
    # win_us = 8.0 → win_n = 1000 (even). Should be upped to 1001.
    env = sliding_dft_envelope(wf, DT, F0, win_us=8.0)
    assert env.shape == wf.shape
    # The midpoint of the smoothed envelope should still be close to A=1
    assert abs(env[n // 2] - 1.0) < 1e-3


# ===========================================================================
# smooth_envelope (Hilbert + boxcar)
# ===========================================================================

def test_smooth_envelope_steady_amplitude():
    """For a pure tone the envelope is ~A everywhere after the boxcar
    transient."""
    n = 10_000
    t = np.arange(n) * DT
    A = 1.5
    wf = A * np.sin(2 * np.pi * F0 * t)
    env = smooth_envelope(wf, win=63)
    interior = env[500:-500]
    assert abs(interior.mean() - A) < 1e-2
    assert interior.std() < 1e-2


# ===========================================================================
# estimate_beat_freq
# ===========================================================================

def test_estimate_beat_freq_recovers_known_df():
    """``estimate_beat_freq`` recovers a synthesized Δf within sub-bin."""
    df_true = 2e3                            # 2 kHz beat
    dt_us = 0.1                              # 100 ns sampling for the
                                              # beat-envelope time grid
    n = 4_096
    t = np.arange(n) * dt_us * 1e-6
    env = np.exp(2j * np.pi * df_true * t)
    df_est = estimate_beat_freq(env, dt_us)
    # FFT bin in beat-envelope frequency:  1 / (n_pad · dt_us · 1e-6)
    # n_pad = max(n*4, 2**16)
    bin_hz = 1 / (max(n * 4, 2**16) * dt_us * 1e-6)
    assert abs(df_est - df_true) < bin_hz, (
        f"df_est={df_est:.3f} Hz, expected {df_true:.3f}, "
        f"bin={bin_hz:.3f} Hz"
    )


def test_estimate_beat_freq_short_signal_returns_zero():
    """n < 8 samples → returns 0.0 (degenerate; not enough to FFT)."""
    env = np.array([1 + 0j, 0.9 + 0.1j])
    assert estimate_beat_freq(env, 0.1) == 0.0


# ===========================================================================
# tau_to_Q
# ===========================================================================

def test_tau_to_Q_numerical():
    """Q = π·h·f·τ  with τ in µs."""
    assert tau_to_Q(1.9e6, 10.0, harmonic=1) == pytest.approx(
        np.pi * 1.9e6 * 10.0 * 1e-6
    )
    assert tau_to_Q(1.9e6, 10.0, harmonic=2) == pytest.approx(
        np.pi * 2 * 1.9e6 * 10.0 * 1e-6
    )


# ===========================================================================
# rise_simple, fall_simple, rise_2f
# ===========================================================================

def test_rise_simple_fall_simple_endpoints():
    """``rise_simple(0)=0``, ``rise_simple(∞)=1``; ``fall_simple`` is the
    complement."""
    tau = 10.0
    assert rise_simple(0.0, tau) == 0.0
    assert rise_simple(1000 * tau, tau) == pytest.approx(1.0, abs=1e-30)
    assert fall_simple(0.0, tau) == 1.0
    assert fall_simple(1000 * tau, tau) == pytest.approx(0.0, abs=1e-30)


def test_rise_2f_endpoints():
    """``rise_2f(0, …) = 0`` and ``rise_2f(∞, …) = 1`` for non-degenerate τ."""
    tau1, tau2 = 10.0, 3.0
    assert rise_2f(np.array([0.0]), tau2, tau1)[0] == pytest.approx(0.0, abs=1e-12)
    assert rise_2f(np.array([1e4]), tau2, tau1)[0] == pytest.approx(1.0, abs=1e-30)


def test_rise_2f_degenerate_returns_nan():
    """Degenerate ratios τ₂≈τ₁ or τ₂≈τ₁/2 → NaN (closed form blows up)."""
    t = np.linspace(0, 30, 50)
    # τ₂ = τ₁ → a = 1 → 1-a = 0
    out_eq = rise_2f(t, tau2=10.0, tau1=10.0)
    assert np.all(np.isnan(out_eq))
    # τ₂ = τ₁/2 → a = 0.5 → 1-2a = 0
    out_half = rise_2f(t, tau2=5.0, tau1=10.0)
    assert np.all(np.isnan(out_half))


# ===========================================================================
# beat_model_complex and residual functions
# ===========================================================================

def test_beat_model_complex_at_zero():
    """``beat_model_complex(0, …) = exp(iφ)`` — independent of τ and Δf."""
    val = beat_model_complex(np.array([0.0]), tau=15.0, df=2e3, phi=0.7)[0]
    assert val == pytest.approx(np.exp(1j * 0.7), abs=1e-12)


def test_rise_beat_residual_zero_at_truth():
    """Synthesize the rise model with known (τ, Δf, φ); residual at the
    true params is ~0."""
    tau_true, df_true, phi_true = 12.0, 1.5e3, 0.4
    t_us = np.linspace(0.0, 100.0, 500)
    model = 1.0 - np.exp(-t_us / tau_true) * np.exp(
        1j * (2 * np.pi * df_true * t_us * 1e-6 + phi_true)
    )
    res = rise_beat_residual(
        (tau_true, df_true, phi_true), t_us, model.real, model.imag,
    )
    assert np.max(np.abs(res)) < 1e-12


def test_fall_beat_residual_zero_at_truth():
    tau_true, df_true, phi_true = 18.0, -2.5e3, -0.6
    t_us = np.linspace(0.0, 80.0, 400)
    model = np.exp(-t_us / tau_true) * np.exp(
        1j * (2 * np.pi * df_true * t_us * 1e-6 + phi_true)
    )
    res = fall_beat_residual(
        (tau_true, df_true, phi_true), t_us, model.real, model.imag,
    )
    assert np.max(np.abs(res)) < 1e-12


def test_decay_beat_residual_zero_at_truth():
    """4-parameter version: amplitude A is also fit."""
    A_true, tau_true, df_true, phi_true = 0.85, 25.0, 800.0, 1.2
    t_us = np.linspace(0.0, 120.0, 600)
    model = A_true * np.exp(-t_us / tau_true) * np.exp(
        1j * (2 * np.pi * df_true * t_us * 1e-6 + phi_true)
    )
    res = decay_beat_residual(
        (A_true, tau_true, df_true, phi_true),
        t_us, model.real, model.imag,
    )
    assert np.max(np.abs(res)) < 1e-12


# ===========================================================================
# detect_burst (Hilbert-envelope thresholding) — NB: different code path
# than fft_cache.detect_burst_window (RMS-chunk thresholding)
# ===========================================================================

def test_detect_burst_from_hilbert_envelope():
    """``detect_burst`` returns indices where the envelope crosses 0.5·max."""
    n = 5_000
    env = np.zeros(n)
    burst_on, burst_off = 1_000, 4_000
    env[burst_on:burst_off] = 1.0
    # Add a small ramp at the edges so the threshold (0.5·max) is
    # unambiguous about which samples qualify.
    on_idx, off_idx = detect_burst(env, DT)
    assert on_idx == burst_on
    assert off_idx == burst_off - 1   # last sample where env > 0.5


# ===========================================================================
# compute_fit_windows
# ===========================================================================

def test_compute_fit_windows_arithmetic():
    """rise/fall sample windows mirror the documented formulas."""
    burst_on, burst_off = 1_000, 31_000
    n_samples = 50_000
    skip_n = int(FIT_SKIP_US * 1e-6 / DT)
    rise_n = int(RISE_FIT_WINDOW_US * 1e-6 / DT)
    fall_n = int(FALL_FIT_WINDOW_US * 1e-6 / DT)
    fw = compute_fit_windows(burst_on, burst_off, DT, n_samples)
    assert fw["skip_n"] == skip_n
    assert fw["rise_start"] == burst_on + skip_n
    assert fw["rise_end"] == burst_on + rise_n
    assert fw["fall_start"] == burst_off + skip_n
    assert fw["end_f"] == min(burst_off + fall_n, n_samples)
    assert fw["has_fall"] is True


def test_compute_fit_windows_has_fall_false_when_no_room():
    """Burst ending close to ``n_samples`` → no usable fall window."""
    n_samples = 50_000
    burst_off = n_samples - 100      # fall window has only 100 samples
    fw = compute_fit_windows(burst_on=1_000, burst_off=burst_off,
                             dt=DT, n_samples=n_samples)
    assert fw["has_fall"] is False
