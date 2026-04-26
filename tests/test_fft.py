"""Unit tests for FFT extraction primitives in fft_cache.py and transient.py.

Synthesizes signals with known harmonic content and verifies the analysis
pipeline recovers the expected amplitudes, phases, and drive frequency.
"""

import numpy as np
import pytest

from ldv_analysis.fft_cache import find_drive_frequency, wrap_phase
from ldv_analysis.transient import sliding_dft_envelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_cycle_setup(f0_target_hz, dt, approx_n_samples):
    """Pick (n_samples, f0) so the window holds exactly K cycles of f0.

    Returns (n_samples, f0_exact) with f0_exact ≈ f0_target.
    """
    n = int(round(approx_n_samples))
    K = int(round(f0_target_hz * n * dt))      # integer cycles in window
    f0_exact = K / (n * dt)                    # adjusted target frequency
    return n, f0_exact


def _make_signal(f0, dt, n_samples, harmonics):
    """Synthesize sum_h A_h sin(2π h f0 t + φ_h)."""
    t = np.arange(n_samples) * dt
    s = np.zeros(n_samples)
    for h, A, phi in harmonics:
        s += A * np.sin(2 * np.pi * h * f0 * t + phi)
    return s


def _dft_at(signal, dt, f):
    """Single-frequency DFT (matches the pipeline's dot-product approach)."""
    n = len(signal)
    tone = np.exp(-2j * np.pi * f * np.arange(n) * dt)
    return signal @ tone


# ---------------------------------------------------------------------------
# find_drive_frequency
# ---------------------------------------------------------------------------

def test_find_drive_frequency_exact_bin():
    """When f0 lands exactly on a bin, recovery is well within sub-bin."""
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 8192)
    s = _make_signal(f0, dt, n, [(1, 1.0, 0.3)])
    f_est = find_drive_frequency(s, dt)
    bin_hz = 1.0 / (n * dt)
    # Exact-bin peak: parabolic interp on tiny numerical residues at
    # neighbours can shift δ slightly. Should still be < 0.001 bin.
    assert abs(f_est - f0) < 1e-3 * bin_hz, (
        f"recovered {f_est}, expected {f0}, err={abs(f_est-f0):.3f} Hz"
    )


@pytest.mark.parametrize("f0_mhz", [1.0, 1.907, 3.845, 5.0])
def test_find_drive_frequency_off_bin(f0_mhz):
    """Off-bin frequency: parabolic interpolation gives ~0.2-bin accuracy."""
    f0 = f0_mhz * 1e6
    dt = 8e-9
    n = 16384                     # ~131 µs window, df ≈ 7.6 kHz
    s = _make_signal(f0, dt, n, [(1, 1.0, 0.3)])
    f_est = find_drive_frequency(s, dt)
    bin_hz = 1.0 / (n * dt)
    assert abs(f_est - f0) < 0.3 * bin_hz, (
        f"recovered {f_est/1e6:.6f} MHz, expected {f0/1e6:.6f} MHz, "
        f"err = {abs(f_est-f0)/bin_hz:.3f} bin"
    )


def test_find_drive_frequency_with_harmonics():
    """Pick the fundamental even when 2f and 3f are present."""
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 16384)
    s = _make_signal(f0, dt, n,
                     [(1, 1.0, 0.0), (2, 0.3, 0.5), (3, 0.1, 1.2)])
    f_est = find_drive_frequency(s, dt)
    bin_hz = 1.0 / (n * dt)
    assert abs(f_est - f0) < 0.3 * bin_hz


def test_find_drive_frequency_with_noise():
    """Recovery still sub-bin under modest white noise."""
    rng = np.random.default_rng(42)
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 16384)
    s = _make_signal(f0, dt, n, [(1, 1.0, 0.0)])
    s = s + 0.05 * rng.standard_normal(n)         # 5 % rms noise
    f_est = find_drive_frequency(s, dt)
    bin_hz = 1.0 / (n * dt)
    assert abs(f_est - f0) < 0.3 * bin_hz


# ---------------------------------------------------------------------------
# DFT amplitude / phase extraction (the pipeline's core math)
# ---------------------------------------------------------------------------

def test_dft_amplitude_three_harmonics():
    """`|DFT|·2/N` recovers each harmonic amplitude when the window holds an
    integer number of cycles of f0 (no spectral leakage)."""
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 8192)
    A = {1: 1.0, 2: 0.3, 3: 0.05}
    s = _make_signal(f0, dt, n, [(h, A[h], 0.0) for h in (1, 2, 3)])
    for h, A_h in A.items():
        amp = np.abs(_dft_at(s, dt, h * f0)) * 2 / n
        assert abs(amp - A_h) < 1e-10, (
            f"h={h}: recovered {amp:.6e}, expected {A_h:.6e}"
        )


def test_dft_phase_recovery():
    """Phase of `DFT(sin(ωt+φ))` is `φ - 90°`; recovered with +90°."""
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 8192)
    for phi_deg in [0, 30, 90, -45, 170, -170]:
        phi = np.deg2rad(phi_deg)
        s = _make_signal(f0, dt, n, [(1, 1.0, phi)])
        dft = _dft_at(s, dt, f0)
        # Σ exp(-iωt)·sin(ωt+φ) = -(i/2)·exp(iφ)·N → angle = φ - π/2
        recovered_phi_deg = wrap_phase(np.degrees(np.angle(dft)) + 90.0)
        err = wrap_phase(np.array([recovered_phi_deg - phi_deg]))[0]
        assert abs(err) < 1e-6, (
            f"phi_in={phi_deg}, recovered={recovered_phi_deg}"
        )


def test_phase_reference_subtraction():
    """Pipeline reports phase relative to Ch1: angle(Ch2) - angle(Ch1)."""
    dt = 8e-9
    n, f0 = _exact_cycle_setup(1.907e6, dt, 8192)
    for delta_deg in [0, 30, -45, 90, 170]:
        phi1 = 0.4
        phi2 = phi1 + np.deg2rad(delta_deg)
        ch1 = _make_signal(f0, dt, n, [(1, 1.0, phi1)])
        ch2 = _make_signal(f0, dt, n, [(1, 0.5, phi2)])
        d1 = _dft_at(ch1, dt, f0)
        d2 = _dft_at(ch2, dt, f0)
        rel_deg = wrap_phase(np.degrees(np.angle(d2) - np.angle(d1)))
        err = wrap_phase(np.array([rel_deg - delta_deg]))[0]
        assert abs(err) < 1e-6


# ---------------------------------------------------------------------------
# wrap_phase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [
    (0, 0),
    (90, 90),
    (-90, -90),
    (180, -180),     # +180 wraps to -180 by convention here
    (-180, -180),
    (181, -179),
    (-181, 179),
    (540, -180),
    (-540, -180),
])
def test_wrap_phase(inp, expected):
    out = wrap_phase(np.array([inp]))[0]
    assert abs(out - expected) < 1e-9


def test_wrap_phase_array_in_range():
    """Output stays within [-180, 180]."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1000, 1000, 1000)
    y = wrap_phase(x)
    assert (y >= -180).all() and (y <= 180).all()


# ---------------------------------------------------------------------------
# sliding_dft_envelope (transient.py)
# ---------------------------------------------------------------------------

def test_sliding_dft_envelope_steady_state():
    """For a continuous tone, envelope magnitude equals signal amplitude."""
    f0 = 1.907e6
    dt = 8e-9
    n = 50000
    A = 0.7
    s = _make_signal(f0, dt, n, [(1, A, 0.0)])
    env = sliding_dft_envelope(s, dt, f0, win_us=2.0)
    interior = env[5000:-5000]    # avoid edge transients
    assert abs(interior.mean() - A) < 1e-3
    assert interior.std() < 1e-3


def test_sliding_dft_envelope_gated_signal():
    """Envelope rises to A inside a gate and is small outside."""
    f0 = 1.907e6
    dt = 8e-9
    n = 100000
    s = _make_signal(f0, dt, n, [(1, 1.0, 0.0)])
    gate = np.zeros(n)
    gate[30000:70000] = 1.0
    s = s * gate
    env = sliding_dft_envelope(s, dt, f0, win_us=2.0)
    win_n = int(2.0e-6 / dt)
    inside = env[40000:60000]
    assert abs(inside.mean() - 1.0) < 5e-3
    outside_left = env[: 30000 - 2 * win_n]
    outside_right = env[70000 + 2 * win_n :]
    assert outside_left.mean() < 1e-3
    assert outside_right.mean() < 1e-3


def test_sliding_dft_envelope_off_frequency_rejected():
    """A sinusoid at f0 produces small envelope at 2*f0."""
    f0 = 1.907e6
    dt = 8e-9
    n = 50000
    s = _make_signal(f0, dt, n, [(1, 1.0, 0.0)])
    env = sliding_dft_envelope(s, dt, 2 * f0, win_us=2.0)
    assert env[5000:-5000].mean() < 0.05
