"""Unit tests for mode-shape fitting in mode_fit.py.

Synthesizes sinusoidal pressure profiles with known amplitude, phase, and
channel center, then verifies fit_mode recovers them. Includes a
regression test for the sigma-clipping bug fixed in commit 6f6a05c.
"""

import numpy as np
import pytest

from ldv_analysis.mode_fit import _mode_shape, _project, fit_mode


W = 0.375e-3   # 375 µm channel width


def _y_grid(n=73, span=W):
    """Symmetric width-axis sample grid in meters, span ≈ channel width."""
    return np.linspace(-span / 2, span / 2, n)


# ---------------------------------------------------------------------------
# _mode_shape — basic geometry
# ---------------------------------------------------------------------------

def test_mode_shape_1f_node_at_center():
    """sin(πy/W) has a node at y=0 and anti-nodes at y=±W/2 (1f mode)."""
    y = np.array([-W / 2, 0.0, W / 2])
    m = _mode_shape(y, W, harmonic=1)
    assert abs(m[0] - (-1.0)) < 1e-12      # sin(-π/2) = -1
    assert abs(m[1]) < 1e-12               # sin(0) = 0  (node at center)
    assert abs(m[2] - 1.0) < 1e-12         # sin(+π/2) = +1
    # Off-center check: y = W/4 → sin(π/4) = √2/2
    m_quarter = _mode_shape(np.array([W / 4]), W, harmonic=1)
    assert abs(m_quarter[0] - np.sin(np.pi / 4)) < 1e-12


def test_mode_shape_2f_peaks_at_center_and_walls():
    """cos(2πy/W) peaks at y=0 and y=±W/2, zeros at y=±W/4."""
    y = np.array([0.0, W / 4, -W / 4, W / 2])
    m = _mode_shape(y, W, harmonic=2)
    assert abs(m[0] - 1.0) < 1e-12
    assert abs(m[1]) < 1e-12
    assert abs(m[2]) < 1e-12
    assert abs(abs(m[3]) - 1.0) < 1e-12


def test_mode_shape_use_abs():
    """use_abs=True returns |mode|, always non-negative."""
    y = np.linspace(-W / 2, W / 2, 50)
    for h in (1, 2, 3):
        m = _mode_shape(y, W, harmonic=h, use_abs=True)
        assert (m >= 0).all()


# ---------------------------------------------------------------------------
# Real-input fits — recover known amplitude
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("harmonic,A_kpa", [(1, 1000.0), (2, 200.0), (3, 50.0)])
def test_fit_mode_real_recovers_amplitude(harmonic, A_kpa):
    """Real |mode| input: fit recovers known amplitude to ~10⁻¹⁰."""
    y = _y_grid()
    A = A_kpa * 1e3       # Pa
    p_real = A * _mode_shape(y, W, harmonic=harmonic, use_abs=True)
    res = fit_mode(y, p_real, W, harmonic=harmonic, center=0.0)
    assert abs(abs(res.p0) - A) < 1e-3, f"recovered {res.p0}, expected {A}"
    assert res.r2 > 0.9999


def test_fit_mode_real_with_noise():
    """Real input with 5 % noise: amplitude recovered to ~1 %."""
    rng = np.random.default_rng(42)
    y = _y_grid()
    A = 1.0e6   # 1 MPa
    p = A * _mode_shape(y, W, harmonic=1, use_abs=True)
    p_noisy = p + 0.05 * A * rng.standard_normal(len(y))
    res = fit_mode(y, p_noisy, W, harmonic=1, center=0.0)
    assert abs(abs(res.p0) - A) / A < 0.02
    assert res.r2 > 0.9


# ---------------------------------------------------------------------------
# Complex-input fits — recover amplitude AND phase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("harmonic", [1, 2, 3])
@pytest.mark.parametrize("phi_deg", [0, 45, 90, -120])
def test_fit_mode_complex_recovers_amplitude_and_phase(harmonic, phi_deg):
    """Complex p₀·exp(iφ)·signed_mode → recover both amplitude and phase."""
    y = _y_grid()
    A = 1.5e6
    phi = np.deg2rad(phi_deg)
    p0_true = A * np.exp(1j * phi)
    p_complex = p0_true * _mode_shape(y, W, harmonic=harmonic, use_abs=False)
    res = fit_mode(y, p_complex, W, harmonic=harmonic, center=0.0)
    assert abs(abs(res.p0) - A) / A < 1e-6
    # Phase comparison wraps over ±180
    err_deg = np.degrees(np.angle(res.p0 / p0_true))
    assert abs(err_deg) < 1e-6


# ---------------------------------------------------------------------------
# Brute-force center search
# ---------------------------------------------------------------------------

def test_fit_mode_center_search_recovers_offset():
    """Signal centered at y₀≠0: brute-force search finds y₀ and recovers A."""
    A = 2.0e6
    y0 = 50e-6   # 50 µm offset
    # Sample over a wider span so the search has room to find the center
    y = np.linspace(-W, W, 201)
    p = A * _mode_shape(y - y0, W, harmonic=1, use_abs=False)
    res = fit_mode(y, p.astype(complex), W, harmonic=1, center=None,
                   n_trial=200)
    assert abs(res.center - y0) < W / 200    # one trial spacing
    assert abs(abs(res.p0) - A) / A < 1e-3
    assert res.r2 > 0.999


# ---------------------------------------------------------------------------
# R² extremes
# ---------------------------------------------------------------------------

def test_fit_mode_r2_perfect_fit():
    """Noise-free synthetic data → R² = 1."""
    y = _y_grid()
    p = 1.0 * _mode_shape(y, W, harmonic=1, use_abs=True)
    res = fit_mode(y, p, W, harmonic=1, center=0.0)
    assert abs(res.r2 - 1.0) < 1e-10


def test_fit_mode_r2_zero_signal():
    """All-zero data → degenerate; r² returned as 0."""
    y = _y_grid()
    p = np.zeros_like(y)
    res = fit_mode(y, p, W, harmonic=1, center=0.0)
    assert res.r2 == 0.0


# ---------------------------------------------------------------------------
# Sigma clipping — regression test for "all-points-rejected" bug
# ---------------------------------------------------------------------------

def test_sigma_clip_does_not_collapse_high_snr():
    """High-SNR signal with sigma_clip=3.0 must NOT reject all points.

    Regression for the bug where iterative clipping shrinks std until even
    good points exceed the tightening threshold and the mask becomes empty.
    See `_project` early-exit (`new_mask.sum() < 3`).
    """
    rng = np.random.default_rng(0)
    y = _y_grid()
    A = 1.0e6
    p = A * _mode_shape(y, W, harmonic=1, use_abs=True)
    p_noisy = p + 1e3 * rng.standard_normal(len(y))   # 0.1 % noise
    res = fit_mode(y, p_noisy, W, harmonic=1, center=0.0, sigma_clip=3.0)
    # We must keep at least 3 points and still recover the amplitude
    assert res.inside.sum() >= 3
    assert abs(abs(res.p0) - A) / A < 0.01
    assert res.r2 > 0.99      # not the 0.0 fallback


def test_sigma_clip_rejects_obvious_outliers():
    """A few large outliers should be clipped; underlying fit unaffected."""
    rng = np.random.default_rng(1)
    y = _y_grid()
    A = 1.0e6
    p = A * _mode_shape(y, W, harmonic=1, use_abs=True)
    p_clean = p.copy()
    # Add 5 outliers at random indices
    out_idx = rng.choice(len(y), 5, replace=False)
    p_outliers = p_clean.copy()
    p_outliers[out_idx] += 5.0 * A    # 5x signal — clear outliers
    res_clip = fit_mode(y, p_outliers, W, harmonic=1, center=0.0,
                        sigma_clip=3.0)
    res_noclip = fit_mode(y, p_outliers, W, harmonic=1, center=0.0,
                          sigma_clip=None)
    # Clipped fit should be closer to true A than unclipped
    err_clip = abs(abs(res_clip.p0) - A) / A
    err_noclip = abs(abs(res_noclip.p0) - A) / A
    assert err_clip < err_noclip
    assert err_clip < 0.02


# ---------------------------------------------------------------------------
# _project — direct LSQ behavior
# ---------------------------------------------------------------------------

def test_project_basic_lsq():
    """`_project` recovers exact p0 on a noise-free signal."""
    y = _y_grid()
    mode = _mode_shape(y, W, harmonic=1, use_abs=True)
    A = 7.5e5
    p0_recovered, mask = _project(A * mode, mode, sigma_clip=None)
    assert abs(p0_recovered - A) < 1e-6
    assert mask.all()


def test_project_zero_mode_returns_zero():
    """All-zero mode → degenerate; returns 0 without error."""
    y = _y_grid()
    mode = np.zeros_like(y)
    data = np.ones_like(y)
    p0, mask = _project(data, mode, sigma_clip=None)
    assert p0 == 0.0
