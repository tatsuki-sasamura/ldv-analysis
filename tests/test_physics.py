"""Unit tests for one-line physics conversions in config.py and transient.py.

Locks down the sign convention of velocity→pressure and the τ→Q formula.
These are the kinds of conversions that, if silently flipped, propagate
into every figure in the pipeline.
"""

import math

import numpy as np
import pytest

from ldv_analysis.config import (
    CHANNEL_HEIGHT,
    DN_DP,
    SENSITIVITY,
    velocity_to_pressure,
)
from ldv_analysis.transient import rise_simple, tau_to_Q


# ---------------------------------------------------------------------------
# velocity_to_pressure — sign convention and magnitude
# ---------------------------------------------------------------------------

def test_velocity_to_pressure_is_negative():
    """+p → +n → +OPL → −v_LDV, so the conversion factor must be negative."""
    factor = velocity_to_pressure(1.907e6, velocity_scale=1.0)
    assert factor < 0


def test_velocity_to_pressure_magnitude():
    """|factor| = velocity_scale / (2π f H dn/dp)."""
    f = 1.907e6
    vel_scale = 1.0
    expected = vel_scale / (2 * math.pi * f * SENSITIVITY)
    factor = velocity_to_pressure(f, velocity_scale=vel_scale)
    assert abs(abs(factor) - expected) / expected < 1e-12


def test_velocity_to_pressure_inverse_frequency():
    """factor ∝ 1/f."""
    f1 = 1.907e6
    f2 = 3.814e6   # exactly 2*f1
    factor1 = velocity_to_pressure(f1)
    factor2 = velocity_to_pressure(f2)
    assert abs(abs(factor1) / abs(factor2) - 2.0) < 1e-12


def test_velocity_to_pressure_velocity_scale_linear():
    """factor ∝ velocity_scale."""
    f = 1.907e6
    f1 = velocity_to_pressure(f, velocity_scale=1.0)
    f2 = velocity_to_pressure(f, velocity_scale=2.5)
    assert abs(f2 / f1 - 2.5) < 1e-12


def test_velocity_to_pressure_round_trip():
    """Apply the conversion and back: original signal recovered."""
    f = 1.907e6
    raw_voltage = 0.05    # V
    pressure = raw_voltage * velocity_to_pressure(f, velocity_scale=1.0)
    back = pressure / velocity_to_pressure(f, velocity_scale=1.0)
    assert abs(back - raw_voltage) < 1e-12


def test_sensitivity_value():
    """Documented constant: SENSITIVITY = CHANNEL_HEIGHT * DN_DP = 2.1e-14 m/Pa."""
    assert abs(SENSITIVITY - CHANNEL_HEIGHT * DN_DP) < 1e-30
    # Documented in CLAUDE.md as 2.1e-14 m/Pa
    assert abs(SENSITIVITY - 2.1e-14) < 1e-20


# ---------------------------------------------------------------------------
# tau_to_Q
# ---------------------------------------------------------------------------

def test_tau_to_Q_formula():
    """Q = π h f τ (with τ in seconds → multiplied by 1e-6 for µs input)."""
    f = 1.907e6
    tau_us = 8.4
    Q = tau_to_Q(f, tau_us, harmonic=1)
    expected = np.pi * 1 * f * tau_us * 1e-6
    assert abs(Q - expected) < 1e-12


@pytest.mark.parametrize("h", [1, 2, 3])
def test_tau_to_Q_harmonic_scales(h):
    """Q scales linearly with the harmonic number."""
    f = 1.907e6
    tau_us = 8.4
    Q_1 = tau_to_Q(f, tau_us, harmonic=1)
    Q_h = tau_to_Q(f, tau_us, harmonic=h)
    assert abs(Q_h / Q_1 - h) < 1e-12


def test_tau_to_Q_realistic_values():
    """Order-of-magnitude check: τ = 8 µs at 1.9 MHz gives Q ≈ 50."""
    Q = tau_to_Q(1.907e6, 8.0, harmonic=1)
    assert 45 < Q < 55


# ---------------------------------------------------------------------------
# rise_simple — first-order ring-up
# ---------------------------------------------------------------------------

def test_rise_simple_zero_at_t0():
    assert abs(rise_simple(0.0, 5.0)) < 1e-12


def test_rise_simple_one_tau_value():
    """At t=τ, the rise reaches 1 - 1/e ≈ 0.6321."""
    assert abs(rise_simple(5.0, 5.0) - (1 - 1 / np.e)) < 1e-12


def test_rise_simple_long_time_limit():
    """At t >> τ, the rise asymptotes to 1."""
    assert abs(rise_simple(50.0, 1.0) - 1.0) < 1e-12
