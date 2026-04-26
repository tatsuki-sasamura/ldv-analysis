"""Unit tests for data-quality filters in filters.py."""

import numpy as np
import pytest

from ldv_analysis.filters import (
    VOLTAGE_QUALITY_FACTOR,
    make_burst_timing_mask,
    make_rssi_mask,
    make_transient_valid_mask,
    make_valid_mask,
    make_voltage_mask,
)
from ldv_analysis.config import RSSI_THRESHOLD


# ---------------------------------------------------------------------------
# make_voltage_mask
# ---------------------------------------------------------------------------

def test_voltage_mask_keeps_above_half_median():
    """Default factor is 0.5; values >= 0.5*median are kept."""
    v = np.array([10.0, 9.0, 11.0, 4.0, 6.0, 10.0, 12.0])
    # median = 10, threshold = 5
    mask = make_voltage_mask(v)
    expected = v >= 5.0
    assert (mask == expected).all()


def test_voltage_mask_quality_factor_constant():
    """Verifies the documented 0.5 quality factor."""
    assert VOLTAGE_QUALITY_FACTOR == 0.5


def test_voltage_mask_uniform_array():
    """Uniform array: all >= 0.5*median, all kept."""
    v = np.full(10, 5.0)
    assert make_voltage_mask(v).all()


def test_voltage_mask_2d_array():
    """Works on 2D array too (median across all elements)."""
    v = np.array([[10.0, 4.0], [10.0, 12.0]])
    mask = make_voltage_mask(v)
    assert mask.shape == v.shape
    # median = 10, threshold = 5
    assert (mask == (v >= 5.0)).all()


# ---------------------------------------------------------------------------
# make_rssi_mask
# ---------------------------------------------------------------------------

def test_rssi_mask_none_passthrough():
    """rssi=None returns None (no RSSI available)."""
    assert make_rssi_mask(None) is None


def test_rssi_mask_threshold_default():
    """Default threshold is RSSI_THRESHOLD; >= passes."""
    rssi = np.array([0.5, 1.0, 1.5, 2.0])
    mask = make_rssi_mask(rssi)
    expected = rssi >= RSSI_THRESHOLD
    assert (mask == expected).all()


def test_rssi_mask_custom_threshold():
    rssi = np.array([0.0, 1.0, 2.0, 3.0])
    mask = make_rssi_mask(rssi, threshold=1.5)
    assert (mask == np.array([False, False, True, True])).all()


def test_rssi_mask_inclusive_at_threshold():
    """Boundary at threshold passes (>= comparison)."""
    rssi = np.array([1.0])
    assert make_rssi_mask(rssi, threshold=1.0).all()


# ---------------------------------------------------------------------------
# make_valid_mask  (voltage AND rssi combined)
# ---------------------------------------------------------------------------

def test_valid_mask_combines_both():
    """Both voltage and RSSI must pass."""
    v = np.array([10.0, 10.0, 10.0, 4.0])         # last fails voltage
    rssi = np.array([2.0, 0.5, 2.0, 2.0])         # 2nd fails rssi
    mask = make_valid_mask(v, rssi, threshold=1.0)
    assert (mask == np.array([True, False, True, False])).all()


def test_valid_mask_no_rssi():
    """No RSSI data → just the voltage mask."""
    v = np.array([10.0, 10.0, 4.0])
    mask = make_valid_mask(v, None)
    assert (mask == np.array([True, True, False])).all()


# ---------------------------------------------------------------------------
# make_transient_valid_mask
# ---------------------------------------------------------------------------

def test_transient_valid_mask_pressure_above_median():
    """Keeps RSSI-valid points whose pressure exceeds the median of valid."""
    rssi = np.array([2.0, 2.0, 2.0, 2.0, 0.5])     # last fails rssi
    p = np.array([1.0, 5.0, 3.0, 7.0, 1e9])        # last excluded by rssi
    mask = make_transient_valid_mask(rssi, p, threshold=1.0)
    # RSSI-valid pressures: [1, 5, 3, 7]; median = 4
    # Keep p > 4 AND rssi_valid → indices 1 and 3
    assert (mask == np.array([False, True, False, True, False])).all()


def test_transient_valid_mask_no_rssi():
    """No RSSI: median taken over all pressure values."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])        # median = 3
    mask = make_transient_valid_mask(None, p)
    assert (mask == np.array([False, False, False, True, True])).all()


# ---------------------------------------------------------------------------
# make_burst_timing_mask
# ---------------------------------------------------------------------------

def test_burst_timing_mask_within_tolerance():
    """All points within ±10 µs of median ON/OFF are kept."""
    on = np.array([100.0, 102.0, 98.0, 105.0, 95.0])
    off = np.array([520.0, 522.0, 518.0, 525.0, 515.0])
    mask = make_burst_timing_mask(on, off, tolerance_us=10.0)
    assert mask.all()


def test_burst_timing_mask_rejects_shifted():
    """Points with shifted burst timing are flagged out."""
    on = np.array([100.0, 100.0, 100.0, 200.0])    # last shifted by 100 µs
    off = np.array([520.0, 520.0, 520.0, 620.0])
    mask = make_burst_timing_mask(on, off, tolerance_us=10.0)
    assert (mask == np.array([True, True, True, False])).all()


def test_burst_timing_mask_rejects_nan():
    """NaN entries are rejected (no valid timing)."""
    on = np.array([100.0, np.nan, 100.0])
    off = np.array([520.0, 520.0, np.nan])
    mask = make_burst_timing_mask(on, off)
    assert (mask == np.array([True, False, False])).all()


def test_burst_timing_mask_all_nan():
    """All-NaN input → all-False mask, no error."""
    on = np.full(5, np.nan)
    off = np.full(5, np.nan)
    mask = make_burst_timing_mask(on, off)
    assert not mask.any()


def test_burst_timing_mask_tolerance_inclusive():
    """Boundary at exactly tolerance_us still passes (<= comparison)."""
    on = np.array([100.0, 110.0, 90.0, 100.0])
    off = np.array([520.0, 520.0, 520.0, 520.0])
    # median(on) = 100; deviations: 0, 10, 10, 0; all <= 10
    mask = make_burst_timing_mask(on, off, tolerance_us=10.0)
    assert mask.all()


def test_burst_timing_mask_custom_tolerance():
    """Tighter tolerance rejects more points."""
    on = np.array([100.0, 105.0, 95.0, 100.0])
    off = np.array([520.0, 525.0, 515.0, 520.0])
    mask_loose = make_burst_timing_mask(on, off, tolerance_us=10.0)
    mask_tight = make_burst_timing_mask(on, off, tolerance_us=1.0)
    assert mask_loose.all()
    # With 1 µs tolerance, the ±5 µs points are rejected
    assert mask_tight.sum() < mask_loose.sum()
