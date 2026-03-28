"""Data quality filters for LDV analysis.

Centralised filtering logic so that all scripts apply identical quality
gates.  Each function returns a boolean mask (True = valid).
"""

from __future__ import annotations

import numpy as np

from ldv_analysis.config import RSSI_THRESHOLD

# Voltage quality: reject points where piezo burst was missed/incomplete.
VOLTAGE_QUALITY_FACTOR = 0.5


def make_voltage_mask(voltage: np.ndarray) -> np.ndarray:
    """Mask points with voltage below ``VOLTAGE_QUALITY_FACTOR × median``.

    Detects missed or incomplete piezo bursts.
    """
    return voltage >= np.median(voltage) * VOLTAGE_QUALITY_FACTOR


def make_rssi_mask(
    rssi: np.ndarray | None,
    threshold: float = RSSI_THRESHOLD,
) -> np.ndarray | None:
    """Mask points with RSSI below *threshold*.

    Returns None when *rssi* is None (no RSSI data available).
    """
    if rssi is None:
        return None
    return rssi >= threshold


def make_valid_mask(
    voltage: np.ndarray,
    rssi: np.ndarray | None,
    threshold: float = RSSI_THRESHOLD,
) -> np.ndarray:
    """Combined voltage + RSSI quality mask."""
    mask = make_voltage_mask(voltage)
    rssi_mask = make_rssi_mask(rssi, threshold)
    if rssi_mask is not None:
        mask &= rssi_mask
    return mask


def make_transient_valid_mask(
    rssi: np.ndarray | None,
    pressure: np.ndarray,
    threshold: float = RSSI_THRESHOLD,
) -> np.ndarray:
    """Validity mask for transient analysis: RSSI + median pressure filter.

    Keeps points with RSSI >= *threshold* AND pressure above the median
    of RSSI-valid points.  Used for selecting the strongest scan point
    for transient envelope analysis.
    """
    rssi_mask = make_rssi_mask(rssi, threshold)
    if rssi_mask is None:
        rssi_mask = np.ones(len(pressure), dtype=bool)
    if rssi_mask.any():
        prs_threshold = np.median(pressure[rssi_mask])
        return rssi_mask & (pressure > prs_threshold)
    return rssi_mask


def make_burst_timing_mask(
    pt_burst_on_us: np.ndarray,
    pt_burst_off_us: np.ndarray,
    *,
    tolerance_us: float = 10.0,
) -> np.ndarray:
    """Mask points whose burst ON/OFF time deviates from the median.

    Detects scan points with shifted burst timing (e.g. residual
    acoustic field from the previous scan point's burst).

    Parameters
    ----------
    pt_burst_on_us : array
        Per-point burst ON time in microseconds.
    pt_burst_off_us : array
        Per-point burst OFF time in microseconds.
    tolerance_us : float
        Maximum allowed deviation from the median ON/OFF time.

    Returns
    -------
    mask : boolean array
        True for points with normal burst timing.
    """
    valid = ~np.isnan(pt_burst_on_us) & ~np.isnan(pt_burst_off_us)
    if not valid.any():
        return valid

    med_on = np.median(pt_burst_on_us[valid])
    med_off = np.median(pt_burst_off_us[valid])

    return (valid
            & (np.abs(pt_burst_on_us - med_on) <= tolerance_us)
            & (np.abs(pt_burst_off_us - med_off) <= tolerance_us))
