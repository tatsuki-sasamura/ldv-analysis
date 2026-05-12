"""Unit tests for the v2 ScanData interface in io_utils.py.

Helper-function tests run in all environments. The integration test
that exercises ``load_scan_tdms`` against a real file is skipped when
``LDV_DATA_ROOT`` is not configured or the example TDMS is absent.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from ldv_analysis.io_utils import (
    ROLE_CURRENT,
    ROLE_DRIVE_VOLTAGE,
    ROLE_LDV_OUTPUT,
    ScanData,
    _detect_drive_freq_from_name,
    _detect_velocity_scale_from_name,
    _detect_vpp_from_name,
    _sort_wf_names,
    load_scan,
    load_scan_tdms,
    load_scan_v2,
)


# ---------------------------------------------------------------------------
# Filename heuristics
# ---------------------------------------------------------------------------

def test_detect_velocity_scale():
    assert _detect_velocity_scale_from_name("test10_1907_5Vpp_1m_s_max") == 0.5
    assert _detect_velocity_scale_from_name("xxx_2m_s_max") == 1.0
    assert _detect_velocity_scale_from_name("xxx_5m_s_max") == 2.5


def test_detect_velocity_scale_missing():
    assert _detect_velocity_scale_from_name("no_pattern") is None


def test_detect_vpp():
    assert _detect_vpp_from_name("test10_1907_5Vpp_1m_s_max") == 5.0
    assert _detect_vpp_from_name("test10_1907_25Vpp_5m_s_max") == 25.0


def test_detect_vpp_missing():
    assert _detect_vpp_from_name("no_voltage_here") is None


def test_detect_drive_freq():
    assert _detect_drive_freq_from_name("test10_1907_5Vpp_1m_s_max") == 1.907e6
    assert _detect_drive_freq_from_name("test5_2015") == 2.015e6


def test_detect_drive_freq_no_match():
    # 5-digit "10000" wouldn't match the 4-digit window
    assert _detect_drive_freq_from_name("test_xx") is None


# ---------------------------------------------------------------------------
# _sort_wf_names — preserve acquisition order from timestamp suffix
# ---------------------------------------------------------------------------

def test_sort_wf_names_by_timestamp():
    names = [
        "WFCh1_20260307_115608_300",
        "WFCh1_20260307_115608_100",
        "WFCh1_20260307_115608_200",
    ]
    out = _sort_wf_names(names)
    assert out == [
        "WFCh1_20260307_115608_100",
        "WFCh1_20260307_115608_200",
        "WFCh1_20260307_115608_300",
    ]


def test_sort_wf_names_short_returns_as_is():
    names = ["short", "name"]
    # Won't crash; returns sorted but doesn't promise meaningful order
    out = _sort_wf_names(names)
    assert sorted(out) == sorted(names)


# ---------------------------------------------------------------------------
# Role constants — spec
# ---------------------------------------------------------------------------

def test_role_constants():
    """Role strings are canonical and stable."""
    assert ROLE_DRIVE_VOLTAGE == "drive_voltage"
    assert ROLE_LDV_OUTPUT == "ldv_output"
    assert ROLE_CURRENT == "current"


# ---------------------------------------------------------------------------
# Dispatcher — extension routing
# ---------------------------------------------------------------------------

def test_load_scan_unknown_extension():
    with pytest.raises(ValueError, match="Unknown scan-data format"):
        load_scan("foo.bar")


def test_load_scan_v2_stub_raises():
    with pytest.raises(NotImplementedError):
        load_scan_v2("anything.h5")


def test_load_scan_tdms_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_scan_tdms("/nonexistent/file.tdms")


# ---------------------------------------------------------------------------
# Integration test against a real TDMS (skipped if data unavailable)
# ---------------------------------------------------------------------------

def _tdms_path() -> Path | None:
    """Locate the smallest test10 TDMS for the integration test.

    Returns None in CI / on machines without the data root.
    """
    root = os.environ.get("LDV_DATA_ROOT")
    if not root:
        # Try .env fallback
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("LDV_DATA_ROOT="):
                    root = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not root:
        return None
    candidate = (
        Path(root) / "20260307experimentB"
        / "test10_1907_5Vpp_1m_s_max.tdms"
    )
    return candidate if candidate.exists() else None


@pytest.mark.skipif(_tdms_path() is None,
                    reason="LDV_DATA_ROOT or test10 5Vpp TDMS not available")
def test_load_scan_tdms_smoke():
    """End-to-end: load a real TDMS, sanity-check shape and metadata."""
    scan = load_scan_tdms(_tdms_path())
    assert isinstance(scan, ScanData)
    assert scan.n_points > 100               # area scan, ~10k points
    assert scan.n_samples > 1000             # MHz waveform, many samples
    assert 1e-9 < scan.dt < 1e-7             # ~8 ns sample interval
    assert scan.pos_x.shape == (scan.n_points,)
    assert scan.pos_y.shape == (scan.n_points,)
    assert scan.metadata["source_format"] == "tdms_v1"
    assert scan.metadata["ldv_velocity_scale_mps_per_v"] == 0.5  # _1m_s_max
    assert scan.metadata["drive_voltage_vpp"] == 5.0
    assert scan.metadata["drive_frequency_hz_nominal"] == 1.907e6
    # Roles available
    avail = scan.metadata["_available_roles"]
    assert ROLE_DRIVE_VOLTAGE in avail
    assert ROLE_LDV_OUTPUT in avail


@pytest.mark.skipif(_tdms_path() is None,
                    reason="LDV_DATA_ROOT or test10 5Vpp TDMS not available")
def test_load_waveforms_shape_and_dtype():
    scan = load_scan_tdms(_tdms_path())
    wf = scan.load_waveforms(ROLE_LDV_OUTPUT, np.array([0, 1, 2]))
    assert wf.shape == (3, scan.n_samples)
    assert np.issubdtype(wf.dtype, np.floating)


@pytest.mark.skipif(_tdms_path() is None,
                    reason="LDV_DATA_ROOT or test10 5Vpp TDMS not available")
def test_load_waveforms_slice_matches_indices():
    """Slice access and explicit indices give identical output."""
    scan = load_scan_tdms(_tdms_path())
    wf_slice = scan.load_waveforms(ROLE_LDV_OUTPUT, slice(0, 3))
    wf_idx = scan.load_waveforms(ROLE_LDV_OUTPUT, np.arange(3))
    np.testing.assert_array_equal(wf_slice, wf_idx)


@pytest.mark.skipif(_tdms_path() is None,
                    reason="LDV_DATA_ROOT or test10 5Vpp TDMS not available")
def test_load_waveforms_unknown_role_raises():
    scan = load_scan_tdms(_tdms_path())
    with pytest.raises(KeyError):
        scan.load_waveforms("not_a_role", slice(0, 1))
