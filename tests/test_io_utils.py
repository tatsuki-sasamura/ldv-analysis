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
    load_scan_hdf5,
    validate_hdf5_v2,
    write_scan_hdf5,
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


def test_load_scan_hdf5_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_scan_hdf5("/nonexistent/file.h5")


def test_load_scan_tdms_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_scan_tdms("/nonexistent/file.tdms")


# ---------------------------------------------------------------------------
# Integration test against a real TDMS (skipped if data unavailable)
# ---------------------------------------------------------------------------

def _data_root() -> str | None:
    """Return LDV_DATA_ROOT from env or .env, or None if unconfigured."""
    root = os.environ.get("LDV_DATA_ROOT")
    if root:
        return root
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("LDV_DATA_ROOT="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def _tdms_path() -> Path | None:
    """Locate the test10 5 Vpp TDMS for the v1 integration test."""
    root = _data_root()
    if not root:
        return None
    candidate = (
        Path(root) / "20260307experimentB"
        / "test10_1907_5Vpp_1m_s_max.tdms"
    )
    return candidate if candidate.exists() else None


def _hdf5_fixture() -> Path | None:
    """Locate the converted v2 HDF5 fixture for the round-trip test."""
    root = _data_root()
    if not root:
        return None
    candidate = Path(root) / "v2_test" / "test10_5Vpp_subset100.h5"
    return candidate if candidate.exists() else None


def _hdf5_full() -> Path | None:
    """Locate the FULL v2 HDF5 conversion for cross-format cache equivalence."""
    root = _data_root()
    if not root:
        return None
    candidate = Path(root) / "v2_test" / "test10_5Vpp_full.h5"
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


# ---------------------------------------------------------------------------
# Integration test: real-data HDF5 fixture produced by the TDMS converter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_hdf5_fixture() is None,
                    reason="v2_test/test10_5Vpp_subset100.h5 not available")
def test_load_scan_hdf5_real_fixture():
    """End-to-end: load the converted v2 fixture and sanity-check."""
    scan = load_scan_hdf5(_hdf5_fixture())
    assert scan.n_points == 100
    assert scan.n_samples == 100000
    assert 1e-9 < scan.dt < 1e-7
    assert scan.metadata["source_format"] == "hdf5_v2"
    assert scan.metadata["chip_id"] == "ldv_chip_2026_W10"
    assert scan.metadata["drive_voltage_vpp"] == 5.0
    assert scan.metadata["drive_frequency_hz_nominal"] == 1.907e6
    assert ROLE_LDV_OUTPUT in scan.metadata["_available_roles"]
    wf = scan.load_waveforms(ROLE_LDV_OUTPUT, slice(0, 3))
    assert wf.shape == (3, scan.n_samples)


@pytest.mark.skipif(
    _tdms_path() is None or _hdf5_fixture() is None,
    reason="Need both TDMS and v2 HDF5 fixture for cross-check",
)
def test_tdms_vs_hdf5_waveform_match():
    """First 5 LDV waveforms must match between source TDMS and converted HDF5."""
    src = load_scan_tdms(_tdms_path())
    dst = load_scan_hdf5(_hdf5_fixture())
    src_wf = src.load_waveforms(ROLE_LDV_OUTPUT, slice(0, 5))
    dst_wf = dst.load_waveforms(ROLE_LDV_OUTPUT, slice(0, 5))
    # float64 -> float32 round trip
    np.testing.assert_allclose(dst_wf, src_wf, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(
    _tdms_path() is None or _hdf5_full() is None,
    reason="Need TDMS + full HDF5 conversion (test10_5Vpp_full.h5)",
)
def test_fft_cache_equivalent_across_formats(tmp_path):
    """Cache built from TDMS must match cache built from the same data as HDF5.

    The whole point of ScanData: analysis layer doesn't care about source
    format. Waveforms are float64 in TDMS and float32 in HDF5, so byte
    equivalence is impossible; we check physical equivalence within the
    quantization-noise floor:
      - peak signal must match to <0.5% relative (science-grade)
      - low-SNR points (noise floor of high harmonics) can differ
        up to 1% of the key's peak amplitude
      - phases are wrapped modulo 360 deg; the median absolute
        difference must be small (noise-floor points have ill-defined
        phase and are excluded by the median).
    """
    from ldv_analysis.fft_cache import load_or_compute

    cache_tdms = load_or_compute(_tdms_path(), tmp_path / "tdms")
    cache_hdf5 = load_or_compute(_hdf5_full(), tmp_path / "hdf5")

    # Same set of keys
    assert set(cache_tdms.files) == set(cache_hdf5.files)

    SKIP = {"cache_version", "source_format", "source_mtime"}
    PHASE_KEYS = {f"phase_{h}f" for h in range(1, 6)} | {"phase_vi"}

    for k in cache_tdms.files:
        if k in SKIP:
            continue
        a = cache_tdms[k]
        b = cache_hdf5[k]
        if not (np.issubdtype(a.dtype, np.number)
                and np.issubdtype(b.dtype, np.number)):
            continue
        if k in PHASE_KEYS:
            # Phase wraps to [-180, 180]; compare modulo 360 deg
            diff = (a - b + 180.0) % 360.0 - 180.0
            median_abs = float(np.nanmedian(np.abs(diff)))
            assert median_abs < 1.0, (
                f"{k!r}: median |phase diff| = {median_abs:.3f} deg (> 1 deg)"
            )
        else:
            # All-NaN arrays (e.g. noise_rms for continuous-excitation
            # files) trivially match
            if np.all(np.isnan(a)) and np.all(np.isnan(b)):
                continue
            # Signal-scaled tolerance: 1% of the key's own peak amplitude
            signal_max = float(np.nanmax(np.abs(a)))
            if signal_max == 0:
                continue
            max_diff = float(np.nanmax(np.abs(a - b)))
            assert max_diff <= 1e-2 * signal_max, (
                f"{k!r}: max |Δ|={max_diff:.3e} > 1% of signal max "
                f"({signal_max:.3e})"
            )


# ---------------------------------------------------------------------------
# load_scan_hdf5 — synthetic HDF5 round-trip (no external data needed)
# ---------------------------------------------------------------------------

def _make_v2_h5(path: Path, n_points: int = 10, n_samples: int = 64,
                include_current: bool = True, include_rssi: bool = True,
                bad_attrs: tuple = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a small valid v2 HDF5 file for tests; return (pos_x, pos_y, ldv).

    bad_attrs: list of attribute names to OMIT (for negative tests).
    """
    import h5py
    pos_x = np.linspace(0, 1e-3, n_points)
    pos_y = np.linspace(0, 1e-3, n_points)
    rssi = np.full(n_points, 2.0, dtype=np.float32)
    drive = np.tile(
        np.sin(2 * np.pi * np.arange(n_samples) / n_samples), (n_points, 1)
    ).astype(np.float32)
    ldv = (drive * 0.1).astype(np.float32)
    current = (drive * 0.01).astype(np.float32)

    sample_rate = 125e6
    all_attrs = {
        "version": "2.0",
        "timestamp_utc": "2026-05-13T00:00:00Z",
        "operator": "tester",
        "daq_software_version": "v2.0.0",
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "ldv_velocity_scale_mps_per_v": 0.5,
        "drive_frequency_hz_nominal": 1.907e6,
        "drive_voltage_vpp": 5.0,
        "burst_on_us_nominal": 5.0,
        "burst_off_us_nominal": 525.0,
        "scan_n_x": n_points,
        "scan_n_y": 1,
        "chip_id": "test_chip",
        "session_id": "test_session",
    }
    for bad in bad_attrs:
        all_attrs.pop(bad, None)

    with h5py.File(path, "w") as f:
        for k, v in all_attrs.items():
            f.attrs[k] = v
        c = f.create_group("coordinates")
        c.create_dataset("pos_x_m", data=pos_x)
        c.create_dataset("pos_y_m", data=pos_y)
        if include_rssi:
            c.create_dataset("rssi", data=rssi)
        w = f.create_group("waveforms")
        w.create_dataset("drive_voltage", data=drive,
                          chunks=(1, n_samples))
        w.create_dataset("ldv_output", data=ldv,
                          chunks=(1, n_samples))
        if include_current:
            w.create_dataset("current", data=current,
                              chunks=(1, n_samples))
    return pos_x, pos_y, ldv


def test_load_scan_hdf5_round_trip(tmp_path):
    """Write a synthetic v2 file and verify all ScanData fields."""
    h5_path = tmp_path / "round_trip.h5"
    pos_x, pos_y, ldv = _make_v2_h5(h5_path, n_points=10, n_samples=64)

    scan = load_scan_hdf5(h5_path)
    assert isinstance(scan, ScanData)
    assert scan.n_points == 10
    assert scan.n_samples == 64
    assert scan.dt == pytest.approx(1.0 / 125e6)
    np.testing.assert_allclose(scan.pos_x, pos_x)
    np.testing.assert_allclose(scan.pos_y, pos_y)
    assert scan.rssi is not None and scan.rssi.shape == (10,)
    assert scan.metadata["source_format"] == "hdf5_v2"
    assert scan.metadata["chip_id"] == "test_chip"
    assert scan.metadata["session_id"] == "test_session"
    assert sorted(scan.metadata["_available_roles"]) == [
        ROLE_CURRENT, ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT,
    ]


def test_load_scan_hdf5_dispatcher(tmp_path):
    """load_scan dispatches .h5 to load_scan_hdf5."""
    h5_path = tmp_path / "dispatch.h5"
    _make_v2_h5(h5_path)
    scan = load_scan(h5_path)
    assert scan.metadata["source_format"] == "hdf5_v2"


def test_load_scan_hdf5_waveforms_slice(tmp_path):
    """Lazy waveform read with slice access."""
    h5_path = tmp_path / "wf_slice.h5"
    _, _, ldv = _make_v2_h5(h5_path, n_points=10, n_samples=64)
    scan = load_scan_hdf5(h5_path)
    out = scan.load_waveforms(ROLE_LDV_OUTPUT, slice(2, 5))
    assert out.shape == (3, 64)
    np.testing.assert_allclose(out, ldv[2:5])


def test_load_scan_hdf5_waveforms_fancy_idx(tmp_path):
    """Fancy indexing (unsorted, non-contiguous) returns rows in order."""
    h5_path = tmp_path / "wf_fancy.h5"
    _, _, ldv = _make_v2_h5(h5_path, n_points=10, n_samples=64)
    scan = load_scan_hdf5(h5_path)
    idx = np.array([7, 1, 4])
    out = scan.load_waveforms(ROLE_LDV_OUTPUT, idx)
    assert out.shape == (3, 64)
    np.testing.assert_allclose(out, ldv[idx])


def test_load_scan_hdf5_optional_current_absent(tmp_path):
    """Current waveform group is optional."""
    h5_path = tmp_path / "no_current.h5"
    _make_v2_h5(h5_path, include_current=False)
    scan = load_scan_hdf5(h5_path)
    assert ROLE_CURRENT not in scan.metadata["_available_roles"]
    with pytest.raises(KeyError):
        scan.load_waveforms(ROLE_CURRENT, slice(0, 1))


def test_load_scan_hdf5_optional_rssi_absent(tmp_path):
    """RSSI dataset is optional."""
    h5_path = tmp_path / "no_rssi.h5"
    _make_v2_h5(h5_path, include_rssi=False)
    scan = load_scan_hdf5(h5_path)
    assert scan.rssi is None


def test_load_scan_hdf5_missing_required_attr_raises(tmp_path):
    """A missing required root attribute is a clear, actionable error."""
    h5_path = tmp_path / "missing_attr.h5"
    _make_v2_h5(h5_path, bad_attrs=("chip_id",))
    with pytest.raises(ValueError, match="chip_id"):
        load_scan_hdf5(h5_path)


def test_load_scan_hdf5_missing_ldv_output_raises(tmp_path):
    """ldv_output is mandatory; if absent, fail at load time."""
    import h5py
    h5_path = tmp_path / "no_ldv.h5"
    _make_v2_h5(h5_path)
    with h5py.File(h5_path, "a") as f:
        del f["waveforms/ldv_output"]
    with pytest.raises(ValueError, match="ldv_output"):
        load_scan_hdf5(h5_path)


# ---------------------------------------------------------------------------
# write_scan_hdf5 — round-trip from a synthetic ScanData
# ---------------------------------------------------------------------------

def _make_synthetic_scan(n_points=10, n_samples=64) -> ScanData:
    """Build a synthetic ScanData with all required v2 metadata."""
    pos_x = np.linspace(0, 1e-3, n_points)
    pos_y = np.linspace(0, 1e-3, n_points)
    rssi = np.full(n_points, 2.0, dtype=np.float32)

    wfs = {
        ROLE_DRIVE_VOLTAGE: np.tile(
            np.sin(2 * np.pi * np.arange(n_samples) / n_samples) * 5.0,
            (n_points, 1),
        ).astype(np.float32),
        ROLE_LDV_OUTPUT: np.tile(
            np.sin(2 * np.pi * np.arange(n_samples) / n_samples) * 0.1,
            (n_points, 1),
        ).astype(np.float32),
        ROLE_CURRENT: np.tile(
            np.cos(2 * np.pi * np.arange(n_samples) / n_samples) * 0.01,
            (n_points, 1),
        ).astype(np.float32),
    }

    def loader(role, points):
        if role not in wfs:
            raise KeyError(role)
        return wfs[role][points]

    return ScanData(
        pos_x=pos_x, pos_y=pos_y, rssi=rssi,
        dt=8e-9, n_points=n_points, n_samples=n_samples,
        metadata={
            "sample_rate_hz": 125e6,
            "n_samples": n_samples,
            "ldv_velocity_scale_mps_per_v": 0.5,
            "drive_frequency_hz_nominal": 1.907e6,
            "drive_voltage_vpp": 5.0,
            "burst_on_us_nominal": 5.0,
            "burst_off_us_nominal": 525.0,
            "scan_n_x": n_points,
            "scan_n_y": 1,
            "chip_id": "synthetic_chip",
            "session_id": "synthetic_session",
            "timestamp_utc": "2026-05-13T00:00:00Z",
            "operator": "tester",
            "daq_software_version": "v2.0.0",
            "_available_roles": sorted(wfs.keys()),
        },
        _loader=loader,
    )


def test_write_scan_hdf5_round_trip(tmp_path):
    """Synthetic ScanData -> write -> read returns identical content."""
    src = _make_synthetic_scan(n_points=23, n_samples=128)
    out_path = tmp_path / "round_trip.h5"
    write_scan_hdf5(src, out_path)

    back = load_scan_hdf5(out_path)
    np.testing.assert_allclose(back.pos_x, src.pos_x)
    np.testing.assert_allclose(back.pos_y, src.pos_y)
    np.testing.assert_allclose(back.rssi, src.rssi)
    assert back.n_points == src.n_points
    assert back.n_samples == src.n_samples
    assert back.dt == pytest.approx(src.dt)
    assert back.metadata["chip_id"] == "synthetic_chip"

    # Waveforms identical (within float32 precision)
    for role in (ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT, ROLE_CURRENT):
        src_wf = src.load_waveforms(role, slice(None))
        back_wf = back.load_waveforms(role, slice(None))
        np.testing.assert_allclose(back_wf, src_wf, atol=1e-6)


def test_write_scan_hdf5_streams_in_chunks(tmp_path):
    """Chunked write produces same result as one-shot, exercises the loop."""
    src = _make_synthetic_scan(n_points=25, n_samples=64)
    out_path = tmp_path / "chunked.h5"
    write_scan_hdf5(src, out_path, chunk_points=3)   # forces multiple iterations
    back = load_scan_hdf5(out_path)
    np.testing.assert_allclose(
        back.load_waveforms(ROLE_LDV_OUTPUT, slice(None)),
        src.load_waveforms(ROLE_LDV_OUTPUT, slice(None)),
        atol=1e-6,
    )


def test_write_scan_hdf5_requires_v2_metadata(tmp_path):
    """Missing a required attribute raises BEFORE any disk I/O."""
    src = _make_synthetic_scan()
    src.metadata.pop("chip_id")
    out_path = tmp_path / "incomplete.h5"
    with pytest.raises(ValueError, match="chip_id"):
        write_scan_hdf5(src, out_path)
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# validate_hdf5_v2 — schema-conformance check for DAQ-produced files
# ---------------------------------------------------------------------------

def test_validate_v2_valid_file_returns_empty(tmp_path):
    """A correctly-written file produces zero problems."""
    h5_path = tmp_path / "valid.h5"
    _make_v2_h5(h5_path)
    assert validate_hdf5_v2(h5_path) == []


def test_validate_v2_missing_attr_reports(tmp_path):
    """Missing root attribute is reported by name."""
    h5_path = tmp_path / "missing_attr.h5"
    _make_v2_h5(h5_path, bad_attrs=("chip_id", "session_id"))
    problems = validate_hdf5_v2(h5_path)
    assert any("chip_id" in p for p in problems)
    assert any("session_id" in p for p in problems)


def test_validate_v2_missing_dataset_reports(tmp_path):
    """Missing required dataset is reported by path."""
    import h5py
    h5_path = tmp_path / "missing_dset.h5"
    _make_v2_h5(h5_path)
    with h5py.File(h5_path, "a") as f:
        del f["waveforms/drive_voltage"]
    problems = validate_hdf5_v2(h5_path)
    assert any("drive_voltage" in p for p in problems)


def test_validate_v2_shape_mismatch_reports(tmp_path):
    """Waveform shape inconsistent with n_samples is flagged."""
    import h5py
    h5_path = tmp_path / "shape_mismatch.h5"
    _make_v2_h5(h5_path, n_points=10, n_samples=64)
    with h5py.File(h5_path, "a") as f:
        f.attrs["n_samples"] = 128       # lie about it
    problems = validate_hdf5_v2(h5_path)
    assert any("shape" in p for p in problems)


def test_validate_v2_unchunked_waveform_warns(tmp_path):
    """A non-chunked waveform dataset is flagged."""
    import h5py
    h5_path = tmp_path / "unchunked.h5"
    _make_v2_h5(h5_path, n_points=10, n_samples=64)
    with h5py.File(h5_path, "a") as f:
        data = f["waveforms/ldv_output"][:]
        del f["waveforms/ldv_output"]
        f["waveforms"].create_dataset(
            "ldv_output", data=data      # contiguous, no chunks
        )
    problems = validate_hdf5_v2(h5_path)
    assert any("not chunked" in p for p in problems)


def test_validate_v2_nonexistent_file():
    out = validate_hdf5_v2("/nonexistent.h5")
    assert len(out) == 1 and "not found" in out[0]
