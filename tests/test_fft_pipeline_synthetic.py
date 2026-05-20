"""Synthetic-data integration test for the FFT cache pipeline (``_compute``).

Builds a small ``ScanData`` in memory with KNOWN per-point amplitudes and
phases at 1f and 2f, writes it to an HDF5 v2 file, then runs
``load_or_compute`` end-to-end and verifies every cached quantity matches
the analytical expectation.

Why this exists
---------------
- ``tests/test_fft.py`` covers the primitives (``find_drive_frequency``,
  the dot-product DFT, ``wrap_phase``) on synthetic data — but in
  isolation, not through ``_compute``'s orchestration.
- ``tests/test_io_utils.py::test_fft_cache_equivalent_across_formats``
  runs the full pipeline, but is regression-to-self: it verifies the
  TDMS and HDF5 read paths agree, not that the output is correct.
- This file plugs the remaining gap: full pipeline (read → burst window
  → drive-freq estimate → per-point chunked DFT → voltage/current/
  impedance scaling) on data whose true 1f / 2f amplitudes and phases
  we know analytically.

The signal is designed so the DFT has no spectral leakage: ``F_DRIVE``
is an integer number of cycles in BOTH the full window and the
continuous-mode steady-state window, so recovered amplitudes match
the synthesized values to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldv_analysis.config import (
    CURRENT_SCALE,
    VOLTAGE_ATTENUATION,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute, wrap_phase
from ldv_analysis.io_utils import (
    ROLE_CURRENT,
    ROLE_DRIVE_VOLTAGE,
    ROLE_LDV_OUTPUT,
    ScanData,
    write_scan_hdf5,
)

# ---------------------------------------------------------------------------
# Parameters chosen so DFT bins line up with the synthesized frequencies
# (zero spectral leakage at f, 2f in BOTH the full window and the
# continuous-mode steady-state window).
# ---------------------------------------------------------------------------
DT = 8e-9                # 125 MS/s — matches Pico5442D default
N_SAMPLES = 25_000       # 200 µs total
F_DRIVE = 1.90e6         # 380 cycles in full window, 342 in ss window
N_POINTS = 25
N_X, N_Y = 5, 5
VEL_SCALE = 0.5          # m/s per V (Polytec 1 m/s decoder, 1 MΩ load)

# detect_burst_window() declares "continuous" when rms_min > 0.3·rms_max.
# Noise-free sine waves satisfy this, so the ss window is predictable.
_MARGIN_S = 10e-6
SS_START = int(_MARGIN_S / DT)
SS_END = N_SAMPLES - SS_START


# ---------------------------------------------------------------------------
# Synthetic-data construction
# ---------------------------------------------------------------------------

def _per_point_params(n_points: int) -> dict:
    """Per-point amplitudes/phases. Smoothly varying so any indexing bug
    leaves a visible spatial pattern."""
    p = np.arange(n_points)
    omega = 2 * np.pi * p / n_points
    deg = np.deg2rad
    return dict(
        # Drive voltage — small variation around 0.5 V; phase per point
        A_drive=0.50 + 0.05 * np.sin(omega),
        phi_drive=0.4 + 0.05 * np.cos(omega),
        # LDV @ 1f — peak 0.30 V, +30° vs drive at p=0
        A_ldv_1=0.20 + 0.10 * np.sin(omega),
        phi_ldv_1=0.4 + 0.05 * np.cos(omega) + deg(30.0),
        # LDV @ 2f — peak 0.07 V, +45° vs drive at p=0
        A_ldv_2=0.05 + 0.02 * np.cos(omega),
        phi_ldv_2=0.4 + 0.05 * np.cos(omega) + deg(45.0),
        # Current — peak 0.012 V, -70° (capacitive)
        A_cur=0.010 + 0.002 * np.sin(omega),
        phi_cur=0.4 + 0.05 * np.cos(omega) + deg(-70.0),
    )


def _build_waveforms(params, n_samples, dt, f):
    """Synthesize (drive, ldv = 1f + 2f, current) waveform arrays."""
    t = np.arange(n_samples) * dt
    twopi = 2 * np.pi
    drive = (params["A_drive"][:, None]
             * np.sin(twopi * f * t + params["phi_drive"][:, None]))
    ldv = (params["A_ldv_1"][:, None]
           * np.sin(twopi * f * t + params["phi_ldv_1"][:, None])
           + params["A_ldv_2"][:, None]
             * np.sin(twopi * 2 * f * t + params["phi_ldv_2"][:, None]))
    current = (params["A_cur"][:, None]
               * np.sin(twopi * f * t + params["phi_cur"][:, None]))
    return drive, ldv, current


def _make_scan(*, include_current: bool = True) -> tuple[ScanData, dict]:
    """Return (ScanData with in-memory waveforms, params dict)."""
    params = _per_point_params(N_POINTS)
    drive, ldv, current = _build_waveforms(params, N_SAMPLES, DT, F_DRIVE)

    pos_x = np.tile(np.linspace(0.0, 1e-3, N_X), N_Y)        # (25,) m
    pos_y = np.repeat(np.linspace(0.0, 2e-3, N_Y), N_X)
    rssi = np.full(N_POINTS, 2.0, dtype=np.float32)

    bank = {ROLE_DRIVE_VOLTAGE: drive, ROLE_LDV_OUTPUT: ldv}
    if include_current:
        bank[ROLE_CURRENT] = current

    def loader(role, points):
        return bank[role][points]

    available = list(bank.keys())
    metadata = {
        # Required v2 root attrs
        "version": "2.0",
        "sample_rate_hz": 1.0 / DT,
        "n_samples": N_SAMPLES,
        "ldv_velocity_scale_mps_per_v": VEL_SCALE,
        "drive_frequency_hz_nominal": F_DRIVE,
        "drive_voltage_vpp": 1.0,
        "burst_on_us_nominal": 0.0,
        "burst_off_us_nominal": N_SAMPLES * DT * 1e6,
        "scan_n_x": N_X,
        "scan_n_y": N_Y,
        "chip_id": "test_synthetic",
        "session_id": "test_fft_pipeline_synthetic",
        "timestamp_utc": "2026-05-20T00:00:00Z",
        "operator": "pytest",
        "daq_software_version": "synthetic",
        # Role discovery
        "_available_roles": available,
        "channel_roles": {r: r for r in available},
    }

    return ScanData(
        pos_x=pos_x,
        pos_y=pos_y,
        rssi=rssi,
        dt=DT,
        n_points=N_POINTS,
        n_samples=N_SAMPLES,
        metadata=metadata,
        _loader=loader,
    ), params


@pytest.fixture
def synthetic_cache(tmp_path):
    """Write synthetic HDF5, run the FFT pipeline, return (cache, params)."""
    scan, params = _make_scan(include_current=True)
    h5_path = tmp_path / "synthetic.h5"
    # float64 keeps the round-trip bit-exact — leakage-free signals
    # then give recovery to ~machine precision.
    write_scan_hdf5(scan, h5_path, waveform_dtype="float64")
    cache = load_or_compute(h5_path, tmp_path / "cache")
    return cache, params


# ---------------------------------------------------------------------------
# Tests — one assertion class per test
# ---------------------------------------------------------------------------

def test_drive_frequency_recovered(synthetic_cache):
    """f_drive recovered within sub-bin precision."""
    cache, _ = synthetic_cache
    df_full = 1.0 / (N_SAMPLES * DT)
    err = abs(float(cache["f_drive"]) - F_DRIVE)
    assert err < 1e-3 * df_full, (
        f"f_drive recovered with {err:.3f} Hz error, bin={df_full:.1f} Hz"
    )


def test_continuous_mode_window_and_noise(synthetic_cache):
    """Continuous excitation → predictable ss window, NaN noise RMS."""
    cache, _ = synthetic_cache
    assert int(cache["ss_start"]) == SS_START
    assert int(cache["ss_end"]) == SS_END
    assert np.all(np.isnan(cache["noise_rms_velocity"]))
    assert np.all(np.isnan(cache["noise_rms_pressure"]))


def test_voltage_1f_recovered(synthetic_cache):
    """voltage_1f = A_drive · VOLTAGE_ATTENUATION (×10 probe)."""
    cache, params = synthetic_cache
    expected = params["A_drive"] * VOLTAGE_ATTENUATION
    np.testing.assert_allclose(cache["voltage_1f"], expected, rtol=1e-7)


def test_velocity_1f_recovered(synthetic_cache):
    """velocity_1f = A_ldv_1 · velocity_scale (mode-shape-like variation)."""
    cache, params = synthetic_cache
    expected = params["A_ldv_1"] * VEL_SCALE
    np.testing.assert_allclose(cache["velocity_1f"], expected, rtol=1e-7)


def test_velocity_2f_recovered_without_1f_leakage(synthetic_cache):
    """2f is recovered cleanly; 1f does NOT alias into the 2f estimate."""
    cache, params = synthetic_cache
    expected = params["A_ldv_2"] * VEL_SCALE
    np.testing.assert_allclose(cache["velocity_2f"], expected, rtol=1e-6)


def test_pressure_matches_velocity_to_pressure_scaling(synthetic_cache):
    """pressure_Nf = velocity_Nf · |velocity_to_pressure(N·f_drive_est)|.

    The scaling uses the ESTIMATED ``f_drive`` (cache["f_drive"]), which
    sits within parabolic-interp residue (<0.1 Hz) of the synthesized
    F_DRIVE. We use the estimated value here so the test is checking
    the relationship encoded in the code, not the absolute scaling.
    """
    cache, _ = synthetic_cache
    f_used = float(cache["f_drive"])
    np.testing.assert_allclose(
        cache["pressure_1f"],
        cache["velocity_1f"] * abs(velocity_to_pressure(f_used)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cache["pressure_2f"],
        cache["velocity_2f"] * abs(velocity_to_pressure(2 * f_used)),
        rtol=1e-12,
    )


def test_current_and_impedance_recovered(synthetic_cache):
    """current_1f = A_cur · CURRENT_SCALE; impedance_1f = V_1f / I_1f."""
    cache, params = synthetic_cache
    expected_i = params["A_cur"] * CURRENT_SCALE
    expected_v = params["A_drive"] * VOLTAGE_ATTENUATION
    np.testing.assert_allclose(cache["current_1f"], expected_i, rtol=1e-7)
    np.testing.assert_allclose(
        cache["impedance_1f"], expected_v / expected_i, rtol=1e-7,
    )


def test_phase_1f_relative_to_drive(synthetic_cache):
    """phase_1f = (φ_ldv_1 - φ_drive) mod 360°."""
    cache, params = synthetic_cache
    expected = wrap_phase(np.degrees(params["phi_ldv_1"] - params["phi_drive"]))
    diff = wrap_phase(cache["phase_1f"] - expected)
    assert np.max(np.abs(diff)) < 1e-4, (
        f"max |Δphase_1f| = {np.max(np.abs(diff)):.3e} deg"
    )


def test_phase_2f_relative_to_drive_1f(synthetic_cache):
    """phase_2f = (φ_ldv_2 - φ_drive) — note: ref is drive @ 1f, not 2f.

    Larger tolerance than phase_1f / phase_vi because phase_2f carries a
    constant bias π·δ·(ss_n−1)·dt from the parabolic-interp residue
    δ ≈ 0.06 Hz in f_drive_est. The bias is real — the test verifies
    it stays well below 0.01° (its observed magnitude ~0.002°), not
    that it's zero.
    """
    cache, params = synthetic_cache
    expected = wrap_phase(np.degrees(params["phi_ldv_2"] - params["phi_drive"]))
    diff = wrap_phase(cache["phase_2f"] - expected)
    assert np.max(np.abs(diff)) < 1e-2


def test_phase_vi_drive_minus_current(synthetic_cache):
    """phase_vi = (φ_drive - φ_cur) — V leads I for capacitive load."""
    cache, params = synthetic_cache
    expected = wrap_phase(np.degrees(params["phi_drive"] - params["phi_cur"]))
    diff = wrap_phase(cache["phase_vi"] - expected)
    assert np.max(np.abs(diff)) < 1e-4


def test_coordinates_and_grid_metadata_pass_through(synthetic_cache):
    """pos_x/pos_y/n_x_meta/n_y_meta/dt/n_samples round-trip unchanged."""
    cache, _ = synthetic_cache
    np.testing.assert_array_equal(
        cache["pos_x"], np.tile(np.linspace(0.0, 1e-3, N_X), N_Y),
    )
    np.testing.assert_array_equal(
        cache["pos_y"], np.repeat(np.linspace(0.0, 2e-3, N_Y), N_X),
    )
    assert int(cache["n_x_meta"]) == N_X
    assert int(cache["n_y_meta"]) == N_Y
    assert float(cache["dt"]) == DT
    assert int(cache["n_samples"]) == N_SAMPLES


def test_pipeline_without_current_channel(tmp_path):
    """If no current dataset, V/I/Z keys are absent but everything else
    still works."""
    scan, params = _make_scan(include_current=False)
    h5_path = tmp_path / "synthetic_no_current.h5"
    write_scan_hdf5(scan, h5_path, waveform_dtype="float64")
    cache = load_or_compute(h5_path, tmp_path / "cache")

    np.testing.assert_allclose(
        cache["velocity_1f"], params["A_ldv_1"] * VEL_SCALE, rtol=1e-7,
    )
    np.testing.assert_allclose(
        cache["voltage_1f"], params["A_drive"] * VOLTAGE_ATTENUATION,
        rtol=1e-7,
    )
    for absent_key in ("current_1f", "impedance_1f", "phase_vi"):
        assert absent_key not in cache.files, (
            f"{absent_key} should not be in cache without current channel"
        )
