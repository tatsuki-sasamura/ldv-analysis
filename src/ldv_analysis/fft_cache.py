# %%
"""Shared FFT cache for burst-mode TDMS scan files.

Computes per-scan-point 1f quantities from the steady-state burst window
and saves results to an .npz file.  Subsequent runs load the cache
directly, skipping the expensive TDMS read + FFT.

Cached quantities
-----------------
- pos_x, pos_y : scan positions (m)
- n_x_meta, n_y_meta : grid dimensions from TDMS metadata
- f_drive : drive frequency (Hz)
- dt, n_samples : waveform timing
- burst_on_us, burst_off_us, ss_start, ss_end : burst / FFT window
- velocity_{n}f, pressure_{n}f, phase_{n}f : Ch2 acoustic at n×f_drive (n=1..5)
- pt_burst_on_us, pt_burst_off_us : per-point burst timing (µs)
- noise_rms_velocity, noise_rms_pressure : post-burst noise RMS (NaN for continuous)
- voltage_1f : Ch1 drive voltage (after attenuation correction)
- current_1f, impedance_1f, phase_vi : Ch4 electrical (if Ch4 exists)
- rssi : RSSI (if available)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldv_analysis.config import (
    CURRENT_SCALE,
    VELOCITY_SCALE,
    VOLTAGE_ATTENUATION,
    velocity_to_pressure,
)
from ldv_analysis.io_utils import (
    ROLE_CURRENT, ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT,
    ScanData, load_scan,
)

# ---------------------------------------------------------------------------
# Cache version — bump to force recomputation of all caches
# ---------------------------------------------------------------------------
# v7: ScanData-based pipeline (format-agnostic). Cache also records
#     source_format / source_mtime so a TDMS and HDF5 with the same
#     stem don't silently share a cache file.
_CACHE_VERSION = 7
MAX_HARMONIC = 5

# ---------------------------------------------------------------------------
# Burst detection / FFT constants
# ---------------------------------------------------------------------------
ENVELOPE_CHUNK = 1000
ON_THRESHOLD_FACTOR = 3.0
RING_UP_US = 100.0      # µs to skip after burst ON
RING_DOWN_US = 10.0      # µs to skip before burst OFF
NOISE_SKIP_US = 100.0    # µs to skip after burst OFF for noise measurement
CHUNK_SIZE = 500
BURST_TIMING_CHUNK = 1000  # samples per RMS chunk for per-point burst detection

def _per_point_burst_timing(
    wf_ch1: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point burst ON/OFF times from Ch1 waveforms.

    Parameters
    ----------
    wf_ch1 : array, shape (n_points_chunk, n_samples)
        Ch1 waveforms for a chunk of scan points.
    dt : float
        Sample interval in seconds.

    Returns
    -------
    on_us, off_us : arrays, shape (n_points_chunk,)
        Burst ON and OFF times in microseconds.  NaN if no burst detected.
    """
    n_pts, n_samp = wf_ch1.shape
    n_env = n_samp // BURST_TIMING_CHUNK
    on_us = np.full(n_pts, np.nan)
    off_us = np.full(n_pts, np.nan)

    for i in range(n_pts):
        rms = np.array([
            np.sqrt(np.mean(
                wf_ch1[i, j * BURST_TIMING_CHUNK:(j + 1) * BURST_TIMING_CHUNK] ** 2))
            for j in range(n_env)
        ])
        rms_max = np.max(rms)
        if rms_max == 0:
            continue
        above = np.where(rms > 0.3 * rms_max)[0]
        if len(above) > 0:
            on_us[i] = float(above[0] * BURST_TIMING_CHUNK * dt * 1e6)
            off_us[i] = float((above[-1] + 1) * BURST_TIMING_CHUNK * dt * 1e6)

    return on_us, off_us


# ---------------------------------------------------------------------------
# FFT utilities (reusable outside fft_cache)
# ---------------------------------------------------------------------------

def find_drive_frequency(waveform: np.ndarray, dt: float) -> float:
    """Find the drive frequency from a waveform using parabolic interpolation.

    Computes the FFT, finds the dominant peak (excluding DC), and refines
    the frequency estimate with log-parabolic interpolation for sub-bin
    accuracy.

    Parameters
    ----------
    waveform : 1-D array
        Time-domain waveform (e.g. Ch1 voltage).
    dt : float
        Sample interval in seconds.

    Returns
    -------
    float
        Drive frequency in Hz.
    """
    fft_full = np.fft.rfft(waveform)
    mag = np.abs(fft_full)
    k = int(np.argmax(mag[1:]) + 1)
    df = 1.0 / (len(waveform) * dt)
    # Parabolic interpolation on log-magnitude for sub-bin accuracy
    if 1 <= k < len(mag) - 1 and mag[k - 1] > 0 and mag[k + 1] > 0:
        alpha = np.log(mag[k - 1])
        beta = np.log(mag[k])
        gamma = np.log(mag[k + 1])
        denom = alpha - 2 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-30 else 0.0
    else:
        delta = 0.0
    return float((k + delta) * df)


def wrap_phase(phase_deg: np.ndarray) -> np.ndarray:
    """Wrap phase angle(s) to [-180, 180] degrees."""
    return (phase_deg + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------

@dataclass
class BurstWindow:
    """Result of burst detection on a waveform."""

    burst_on_us: float   # burst ON time (µs)
    burst_off_us: float  # burst OFF time (µs)
    ss_start: int        # steady-state FFT window start (sample index)
    ss_end: int          # steady-state FFT window end (sample index)
    continuous: bool     # True if no burst detected (continuous excitation)
    noise_start: int     # post-burst noise start (sample index), -1 if N/A
    has_noise: bool      # True if usable post-burst noise segment exists


def detect_burst_window(
    ch1_ref: np.ndarray,
    n_samples: int,
    dt: float,
) -> BurstWindow:
    """Detect burst ON/OFF boundaries and compute the steady-state FFT window.

    Uses the RMS envelope of a reference Ch1 waveform to distinguish
    continuous excitation from burst mode, and determines the analysis
    window accordingly.

    Parameters
    ----------
    ch1_ref : 1-D array
        Reference Ch1 waveform (strongest signal among probed points).
    n_samples : int
        Total number of samples per waveform.
    dt : float
        Sample interval in seconds.

    Returns
    -------
    BurstWindow
        Burst timing and FFT window parameters.
    """
    n_env_chunks = n_samples // ENVELOPE_CHUNK
    rms_env = np.array([
        np.sqrt(np.mean(
            ch1_ref[i * ENVELOPE_CHUNK:(i + 1) * ENVELOPE_CHUNK] ** 2))
        for i in range(n_env_chunks)
    ])
    rms_min = np.min(rms_env)
    rms_max = np.max(rms_env)
    continuous = rms_min > 0.3 * rms_max  # flat envelope → continuous

    if continuous:
        margin = int(10e-6 / dt)  # 10 µs
        burst_on_us = 0.0
        burst_off_us = float(n_samples * dt * 1e6)
        ss_start = margin
        ss_end = n_samples - margin
        print(f"  Continuous excitation detected")
    else:
        noise_floor = np.median(rms_env[-max(n_env_chunks // 10, 1):])
        on_mask = rms_env > ON_THRESHOLD_FACTOR * noise_floor
        on_indices = np.where(on_mask)[0]
        burst_on_us = float(on_indices[0] * ENVELOPE_CHUNK * dt * 1e6)
        burst_off_us = float((on_indices[-1] + 1) * ENVELOPE_CHUNK * dt * 1e6)
        print(f"  Burst ON: {burst_on_us:.0f}--{burst_off_us:.0f} us")

        ss_start = int((burst_on_us + RING_UP_US) * 1e-6 / dt)
        ss_end = int((burst_off_us - RING_DOWN_US) * 1e-6 / dt)

    ss_n = ss_end - ss_start
    print(f"  FFT window: {ss_start * dt * 1e6:.0f}"
          f"--{ss_end * dt * 1e6:.0f} us "
          f"({ss_n} samples, df = {1/(ss_n*dt):.0f} Hz)")

    # Post-burst noise segment (burst mode only)
    noise_start = -1
    has_noise = False
    if not continuous:
        burst_off_sample = int(burst_off_us * 1e-6 / dt)
        noise_start = burst_off_sample + int(NOISE_SKIP_US * 1e-6 / dt)
        noise_n = n_samples - noise_start
        has_noise = noise_n > 100
        if has_noise:
            print(f"  Noise segment: {noise_start * dt * 1e6:.0f}"
                  f"--{n_samples * dt * 1e6:.0f} us ({noise_n} samples)")
        else:
            print("  No usable post-burst noise segment")

    return BurstWindow(
        burst_on_us=burst_on_us,
        burst_off_us=burst_off_us,
        ss_start=ss_start,
        ss_end=ss_end,
        continuous=continuous,
        noise_start=noise_start,
        has_noise=has_noise,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_velocity_scale(tdms_path: str | Path) -> float:
    """Detect LDV velocity scale from TDMS filename pattern.

    Looks for ``_(\\d+)m_s_max`` in the filename stem.  The Polytec
    decoder full-scale range (m/s) maps to m/s/V by dividing by 2
    (PicoScope 1 MΩ input sees 2× the open-circuit voltage).

    Falls back to :data:`VELOCITY_SCALE` (1.0 m/s/V) if the pattern
    is not found.
    """
    m = re.search(r"_(\d+)m_s_max", Path(tdms_path).stem)
    if m:
        return int(m.group(1)) / 2.0
    return VELOCITY_SCALE


# Backward-compatible alias: channel-number -> role mapping
_CHANNEL_TO_ROLE = {
    1: ROLE_DRIVE_VOLTAGE,
    2: ROLE_LDV_OUTPUT,
    4: ROLE_CURRENT,
}


def load_or_compute(
    scan_path: str | Path,
    cache_dir: str | Path,
    velocity_scale: float | None = None,
) -> np.lib.npyio.NpzFile:
    """Load FFT cache if available, otherwise compute from the scan file.

    Format-agnostic: ``scan_path`` may be ``.tdms`` (v1) or ``.h5`` (v2).
    The reader is chosen by extension via ``load_scan()``.

    Parameters
    ----------
    scan_path : str or Path
        Path to the acquisition file (.tdms or .h5).
    cache_dir : str or Path
        Directory where cache .npz files are stored.
    velocity_scale : float or None
        LDV decoder scale in m/s/V. If *None*, read from
        ``scan.metadata["ldv_velocity_scale_mps_per_v"]``; for v1 TDMS
        this is auto-detected from the filename pattern.
    """
    scan_path = Path(scan_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"_fft_cache_{scan_path.stem}.npz"

    source_format = "tdms_v1" if scan_path.suffix.lower() == ".tdms" else "hdf5_v2"
    source_mtime = scan_path.stat().st_mtime if scan_path.exists() else 0.0

    if cache_path.exists():
        cache = np.load(cache_path)
        version = int(cache["cache_version"]) if "cache_version" in cache else 1
        cached_format = (
            str(cache["source_format"]) if "source_format" in cache else None
        )
        cached_mtime = (
            float(cache["source_mtime"]) if "source_mtime" in cache else 0.0
        )
        if (version >= _CACHE_VERSION
                and "dt" in cache and "phase_2f" in cache
                and "noise_rms_velocity" in cache
                and cached_format == source_format
                and abs(cached_mtime - source_mtime) < 1.0):
            if velocity_scale is not None:
                cached_scale = float(cache["velocity_scale"])
                if not np.isclose(cached_scale, velocity_scale):
                    print(f"  Cache scale mismatch "
                          f"({cached_scale} vs {velocity_scale}), recomputing...")
                else:
                    print(f"  Using FFT cache: {cache_path.name}")
                    return cache
            else:
                print(f"  Using FFT cache: {cache_path.name}")
                return cache
        else:
            print("  Cache outdated, recomputing...")

    return _compute(scan_path, cache_path, velocity_scale)


def load_point_waveforms(
    scan_path: str | Path,
    point_index: int,
    roles: tuple[str, ...] = (ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT),
    *,
    channels: tuple[int, ...] | None = None,  # deprecated
) -> tuple[dict[str, np.ndarray], float]:
    """Load raw waveforms for one scan point, by semantic role.

    Format-agnostic: dispatches by extension to TDMS or HDF5 readers.

    Parameters
    ----------
    scan_path : str or Path
        Acquisition file path (.tdms or .h5).
    point_index : int
        Scan point index (0-based).
    roles : tuple of str
        Channel roles to load (e.g. ``("drive_voltage", "ldv_output")``).
    channels : tuple of int, optional
        Backward-compat shim for the old channel-number API. Maps
        1→drive_voltage, 2→ldv_output, 4→current. Prefer ``roles``.

    Returns
    -------
    waveforms : dict mapping role to 1-D ndarray
    dt : float (seconds)
    """
    if channels is not None:
        roles = tuple(_CHANNEL_TO_ROLE[ch] for ch in channels)

    scan = load_scan(scan_path)
    point_indices = np.array([point_index], dtype=int)
    return (
        {role: scan.load_waveforms(role, point_indices)[0] for role in roles},
        scan.dt,
    )


# ---------------------------------------------------------------------------
# Internal: _select_burst_reference and _compute
# ---------------------------------------------------------------------------

def _select_burst_reference(
    scan: ScanData,
) -> tuple[int, np.ndarray]:
    """Probe ~5 scan points to pick the best drive_voltage reference.

    Point 0 may be at channel edge with near-zero signal; the strongest
    point gives a cleaner burst envelope. Format-agnostic via
    ``scan.load_waveforms``.
    """
    n_points = scan.n_points
    n_samples = scan.n_samples
    n_env_chunks = n_samples // ENVELOPE_CHUNK
    probe_indices = sorted(set(
        [0, n_points // 4, n_points // 2, 3 * n_points // 4, n_points - 1]
    ))
    waveforms = scan.load_waveforms(
        ROLE_DRIVE_VOLTAGE, np.array(probe_indices)
    )
    best_rms = -1.0
    ref_idx = probe_indices[0]
    ch1_ref = waveforms[0]
    for k, pi in enumerate(probe_indices):
        wf = waveforms[k]
        rms_env = np.array([
            np.sqrt(np.mean(wf[i * ENVELOPE_CHUNK:(i + 1) * ENVELOPE_CHUNK] ** 2))
            for i in range(n_env_chunks)
        ])
        peak = float(np.max(rms_env))
        if peak > best_rms:
            best_rms = peak
            ref_idx = pi
            ch1_ref = wf
    print(f"  Burst ref: point {ref_idx} (of {n_points}), peak RMS = {best_rms:.4f}")
    return ref_idx, ch1_ref


def _compute(scan_path: Path, cache_path: Path,
             velocity_scale: float | None = None) -> np.lib.npyio.NpzFile:
    """Read scan, compute 1f-5f FFT quantities, save and return cache.

    Format-agnostic: ``scan_path`` may be ``.tdms`` or ``.h5``.
    """
    scan = load_scan(scan_path)

    # velocity_scale: prefer caller > scan metadata > legacy filename heuristic
    if velocity_scale is None:
        velocity_scale = scan.metadata.get("ldv_velocity_scale_mps_per_v")
    if velocity_scale is None:
        velocity_scale = detect_velocity_scale(scan_path)

    n_x_meta = int(scan.metadata.get("scan_n_x", scan.metadata.get("n_x", 0)) or 0)
    n_y_meta = int(scan.metadata.get("scan_n_y", scan.metadata.get("n_y", 0)) or 0)
    print(f"  Grid: {n_x_meta} x {n_y_meta} y")

    pos_x = scan.pos_x
    pos_y = scan.pos_y
    n_points = scan.n_points
    rssi = scan.rssi
    dt = scan.dt
    n_samples = scan.n_samples
    print(f"  {n_points} points, {n_samples} samples, dt = {dt*1e9:.0f} ns")

    has_current = ROLE_CURRENT in scan.metadata.get("_available_roles", [])

    # --- Burst detection ---
    _, ch1_ref = _select_burst_reference(scan)
    bw = detect_burst_window(ch1_ref, n_samples, dt)

    # --- Drive frequency ---
    f_drive = find_drive_frequency(ch1_ref, dt)
    print(f"  Drive: {f_drive / 1e6:.6f} MHz")
    del ch1_ref

    # --- Chunked DFT processing ---
    ss_n = bw.ss_end - bw.ss_start
    ss_time = np.arange(ss_n) * dt
    tones = {h: np.exp(-2j * np.pi * h * f_drive * ss_time)
             for h in range(1, MAX_HARMONIC + 1)}

    velocity = {h: np.empty(n_points) for h in range(1, MAX_HARMONIC + 1)}
    pressure = {h: np.empty(n_points) for h in range(1, MAX_HARMONIC + 1)}
    phase = {h: np.empty(n_points) for h in range(1, MAX_HARMONIC + 1)}
    voltage_1f = np.empty(n_points)
    noise_rms_velocity = np.full(n_points, np.nan)
    noise_rms_pressure = np.full(n_points, np.nan)
    pt_burst_on_us = np.full(n_points, np.nan)
    pt_burst_off_us = np.full(n_points, np.nan)
    if has_current:
        current_1f = np.empty(n_points)
        impedance_1f = np.empty(n_points)
        phase_vi = np.empty(n_points)

    n_chunks = (n_points + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Computing FFT in {n_chunks} chunks...")

    for ci in range(n_chunks):
        i0 = ci * CHUNK_SIZE
        i1 = min(i0 + CHUNK_SIZE, n_points)

        wf1 = scan.load_waveforms(ROLE_DRIVE_VOLTAGE, slice(i0, i1))
        wf2 = scan.load_waveforms(ROLE_LDV_OUTPUT, slice(i0, i1))

        # Per-point burst timing from drive voltage
        _on, _off = _per_point_burst_timing(wf1, dt)
        pt_burst_on_us[i0:i1] = _on
        pt_burst_off_us[i0:i1] = _off

        # Exact-frequency DFT via dot product (no scalloping loss)
        ss_seg1 = wf1[:, bw.ss_start:bw.ss_end]
        ss_seg2 = wf2[:, bw.ss_start:bw.ss_end]
        dft1 = ss_seg1 @ tones[1]  # drive_voltage at 1f (phase reference)

        for h in range(1, MAX_HARMONIC + 1):
            dft2_h = ss_seg2 @ tones[h]
            vel_h = np.abs(dft2_h) * 2 / ss_n * velocity_scale
            velocity[h][i0:i1] = vel_h
            pressure[h][i0:i1] = vel_h * abs(velocity_to_pressure(h * f_drive))
            phase[h][i0:i1] = wrap_phase(
                np.degrees(np.angle(dft2_h) - np.angle(dft1)))

        # Post-burst noise RMS (ldv_output)
        if bw.has_noise:
            noise_rms = np.sqrt(np.mean(wf2[:, bw.noise_start:] ** 2, axis=1))
            noise_rms_velocity[i0:i1] = noise_rms * velocity_scale

        # Drive voltage 1f (with probe attenuation)
        voltage_1f[i0:i1] = np.abs(dft1) * 2 / ss_n * VOLTAGE_ATTENUATION

        # Current
        if has_current:
            wf4 = scan.load_waveforms(ROLE_CURRENT, slice(i0, i1))
            dft4 = wf4[:, bw.ss_start:bw.ss_end] @ tones[1]
            cur = np.abs(dft4) * 2 / ss_n * CURRENT_SCALE
            current_1f[i0:i1] = cur
            impedance_1f[i0:i1] = voltage_1f[i0:i1] / cur
            phase_vi[i0:i1] = wrap_phase(
                np.degrees(np.angle(dft1) - np.angle(dft4)))

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci + 1}/{n_chunks} done")

    if bw.has_noise:
        noise_rms_pressure = noise_rms_velocity * abs(velocity_to_pressure(f_drive))

    # --- Assemble and save ---
    source_format = scan.metadata.get(
        "source_format",
        "tdms_v1" if scan_path.suffix.lower() == ".tdms" else "hdf5_v2",
    )
    source_mtime = scan_path.stat().st_mtime

    arrays = dict(
        cache_version=np.array(_CACHE_VERSION),
        source_format=np.array(source_format),
        source_mtime=np.array(source_mtime),
        velocity_scale=np.array(velocity_scale),
        pos_x=pos_x, pos_y=pos_y,
        n_x_meta=np.array(n_x_meta), n_y_meta=np.array(n_y_meta),
        f_drive=np.array(f_drive),
        dt=np.array(dt), n_samples=np.array(n_samples),
        burst_on_us=np.array(bw.burst_on_us),
        burst_off_us=np.array(bw.burst_off_us),
        pt_burst_on_us=pt_burst_on_us,
        pt_burst_off_us=pt_burst_off_us,
        ss_start=np.array(bw.ss_start), ss_end=np.array(bw.ss_end),
        noise_rms_velocity=noise_rms_velocity,
        noise_rms_pressure=noise_rms_pressure,
        voltage_1f=voltage_1f,
    )
    for h in range(1, MAX_HARMONIC + 1):
        suffix = f"_{h}f"
        arrays[f"velocity{suffix}"] = velocity[h]
        arrays[f"pressure{suffix}"] = pressure[h]
        arrays[f"phase{suffix}"] = phase[h]
    if rssi is not None:
        arrays["rssi"] = rssi
    if has_current:
        arrays["current_1f"] = current_1f
        arrays["impedance_1f"] = impedance_1f
        arrays["phase_vi"] = phase_vi

    np.savez(cache_path, **arrays)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  Cache saved: {cache_path.name} ({size_mb:.1f} MB)")
    return np.load(cache_path)
