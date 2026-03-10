# %%
"""Shared FFT cache for burst-mode TDMS scan files.

Computes per-scan-point 1f quantities from the steady-state burst window
and saves results to an .npz file.  Subsequent runs load the cache
directly, skipping the expensive TDMS read + FFT.

Cached quantities
-----------------
- pos_x, pos_y : scan positions (mm)
- n_x_meta, n_y_meta : grid dimensions from TDMS metadata
- f_drive : drive frequency (Hz)
- dt, n_samples : waveform timing
- burst_on_us, burst_off_us, ss_start, ss_end : burst / FFT window
- velocity_1f, pressure_1f, phase_1f : Ch2 acoustic
- voltage_1f : Ch1 drive voltage (after attenuation correction)
- current_1f, impedance_1f, phase_vi : Ch4 electrical (if Ch4 exists)
- rssi : RSSI (if available)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from nptdms import TdmsFile

from ldv_analysis.config import (
    CURRENT_SCALE,
    SENSITIVITY,
    VELOCITY_SCALE,
    VOLTAGE_ATTENUATION,
)
from ldv_analysis.io_utils import load_tdms_file

# ---------------------------------------------------------------------------
# Burst detection / FFT constants
# ---------------------------------------------------------------------------
ENVELOPE_CHUNK = 1000
ON_THRESHOLD_FACTOR = 3.0
RING_UP_US = 100.0      # µs to skip after burst ON
RING_DOWN_US = 10.0      # µs to skip before burst OFF
CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_compute(
    tdms_path: str | Path,
    cache_dir: str | Path,
) -> np.lib.npyio.NpzFile:
    """Load FFT cache if available, otherwise compute from TDMS.

    Parameters
    ----------
    tdms_path : str or Path
        Path to the TDMS file.
    cache_dir : str or Path
        Directory where cache .npz files are stored.
    """
    tdms_path = Path(tdms_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"_fft_cache_{tdms_path.stem}.npz"

    if cache_path.exists():
        cache = np.load(cache_path)
        if "dt" in cache:          # new-format cache with timing info
            print(f"  Using FFT cache: {cache_path.name}")
            return cache
        print("  Cache outdated, recomputing...")

    return _compute(tdms_path, cache_path)


def load_point_waveforms(
    tdms_path: str | Path,
    point_index: int,
    channels: tuple[int, ...] = (1, 2),
) -> tuple[dict[int, np.ndarray], float]:
    """Load raw waveforms for a single scan point (memory-efficient).

    Uses streaming TDMS access to avoid loading the entire file into RAM.

    Parameters
    ----------
    tdms_path : str or Path
        Path to the TDMS file.
    point_index : int
        Scan point index (0-based).
    channels : tuple of int
        Channel numbers to load (e.g. (1, 2) or (1, 2, 4)).

    Returns
    -------
    waveforms : dict mapping channel number to 1-D ndarray
    dt : float (seconds)
    """
    tdms_path = Path(tdms_path)
    result: dict[int, np.ndarray] = {}
    dt = 8e-9
    with TdmsFile.open(str(tdms_path)) as f:
        wf_group = f["Waveforms"]
        for ch in channels:
            prefix = f"WFCh{ch}"
            names = sorted(
                [c.name for c in wf_group.channels()
                 if c.name.startswith(prefix)])
            ch_obj = wf_group[names[point_index]]
            result[ch] = ch_obj[:]
            dt = ch_obj.properties.get("wf_increment", 8e-9)
    return result, dt


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _compute(tdms_path: Path, cache_path: Path) -> np.lib.npyio.NpzFile:
    """Read TDMS, compute 1f FFT quantities, save and return cache."""
    tdms_file, metadata = load_tdms_file(tdms_path)
    n_x_meta = int(metadata.get("n_x", 0))
    n_y_meta = int(metadata.get("n_y", 0))
    print(f"  Grid: {n_x_meta} x × {n_y_meta} y")

    scan = tdms_file["ScanData"]
    pos_x = scan["PosX"][:]
    pos_y = scan["PosY"][:]
    n_points = len(pos_x)

    rssi = None
    scan_ch_names = [ch.name for ch in scan.channels()]
    if "RSSI" in scan_ch_names:
        rssi = scan["RSSI"][:n_points]

    # Detect available waveform channels
    wf_group = tdms_file["Waveforms"]
    all_wf_names = [ch.name for ch in wf_group.channels()]
    has_ch4 = any(n.startswith("WFCh4") for n in all_wf_names)

    # Channel name lists (needed for probing and chunked processing)
    ch1_names = sorted([n for n in all_wf_names if n.startswith("WFCh1")])

    # Timing from first waveform
    _ch0 = wf_group[ch1_names[0]]
    dt = _ch0.properties.get("wf_increment", 8e-9)
    n_samples = len(_ch0)
    print(f"  {n_points} points, {n_samples} samples, dt = {dt*1e9:.0f} ns")

    # Probe several points to find best reference for burst detection.
    # Point 0 may be at channel edge with near-zero signal, which causes
    # burst misdetection (appears continuous → wrong FFT window).
    n_env_chunks = n_samples // ENVELOPE_CHUNK
    probe_indices = sorted(set(
        [0, n_points // 4, n_points // 2, 3 * n_points // 4, n_points - 1]
    ))
    best_rms = -1.0
    ref_idx = 0
    ch1_ref = _ch0[:]
    for pi in probe_indices:
        wf = wf_group[ch1_names[pi]][:]
        rms_env = np.array([
            np.sqrt(np.mean(wf[i * ENVELOPE_CHUNK:(i + 1) * ENVELOPE_CHUNK] ** 2))
            for i in range(n_env_chunks)
        ])
        peak = float(np.max(rms_env))
        if peak > best_rms:
            best_rms = peak
            ref_idx = pi
            ch1_ref = wf
    del _ch0
    print(f"  Burst ref: point {ref_idx} (of {n_points}), peak RMS = {best_rms:.4f}")
    rms_env = np.array([
        np.sqrt(np.mean(
            ch1_ref[i * ENVELOPE_CHUNK:(i + 1) * ENVELOPE_CHUNK] ** 2))
        for i in range(n_env_chunks)
    ])
    # Detect continuous vs burst: if RMS envelope is flat, it's continuous
    rms_min = np.min(rms_env)
    rms_max = np.max(rms_env)
    continuous = rms_min > 0.3 * rms_max  # flat envelope → continuous

    if continuous:
        # Use full waveform, skip a small margin at edges
        margin = int(10e-6 / dt)  # 10 µs
        burst_on_us = 0.0
        burst_off_us = float(n_samples * dt * 1e6)
        ss_start = margin
        ss_end = n_samples - margin
        ss_n = ss_end - ss_start
        print(f"  Continuous excitation detected")
        print(f"  FFT window: {ss_start * dt * 1e6:.0f}"
              f"--{ss_end * dt * 1e6:.0f} us "
              f"({ss_n} samples, df = {1/(ss_n*dt):.0f} Hz)")
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
        print(f"  FFT window: {burst_on_us + RING_UP_US:.0f}"
              f"--{burst_off_us - RING_DOWN_US:.0f} us "
              f"({ss_n} samples, df = {1/(ss_n*dt):.0f} Hz)")

    # Drive frequency from full-record Ch1 (parabolic interpolation for sub-bin accuracy)
    fft_ch1_full = np.fft.rfft(ch1_ref)
    mag = np.abs(fft_ch1_full)
    k = int(np.argmax(mag[1:]) + 1)
    df_full = 1.0 / (n_samples * dt)
    alpha = np.log(mag[k - 1])
    beta = np.log(mag[k])
    gamma = np.log(mag[k + 1])
    delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    f_drive = float((k + delta) * df_full)
    print(f"  Drive: {f_drive / 1e6:.6f} MHz")

    # Exact-frequency DFT tone (avoids scalloping loss from nearest-bin FFT)
    tone = np.exp(-2j * np.pi * f_drive * np.arange(ss_n) * dt)

    # Output arrays
    velocity_1f = np.empty(n_points)
    pressure_1f = np.empty(n_points)
    phase_1f = np.empty(n_points)
    voltage_1f = np.empty(n_points)
    if has_ch4:
        current_1f = np.empty(n_points)
        impedance_1f = np.empty(n_points)
        phase_vi = np.empty(n_points)

    # Channel name lists (ch1_names already created above for probing)
    ch2_names = sorted([n for n in all_wf_names if n.startswith("WFCh2")])
    if has_ch4:
        ch4_names = sorted(
            [n for n in all_wf_names if n.startswith("WFCh4")])

    n_chunks = (n_points + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Computing FFT in {n_chunks} chunks...")

    for ci in range(n_chunks):
        i0 = ci * CHUNK_SIZE
        i1 = min(i0 + CHUNK_SIZE, n_points)
        chunk_n = i1 - i0

        wf1 = np.empty((chunk_n, n_samples))
        wf2 = np.empty((chunk_n, n_samples))
        for j in range(chunk_n):
            wf1[j] = wf_group[ch1_names[i0 + j]][:]
            wf2[j] = wf_group[ch2_names[i0 + j]][:]

        # Exact-frequency DFT via dot product (no scalloping loss)
        dft1 = wf1[:, ss_start:ss_end] @ tone
        dft2 = wf2[:, ss_start:ss_end] @ tone

        # Ch2 acoustic
        vel = np.abs(dft2) * 2 / ss_n * VELOCITY_SCALE
        velocity_1f[i0:i1] = vel
        pressure_1f[i0:i1] = vel / (2 * np.pi * f_drive * SENSITIVITY)

        diff = np.degrees(np.angle(dft2) - np.angle(dft1))
        phase_1f[i0:i1] = (diff + 180) % 360 - 180

        # Ch1 voltage (with probe attenuation)
        voltage_1f[i0:i1] = np.abs(dft1) * 2 / ss_n * VOLTAGE_ATTENUATION

        # Ch4 current
        if has_ch4:
            wf4 = np.empty((chunk_n, n_samples))
            for j in range(chunk_n):
                wf4[j] = wf_group[ch4_names[i0 + j]][:]
            dft4 = wf4[:, ss_start:ss_end] @ tone

            cur = np.abs(dft4) * 2 / ss_n * CURRENT_SCALE
            current_1f[i0:i1] = cur
            impedance_1f[i0:i1] = voltage_1f[i0:i1] / cur

            ph = np.degrees(np.angle(dft1) - np.angle(dft4))
            phase_vi[i0:i1] = (ph + 180) % 360 - 180

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            print(f"    chunk {ci + 1}/{n_chunks} done")

    # Assemble and save
    arrays = dict(
        pos_x=pos_x, pos_y=pos_y,
        n_x_meta=np.array(n_x_meta), n_y_meta=np.array(n_y_meta),
        f_drive=np.array(f_drive),
        dt=np.array(dt), n_samples=np.array(n_samples),
        burst_on_us=np.array(burst_on_us), burst_off_us=np.array(burst_off_us),
        ss_start=np.array(ss_start), ss_end=np.array(ss_end),
        velocity_1f=velocity_1f, pressure_1f=pressure_1f, phase_1f=phase_1f,
        voltage_1f=voltage_1f,
    )
    if rssi is not None:
        arrays["rssi"] = rssi
    if has_ch4:
        arrays["current_1f"] = current_1f
        arrays["impedance_1f"] = impedance_1f
        arrays["phase_vi"] = phase_vi

    np.savez(cache_path, **arrays)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  Cache saved: {cache_path.name} ({size_mb:.1f} MB)")
    return np.load(cache_path)
