"""Shared loaders for the PRL figure scripts.

Three functions:

- ``low_drive_mask``: contract data-defined low-drive selector
  ``|P_2f/P_1f| < LOW_DRIVE_P2_OVER_P1``. Replaces hardcoded V<=30 Vpp
  windows in the legacy scripts.
- ``load_ladder``: read ``harmonic_ladder.npz`` produced by
  ``harmonic_ladder.py`` and attach low-drive mask + per-harmonic
  acoustic energy with propagated 1-sigma.
- ``compute_pin_time``: contract-primary time-domain ``P_in =
  <v(t) i(t)>`` over the burst steady-state window. The only genuinely
  new low-level primitive in the MVP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from conventions import (
    BETA,
    C_0,
    LADDER_NPZ,
    LOW_DRIVE_P2_OVER_P1,
    RHO_0,
)
from ldv_analysis.config import CURRENT_SCALE, VOLTAGE_ATTENUATION
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.io_utils import ROLE_CURRENT, ROLE_DRIVE_VOLTAGE


def low_drive_mask(p_kpa: np.ndarray) -> np.ndarray:
    """Contract low-drive selector.

    Parameters
    ----------
    p_kpa : array, shape (N, H)
        Per-drive per-harmonic pressure amplitude in kPa, with column 0
        = P_1f and column 1 = P_2f (the canonical ``harmonic_ladder.npz``
        layout).

    Returns
    -------
    mask : bool array, shape (N,)
        True where ``|P_2f/P_1f| < LOW_DRIVE_P2_OVER_P1`` and both values
        are finite and positive.
    """
    p1 = p_kpa[:, 0]
    p2 = p_kpa[:, 1]
    finite = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0)
    ratio = np.where(finite, np.abs(p2) / np.where(p1 > 0, p1, np.nan), np.nan)
    return finite & (ratio < LOW_DRIVE_P2_OVER_P1)


def load_ladder(path: Path = LADDER_NPZ) -> dict:
    """Read ``harmonic_ladder.npz`` and attach derived fields.

    Adds:

    - ``low_drive`` (bool[N]) : contract low-drive window mask
    - ``e_n_j_m3`` (float[N,H]) : ``|P_nf|^2 / (4 rho c^2)``  acoustic
      energy density per harmonic
    - ``e_n_sigma_j_m3`` (float[N,H]) : 1-sigma propagated from
      ``p_std_kpa`` via ``sigma_E = 2 |P| sigma_P / (4 rho c^2)``
    - ``m_mach`` (float[N]) : Mach number ``M = P_1f / (rho c^2)``
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run experiments/2026W21/harmonic_ladder.py first.")
    npz = np.load(path, allow_pickle=False)
    d = {k: npz[k] for k in npz.files}

    p_pa = d["p_kpa"] * 1e3
    s_pa = d["p_std_kpa"] * 1e3
    d["e_n_j_m3"] = p_pa**2 / (4.0 * RHO_0 * C_0**2)
    d["e_n_sigma_j_m3"] = 2.0 * np.abs(p_pa) * s_pa / (4.0 * RHO_0 * C_0**2)
    d["m_mach"] = p_pa[:, 0] / (RHO_0 * C_0**2)
    d["low_drive"] = low_drive_mask(d["p_kpa"])
    d["beta"] = BETA
    return d


def compute_pin_time(
    scan_path: Path,
    cache_dir: Path,
    n_points: int = 5,
) -> Tuple[float, float]:
    """Time-domain input power ``P_in = <v(t) i(t)>`` (contract primary).

    Loads the drive_voltage (Ch1) and current (Ch4) waveforms for
    ``n_points`` mid-channel scan points, restricts to the burst
    steady-state window already in the cache (``ss_start:ss_end``),
    DC-removes each channel, takes the element-wise product, and
    averages.

    Electrical channels are spatially uniform across the LDV scan grid
    (Ch1 and Ch4 are measured at the PZT, independent of the laser
    position), so a few points are enough and their dispersion gives
    the SEM. Five points is overkill for the mean (single point would
    give the same answer to a few parts per thousand) but cheap; the
    spread across the 5 points is what the SEM reports as a
    self-consistency check.

    Parameters
    ----------
    scan_path : Path
        One acquisition file (.h5 or .tdms) in the cascade scan.
    cache_dir : Path
        FFT cache directory for this scan (used to recover the steady-
        state window ``ss_start``/``ss_end``; the cache must already
        exist, no recompute is attempted).
    n_points : int
        Number of evenly-spaced scan points to sample.

    Returns
    -------
    p_in_w : float
        Mean ``<v i>`` across the ``n_points``, in watts.
    p_in_sem_w : float
        Standard error of the mean across the ``n_points``.

    Notes
    -----
    Raw waveforms from ``load_point_waveforms`` are at scope voltages.
    Multiply Ch1 by ``VOLTAGE_ATTENUATION`` (x10 probe) and Ch4 by
    ``CURRENT_SCALE`` (0.2 A/V) to get physical V and A.
    """
    cache = load_or_compute(scan_path, cache_dir, velocity_scale=None)
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    n_total = int(cache["voltage_1f"].size)
    n_use = min(n_points, n_total)
    pts = np.linspace(0, n_total - 1, n_use, dtype=int)

    pin_per_pt = np.empty(n_use)
    for k, idx in enumerate(pts):
        wf, _ = load_point_waveforms(
            scan_path, int(idx),
            roles=(ROLE_DRIVE_VOLTAGE, ROLE_CURRENT),
        )
        v = wf[ROLE_DRIVE_VOLTAGE][ss_start:ss_end] * VOLTAGE_ATTENUATION  # V
        i = wf[ROLE_CURRENT][ss_start:ss_end] * CURRENT_SCALE             # A
        v = v - v.mean()
        i = i - i.mean()
        pin_per_pt[k] = float(np.mean(v * i))                              # W

    p_in_w = float(pin_per_pt.mean())
    if n_use >= 2:
        p_in_sem_w = float(pin_per_pt.std(ddof=1) / np.sqrt(n_use))
    else:
        p_in_sem_w = 0.0
    return p_in_w, p_in_sem_w


def compute_pin_spectral(scan_path: Path, cache_dir: Path) -> Tuple[float, float]:
    """Diagnostic spectral input power ``(1/2) V_1f I_1f cos(phi_vi)``.

    For Fig S1 cross-check vs ``compute_pin_time``. Already computed at
    every scan point inside the FFT cache; this just returns the
    grid-median + median absolute deviation as the spread.
    """
    cache = load_or_compute(scan_path, cache_dir, velocity_scale=None)
    if "current_1f" not in cache.files:
        return float("nan"), float("nan")
    v = np.asarray(cache["voltage_1f"])
    i = np.asarray(cache["current_1f"])
    phi = np.radians(np.asarray(cache["phase_vi"]))
    pin_pt = 0.5 * v * i * np.cos(phi)
    finite = np.isfinite(pin_pt)
    if not np.any(finite):
        return float("nan"), float("nan")
    med = float(np.median(pin_pt[finite]))
    mad = float(np.median(np.abs(pin_pt[finite] - med)))
    return med, 1.4826 * mad / np.sqrt(np.count_nonzero(finite))


__all__ = [
    "low_drive_mask",
    "load_ladder",
    "compute_pin_time",
    "compute_pin_spectral",
]
