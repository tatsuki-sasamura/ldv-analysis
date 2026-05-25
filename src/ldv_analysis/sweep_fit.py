"""Per-axial mode-shape fitting for v2 frequency sweeps.

Shared logic so the freq-sweep summary (``freq_vs_current.py``) and the
voltage cascade (``vpp_vs_pressure.py``) extract peak pressures the same
way.

For a 2D area scan the ``|sin(pi y/W)|`` (1f) / ``|cos(2 pi y/W)|`` (2f)
mode is fit at each axial position -- like ``pressure_map_2d.py`` -- and
the reported value is taken at the axial slice that maximizes P_1f.  For
a 1D line scan (a single axial position) a single lumped fit over all
valid points is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brute, fmin

from ldv_analysis.config import RSSI_THRESHOLD
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import (
    _mode_shape, _project, _r2, fit_columns, fit_mode_1f, fit_mode_2f,
)

Geom = tuple[float, float]  # tilted channel center: center(y) = a*y + b


def detect_channel_geometry(pos_x, pos_y, rssi, pressure_abs, hw) -> Geom:
    """Detect a tilted channel center, ``center(y) = a*y + b``.

    Mirrors ``pressure_map_2d.py``: maximize mean RSSI inside a strip of
    fixed width ``2*hw`` (or, with no RSSI, minimize out-of-channel
    pressure^2).  Positions in meters; returns ``(a, b)``.
    """
    x_min, x_max = pos_x.min(), pos_x.max()
    y_min, y_max = pos_y.min(), pos_y.max()
    y_span = max(y_max - y_min, 1e-12)
    c_lo, c_hi = x_min + hw, x_max - hw

    if rssi is not None:
        def cost(params):
            c_left, c_right = params
            center = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
            inside = np.abs(pos_x - center) <= hw
            if np.sum(inside) < 10:
                return 0.0
            return -np.mean(rssi[inside])
        res = brute(cost, ranges=((c_lo, c_hi), (c_lo, c_hi)), Ns=100,
                    finish=fmin)
    else:
        prs_sq = pressure_abs ** 2

        def cost(params):
            c_left, c_right = params
            center = c_left + (c_right - c_left) / y_span * (pos_y - y_min)
            outside = np.abs(pos_x - center) > hw
            return np.nansum(prs_sq[outside])
        res = brute(cost, ranges=((c_lo, c_hi), (c_lo, c_hi)), Ns=100,
                    finish=None)

    c_left, c_right = float(res[0]), float(res[1])
    a = (c_right - c_left) / y_span
    b = c_left - a * y_min
    return a, b


def fit_complex_column(col_c, width_grid, channel_width, harmonic):
    """Project one gridded complex column onto the signed mode shape.

    ``fit_columns`` returns only the real magnitude per axial position;
    this recovers the complex ``p0`` (for phase) and R^2 at one column.
    Returns ``(p0 complex, r2)``.
    """
    mode = _mode_shape(width_grid, channel_width, harmonic, use_abs=False)
    finite = ~np.isnan(col_c)
    if finite.sum() < 3:
        return 0j, 0.0
    # Floor the clip at half the column: complex residuals are amplitude-
    # weighted, so at high pressure small per-point phase scatter can
    # otherwise spiral the clip down to a few points and yield a garbage R2.
    p0, clip = _project(col_c[finite], mode[finite], sigma_clip=3.0,
                        min_keep_frac=0.5)
    pred = p0 * mode[finite]
    r2 = _r2(col_c[finite][clip], pred[clip])
    return complex(p0), r2


@dataclass
class AxialFit:
    """Mode fit at the best axial slice (2D) or the lumped fit (1D)."""

    p1_mag: float          # |P_1f| (Pa)
    p1_complex: complex     # complex P_1f (Pa)
    r2_1: float
    p2_mag: float          # |P_2f| (Pa); nan if no 2f channel
    p2_complex: complex
    r2_2: float
    x_best_mm: float        # best axial position (mm); nan for 1D
    is_1d: bool
    # Width profile at the chosen slice, for mode-shape plots:
    center: float           # channel center (m); 0 for 2D (grid centered)
    y_c: np.ndarray         # centered width positions (m)
    p: np.ndarray           # |P_1f| at those positions (Pa)
    phase_deg: np.ndarray   # phase of P_1f at those positions (deg)


def fit_axial(cache, valid, channel_width, *, geom: Geom | None = None
              ) -> tuple[AxialFit, Geom | None]:
    """Fit the mode at the best axial slice (2D) or lumped (1D).

    Parameters
    ----------
    cache : npz-like
        FFT cache (output of :func:`ldv_analysis.fft_cache.load_or_compute`).
    valid : array of bool
        Per-point validity mask.
    channel_width : float
        Channel width in meters.
    geom : (a, b) or None
        Reuse a previously detected tilted center to skip re-detection
        (every frequency in a sweep shares the same physical scan grid).

    Returns
    -------
    (fit, geom)
        ``geom`` is the geometry used (newly detected for a 2D scan if
        None was passed), so the caller can thread it through a sweep.
    """
    hw = channel_width / 2
    pos = np.asarray(cache["pos_x"])        # width direction (m)
    pos_y = np.asarray(cache["pos_y"])      # length / axial direction (m)
    n_x = int(cache["n_x_meta"])
    n_y = int(cache["n_y_meta"])
    rssi = np.asarray(cache["rssi"]) if "rssi" in cache.files else None

    P1_abs = np.asarray(cache["pressure_1f"])
    P1_ph = np.asarray(cache["phase_1f"])
    P1c = P1_abs * np.exp(1j * np.radians(P1_ph))
    has_2f = "pressure_2f" in cache.files and "phase_2f" in cache.files
    if has_2f:
        P2_abs = np.asarray(cache["pressure_2f"])
        P2_ph = np.asarray(cache["phase_2f"])
        P2c = P2_abs * np.exp(1j * np.radians(P2_ph))

    is_1d = (n_x <= 1) or (n_y <= 1)

    if is_1d:
        r1 = fit_mode_1f(pos[valid], P1c[valid], channel_width)
        p1_mag = abs(r1.p0)
        if has_2f:
            r2 = fit_mode_2f(pos[valid], P2c[valid], channel_width, r1.center)
            p2_mag, p2c, r2_2 = abs(r2.p0), r2.p0, r2.r2
        else:
            p2_mag, p2c, r2_2 = float("nan"), 0j, float("nan")
        fit = AxialFit(
            p1_mag=p1_mag, p1_complex=r1.p0, r2_1=r1.r2,
            p2_mag=p2_mag, p2_complex=p2c, r2_2=r2_2,
            x_best_mm=float("nan"), is_1d=True,
            center=r1.center, y_c=pos[valid] - r1.center,
            p=P1_abs[valid], phase_deg=P1_ph[valid],
        )
        return fit, geom

    # 2D: per-axial fit, report the slice that maximizes P_1f.
    if geom is None:
        P1g = P1_abs.copy()
        P1g[~valid] = np.nan
        geom = detect_channel_geometry(pos, pos_y, rssi, P1g, hw)
    a_opt, b_opt = geom

    pos_x_c = pos - (a_opt * pos_y + b_opt)
    inside_c = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(
        pos_width_c=pos_x_c, pos_length=pos_y,
        n_scan_width=n_x, n_scan_length=n_y,
        channel_width=channel_width, raw_width_span=pos.max() - pos.min(),
        inside=inside_c, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    grid_mag1 = cg.to_grid(np.where(valid, P1_abs, np.nan))
    p0_y1 = fit_columns(grid_mag1, cg.width_grid, channel_width,
                        harmonic=1, sigma_clip=3.0)
    if np.all(np.isnan(p0_y1)):
        raise ValueError("no valid axial column to fit")
    best = int(np.nanargmax(p0_y1))
    p1_mag = float(p0_y1[best])

    grid_re1 = cg.to_grid(np.where(valid, P1c.real, np.nan))
    grid_im1 = cg.to_grid(np.where(valid, P1c.imag, np.nan))
    col1 = grid_re1[:, best] + 1j * grid_im1[:, best]
    p1c, r2_1 = fit_complex_column(col1, cg.width_grid, channel_width, 1)

    if has_2f:
        grid_mag2 = cg.to_grid(np.where(valid, P2_abs, np.nan))
        p0_y2 = fit_columns(grid_mag2, cg.width_grid, channel_width,
                            harmonic=2, sigma_clip=3.0)
        p2_mag = float(p0_y2[best])
        grid_re2 = cg.to_grid(np.where(valid, P2c.real, np.nan))
        grid_im2 = cg.to_grid(np.where(valid, P2c.imag, np.nan))
        col2 = grid_re2[:, best] + 1j * grid_im2[:, best]
        p2c, r2_2 = fit_complex_column(col2, cg.width_grid, channel_width, 2)
    else:
        p2_mag, p2c, r2_2 = float("nan"), 0j, float("nan")

    fit = AxialFit(
        p1_mag=p1_mag, p1_complex=p1c, r2_1=r2_1,
        p2_mag=p2_mag, p2_complex=p2c, r2_2=r2_2,
        x_best_mm=float(cg.length_grid[best] * 1e3), is_1d=False,
        center=0.0, y_c=cg.width_grid, p=grid_mag1[:, best],
        phase_deg=np.degrees(np.angle(col1)),
    )
    return fit, geom


@dataclass
class SweepPeaks:
    """Per-frequency peak pressures across one sweep directory."""

    freq_mhz: np.ndarray
    p1_kpa: np.ndarray
    p2_kpa: np.ndarray
    r2_1: np.ndarray
    x_best_mm: np.ndarray
    peak_p1_kpa: float
    peak_p1_freq_mhz: float
    peak_p2_kpa: float
    peak_p2_freq_mhz: float
    is_1d: bool


def sweep_peaks(run_dir: Path, channel_width: float, cache_dir: Path
                ) -> SweepPeaks:
    """Fit every HDF5 in *run_dir* and return per-frequency peak pressures.

    Geometry is detected once (from the first 2D file) and reused.  Uses
    *cache_dir* for the shared FFT cache so repeated runs are fast.
    """
    import h5py

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in run_dir.glob("*.h5")
                   if not p.name.endswith(".inprogress"))

    geom: Geom | None = None
    f_hz: list[float] = []
    p1: list[float] = []
    p2: list[float] = []
    r2_1: list[float] = []
    xb: list[float] = []
    is_1d = False

    for p in files:
        with h5py.File(p, "r") as f:
            f_nom = float(f.attrs["drive_frequency_hz_nominal"])
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if int(np.sum(valid)) < 3:
            continue
        fit, geom = fit_axial(c, valid, channel_width, geom=geom)
        is_1d = fit.is_1d
        f_hz.append(f_nom)
        p1.append(fit.p1_mag / 1e3)
        p2.append(fit.p2_mag / 1e3)
        r2_1.append(fit.r2_1)
        xb.append(fit.x_best_mm)

    if not f_hz:
        raise ValueError(f"no usable frequency files in {run_dir}")

    order = np.argsort(f_hz)
    fm = np.asarray(f_hz)[order] / 1e6
    p1a = np.asarray(p1)[order]
    p2a = np.asarray(p2)[order]
    r2a = np.asarray(r2_1)[order]
    xba = np.asarray(xb)[order]

    i1 = int(np.nanargmax(p1a))
    i2 = int(np.nanargmax(p2a))
    return SweepPeaks(
        freq_mhz=fm, p1_kpa=p1a, p2_kpa=p2a, r2_1=r2a, x_best_mm=xba,
        peak_p1_kpa=float(p1a[i1]), peak_p1_freq_mhz=float(fm[i1]),
        peak_p2_kpa=float(p2a[i2]), peak_p2_freq_mhz=float(fm[i2]),
        is_1d=is_1d,
    )
