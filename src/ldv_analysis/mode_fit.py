"""Mode-shape fitting for acoustofluidic channel line scans and 2D maps.

Provides functions to fit sinusoidal pressure profiles across the channel
width, with optional brute-force channel centre search for line scans.

1f mode: p(y) = p0 * |sin(π y / W)|   (half-wavelength)
2f mode: p(y) = p0 * |cos(2π y / W)|  (full-wavelength)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModeFitResult:
    """Result of a mode-shape fit to line-scan data."""

    p0: float  # pressure amplitude (Pa)
    centre: float  # channel centre position (mm)
    r2: float  # goodness of fit
    inside: np.ndarray  # boolean mask of points inside channel


def fit_mode_1f(
    positions_mm: np.ndarray,
    pressure: np.ndarray,
    channel_width_mm: float,
    centre: float | None = None,
    n_trial: int = 200,
) -> ModeFitResult:
    """Fit p(y) = p0 * |sin(π y/W)| to line-scan data.

    Parameters
    ----------
    positions_mm : array
        Scan positions in mm.
    pressure : array
        Pressure values in Pa (same length as positions_mm).
    channel_width_mm : float
        Known channel width in mm.
    centre : float or None
        Channel centre in mm.  If None, brute-force search over n_trial
        candidates to find the centre that maximises p0.
    n_trial : int
        Number of trial centres for brute-force search.
    """
    W = channel_width_mm * 1e-3  # m
    hw = channel_width_mm / 2  # mm
    k = np.pi / W

    if centre is None:
        # Brute-force centre search
        trials = np.linspace(
            positions_mm.min() + hw, positions_mm.max() - hw, n_trial
        )
        best_p0 = 0.0
        best_c = trials[0] if len(trials) > 0 else float(positions_mm.mean())
        for c in trials:
            y_c = (positions_mm - c) * 1e-3  # mm → m, centred
            inside = np.abs(y_c) <= W / 2
            if inside.sum() < 3:
                continue
            sin_prof = np.abs(np.sin(k * y_c[inside]))
            denom = np.sum(sin_prof**2)
            if denom > 0:
                p0_cand = float(np.sum(pressure[inside] * sin_prof) / denom)
                if p0_cand > best_p0:
                    best_p0 = p0_cand
                    best_c = float(c)
        centre = best_c

    # Final fit at best centre
    y_c = (positions_mm - centre) * 1e-3
    inside = np.abs(y_c) <= W / 2
    sin_prof = np.abs(np.sin(k * y_c[inside]))
    denom = np.sum(sin_prof**2)
    p0 = float(np.sum(pressure[inside] * sin_prof) / denom) if denom > 0 else 0.0
    r2 = _r2(pressure[inside], p0 * sin_prof)

    return ModeFitResult(p0=p0, centre=centre, r2=r2, inside=inside)


def fit_mode_2f(
    positions_mm: np.ndarray,
    pressure: np.ndarray,
    channel_width_mm: float,
    centre: float,
) -> ModeFitResult:
    """Fit p(y) = p0 * |cos(2π y/W)| to line-scan data.

    Parameters
    ----------
    positions_mm : array
        Scan positions in mm.
    pressure : array
        Pressure values in Pa.
    channel_width_mm : float
        Known channel width in mm.
    centre : float
        Channel centre in mm (reuse from 1f fit).
    """
    W = channel_width_mm * 1e-3
    hw = channel_width_mm / 2
    k = 2 * np.pi / W

    y_c = (positions_mm - centre) * 1e-3
    inside = np.abs(y_c) <= W / 2
    cos_prof = np.abs(np.cos(k * y_c[inside]))
    denom = np.sum(cos_prof**2)
    p0 = float(np.sum(pressure[inside] * cos_prof) / denom) if denom > 0 else 0.0
    r2 = _r2(pressure[inside], p0 * cos_prof)

    return ModeFitResult(p0=p0, centre=centre, r2=r2, inside=inside)


def fit_columns(
    grid: np.ndarray,
    width_positions_m: np.ndarray,
    channel_width_m: float,
    harmonic: int = 1,
    quality_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit mode shape at each axial position in a 2D grid.

    Parameters
    ----------
    grid : array, shape (n_width, n_length)
        Pressure grid in Pa.
    width_positions_m : array, shape (n_width,)
        Centred width positions in metres.
    channel_width_m : float
        Channel width in metres.
    harmonic : int
        1 → |sin(π y/W)|, 2 → |cos(2π y/W)|.
    quality_mask : bool array, shape (n_width,), optional
        Points to include (True = use).

    Returns
    -------
    p0 : array, shape (n_length,)
        Fitted pressure amplitude at each axial position (Pa).
    """
    n_width, n_length = grid.shape

    if harmonic == 2:
        k = 2 * np.pi / channel_width_m
        mode = np.abs(np.cos(k * width_positions_m))
    else:
        k = np.pi / channel_width_m
        mode = np.abs(np.sin(k * width_positions_m))

    if quality_mask is None:
        quality_mask = np.ones(n_width, dtype=bool)

    p0 = np.full(n_length, np.nan)
    for j in range(n_length):
        col = grid[:, j]
        valid = ~np.isnan(col) & quality_mask
        if valid.sum() > 3:
            p0[j] = np.sum(col[valid] * mode[valid]) / np.sum(mode[valid] ** 2)

    return p0


def make_quality_mask(
    n_width: int,
    edge_margin: int = 1,
    rssi_grid: np.ndarray | None = None,
    rssi_threshold: float = 1.0,
) -> np.ndarray:
    """Build boolean mask excluding edge points and low-RSSI columns.

    Parameters
    ----------
    n_width : int
        Number of width grid points.
    edge_margin : int
        Points to exclude at each edge.
    rssi_grid : array, shape (n_width, n_length), optional
        RSSI grid — per-column median is compared to threshold.
    rssi_threshold : float
        Minimum acceptable RSSI (V).

    Returns
    -------
    mask : bool array, shape (n_width,)
    """
    mask = np.ones(n_width, dtype=bool)
    mask[:edge_margin] = False
    mask[-edge_margin:] = False

    if rssi_grid is not None:
        rssi_col_median = np.nanmedian(rssi_grid, axis=1)
        mask &= rssi_col_median >= rssi_threshold

    return mask


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    if len(observed) < 3:
        return 0.0
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
