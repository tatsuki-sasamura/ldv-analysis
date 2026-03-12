"""Mode-shape fitting for acoustofluidic channel line scans and 2D maps.

Provides functions to fit sinusoidal pressure profiles across the channel
width, with optional brute-force channel centre search for line scans.

1f mode: p(y) = p0 * sin(π y / W)   (half-wavelength, signed)
2f mode: p(y) = p0 * |cos(2π y / W)|  (full-wavelength)

All positions and widths are in SI units (metres).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModeFitResult:
    """Result of a mode-shape fit to line-scan data."""

    p0: complex  # complex pressure amplitude (Pa)
    centre: float  # channel centre position (m)
    r2: float  # goodness of fit
    inside: np.ndarray  # boolean mask of points inside channel


def fit_mode_1f(
    positions: np.ndarray,
    pressure: np.ndarray,
    channel_width: float,
    centre: float | None = None,
    n_trial: int = 200,
) -> ModeFitResult:
    """Fit p(y) = p0 * sin(π y/W) to line-scan data.

    Parameters
    ----------
    positions : array
        Scan positions in m.
    pressure : array
        Complex pressure in Pa (same length as positions).
    channel_width : float
        Known channel width in m.
    centre : float or None
        Channel centre in m.  If None, brute-force search over n_trial
        candidates to find the centre that maximises R².
    n_trial : int
        Number of trial centres for brute-force search.
    """
    W = channel_width
    hw = channel_width / 2
    k = np.pi / W

    if centre is None:
        # Brute-force centre search — maximise R²
        trials = np.linspace(
            positions.min() + hw, positions.max() - hw, n_trial
        )
        best_r2 = -np.inf
        best_c = trials[0] if len(trials) > 0 else float(positions.mean())
        for c in trials:
            y_c = positions - c
            inside = np.abs(y_c) <= W / 2
            if inside.sum() < 3:
                continue
            sin_prof = np.sin(k * y_c[inside])
            denom = np.sum(sin_prof**2)
            if denom <= 0:
                continue
            p0_c = np.sum(pressure[inside] * sin_prof) / denom
            r2_c = _r2(pressure[inside], p0_c * sin_prof)
            if r2_c > best_r2:
                best_r2 = r2_c
                best_c = float(c)
        centre = best_c

    # Final fit at best centre
    assert centre is not None
    y_c = positions - centre
    inside = np.abs(y_c) <= W / 2
    sin_prof = np.sin(k * y_c[inside])
    denom = np.sum(sin_prof**2)
    p0 = complex(np.sum(pressure[inside] *
                 sin_prof) / denom) if denom > 0 else 0j
    r2 = _r2(pressure[inside], p0 * sin_prof)

    return ModeFitResult(p0=p0, centre=centre, r2=r2, inside=inside)


def fit_mode_2f(
    positions: np.ndarray,
    pressure: np.ndarray,
    channel_width: float,
    centre: float,
) -> ModeFitResult:
    """Fit p(y) = p0 * cos(2π y/W) to line-scan data.

    Parameters
    ----------
    positions : array
        Scan positions in m.
    pressure : array
        Complex pressure in Pa.
    channel_width : float
        Known channel width in m.
    centre : float
        Channel centre in m (reuse from 1f fit).
    """
    W = channel_width
    k = 2 * np.pi / W

    y_c = positions - centre
    inside = np.abs(y_c) <= W / 2
    cos_prof = np.cos(k * y_c[inside])  # signed
    denom = np.sum(cos_prof**2)
    p0 = complex(np.sum(pressure[inside] *
                 cos_prof) / denom) if denom > 0 else 0j
    r2 = _r2(pressure[inside], p0 * cos_prof)

    return ModeFitResult(p0=p0, centre=centre, r2=r2, inside=inside)


def fit_columns(
    grid: np.ndarray,
    width_positions_m: np.ndarray,
    channel_width_m: float,
    harmonic: int = 1,
) -> np.ndarray:
    """Fit mode shape at each axial position in a 2D grid.

    Parameters
    ----------
    grid : array, shape (n_width, n_length)
        Pressure grid in Pa.  NaN entries are skipped automatically.
    width_positions_m : array, shape (n_width,)
        Centred width positions in metres.
    channel_width_m : float
        Channel width in metres.
    harmonic : int
        1 → |sin(π y/W)|, 2 → |cos(2π y/W)|.

    Returns
    -------
    p0 : array, shape (n_length,)
        Fitted pressure amplitude at each axial position (Pa).
    """
    _, n_length = grid.shape

    if harmonic == 2:
        k = 2 * np.pi / channel_width_m
        mode = np.abs(np.cos(k * width_positions_m))
    else:
        k = np.pi / channel_width_m
        mode = np.abs(np.sin(k * width_positions_m))

    p0 = np.full(n_length, np.nan)
    for j in range(n_length):
        col = grid[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 3:
            p0[j] = np.sum(col[valid] * mode[valid]) / np.sum(mode[valid] ** 2)

    return p0


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute R² (coefficient of determination).

    Works for both real and complex arrays.  For complex data the
    sum-of-squares use ``|·|²`` so that the result is always real.
    """
    if len(observed) < 3:
        return 0.0
    ss_res = float(np.sum(np.abs(observed - predicted) ** 2))
    ss_tot = float(np.sum(np.abs(observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
