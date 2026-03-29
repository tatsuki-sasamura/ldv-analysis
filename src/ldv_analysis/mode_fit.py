"""Mode-shape fitting for acoustofluidic channel line scans and 2D maps.

Provides functions to fit sinusoidal pressure profiles across the channel
width, with optional brute-force channel centre search for line scans.

Mode shape convention:
  odd  harmonic (1, 3, 5, …): sin(h π y / W)
  even harmonic (2, 4, …):    cos(h π y / W)

Complex input uses the signed mode shape; real input uses |mode|.
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


def _mode_shape(y_c: np.ndarray, channel_width: float, harmonic: int,
                use_abs: bool = False) -> np.ndarray:
    """Compute mode shape for a given harmonic.

    Odd harmonics use sin, even use cos.  If *use_abs*, return |mode|
    (for real amplitude data); otherwise return signed mode (for complex).
    """
    k = harmonic * np.pi / channel_width
    if harmonic % 2 == 1:
        mode = np.sin(k * y_c)
    else:
        mode = np.cos(k * y_c)
    return np.abs(mode) if use_abs else mode


def _project(data: np.ndarray, mode: np.ndarray,
             sigma_clip: float | None = None,
             max_iter: int = 5) -> tuple[complex | float, np.ndarray]:
    """Core LSQ projection with optional sigma clipping.

    Returns (p0, mask) where mask is True for retained points.
    """
    mask = np.ones(len(data), dtype=bool)
    for _ in range(max_iter if sigma_clip else 1):
        m = mode[mask]
        denom = np.sum(np.abs(m) ** 2)
        if denom <= 0:
            return 0j if np.iscomplexobj(data) else 0.0, mask
        p0 = np.sum(data[mask] * m) / denom
        if sigma_clip is None:
            break
        residual = np.abs(data - p0 * mode)
        threshold = sigma_clip * np.std(residual[mask])
        new_mask = mask & (residual <= threshold)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return p0, mask


def fit_mode(
    positions: np.ndarray,
    pressure: np.ndarray,
    channel_width: float,
    harmonic: int = 1,
    *,
    centre: float | None = None,
    n_trial: int = 200,
    sigma_clip: float | None = None,
) -> ModeFitResult:
    """Fit p(y) = p0 * mode_shape(y) to line-scan data.

    Parameters
    ----------
    positions : array
        Scan positions in m.
    pressure : array
        Pressure in Pa (complex or real).  Complex input uses signed
        mode shape; real input uses |mode|.
    channel_width : float
        Known channel width in m.
    harmonic : int
        Harmonic number (1, 2, 3, …).  Odd → sin, even → cos.
    centre : float or None
        Channel centre in m.  If None, brute-force search over *n_trial*
        candidates to find the centre that maximises R².
    n_trial : int
        Number of trial centres for brute-force search.
    sigma_clip : float or None
        If set, iteratively reject points with residuals exceeding
        *sigma_clip* × std(residual).
    """
    W = channel_width
    hw = W / 2
    is_complex = np.iscomplexobj(pressure)

    if centre is None:
        # Brute-force centre search — maximise R²
        trials = np.linspace(
            positions.min() + hw, positions.max() - hw, n_trial
        )
        best_r2 = -np.inf
        best_c = trials[0] if len(trials) > 0 else float(positions.mean())
        for c in trials:
            y_c = positions - c
            inside = np.abs(y_c) <= hw
            if inside.sum() < 3:
                continue
            mode = _mode_shape(y_c[inside], W, harmonic, use_abs=not is_complex)
            denom = np.sum(np.abs(mode) ** 2)
            if denom <= 0:
                continue
            p0_c = np.sum(pressure[inside] * mode) / denom
            r2_c = _r2(pressure[inside], p0_c * mode)
            if r2_c > best_r2:
                best_r2 = r2_c
                best_c = float(c)
        centre = best_c

    # Final fit at best centre
    y_c = positions - centre
    inside = np.abs(y_c) <= hw
    mode = _mode_shape(y_c[inside], W, harmonic, use_abs=not is_complex)
    p0, clip_mask = _project(pressure[inside], mode, sigma_clip=sigma_clip)
    if is_complex:
        p0 = complex(p0)
    predicted = p0 * mode
    r2 = _r2(pressure[inside][clip_mask], predicted[clip_mask])

    return ModeFitResult(p0=p0, centre=centre, r2=r2, inside=inside)


def fit_mode_1f(
    positions: np.ndarray,
    pressure: np.ndarray,
    channel_width: float,
    centre: float | None = None,
    n_trial: int = 200,
) -> ModeFitResult:
    """Fit 1f mode shape.  Wrapper around :func:`fit_mode`."""
    return fit_mode(positions, pressure, channel_width, 1,
                    centre=centre, n_trial=n_trial)


def fit_mode_2f(
    positions: np.ndarray,
    pressure: np.ndarray,
    channel_width: float,
    centre: float,
) -> ModeFitResult:
    """Fit 2f mode shape.  Wrapper around :func:`fit_mode`."""
    return fit_mode(positions, pressure, channel_width, 2, centre=centre)


def fit_columns(
    grid: np.ndarray,
    width_positions_m: np.ndarray,
    channel_width_m: float,
    harmonic: int = 1,
    *,
    return_sigma: bool = False,
    sigma_clip: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        Odd (1, 3, 5, …) → |sin(h π y/W)|, even (2, 4, …) → |cos(h π y/W)|.
    return_sigma : bool
        If True, also return the standard error of p0 at each column.
    sigma_clip : float or None
        If set, iteratively reject outlier points per column.

    Returns
    -------
    p0 : array, shape (n_length,)
        Fitted pressure amplitude at each axial position (Pa).
    sigma_p0 : array, shape (n_length,)
        Standard error of p0 (only if *return_sigma* is True).
    """
    _, n_length = grid.shape
    mode = _mode_shape(width_positions_m, channel_width_m, harmonic, use_abs=True)

    p0 = np.full(n_length, np.nan)
    sigma_p0 = np.full(n_length, np.nan)
    for j in range(n_length):
        col = grid[:, j]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid > 3:
            p0_j, clip_mask = _project(col[valid], mode[valid],
                                       sigma_clip=sigma_clip)
            p0[j] = float(p0_j)
            m = mode[valid][clip_mask]
            residual = col[valid][clip_mask] - p0[j] * m
            n_clipped = clip_mask.sum()
            if n_clipped > 1:
                denom = np.sum(m ** 2)
                sigma_p0[j] = np.sqrt(
                    np.sum(residual**2) / ((n_clipped - 1) * denom))

    if return_sigma:
        return p0, sigma_p0
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
