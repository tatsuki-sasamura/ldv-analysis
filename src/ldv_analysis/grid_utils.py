"""Grid utilities for mapping flat scan arrays to 2D channel grids."""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_to_grid(
    w_idx: np.ndarray,
    l_idx: np.ndarray,
    inside: np.ndarray,
    n_width: int,
    n_length: int,
    *,
    rssi: np.ndarray | None = None,
    rssi_threshold: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a function that maps flat scan arrays onto a 2D grid.

    Parameters
    ----------
    w_idx : array of int
        Width-axis grid index for each scan point.
    l_idx : array of int
        Length-axis grid index for each scan point.
    inside : array of bool
        True for points inside the channel.
    n_width, n_length : int
        Grid dimensions.
    rssi : array or None
        RSSI values; points below *rssi_threshold* are excluded.
    rssi_threshold : float
        Minimum RSSI to include (default 1.0 V).

    Returns
    -------
    to_grid : callable
        ``to_grid(values) -> np.ndarray`` of shape (n_width, n_length).
        Points outside the channel, with NaN values, or below RSSI
        threshold are set to NaN.  Edge rows (first and last) are
        forced to NaN to avoid boundary artefacts in mode-shape fits
        (2 rows blanked on each side).
    """
    def to_grid(values: np.ndarray) -> np.ndarray:
        grid = np.full((n_width, n_length), np.nan)
        mask = inside & ~np.isnan(values)
        if rssi is not None:
            mask &= rssi >= rssi_threshold
        grid[w_idx[mask], l_idx[mask]] = values[mask]
        grid[:2, :] = np.nan
        grid[-2:, :] = np.nan
        return grid
    return to_grid
