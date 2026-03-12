"""Grid utilities for mapping flat scan arrays to 2D channel grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ChannelGrid:
    """2D grid structure for channel scan data.

    All positions in metres.
    """

    length_grid: np.ndarray   # (n_length,) m
    width_grid: np.ndarray    # (n_width,) m, centred
    n_width: int
    n_length: int
    to_grid: Callable[[np.ndarray], np.ndarray]  # flat array → 2D grid


def make_channel_grid(
    pos_width_c: np.ndarray,
    pos_length: np.ndarray,
    n_scan_width: int,
    n_scan_length: int,
    channel_width: float,
    raw_width_span: float,
    inside: np.ndarray,
    *,
    rssi: np.ndarray | None = None,
    rssi_threshold: float = 1.0,
) -> ChannelGrid:
    """Build a centred 2D grid from flat scan arrays.

    Parameters
    ----------
    pos_width_c : array
        Centred width positions (m), one per scan point.
    pos_length : array
        Length positions (m), one per scan point.
    n_scan_width : int
        Number of scan points in the width direction (from TDMS metadata).
    n_scan_length : int
        Number of scan points in the length direction (from TDMS metadata).
    channel_width : float
        Known channel width (m).
    raw_width_span : float
        Total span of the raw (un-centred) width positions (m).
    inside : array of bool
        True for points inside the channel.
    rssi : array or None
        RSSI values; points below *rssi_threshold* are excluded.
    rssi_threshold : float
        Minimum RSSI to include (default 1.0 V).
    """
    hw = channel_width / 2

    # Length grid
    y_min, y_max = pos_length.min(), pos_length.max()
    length_grid = np.linspace(y_min, y_max, n_scan_length)
    l_idx = np.argmin(
        np.abs(pos_length[:, None] - length_grid[None, :]), axis=1)

    # Width grid (centred on channel)
    scan_step = raw_width_span / max(n_scan_width - 1, 1)
    n_width = max(int(round(channel_width / scan_step)), 2)
    half_step = channel_width / n_width / 2
    width_grid = np.linspace(-hw + half_step, hw - half_step, n_width)
    w_idx = np.argmin(
        np.abs(pos_width_c[:, None] - width_grid[None, :]), axis=1)

    to_grid = make_to_grid(w_idx, l_idx, inside, n_width, n_scan_length,
                           rssi=rssi, rssi_threshold=rssi_threshold)

    return ChannelGrid(
        length_grid=length_grid,
        width_grid=width_grid,
        n_width=n_width,
        n_length=n_scan_length,
        to_grid=to_grid,
    )


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
