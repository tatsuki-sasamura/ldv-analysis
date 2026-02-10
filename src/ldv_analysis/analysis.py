"""Analysis utilities for scanning LDV data.

Provides functions for converting amplitude/phase to complex representation,
computing spatial vibration maps, and generating statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def amplitude_phase_to_complex(
    amplitude: np.ndarray,
    phase_deg: np.ndarray,
) -> np.ndarray:
    """Convert amplitude and phase (degrees) to complex representation.

    Parameters
    ----------
    amplitude : ndarray
        Vibration amplitude at each scan point.
    phase_deg : ndarray
        Phase in degrees at each scan point.

    Returns
    -------
    complex_data : ndarray
        Complex vibration data (amplitude * exp(j*phase)).
    """
    phase_rad = np.deg2rad(phase_deg)
    return amplitude * np.exp(1j * phase_rad)


def normalize_phase(phase_deg: np.ndarray, ref_index: int = 0) -> np.ndarray:
    """Normalize phase relative to a reference point.

    Parameters
    ----------
    phase_deg : ndarray
        Phase values in degrees.
    ref_index : int
        Index of the reference point (default: first point).

    Returns
    -------
    normalized : ndarray
        Phase normalized to [-180, 180] relative to reference.
    """
    shifted = phase_deg - phase_deg[ref_index]
    return (shifted + 180) % 360 - 180


def compute_spatial_map(
    df: pd.DataFrame,
    value_col: str = "amp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape scan data into a 2D spatial map.

    Parameters
    ----------
    df : DataFrame
        Scan data with columns pos_x, pos_y, and *value_col*.
    value_col : str
        Column name to map (e.g. "amp", "phase", "freq").

    Returns
    -------
    x_grid : ndarray, shape (n_y, n_x)
        X positions on the grid.
    y_grid : ndarray, shape (n_y, n_x)
        Y positions on the grid.
    values : ndarray, shape (n_y, n_x)
        Values reshaped to the scan grid.
    """
    x_unique = np.sort(df["pos_x"].unique())
    y_unique = np.sort(df["pos_y"].unique())
    n_x = len(x_unique)
    n_y = len(y_unique)

    # Create lookup for grid indices
    x_idx = {v: i for i, v in enumerate(x_unique)}
    y_idx = {v: i for i, v in enumerate(y_unique)}

    values = np.full((n_y, n_x), np.nan)
    for _, row in df.iterrows():
        xi = x_idx[row["pos_x"]]
        yi = y_idx[row["pos_y"]]
        values[yi, xi] = row[value_col]

    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    return x_grid, y_grid, values


def compute_scan_statistics(df: pd.DataFrame, source_file: str = "") -> dict[str, Any]:
    """Compute summary statistics for a scan dataset.

    Parameters
    ----------
    df : DataFrame
        Scan data with columns: amp, phase, freq, pos_x, pos_y.
    source_file : str
        Source filename for labeling.

    Returns
    -------
    stats : dict
        Summary statistics including amplitude mean/max/std, phase range,
        frequency mean/std, spatial extent.
    """
    stats: dict[str, Any] = {"source_file": source_file}

    if "amp" in df.columns:
        stats["amp_mean"] = df["amp"].mean()
        stats["amp_max"] = df["amp"].max()
        stats["amp_min"] = df["amp"].min()
        stats["amp_std"] = df["amp"].std()

    if "phase" in df.columns:
        stats["phase_mean"] = df["phase"].mean()
        stats["phase_std"] = df["phase"].std()
        stats["phase_range"] = df["phase"].max() - df["phase"].min()

    if "freq" in df.columns:
        stats["freq_mean"] = df["freq"].mean()
        stats["freq_std"] = df["freq"].std()

    if "pos_x" in df.columns:
        stats["x_min"] = df["pos_x"].min()
        stats["x_max"] = df["pos_x"].max()
        stats["x_range"] = df["pos_x"].max() - df["pos_x"].min()

    if "pos_y" in df.columns:
        stats["y_min"] = df["pos_y"].min()
        stats["y_max"] = df["pos_y"].max()

    stats["n_points"] = len(df)
    return stats


def export_scan_data(
    df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "scan",
) -> Path:
    """Export scan data to CSV.

    Parameters
    ----------
    df : DataFrame
        Scan data to export.
    output_dir : str or Path
        Output directory.
    prefix : str
        Filename prefix.

    Returns
    -------
    output_path : Path
        Path to the exported CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_data.csv"
    df.to_csv(output_path, index=False)
    return output_path
