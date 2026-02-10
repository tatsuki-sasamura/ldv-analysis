"""TDMS file I/O utilities for scanning LDV data.

Handles loading National Instruments TDMS files produced by scanning
vibrometer systems. Extracts scan metadata, per-point measurement data,
and raw waveforms.

Typical TDMS structure
----------------------
- Info group     : scan grid metadata (NumberOfXPositions, NumberOfYPositions, ...)
- ScanData group : per-point Freq/Amp/Phase for each channel, plus PosX/PosY/Z
- Waveforms group: raw time-domain waveforms per channel per scan point
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nptdms import TdmsFile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tdms_file(path: str | Path) -> tuple[TdmsFile, dict[str, Any]]:
    """Load a TDMS file and extract metadata.

    Parameters
    ----------
    path : str or Path
        Path to the TDMS file.

    Returns
    -------
    tdms_file : TdmsFile
        The loaded TDMS file object.
    metadata : dict
        Scan metadata extracted from the Info group (n_x, n_y, n_freq, n_amp,
        meander).
    """
    path = Path(path)
    f = TdmsFile.read(path)
    metadata = _extract_metadata(f)
    return f, metadata


def load_scan_data(path: str | Path, channel: int = 1) -> pd.DataFrame:
    """Load scan data for a specific channel as a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the TDMS file.
    channel : int
        Channel number (1-4).

    Returns
    -------
    df : DataFrame
        Columns: pos_x, pos_y, freq, amp, phase, plus any available metadata
        columns (rssi, z_actual, set_freq, set_amp).
    """
    f, metadata = load_tdms_file(path)
    scan = f["ScanData"]

    n_points = len(scan[f"Ch{channel}Freq"])

    data = {
        "pos_x": scan["PosX"][:n_points],
        "pos_y": scan["PosY"][:n_points],
        "freq": scan[f"Ch{channel}Freq"][:n_points],
        "amp": scan[f"Ch{channel}Amp"][:n_points],
        "phase": scan[f"Ch{channel}Phase"][:n_points],
    }

    # Optional columns (may not exist in all files)
    for col_name, tdms_name in [
        ("rssi", "RSSI"),
        ("z_actual", "ZPosActual"),
        ("set_freq", "SetFreqCh1"),
        ("set_amp", "SetAmpCh1"),
    ]:
        try:
            data[col_name] = scan[tdms_name][:n_points]
        except KeyError:
            pass

    df = pd.DataFrame(data)
    df["point_index"] = np.arange(n_points)
    return df


def extract_scan_grid(path: str | Path) -> dict[str, Any]:
    """Extract the scan grid positions and dimensions.

    Parameters
    ----------
    path : str or Path
        Path to the TDMS file.

    Returns
    -------
    grid : dict
        Keys: x_positions, y_positions, x_unique, y_unique, n_x, n_y, shape.
    """
    f, metadata = load_tdms_file(path)
    scan = f["ScanData"]

    pos_x = scan["PosX"][:]
    pos_y = scan["PosY"][:]

    x_unique = np.sort(np.unique(pos_x))
    y_unique = np.sort(np.unique(pos_y))

    return {
        "x_positions": pos_x,
        "y_positions": pos_y,
        "x_unique": x_unique,
        "y_unique": y_unique,
        "n_x": len(x_unique),
        "n_y": len(y_unique),
        "shape": (len(y_unique), len(x_unique)),
    }


def extract_waveforms(
    path_or_file: str | Path | TdmsFile,
    channel: int = 1,
    max_points: int | None = None,
) -> tuple[np.ndarray, float]:
    """Extract raw waveform data for a channel.

    Parameters
    ----------
    path_or_file : str, Path, or TdmsFile
        Path to the TDMS file, or an already-loaded TdmsFile object.
        Passing a TdmsFile avoids re-reading the file from disk.
    channel : int
        Channel number (1-4).
    max_points : int or None
        Maximum number of scan points to load. None = all.

    Returns
    -------
    waveforms : ndarray, shape (n_scan_points, n_samples)
        Raw waveform data.
    dt : float
        Time increment between samples (seconds).
    """
    if isinstance(path_or_file, TdmsFile):
        f = path_or_file
    else:
        f, _ = load_tdms_file(path_or_file)
    wf_group = f["Waveforms"]

    prefix = f"WFCh{channel}"
    channels = [ch for ch in wf_group.channels() if ch.name.startswith(prefix)]

    if max_points is not None:
        channels = channels[:max_points]

    if not channels:
        raise ValueError(f"No waveform channels found for Ch{channel}")

    dt = channels[0].properties.get("wf_increment", 8e-9)
    n_samples = len(channels[0])

    waveforms = np.empty((len(channels), n_samples), dtype=np.float64)
    for i, ch in enumerate(channels):
        waveforms[i] = ch[:]

    return waveforms, dt


def list_tdms_files(directory: str | Path) -> list[Path]:
    """List all TDMS files in a directory, sorted by name.

    Parameters
    ----------
    directory : str or Path
        Directory to search.

    Returns
    -------
    files : list of Path
        Sorted list of TDMS file paths (excludes *_index files).
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.tdms"))
    # Exclude index files
    files = [f for f in files if not f.name.endswith("_index")]
    return files


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_metadata(f: TdmsFile) -> dict[str, Any]:
    """Extract scan metadata from the Info group."""
    metadata: dict[str, Any] = {}

    try:
        info = f["Info"]
        channel_map = {
            "n_x": "NumberOfXPositions",
            "n_y": "NumberOfYPositions",
            "n_freq": "NumberOfFrequencies",
            "n_amp": "NumberOfAmplitudes",
            "meander": "Meander",
        }
        for key, ch_name in channel_map.items():
            try:
                metadata[key] = int(info[ch_name][0])
            except (KeyError, IndexError):
                pass
    except KeyError:
        pass

    # Count available channels from ScanData
    try:
        scan = f["ScanData"]
        ch_count = 0
        for i in range(1, 9):
            try:
                scan[f"Ch{i}Freq"]
                ch_count += 1
            except KeyError:
                break
        metadata["n_channels"] = ch_count
    except KeyError:
        pass

    return metadata
