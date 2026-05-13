"""TDMS file I/O utilities for scanning LDV data.

Handles loading National Instruments TDMS files produced by scanning
vibrometer systems. Extracts scan metadata, per-point measurement data,
and raw waveforms.

Typical TDMS structure
----------------------
- Info group     : scan grid metadata (NumberOfXPositions, NumberOfYPositions, ...)
- ScanData group : per-point Freq/Amp/Phase for each channel, plus PosX/PosY/Z
- Waveforms group: raw time-domain waveforms per channel per scan point

For v2 (rebuilt DAQ): use the ``ScanData``/``load_scan`` interface in the
second half of this file. It provides a format-agnostic view that v2
code can target now and the new format can implement when ready. See
``plans/data_format_v2.md`` for the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# nptdms is imported lazily inside the TDMS-specific functions so that
# HDF5-only environments can install without nptdms.


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
    from nptdms import TdmsFile
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
    t_range_s: tuple[float, float] | None = None,
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
    t_range_s : (start, end) in seconds, or None
        Time range to load. None = all samples.

    Returns
    -------
    waveforms : ndarray, shape (n_scan_points, n_samples)
        Raw waveform data.
    dt : float
        Time increment between samples (seconds).
    """
    from nptdms import TdmsFile
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
    n_total = len(channels[0])

    if t_range_s is not None:
        i_start = max(int(t_range_s[0] / dt), 0)
        i_end = min(int(t_range_s[1] / dt), n_total)
    else:
        i_start = 0
        i_end = n_total

    n_samples = i_end - i_start
    waveforms = np.empty((len(channels), n_samples), dtype=np.float64)
    for i, ch in enumerate(channels):
        waveforms[i] = ch[i_start:i_end]

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


# ---------------------------------------------------------------------------
# v2 interface: ScanData + load_scan (format-agnostic)
# ---------------------------------------------------------------------------
#
# This is the seam the new DAQ format will plug into. ``ScanData`` holds
# coordinates, dt, metadata, and a lazy ``load_waveforms`` accessor;
# downstream analysis consumes only this interface. ``load_scan_tdms``
# wraps the existing TDMS reader. ``load_scan_hdf5`` is the (future) reader
# for the rebuilt DAQ. ``load_scan`` dispatches by file extension.
#
# See ``plans/data_format_v2.md`` for the schema and metadata keys.


# Canonical channel roles in the analysis layer
ROLE_DRIVE_VOLTAGE = "drive_voltage"
ROLE_LDV_OUTPUT = "ldv_output"
ROLE_CURRENT = "current"


@dataclass
class ScanData:
    """Format-agnostic view of one scan acquisition.

    Coordinates and metadata are eagerly loaded. Waveforms are accessed
    lazily through ``load_waveforms`` so large files (~30 GB) do not
    require everything in RAM.

    Attributes
    ----------
    pos_x, pos_y : (N_points,) float arrays in meters
    rssi : (N_points,) float array or None (no RSSI in the file)
    dt : sample interval in seconds (shared across channels)
    n_points : int
    n_samples : int per point, per channel
    metadata : dict — see plans/data_format_v2.md for canonical keys
    """

    pos_x: np.ndarray
    pos_y: np.ndarray
    rssi: np.ndarray | None
    dt: float
    n_points: int
    n_samples: int
    metadata: dict
    _loader: Callable[[str, slice | np.ndarray], np.ndarray] = field(
        repr=False
    )

    def load_waveforms(
        self, role: str, points: slice | np.ndarray
    ) -> np.ndarray:
        """Return ``(k, n_samples)`` waveforms for ``k`` requested points.

        Parameters
        ----------
        role : str
            One of ``ROLE_DRIVE_VOLTAGE``, ``ROLE_LDV_OUTPUT``,
            ``ROLE_CURRENT``. The set of available roles depends on the
            underlying file; check ``self.metadata["channel_roles"]``.
        points : slice or 1-D index array
            Which scan points to fetch.
        """
        return self._loader(role, points)


def _sort_wf_names(names: list[str]) -> list[str]:
    """Sort TDMS waveform channel names by their time component.

    Names look like ``WFCh2_20260307_115608_123``; the timestamp suffix
    determines acquisition order. Matches the helper in fft_cache.py.
    """
    def key(name: str) -> tuple:
        parts = name.rsplit("_", 3)
        if len(parts) >= 4:
            return parts[-3], parts[-2], parts[-1]
        return ("", "", name)
    return sorted(names, key=key)


def _detect_velocity_scale_from_name(stem: str) -> float | None:
    """LDV velocity scale (m/s per V) from filename pattern ``_Nm_s_max``.

    The Polytec decoder full-scale ±N m/s on a 1 MΩ PicoScope input
    appears as ±2 V, so m/s per V is N/2.
    """
    m = re.search(r"_(\d+)m_s_max", stem)
    if m:
        return int(m.group(1)) / 2.0
    return None


def _detect_vpp_from_name(stem: str) -> float | None:
    """Drive Vpp from filename pattern ``_NVpp``."""
    m = re.search(r"_(\d+)Vpp", stem)
    if m:
        return float(m.group(1))
    return None


def _detect_drive_freq_from_name(stem: str) -> float | None:
    """Nominal drive frequency (Hz) from filename digits like ``_1907``.

    Heuristic: the first integer between 1000 and 9999 kHz. Falls back
    to None if no plausible match — pipeline should detect via FFT then.
    """
    for m in re.finditer(r"_(\d{4})(?:_|kHz|\.tdms)", stem + ".tdms"):
        val = int(m.group(1))
        if 1000 <= val <= 9999:
            return val * 1e3
    return None


def load_scan_tdms(path: str | Path) -> ScanData:
    """Adapter: produce ``ScanData`` from a v1 TDMS file.

    Wraps the legacy TDMS layout (PosX/PosY/RSSI in ScanData group,
    WFCh1/WFCh2/WFCh4 waveform channels) and infers acquisition
    metadata from filename heuristics where the file itself does not
    carry the field explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    from nptdms import TdmsFile

    role_to_chnum = {
        ROLE_DRIVE_VOLTAGE: 1,
        ROLE_LDV_OUTPUT: 2,
        ROLE_CURRENT: 4,
    }

    with TdmsFile.open(str(path)) as f:
        scan = f["ScanData"]
        pos_x = scan["PosX"][:] * 1e-3   # mm -> m
        pos_y = scan["PosY"][:] * 1e-3
        n_points = len(pos_x)
        try:
            rssi = scan["RSSI"][:n_points]
        except KeyError:
            rssi = None

        # Optional grid shape from the Info group
        n_x_meta = 0
        n_y_meta = 0
        try:
            info = f["Info"]
            try:
                n_x_meta = int(info["NumberOfXPositions"][0])
            except (KeyError, IndexError):
                pass
            try:
                n_y_meta = int(info["NumberOfYPositions"][0])
            except (KeyError, IndexError):
                pass
        except KeyError:
            pass

        wf_group = f["Waveforms"]
        all_names = [c.name for c in wf_group.channels()]

        # Discover which channels are present and which we will expose
        present_roles: dict[str, list[str]] = {}
        for role, chnum in role_to_chnum.items():
            prefix = f"WFCh{chnum}"
            names = _sort_wf_names(
                [n for n in all_names if n.startswith(prefix)]
            )
            if names:
                present_roles[role] = names

        if ROLE_LDV_OUTPUT not in present_roles:
            raise ValueError(
                f"{path.name}: no {ROLE_LDV_OUTPUT} (Ch2) waveforms found"
            )

        ref_ch = wf_group[present_roles[ROLE_LDV_OUTPUT][0]]
        dt = float(ref_ch.properties.get("wf_increment", 8e-9))
        n_samples = len(ref_ch)

    # ----- filename-encoded metadata (v2 moves these into the file) -----
    stem = path.stem
    vel_scale = _detect_velocity_scale_from_name(stem)
    vpp = _detect_vpp_from_name(stem)
    f_drive_nom = _detect_drive_freq_from_name(stem)

    metadata: dict[str, Any] = {
        "source_format": "tdms_v1",
        "source_path": str(path),
        "channel_roles": dict(
            zip(role_to_chnum.keys(), role_to_chnum.values())
        ),
        "_available_roles": sorted(present_roles),
        "n_x": n_x_meta,
        "n_y": n_y_meta,
        "scan_n_x": n_x_meta,
        "scan_n_y": n_y_meta,
        "ldv_velocity_scale_mps_per_v": vel_scale,
        "drive_voltage_vpp": vpp,
        "drive_frequency_hz_nominal": f_drive_nom,
        "sample_rate_hz": 1.0 / dt,
    }

    # Lazy loader closure: opens TDMS, reads requested points for role.
    def loader(role: str, points: slice | np.ndarray) -> np.ndarray:
        if role not in present_roles:
            raise KeyError(
                f"{role!r} not available in {path.name}; "
                f"available: {sorted(present_roles)}"
            )
        names = present_roles[role]
        if isinstance(points, slice):
            idx = list(range(*points.indices(n_points)))
        else:
            idx = list(np.asarray(points, dtype=int))
        out = np.empty((len(idx), n_samples), dtype=np.float64)
        with TdmsFile.open(str(path)) as f:
            wf_group = f["Waveforms"]
            for i, p in enumerate(idx):
                out[i] = wf_group[names[p]][:]
        return out

    return ScanData(
        pos_x=pos_x,
        pos_y=pos_y,
        rssi=rssi,
        dt=dt,
        n_points=n_points,
        n_samples=n_samples,
        metadata=metadata,
        _loader=loader,
    )


_V2_HDF5_ROLE_DATASETS = {
    ROLE_DRIVE_VOLTAGE: "waveforms/drive_voltage",
    ROLE_LDV_OUTPUT:    "waveforms/ldv_output",
    ROLE_CURRENT:       "waveforms/current",
}

# Required root attributes per plans/data_format_v2.md
_V2_REQUIRED_ATTRS = (
    "sample_rate_hz",
    "n_samples",
    "ldv_velocity_scale_mps_per_v",
    "drive_frequency_hz_nominal",
    "drive_voltage_vpp",
    "burst_on_us_nominal",
    "burst_off_us_nominal",
    "scan_n_x",
    "scan_n_y",
    "chip_id",
    "session_id",
    "timestamp_utc",
    "operator",
    "daq_software_version",
)


def _decode_attr(value):
    """h5py returns bytes for string attrs in some versions — normalize."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_scan_hdf5(path: str | Path) -> ScanData:
    """Reader for the v2 (HDF5) acquisition format.

    Schema: see ``plans/data_format_v2.md``.
    Reads coordinates and root attributes eagerly; waveforms remain
    lazy via ``ScanData.load_waveforms(role, points)``.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(str(path), "r") as f:
        attrs = {k: _decode_attr(v) for k, v in f.attrs.items()}

        missing = [k for k in _V2_REQUIRED_ATTRS if k not in attrs]
        if missing:
            raise ValueError(
                f"{path.name}: missing required attributes {missing}"
            )

        coords = f["coordinates"]
        pos_x = coords["pos_x_m"][:]
        pos_y = coords["pos_y_m"][:]
        n_points = len(pos_x)
        if len(pos_y) != n_points:
            raise ValueError(
                f"{path.name}: pos_x/pos_y length mismatch "
                f"({len(pos_x)} vs {len(pos_y)})"
            )
        rssi = coords["rssi"][:] if "rssi" in coords else None

        # Discover which waveform roles are present
        wf = f.get("waveforms")
        if wf is None:
            raise ValueError(f"{path.name}: missing 'waveforms' group")
        available_roles: list[str] = []
        for role, dset_path in _V2_HDF5_ROLE_DATASETS.items():
            if dset_path.split("/", 1)[1] in wf:
                available_roles.append(role)
        if ROLE_LDV_OUTPUT not in available_roles:
            raise ValueError(
                f"{path.name}: required waveform 'ldv_output' not found"
            )

        n_samples = int(attrs["n_samples"])
        # Cross-check against the dataset shape
        dset_shape = wf[_V2_HDF5_ROLE_DATASETS[ROLE_LDV_OUTPUT].split("/", 1)[1]].shape
        if dset_shape != (n_points, n_samples):
            raise ValueError(
                f"{path.name}: ldv_output shape {dset_shape} != "
                f"(n_points={n_points}, n_samples={n_samples})"
            )

    sample_rate_hz = float(attrs["sample_rate_hz"])
    dt = 1.0 / sample_rate_hz

    metadata: dict[str, Any] = dict(attrs)
    metadata["source_format"] = "hdf5_v2"
    metadata["source_path"] = str(path)
    metadata["channel_roles"] = {r: r for r in available_roles}
    metadata["_available_roles"] = sorted(available_roles)

    # Lazy loader: opens HDF5 per call, reads requested rows of the
    # role dataset. h5py supports fancy indexing on the leading axis.
    def loader(role: str, points: slice | np.ndarray) -> np.ndarray:
        if role not in available_roles:
            raise KeyError(
                f"{role!r} not available in {path.name}; "
                f"available: {available_roles}"
            )
        dset_path = _V2_HDF5_ROLE_DATASETS[role]
        with h5py.File(str(path), "r") as f:
            dset = f[dset_path]
            if isinstance(points, slice):
                out = dset[points]
            else:
                idx = np.asarray(points, dtype=int)
                # h5py requires sorted, unique indices for fancy indexing
                sort_order = np.argsort(idx)
                sorted_idx = idx[sort_order]
                raw = dset[sorted_idx, :]
                inv = np.empty_like(sort_order)
                inv[sort_order] = np.arange(len(idx))
                out = raw[inv]
            return np.asarray(out)

    return ScanData(
        pos_x=pos_x,
        pos_y=pos_y,
        rssi=rssi,
        dt=dt,
        n_points=n_points,
        n_samples=n_samples,
        metadata=metadata,
        _loader=loader,
    )


def load_scan(path: str | Path) -> ScanData:
    """Dispatch to the correct loader based on file extension.

    .tdms -> load_scan_tdms (v1)
    .h5 / .hdf5 -> load_scan_hdf5 (v2 schema)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tdms":
        return load_scan_tdms(path)
    if suffix in {".h5", ".hdf5"}:
        return load_scan_hdf5(path)
    raise ValueError(f"Unknown scan-data format: {path.suffix!r} ({path})")


# Required v2 datasets per the schema (must be present)
_V2_REQUIRED_DATASETS = (
    "coordinates/pos_x_m",
    "coordinates/pos_y_m",
    "waveforms/drive_voltage",
    "waveforms/ldv_output",
)


def validate_hdf5_v2(path: str | Path) -> list[str]:
    """Check an HDF5 file against the v2 schema.

    Returns an empty list if the file is valid, otherwise a list of
    human-readable problem descriptions. Run this on the first file
    each new DAQ produces; ship green.

    Checks: file opens, required root attrs present, required datasets
    present, position/rssi/waveform shapes consistent, waveforms
    chunked for lazy reads.
    """
    import h5py

    problems: list[str] = []
    path = Path(path)
    if not path.exists():
        return [f"file not found: {path}"]

    try:
        f = h5py.File(str(path), "r")
    except Exception as e:
        return [f"failed to open as HDF5: {e}"]

    with f:
        attrs = dict(f.attrs)
        for key in _V2_REQUIRED_ATTRS:
            if key not in attrs:
                problems.append(f"missing required attribute {key!r}")

        for ds_path in _V2_REQUIRED_DATASETS:
            if ds_path not in f:
                problems.append(f"missing required dataset {ds_path!r}")

        # Cross-check shapes only when basics are present
        if "coordinates/pos_x_m" in f and "coordinates/pos_y_m" in f:
            n_x = len(f["coordinates/pos_x_m"])
            n_y = len(f["coordinates/pos_y_m"])
            if n_x != n_y:
                problems.append(
                    f"pos_x_m and pos_y_m differ in length: {n_x} vs {n_y}"
                )
            if "coordinates/rssi" in f:
                n_r = len(f["coordinates/rssi"])
                if n_r != n_x:
                    problems.append(
                        f"rssi length {n_r} != positions length {n_x}"
                    )

            n_samples_attr = int(attrs.get("n_samples", 0))
            for role_path in (
                "waveforms/drive_voltage",
                "waveforms/ldv_output",
                "waveforms/current",
            ):
                if role_path not in f:
                    continue
                dset = f[role_path]
                shape = dset.shape
                if shape != (n_x, n_samples_attr):
                    problems.append(
                        f"{role_path} shape {shape} != "
                        f"(n_points={n_x}, n_samples={n_samples_attr})"
                    )
                if dset.chunks is None:
                    problems.append(
                        f"{role_path} is not chunked; lazy reads will be slow"
                    )

    return problems


def write_scan_hdf5(
    scan: ScanData,
    path: str | Path,
    *,
    chunk_points: int = 100,
    compression: str | None = None,
    waveform_dtype: str = "float32",
) -> Path:
    """Write a ``ScanData`` to the v2 HDF5 format.

    Used by the TDMS→v2 converter and by any future ScanData producer.
    Streams waveforms in blocks of ``chunk_points`` so memory usage is
    bounded regardless of file size. The required v2 root attributes
    must be present in ``scan.metadata`` (see
    ``plans/data_format_v2.md``); a missing one raises ``ValueError``
    before any disk I/O.

    Parameters
    ----------
    waveform_dtype : str
        NumPy dtype for stored waveforms (default ``"float32"``).
        ``"float32"`` halves disk usage and is well below the LDV ADC
        noise floor — the right choice for production. Use
        ``"float64"`` when byte-exact round-tripping with TDMS is
        wanted (e.g. for the cross-format cache equivalence test).
    """
    import h5py

    path = Path(path)

    # Fill derivable fields from the ScanData dataclass if absent
    metadata = dict(scan.metadata)
    metadata.setdefault("sample_rate_hz", 1.0 / scan.dt)
    metadata.setdefault("n_samples", scan.n_samples)

    missing = [k for k in _V2_REQUIRED_ATTRS if k not in metadata]
    if missing:
        raise ValueError(
            f"scan.metadata missing required v2 attributes {missing}"
        )

    available_roles = scan.metadata.get(
        "_available_roles", [ROLE_DRIVE_VOLTAGE, ROLE_LDV_OUTPUT]
    )
    if ROLE_LDV_OUTPUT not in available_roles:
        raise ValueError(
            f"scan must provide {ROLE_LDV_OUTPUT!r} role; got {available_roles}"
        )

    # Keys we don't write to the output file (transient / format-specific)
    SKIP_ATTR_KEYS = {
        "source_format", "source_path", "channel_roles", "_available_roles"
    }

    with h5py.File(str(path), "w") as f:
        # Root attributes (using the augmented metadata dict)
        for key, value in metadata.items():
            if key in SKIP_ATTR_KEYS:
                continue
            if value is None:
                continue
            f.attrs[key] = value
        f.attrs["version"] = "2.0"

        # Coordinates
        coords = f.create_group("coordinates")
        coords.create_dataset("pos_x_m", data=scan.pos_x.astype(np.float64))
        coords.create_dataset("pos_y_m", data=scan.pos_y.astype(np.float64))
        if scan.rssi is not None:
            coords.create_dataset("rssi", data=np.asarray(scan.rssi))

        # Waveforms: pre-allocate then stream chunked writes
        wf = f.create_group("waveforms")
        np_dtype = np.dtype(waveform_dtype)
        kwargs = dict(
            shape=(scan.n_points, scan.n_samples),
            dtype=np_dtype,
            chunks=(1, scan.n_samples),
        )
        if compression:
            kwargs["compression"] = compression

        dsets = {}
        for role in available_roles:
            dset_name = _V2_HDF5_ROLE_DATASETS[role].split("/", 1)[1]
            dsets[role] = wf.create_dataset(dset_name, **kwargs)

        for i0 in range(0, scan.n_points, chunk_points):
            i1 = min(i0 + chunk_points, scan.n_points)
            for role, dset in dsets.items():
                block = scan.load_waveforms(role, slice(i0, i1))
                dset[i0:i1] = block.astype(np_dtype, copy=False)

    return path
