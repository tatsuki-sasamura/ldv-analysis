"""Refracto-vibrometry analysis of acoustofluidic devices from scanning LDV TDMS data."""

__version__ = "0.1.0"

from .analysis import (
    amplitude_phase_to_complex,
    compute_scan_statistics,
    compute_spatial_map,
    export_scan_data,
    normalize_phase,
)
from .io_utils import (
    # v2 (format-agnostic) interface
    ROLE_CURRENT,
    ROLE_DRIVE_VOLTAGE,
    ROLE_LDV_OUTPUT,
    ScanData,
    load_scan,
    load_scan_hdf5,
    load_scan_tdms,
    validate_hdf5_v2,
    write_scan_hdf5,
    # Legacy TDMS-specific helpers
    extract_scan_grid,
    extract_waveforms,
    list_tdms_files,
    load_scan_data,
    load_tdms_file,
)

__all__ = [
    # v2 scan interface
    "ScanData",
    "load_scan",
    "load_scan_tdms",
    "load_scan_hdf5",
    "validate_hdf5_v2",
    "write_scan_hdf5",
    "ROLE_DRIVE_VOLTAGE",
    "ROLE_LDV_OUTPUT",
    "ROLE_CURRENT",
    # legacy TDMS
    "load_tdms_file",
    "load_scan_data",
    "extract_scan_grid",
    "extract_waveforms",
    "list_tdms_files",
    # analysis
    "amplitude_phase_to_complex",
    "normalize_phase",
    "compute_spatial_map",
    "compute_scan_statistics",
    "export_scan_data",
    # config — import as ldv_analysis.config
    # fft_cache — import as ldv_analysis.fft_cache
    # layout — import as ldv_analysis.layout
    # plotting — import as ldv_analysis.plotting
]
