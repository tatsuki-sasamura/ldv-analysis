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
    extract_scan_grid,
    extract_waveforms,
    list_tdms_files,
    load_scan_data,
    load_tdms_file,
)

__all__ = [
    # io_utils
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
]
