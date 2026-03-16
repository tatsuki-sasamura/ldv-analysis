# %%
"""Shared configuration for LDV analysis scripts.

Configures paths, measurement parameters, and visualization settings
for refracto-vibrometry analysis of acoustofluidic devices.

Measurement principle
---------------------
The LDV (laser Doppler vibrometer) beam passes through a water-filled
microchannel on the acoustofluidic chip. The acoustic pressure field
driven by the piezo causes density (and therefore refractive index)
changes in the water. The LDV decoder interprets the resulting optical
path length modulation as apparent surface velocity / displacement.
This technique is known as *refracto-vibrometry*.

Ch2 and Ch3 therefore do NOT represent the mechanical vibration of a
surface; they represent the pressure-induced refractive-index change
integrated along the laser path through the fluid.

Conversion to acoustic pressure (decoder calibrated for air, no 1/n):
    d_apparent = -H * (dn/dp) * p
    v_apparent = -2*pi*f * H * (dn/dp) * p
    => p = -d_apparent / (H * dn/dp)
       p = -v_apparent / (2*pi*f * H * dn/dp)

Sign convention: positive pressure increases refractive index (dn/dp > 0),
which increases OPL, which the LDV interprets as the surface receding
(negative apparent velocity).  Hence the minus sign.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers styles

plt.style.use(["science", "ieee"])

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET = os.environ.get("LDV_DATASET", "1d_line_scan")
DATA_DIR = ROOT_DIR / "data" / DATASET / "raw"
CONVERTED_DIR = ROOT_DIR / "data" / DATASET / "converted"

# External data root — for TDMS files stored outside the repo (e.g. OneDrive).
# Resolved from (in order): LDV_DATA_ROOT env var → .env file → fallback.
_DEFAULT_DATA_ROOT = "C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data"

def _resolve_data_root() -> Path:
    """Read LDV_DATA_ROOT from env or .env file at repo root."""
    val = os.environ.get("LDV_DATA_ROOT")
    if val:
        return Path(val)
    env_file = ROOT_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "LDV_DATA_ROOT":
                return Path(value.strip().strip("\"'"))
    return Path(_DEFAULT_DATA_ROOT)

LDV_DATA_ROOT = _resolve_data_root()


def get_data_dir(experiment: str) -> Path:
    """Resolve an experiment subdirectory under LDV_DATA_ROOT.

    Parameters
    ----------
    experiment : str
        Subdirectory name, e.g. "20260303experimentA".

    Returns
    -------
    Path
        ``LDV_DATA_ROOT / experiment``.  Existence is NOT checked here so that
        scripts can import config without the data being present.
    """
    return LDV_DATA_ROOT / experiment


def load_channel_geometry(dataset: str, cache_dir: Path) -> dict:
    """Load calibrated channel geometry for a dataset.

    Returns a dict with keys: centre_left_m, centre_right_m,
    y_min_m, y_max_m, tilt_deg.  Legacy ``_mm`` keys are accepted
    and auto-converted to ``_m``.

    Raises FileNotFoundError if no geometry file exists.
    """
    import json
    geom_path = cache_dir / f"channel_geometry_{dataset}.json"
    if not geom_path.exists():
        raise FileNotFoundError(
            f"No geometry file: {geom_path}\n"
            f"Run calibrate_geometry.py first.")
    with open(geom_path) as f:
        geom = json.load(f)
    # Accept legacy _mm keys and convert to _m
    if "centre_left_mm" in geom and "centre_left_m" not in geom:
        for key in ("centre_left", "centre_right", "y_min", "y_max"):
            geom[f"{key}_m"] = geom[f"{key}_mm"] * 1e-3
    return geom


def channel_centre_func(geom: dict):
    """Return a function centre(pos_y) from geometry dict.

    All positions in metres.

    Usage::

        geom = load_channel_geometry("20260307experimentB", cache_dir)
        centre = channel_centre_func(geom)
        pos_x_c = pos_x - centre(pos_y)
    """
    import numpy as _np
    c_left = geom["centre_left_m"]
    c_right = geom["centre_right_m"]
    y_min = geom["y_min_m"]
    y_max = geom["y_max_m"]
    a = (c_right - c_left) / (y_max - y_min)
    b = c_left - a * y_min

    def _centre(pos_y):
        return a * _np.asarray(pos_y) + b

    return _centre


def get_output_dir(script_file: str) -> Path:
    """Get output directory for a script (creates subfolder based on script name).

    Output is placed next to the script: <experiment_dir>/output/<script_name>/
    """
    script_path = Path(script_file).resolve()
    script_name = script_path.stem
    out_dir = script_path.parent / "output" / script_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# Channel definitions
# =============================================================================
# The TDMS files contain 4 measurement channels (ScanData group).
# Each channel has Freq, Amp, and Phase sub-channels.
#
# Ch1: Applied voltage to piezo (x10 attenuation)
#       -> Actual voltage = measured_value * 10
#       -> Use this to identify driving amplitude (piezo driven from an
#          external function generator, NOT from the vibrometer's output).
#       -> SetFreqCh1/SetAmpCh1 in the file are IRRELEVANT (not the actual
#          drive). The Polytec software can generate a drive signal, but it
#          was not used; an external function generator drove the piezo
#          instead. Therefore all ScanData summary fields (Ch*Freq, Ch*Amp,
#          Ch*Phase) reflect the Polytec's internal configuration, not the
#          actual measurement. Always extract frequency and amplitude from
#          the raw waveforms via FFT.
#
# Ch2: LDV velocity decoder output (refracto-vibrometry)
#       -> Apparent velocity from refractive-index modulation in the
#          water-filled microchannel, NOT actual surface velocity.
#       -> Polytec VIB-A-511 decoder sensitivity: 2 m/s/V @ 50 Ω
#       -> PicoScope 5442D input impedance: 1 MΩ (fixed, not switchable)
#          No 50 Ω termination → scope sees full open-circuit voltage
#          (2× the voltage that would appear across a 50 Ω load).
#          Effective scale = 2 / 2 = 1 m/s per V.
#       -> Scale: 1 m/s per V (default)
#       -> EXCEPTION: 6vpp.tdms and test5.tdms use 0.5 m/s per V
#
# Ch3: LDV displacement decoder output (refracto-vibrometry)
#       -> Apparent displacement from optical path length change,
#          NOT actual surface displacement.
#       -> Scale: 0.5 µm/V (verified by cross-check with Ch2 via v = 2πf·d;
#          most files give 0.48–0.52 µm/V)
#       -> EXCEPTION: 6vpp.tdms and test5.tdms use 0.25 µm/V
#
# Ch4: Driving current to piezo
#       -> Scale: 0.2 A/V
#
# Ch1 and Ch4 are approximately constant within each file (rel. std < 0.5%).

CHANNEL_VELOCITY = 2
CHANNEL_DISPLACEMENT = 3
CHANNEL_VOLTAGE = 1
CHANNEL_CURRENT = 4

# Physical scale factors (decoder output -> apparent physical units)
VELOCITY_SCALE = 1.0        # m/s per V (apparent velocity, refracto-vibrometry)
DISPLACEMENT_SCALE = 0.5e-6 # m per V (apparent displacement, 0.5 µm/V)
VOLTAGE_ATTENUATION = 10    # x10 attenuation on voltage probe
CURRENT_SCALE = 0.2         # A per V

# =============================================================================
# Refracto-vibrometry parameters
# =============================================================================
# Microchannel and medium properties for pressure conversion.
# LDV decoder is calibrated for air (no medium correction), so:
#   p = -d_apparent / (H * dn/dp)
#   p = -v_apparent / (2*pi*f * H * dn/dp)

CHANNEL_HEIGHT = 150e-6     # m — microchannel depth (150 µm)
REFRACTIVE_INDEX = 1.33     # water at visible wavelength
DN_DP = 1.4e-10             # Pa^-1 — piezo-optic coefficient of water at MHz
SENSITIVITY = CHANNEL_HEIGHT * DN_DP  # m/Pa — apparent displacement per unit pressure (2.1e-14)

# Fluid properties (water, 25 °C)
RHO = 1004.0              # kg/m³
C_SOUND = 1508.0           # m/s

# Microchannel geometry
CHANNEL_WIDTH = 0.375e-3   # m (375 µm)

# LDV signal quality
RSSI_THRESHOLD = 1.0       # V — minimum RSSI for valid signal

# =============================================================================
# Measurement Parameters
# =============================================================================

# Drive frequency is NOT known a priori — must be determined from FFT of
# waveform data (Waveforms group). The ScanData summary channels report the
# Polytec's internal configuration, NOT the actual drive. The Polytec
# software has its own signal generator, but it was not used in this
# experiment; an external function generator drove the piezo instead.

# LDV decoder bandwidth: 6 MHz.
# At the drive frequency (~1.97 MHz), 1f and 2f (~3.94 MHz) are within
# bandwidth. 3f (~5.91 MHz) is near the edge; 4f+ is beyond.
# Consequence: Ch3 (displacement decoder) rolls off at 3f+, so Ch2
# (velocity decoder) is more reliable for higher harmonics.
# Verified: d(Ch3)/dt ≈ Ch2 holds to ~1% at 1f and 2f across all files.

# Number of channels in the data
N_CHANNELS = 4

# =============================================================================
# File selection
# =============================================================================
# 6vpp.tdms and test5.tdms are exploratory measurements with inconsistent
# config (different Ch2/Ch3 scaling, different scan grid). Exclude them
# from the main analysis dataset.

_EXCLUDED_FILES = {
    "1d_line_scan": {"6vpp.tdms", "test5.tdms"},
}
EXCLUDED_FILES = _EXCLUDED_FILES.get(DATASET, set())

print(f"[config] Dataset: {DATASET}")

# =============================================================================
# Visualization
# =============================================================================

# Colormap for amplitude maps
AMP_CMAP = "viridis"

# Colormap for phase maps
PHASE_CMAP = "twilight"

# Default figure size (from science+ieee style)
DEFAULT_FIGSIZE = plt.rcParams["figure.figsize"]  # (3.3, 2.5)

# Figure DPI for saved images (from science+ieee style)
FIG_DPI = plt.rcParams["figure.dpi"]  # 600

# Override savefig.bbox so figures respect the specified figsize
plt.rcParams["savefig.bbox"] = "standard"


# =============================================================================
# Layout-aware figsize (delegated to layout module)
# =============================================================================

from ldv_analysis.layout import figsize_for_layout  # noqa: F401, E402
