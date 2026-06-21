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

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication style.  Full ["science", "ieee"] enables LaTeX usetex
# which needs a system LaTeX install; on machines without LaTeX we
# fall back to the "no-latex" variant so the typography/colors apply
# but matplotlib's mathtext does the rendering.  Skipped entirely if
# LDV_NO_STYLE=1 (tests, CI).
if os.environ.get("LDV_NO_STYLE") != "1":
    import shutil
    import scienceplots  # noqa: F401 — registers styles
    if shutil.which("latex"):
        plt.style.use(["science", "ieee"])
    else:
        plt.style.use(["science", "ieee", "no-latex"])

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
_DEFAULT_MANUSCRIPT_DIR = ""

def _resolve_env(key: str, default: str = "") -> str:
    """Read a config key from env var or .env file at repo root."""
    val = os.environ.get(key)
    if val:
        return val
    env_file = ROOT_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip("\"'")
    return default

LDV_DATA_ROOT = Path(_resolve_env("LDV_DATA_ROOT", _DEFAULT_DATA_ROOT))
MANUSCRIPT_DIR = Path(_resolve_env("MANUSCRIPT_DIR", _DEFAULT_MANUSCRIPT_DIR)) if _resolve_env("MANUSCRIPT_DIR", _DEFAULT_MANUSCRIPT_DIR) else None

# Optional shared root for FFT caches.  When set (env or .env), scripts
# using ``get_cache_dir`` will route caches to
# ``LDV_CACHE_ROOT/<scan_dir_name>/fft_cache/`` so a single cache tree
# can be shared between machines (e.g. via OneDrive).  When unset, the
# legacy per-script layout is preserved.
_DEFAULT_CACHE_ROOT = ""
_cache_root_str = _resolve_env("LDV_CACHE_ROOT", _DEFAULT_CACHE_ROOT)
LDV_CACHE_ROOT = Path(_cache_root_str) if _cache_root_str else None


def get_cache_dir(scan_dir_name: str, script_file: str) -> Path:
    """Resolve the FFT cache directory for one scan.

    - With ``LDV_CACHE_ROOT`` set:
      ``LDV_CACHE_ROOT/<scan_dir_name>/fft_cache/``
    - Without (legacy):
      ``<script_dir>/output/<scan_dir_name>/fft_cache/``

    Either way the directory is created if missing.
    """
    if LDV_CACHE_ROOT is not None:
        cache_dir = LDV_CACHE_ROOT / scan_dir_name / "fft_cache"
    else:
        script_dir = Path(script_file).resolve().parent
        cache_dir = script_dir / "output" / scan_dir_name / "fft_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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


# Pre-cleanup geometry JSONs used the British spelling "centre" for the
# channel-center keys. We accept those keys and rewrite them to the new
# spelling so downstream code can rely on `geom["center_left_m"]` etc.
_LEGACY_SPELL = "centre"


def _swap_legacy_spelling(geom: dict) -> dict:
    """In-place: rename ``<legacy>_left_m`` / ``_right_m`` / ``_mm`` keys
    to the new ``center_*`` spelling. Returns the same dict for chaining.
    """
    for side in ("left", "right"):
        for suffix in ("_m", "_mm"):
            old = f"{_LEGACY_SPELL}_{side}{suffix}"
            new = f"center_{side}{suffix}"
            if old in geom and new not in geom:
                geom[new] = geom.pop(old)
    return geom


def load_channel_geometry(dataset: str, cache_dir: Path) -> dict:
    """Load calibrated channel geometry for a dataset.

    Returns a dict with keys: center_left_m, center_right_m,
    y_min_m, y_max_m, tilt_deg.  Pre-cleanup geometry JSONs that used
    the British spelling for the channel-center keys are silently
    accepted and renamed to the new spelling.  Legacy ``_mm`` keys are
    also accepted and converted to ``_m``.  After loading, downstream
    consumers can always rely on ``geom["center_left_m"]`` /
    ``geom["center_right_m"]``.

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
    _swap_legacy_spelling(geom)
    # Accept legacy _mm keys and convert to _m
    if "center_left_mm" in geom and "center_left_m" not in geom:
        for key in ("center_left", "center_right", "y_min", "y_max"):
            geom[f"{key}_m"] = geom[f"{key}_mm"] * 1e-3
    return geom


def channel_center_func(geom: dict):
    """Return a function center(pos_y) from geometry dict.

    All positions in meters.

    Usage::

        geom = load_channel_geometry("20260307experimentB", cache_dir)
        center = channel_center_func(geom)
        pos_x_c = pos_x - center(pos_y)
    """
    import numpy as _np
    c_left = geom["center_left_m"]
    c_right = geom["center_right_m"]
    y_min = geom["y_min_m"]
    y_max = geom["y_max_m"]
    a = (c_right - c_left) / (y_max - y_min)
    b = c_left - a * y_min

    def _center(pos_y):
        return a * _np.asarray(pos_y) + b

    return _center


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
# Adiabatic piezo-optic coefficient of water at 633 nm, 20 °C.
# Literature value 1.48e-10 (diposit.ub.edu uses 1.51e-10 at 532 nm; IAPWS-95
# + Lorentz–Lorenz consistent with 1.43-1.54e-10 across 400-1064 nm).
# Confirmed essentially frequency-independent from kHz to tens of MHz —
# any MHz-band variation candidate (which we briefly considered as a
# possible source of the LDV/PTV gap) is ruled out by literature review
# 2026-05-21; see reports/archive/2026-05-21_glass_pressure_self_verification.md.
# Previous value was 1.4e-10 (Coppens 1971 round number); refining to
# the 633 nm value reduces every LDV-reported pressure by ~5.4%.
DN_DP = 1.48e-10            # Pa^-1
SENSITIVITY = CHANNEL_HEIGHT * DN_DP  # m/Pa — apparent displacement per unit pressure (2.22e-14)


def velocity_to_pressure(f_hz: float, velocity_scale: float = 1.0) -> float:
    """Signed scale factor: multiply by raw voltage (V) to get pressure (Pa).

    +p → +n → +OPL → −v_LDV, hence the minus sign.

    Parameters
    ----------
    f_hz : float
        Drive frequency in Hz.
    velocity_scale : float
        LDV decoder scale in m/s per V (default 1.0).

    Returns
    -------
    float
        Conversion factor such that ``pressure = factor * V_ch2``.
    """
    import math
    return -velocity_scale / (2 * math.pi * f_hz * SENSITIVITY)


# Fluid properties (water)
RHO = 1000.0               # kg/m³
C_SOUND = 1500.0           # m/s

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
