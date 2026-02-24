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
    d_apparent = H * (dn/dp) * p
    v_apparent = 2*pi*f * H * (dn/dp) * p
    => p = d_apparent / (H * dn/dp)
       p = v_apparent / (2*pi*f * H * dn/dp)
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

ROOT_DIR = Path(__file__).parent.parent
DATASET = os.environ.get("LDV_DATASET", "1d_line_scan")
DATA_DIR = ROOT_DIR / "data" / DATASET / "raw"
CONVERTED_DIR = ROOT_DIR / "data" / DATASET / "converted"
OUTPUT_DIR = ROOT_DIR / "output" / DATASET


def get_output_dir(script_file: str) -> Path:
    """Get output directory for a script (creates subfolder based on script name)."""
    script_name = Path(script_file).stem
    out_dir = OUTPUT_DIR / script_name
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
#       -> Use this to identify driving amplitude (piezo driven from manual
#          wave source, NOT from the vibrometer's controlled output).
#       -> SetFreqCh1/SetAmpCh1 in the file are IRRELEVANT (not the actual drive).
#
# Ch2: LDV velocity decoder output (refracto-vibrometry)
#       -> Apparent velocity from refractive-index modulation in the
#          water-filled microchannel, NOT actual surface velocity.
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
#   p = d_apparent / (H * dn/dp)
#   p = v_apparent / (2*pi*f * H * dn/dp)

CHANNEL_HEIGHT = 150e-6     # m — microchannel depth (150 µm)
REFRACTIVE_INDEX = 1.33     # water at visible wavelength
DN_DP = 1.4e-10             # Pa^-1 — piezo-optic coefficient of water at MHz
SENSITIVITY = CHANNEL_HEIGHT * DN_DP  # m/Pa — apparent displacement per unit pressure (2.1e-14)

# =============================================================================
# Measurement Parameters
# =============================================================================

# Drive frequency is NOT known a priori — must be determined from FFT of
# waveform data (Waveforms group). The ScanData "Freq" channels report the
# vibrometer's internal system configuration, NOT the actual drive frequency
# (the piezo was driven by an external manual wave source).

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
# Layout-aware figsize
# =============================================================================
# tight_layout overhead (margins, gaps) is constant in absolute inches
# (set by font size in points). We measure from reference figures so that
# figsize_for_layout(n_rows, n_cols) gives each subplot the same axes area
# as a 1x1 figure at DEFAULT_FIGSIZE.
#
# Model:
#   fig_h = margin_v + n_rows * ax_h + (n_rows - 1) * gap_v
#   fig_w = margin_h + n_cols * ax_w + (n_cols - 1) * gap_h
#
# Gaps differ between shared and non-shared axes because sharex removes
# redundant xlabels (reducing vertical gap) and sharey removes redundant
# ylabels (reducing horizontal gap).

import numpy as _np

# Representative labels matching the analysis scripts
_XLABEL = "X position (mm)"
_YLABEL = "Y value (unit)"
_TITLE = "Title text"


def _measure_layout_overhead():
    """Measure tight_layout margins and gaps from reference figures."""
    _w, _h = plt.rcParams["figure.figsize"]
    _x = _np.linspace(0, 1, 10)
    _y = _np.zeros(10)

    def _axes_size(fig, ax):
        bbox = ax.get_position()
        return bbox.width * fig.get_figwidth(), bbox.height * fig.get_figheight()

    # 1x1 reference: total margin
    fig, ax = plt.subplots(1, 1, figsize=(_w, _h))
    ax.plot(_x, _y)
    ax.set_xlabel(_XLABEL); ax.set_ylabel(_YLABEL); ax.set_title(_TITLE)
    fig.tight_layout()
    ref_w, ref_h = _axes_size(fig, ax)
    plt.close(fig)

    margin_h = _w - ref_w
    margin_v = _h - ref_h

    # 2x1 sharex=True: vertical gap (shared)
    fig, axes = plt.subplots(2, 1, figsize=(_w, 2 * _h), sharex=True)
    for a in axes:
        a.plot(_x, _y); a.set_ylabel(_YLABEL)
    axes[-1].set_xlabel(_XLABEL); axes[0].set_title(_TITLE)
    fig.tight_layout()
    _, h2_shared = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_v_shared = 2 * _h - margin_v - 2 * h2_shared

    # 2x1 sharex=False: vertical gap (independent)
    fig, axes = plt.subplots(2, 1, figsize=(_w, 2 * _h), sharex=False)
    for a in axes:
        a.plot(_x, _y); a.set_ylabel(_YLABEL); a.set_xlabel(_XLABEL)
    axes[0].set_title(_TITLE)
    fig.tight_layout()
    _, h2_indep = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_v_indep = 2 * _h - margin_v - 2 * h2_indep

    # 1x2 sharey=True: horizontal gap (shared)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _w, _h), sharey=True)
    for a in axes:
        a.plot(_x, _y); a.set_xlabel(_XLABEL)
    axes[0].set_ylabel(_YLABEL); axes[0].set_title(_TITLE)
    fig.tight_layout()
    w2_shared, _ = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_h_shared = max(2 * _w - margin_h - 2 * w2_shared, 0)

    # 1x2 sharey=False: horizontal gap (independent)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _w, _h), sharey=False)
    for a in axes:
        a.plot(_x, _y); a.set_xlabel(_XLABEL); a.set_ylabel(_YLABEL)
    axes[0].set_title(_TITLE)
    fig.tight_layout()
    w2_indep, _ = _axes_size(fig, axes[0])
    plt.close(fig)
    gap_h_indep = max(2 * _w - margin_h - 2 * w2_indep, 0)

    return (ref_w, ref_h, margin_h, margin_v,
            gap_h_shared, gap_h_indep, gap_v_shared, gap_v_indep)


(_REF_AX_W, _REF_AX_H, _MARGIN_H, _MARGIN_V,
 _GAP_H_SHARED, _GAP_H_INDEP, _GAP_V_SHARED, _GAP_V_INDEP,
 ) = _measure_layout_overhead()


def figsize_for_layout(n_rows=1, n_cols=1, ax_w_scale=1.0, ax_h_scale=1.0,
                       sharex=False, sharey=False):
    """Compute figsize so each subplot's axes area matches the 1x1 reference.

    Parameters
    ----------
    n_rows, n_cols : int
        Subplot grid dimensions.
    ax_w_scale, ax_h_scale : float
        Multiplier on the reference axes width/height (e.g. 2.0 for a
        double-wide single plot).
    sharex, sharey : bool
        Whether axes sharing is used (affects inter-subplot gaps).
    """
    target_w = ax_w_scale * _REF_AX_W
    target_h = ax_h_scale * _REF_AX_H
    gap_h = _GAP_H_SHARED if sharey else _GAP_H_INDEP
    gap_v = _GAP_V_SHARED if sharex else _GAP_V_INDEP
    fig_w = _MARGIN_H + n_cols * target_w + max(n_cols - 1, 0) * gap_h
    fig_h = _MARGIN_V + n_rows * target_h + max(n_rows - 1, 0) * gap_v
    return (fig_w, fig_h)
