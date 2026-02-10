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

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
CONVERTED_DIR = ROOT_DIR / "data" / "converted"
OUTPUT_DIR = ROOT_DIR / "output"


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

EXCLUDED_FILES = {"6vpp.tdms", "test5.tdms"}

# =============================================================================
# Visualization
# =============================================================================

# Colormap for amplitude maps
AMP_CMAP = "viridis"

# Colormap for phase maps
PHASE_CMAP = "twilight"

# Figure DPI for saved images
FIG_DPI = 150
