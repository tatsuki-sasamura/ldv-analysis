"""Pre-registered analysis conventions for the PRL figure pipeline.

Mirrors the §3 "Analysis conventions" table in
``nonlinearphysics-manuscript/prl/analysis_contract.md``. Every constant
used by ``figN.py`` lives here; duplicated literals are a bug.

Where a value also exists in ``ldv_analysis.config`` (water properties,
channel geometry, calibration), it is re-exported from there so there
is one source of truth, not two.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the W21 experiment scripts importable for SCANS reuse.
_HERE = Path(__file__).resolve()
_W21 = _HERE.parents[1]
if str(_W21) not in sys.path:
    sys.path.insert(0, str(_W21))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_HEIGHT,
    CHANNEL_WIDTH,
    C_SOUND,
    DN_DP,
    LDV_DATA_ROOT,
    MANUSCRIPT_DIR,
    RHO,
    get_cache_dir,
    velocity_to_pressure,
)
from vpp_vs_pressure import DATA_ROOT, SCANS  # noqa: E402

# ---------------------------------------------------------------------------
# Water / acoustic constants (contract §3 — Physical-parameter ground truth)
# ---------------------------------------------------------------------------
RHO_0 = RHO          # kg/m^3 — Pátek 2009 at T_0=298.15 K (997.047)
C_0 = C_SOUND        # m/s — Pátek 2009 at T_0=298.15 K (1496.70)
# Nonlinear acoustics of water (contract §3 ground-truth table)
B_OVER_A = 5.0
BETA = 1.0 + 0.5 * B_OVER_A  # 3.5

# ---------------------------------------------------------------------------
# Selectors and thresholds (contract §3 — Analysis conventions)
# ---------------------------------------------------------------------------
# Low-drive window is DATA-DEFINED, not hardcoded voltages:
LOW_DRIVE_P2_OVER_P1 = 0.10
MIN_LOW_DRIVE_POINTS = 3        # below this, drop the V^n claim (Fig 2a fallback)

# 4f / 5f adoption in the force series (Fig 3, contract §3):
HARMONIC_SNR_MIN = 5.0
HARMONIC_E_RATIO_SIGMA = 1.0    # E_n/E_1 must exceed its 1-sigma band

# Drive purity bound (Fig S1):
DRIVE_PURITY_THRESHOLD = 0.005  # AFG V_nf/V_1f, n >= 2

# Noise-bias / censoring (contract §3 noise policy):
NOISE_SIGMA_MIN = 3.0           # below this, censor as NaN (never zero)

# Local-linearity thresholds (Fig 3 sensitivity):
LIN_THRESHOLD_PRIMARY = 0.05
LIN_THRESHOLD_SENSITIVITY = 0.10

# ---------------------------------------------------------------------------
# Canonical run set
# ---------------------------------------------------------------------------
# 12 cascade scans (10..120 Vpp), reused from vpp_vs_pressure.SCANS
# (single source so adding a Vpp point is one edit, not three).
CASCADE_SCANS = tuple(SCANS)
W21_DATA_ROOT = DATA_ROOT  # LDV_DATA_ROOT / "output" / "W21"

# Resonance sweeps consumed by Fig S2.
FSWEEP_1F = "sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"
# Preferred 2f sweep: fine 2 kHz step covers BOTH 2f modes at 3.794 + 3.817 MHz
# (per reports/2026-06-18_f2_eigenmode_pin.md). The cascade drives at
# 2*f_1f ~ 3.804 MHz, between the two modes — single-pole fit will misfit.
FSWEEP_2F_FINE = "sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_20260530_072344"
# Fallback if FINE is unavailable: coarser 3.70-3.90 MHz sweep
FSWEEP_2F_COARSE = "sample_101x77_fsweep_3p7to3p9_60Vpp_20260530_013206"

# Pre-pinned cavity-mode frequencies (initial seeds for two-pole fit).
# Locked by f2_eigenmode_scan.py + the 2026-06-18 eigenmode-pin report.
F2_MODE_A_HZ = 3.794e6
F2_MODE_B_HZ = 3.817e6

# ---------------------------------------------------------------------------
# Output and intermediate paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
W21_OUT = ROOT / "experiments" / "2026W21" / "output"
PIPE_OUT = W21_OUT / "prl_pipeline"
PIPE_OUT.mkdir(parents=True, exist_ok=True)

# Inputs already produced by upstream scripts:
LADDER_NPZ = W21_OUT / "harmonic_ladder.npz"
DRIVE_PURITY_CSV = W21_OUT / "drive_purity_b1" / "drive_purity_b1.csv"

# Re-export config helpers used by figN.py:
__all__ = [
    # water + acoustics
    "RHO_0", "C_0", "BETA", "B_OVER_A",
    # geometry / calibration
    "CHANNEL_WIDTH", "CHANNEL_HEIGHT", "DN_DP", "velocity_to_pressure",
    # selectors
    "LOW_DRIVE_P2_OVER_P1", "MIN_LOW_DRIVE_POINTS",
    "HARMONIC_SNR_MIN", "HARMONIC_E_RATIO_SIGMA",
    "DRIVE_PURITY_THRESHOLD", "NOISE_SIGMA_MIN",
    "LIN_THRESHOLD_PRIMARY", "LIN_THRESHOLD_SENSITIVITY",
    # run set
    "CASCADE_SCANS", "W21_DATA_ROOT", "LDV_DATA_ROOT",
    "FSWEEP_1F", "FSWEEP_2F_FINE", "FSWEEP_2F_COARSE",
    "F2_MODE_A_HZ", "F2_MODE_B_HZ",
    # paths
    "ROOT", "W21_OUT", "PIPE_OUT", "LADDER_NPZ", "DRIVE_PURITY_CSV",
    "MANUSCRIPT_DIR", "get_cache_dir",
]
