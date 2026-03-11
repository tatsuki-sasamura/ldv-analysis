# %%
"""BVD (Butterworth–Van Dyke) equivalent circuit extraction.

Fits a multi-branch BVD model to the measured electrical impedance from
the fine frequency sweep (test5).  The model:

    Y(f) = jωC0 + Σ_n 1 / [R_n + j(ωL_n - 1/(ωC_n))]
    Z(f) = 1 / Y(f)

Two motional branches are used for the two visible resonance peaks
(~1.910 and ~1.970 MHz).

Produces:
  1. Measured vs fitted |Z|(f) and θ(f)
  2. Table of extracted parameters: C0, f_n, Q_n, R_n
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from ldv_analysis.config import FIG_DPI, figsize_for_layout, get_data_dir, get_output_dir
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260306experimentA")
FILE_PATTERN = "test5_*.tdms"

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load impedance data
# =============================================================================

tdms_files = sorted(DATA_DIR.glob(FILE_PATTERN))
tdms_files = [f for f in tdms_files if not f.name.endswith("_index")]
print(f"Found {len(tdms_files)} sweep files\n")

if not tdms_files:
    print("No files found. Exiting.")
    sys.exit(0)

all_freqs = []   # Hz
all_Z_mag = []   # ohm
all_Z_phase = [] # deg

for tdms_path in tdms_files:
    cache = load_or_compute(tdms_path, CACHE_DIR)

    has_ch4 = "current_1f" in cache
    if not has_ch4:
        continue

    f_drive = float(cache["f_drive"])
    V = cache["voltage_1f"]
    I = cache["current_1f"]
    phase_vi = cache["phase_vi"]

    valid = V > np.median(V) * 0.5

    all_freqs.append(f_drive)
    all_Z_mag.append(float(np.median(V[valid] / I[valid])))
    all_Z_phase.append(float(np.median(phase_vi[valid])))

freq = np.array(all_freqs)
sort_f = np.argsort(freq)
freq = freq[sort_f]
Z_mag = np.array(all_Z_mag)[sort_f]
Z_phase = np.array(all_Z_phase)[sort_f]

# Complex impedance
Z_data = Z_mag * np.exp(1j * np.radians(Z_phase))

print(f"\n{len(freq)} frequencies, {freq[0]/1e6:.3f}--{freq[-1]/1e6:.3f} MHz")
print(f"|Z| range: {Z_mag.min():.0f}--{Z_mag.max():.0f} ohm")

# %%
# =============================================================================
# BVD model
# =============================================================================

def bvd_impedance(f, C0, f_res, Q, R):
    """Single-branch BVD model: C0 in parallel with series RLC.

    Parameters: C0 (F), f_res (Hz), Q (dimensionless), R (ohm).
    Derived: L = Q*R/(2*pi*f_res), C_mot = 1/(2*pi*f_res*Q*R).
    """
    omega = 2 * np.pi * f
    omega_r = 2 * np.pi * f_res
    L = Q * R / omega_r
    C_mot = 1 / (omega_r * Q * R)
    Y = 1j * omega * C0 + 1 / (R + 1j * (omega * L - 1 / (omega * C_mot)))
    return 1 / Y


def residual(params, f, Z_data):
    """Complex residual for least_squares (split real/imag)."""
    Z_model = bvd_impedance(f, *params)
    diff = (Z_model - Z_data) / np.abs(Z_data)
    return np.concatenate([diff.real, diff.imag])


# %%
# =============================================================================
# Initial guesses
# =============================================================================

# Focus on transducer resonance (~1.97 MHz), ignore chip resonance (~1.91 MHz)
FIT_FMIN = 1.940e6
fit_mask = freq >= FIT_FMIN
freq_fit = freq[fit_mask]
Z_data_fit = Z_data[fit_mask]
Z_mag_fit = Z_mag[fit_mask]
print(f"Fitting range: {freq_fit[0]/1e6:.3f}--{freq_fit[-1]/1e6:.3f} MHz "
      f"({len(freq_fit)} points)")

# f_res from peak conductance G = Re(1/Z)
Y_fit = 1 / Z_data_fit
G_fit = np.real(Y_fit)
f_res_guess = freq_fit[np.argmax(G_fit)]

# Q from susceptance extrema: B = Im(1/Z) has max/min at f_res +/- df/2
B_fit = np.imag(Y_fit)
i_lo = int(min(np.argmax(B_fit), np.argmin(B_fit)))
i_hi = int(max(np.argmax(B_fit), np.argmin(B_fit)))
Q_guess = f_res_guess / (freq_fit[i_hi] - freq_fit[i_lo])
print(f"  f_res (peak G): {f_res_guess/1e6:.4f} MHz")
print(f"  Q (susceptance FWHM): {Q_guess:.0f}")

# C0: at series resonance motional reactance cancels, so B(f_res) = w*C0
i_Gmax = np.argmax(G_fit)
C0_guess = B_fit[i_Gmax] / (2 * np.pi * freq_fit[i_Gmax])

# R: at series resonance |Z| ~ R (motional branch dominates)
R_guess = Z_mag_fit[np.argmin(Z_mag_fit)]

# params: [C0, f_res, Q, R]
p0 = [C0_guess, f_res_guess, Q_guess, R_guess]

print(f"  Initial: C0={C0_guess*1e12:.1f} pF, f={f_res_guess/1e6:.3f} MHz, "
      f"Q={Q_guess}, R={R_guess:.0f} ohm")

# %%
# =============================================================================
# Fit
# =============================================================================

# Bounds: [C0, f_res, Q, R]
lb = [1e-15, 1.94e6, 10,  10]
ub = [1e-9,  2.05e6, 2000, 2000]

result = least_squares(residual, p0, args=(freq_fit, Z_data_fit),
                       bounds=(lb, ub), method="trf", max_nfev=10000)

C0, f_res_fit, Q_fit, R_fit = result.x
print(f"\nFit converged: {result.success}, cost = {result.cost:.4f}")

# Derived RLC
L_fit = Q_fit * R_fit / (2 * np.pi * f_res_fit)
C_mot = 1 / (2 * np.pi * f_res_fit * Q_fit * R_fit)

print(f"\n  f_res = {f_res_fit/1e6:.4f} MHz")
print(f"  Q     = {Q_fit:.0f}")
print(f"  R     = {R_fit:.1f} ohm")
print(f"  L     = {L_fit*1e3:.3f} mH")
print(f"  C_mot = {C_mot*1e12:.3f} pF")
print(f"  C0    = {C0*1e12:.2f} pF")

# %%
# =============================================================================
# Plot: measured vs fitted |Z| and phase
# =============================================================================

f_fine = np.linspace(freq_fit[0], freq_fit[-1], 500)
Z_fit_fine = bvd_impedance(f_fine, *result.x)
Z_fit_at_data = bvd_impedance(freq_fit, *result.x)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True,
)

# |Z| -- show full range as context, fit only in fit range
ax1.plot(freq / 1e6, Z_mag, "o", markersize=3, alpha=0.3, color="0.5",
         label="Outside fit range")
ax1.plot(freq_fit / 1e6, Z_mag_fit, "o", markersize=3, label="Measured")
ax1.plot(f_fine / 1e6, np.abs(Z_fit_fine), "-", linewidth=0.8, color="C3",
         label=f"BVD: $Q={Q_fit:.0f}$, $f_s={f_res_fit/1e6:.4f}$ MHz")
ax1.set_ylabel(r"$|Z|$ ($\Omega$)")
ax1.set_title("BVD impedance fit --- transducer resonance")
ax1.legend(fontsize=5)
ax1.grid(True, alpha=0.3)

# Phase
ax2.plot(freq / 1e6, Z_phase, "o", markersize=3, alpha=0.3, color="0.5")
ax2.plot(freq_fit / 1e6, Z_phase[fit_mask], "o", markersize=3)
ax2.plot(f_fine / 1e6, np.degrees(np.angle(Z_fit_fine)), "-", linewidth=0.8,
         color="C3")
ax2.set_ylabel(r"Phase ($^\circ$)")
ax2.set_xlabel("Frequency (MHz)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "bvd_impedance_fit.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Fit quality
# =============================================================================

rel_err_mag = np.abs(np.abs(Z_fit_at_data) - Z_mag_fit) / Z_mag_fit * 100
rel_err_phase = np.abs(np.degrees(np.angle(Z_fit_at_data)) - Z_phase[fit_mask])

print(f"\nResidual |Z|: median = {np.median(rel_err_mag):.1f}%, "
      f"max = {np.max(rel_err_mag):.1f}%")
print(f"Residual phase: median = {np.median(rel_err_phase):.1f} deg, "
      f"max = {np.max(rel_err_phase):.1f} deg")

# %%
print("\n=== Done ===")
