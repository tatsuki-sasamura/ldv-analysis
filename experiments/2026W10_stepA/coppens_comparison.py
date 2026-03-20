# %%
"""Compare measured harmonic ratios with Coppens cascade prediction.

Coppens (1975):
  P_2f / P_1f = (1/4) * beta * Q_2 * M
  P_3f / P_1f = (1/16) * beta^2 * Q_2 * Q_3 * M^2

where beta = coefficient of nonlinearity, Q_n = quality factor at nf,
M = p_0 / (rho * c^2) is the acoustic Mach number of the fundamental.

Loads data from voltage_sweep_1d.py's processing loop (same test12 dataset).

Usage:
    python coppens_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    C_SOUND,
    FIG_DPI,
    RHO,
    figsize_for_layout,
    get_output_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.mode_fit import fit_mode_1f, fit_mode_2f

# %%
# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("G:/My Drive/260318_ldv")

FILES = [
    ("test12_5Vpp1907kHz.tdms", 5),
    ("test12_10Vpp1907kHz.tdms", 10),
    ("test12_15Vpp1907kHz.tdms", 15),
    ("test12_20Vpp1907kHz.tdms", 20),
    ("test12_25Vpp1907kHz.tdms", 25),
    ("test12_30Vpp1907kHz.tdms", 30),
    ("test12_35Vpp1907kHz_5m_s_max.tdms", 35),
    ("test12_40Vpp1907kHz_5m_s_max.tdms", 40),
    ("test12_45Vpp1907kHz_5m_s_max.tdms", 45),
]

# Coppens parameters
BETA = 3.5     # coefficient of nonlinearity for water
Q_2F = 100     # quality factor at the 2f frequency
Q_3F = 100     # quality factor at the 3f frequency (assumed equal)

# PIV-calibrated pressure ratio
PIV_LDV_RATIO = 1.72

# %%
# =============================================================================
# Load and process data
# =============================================================================

vpps = []
p0_1f_arr = []
p0_2f_arr = []
p0_3f_arr = []

for fname, vpp in FILES:
    tdms_path = DATA_DIR / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    cache = load_or_compute(tdms_path, CACHE_DIR)
    pos_y = cache["pos_x"]
    pressure_1f = cache["pressure_1f"]
    pressure_2f = cache["pressure_2f"]
    phase_1f = cache["phase_1f"]
    phase_2f = cache["phase_2f"]
    V = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None
    valid = make_valid_mask(V, rssi)

    f_drive = float(cache["f_drive"])
    dt = float(cache["dt"])
    ss_start, ss_end = int(cache["ss_start"]), int(cache["ss_end"])
    ss_n = ss_end - ss_start

    # 1f and 2f mode-shape fits
    p1f_c = pressure_1f[valid] * np.exp(1j * np.radians(phase_1f[valid]))
    res_1f = fit_mode_1f(pos_y[valid], p1f_c, CHANNEL_WIDTH)
    p2f_c = pressure_2f[valid] * np.exp(1j * np.radians(phase_2f[valid]))
    res_2f = fit_mode_2f(pos_y[valid], p2f_c, CHANNEL_WIDTH, res_1f.centre)

    # 3f pressure from raw Ch2 waveforms
    vel_scale = float(cache["velocity_scale"])
    prs_3f = np.zeros(len(pos_y))
    tone_3f = np.exp(-2j * np.pi * 3 * f_drive * np.arange(ss_n) * dt)
    from ldv_analysis.io_utils import load_tdms_file
    tdms_file, _ = load_tdms_file(tdms_path)
    wfg = tdms_file["Waveforms"]
    ch2_names = sorted([ch.name for ch in wfg.channels()
                        if ch.name.startswith("WFCh2")])
    for idx in range(len(ch2_names)):
        wf = wfg[ch2_names[idx]][ss_start:ss_end]
        vel_3f = np.abs(wf @ tone_3f) * 2 / ss_n * vel_scale
        prs_3f[idx] = vel_3f * abs(velocity_to_pressure(3 * f_drive))
    del tdms_file

    # 3f mode-shape fit: |sin(3πy_c/W)|
    y_c = pos_y[valid] - res_1f.centre
    k_3f = 3 * np.pi / CHANNEL_WIDTH
    mode_3f = np.abs(np.sin(k_3f * y_c))
    denom = np.sum(mode_3f**2)
    p0_3f = float(np.sum(prs_3f[valid] * mode_3f) / denom) if denom > 0 else 0

    vpps.append(vpp)
    p0_1f_arr.append(abs(res_1f.p0))
    p0_2f_arr.append(abs(res_2f.p0))
    p0_3f_arr.append(p0_3f)

    print(f"  {vpp:2d} Vpp: p0_1f={abs(res_1f.p0)/1e3:.0f} kPa, "
          f"p0_2f={abs(res_2f.p0)/1e3:.0f} kPa, "
          f"p0_3f={p0_3f/1e3:.1f} kPa")

vpps = np.array(vpps)
P0_1F = np.array(p0_1f_arr)
P0_2F = np.array(p0_2f_arr)
P0_3F = np.array(p0_3f_arr)

# %%
# =============================================================================
# Compute ratios and predictions
# =============================================================================

ratio_2f = P0_2F / P0_1F
ratio_3f = P0_3F / P0_1F
M = P0_1F / (RHO * C_SOUND**2)
M_piv = M / PIV_LDV_RATIO

M_fine = np.linspace(0, M.max() * 1.15, 200)
coppens_2f = 0.25 * BETA * Q_2F * M_fine
coppens_3f = (1.0 / 16) * BETA**2 * Q_2F * Q_3F * M_fine**2

print(f"\nCoppens 2f: P_2f/P_1f = (1/4) * beta * Q_2 * M")
print(f"Coppens 3f: P_3f/P_1f = (1/16) * beta^2 * Q_2 * Q_3 * M^2")
print(f"  beta = {BETA}, Q_2 = {Q_2F}, Q_3 = {Q_3F}")
print()
print(f"{'Vpp':>5} {'M':>10} {'2f/1f':>10} {'Cop 2f':>10} {'ratio':>8}"
      f" {'3f/1f':>10} {'Cop 3f':>10} {'ratio':>8}")
print("-" * 80)
for i in range(len(vpps)):
    cop2 = 0.25 * BETA * Q_2F * M[i]
    r2 = ratio_2f[i] / cop2 if cop2 > 0 else 0
    cop3 = (1.0 / 16) * BETA**2 * Q_2F * Q_3F * M[i]**2
    r3 = ratio_3f[i] / cop3 if cop3 > 0 else 0
    print(f"{vpps[i]:5d} {M[i]:10.5f} {ratio_2f[i]:10.4f} {cop2:10.4f} {r2:8.3f}"
          f" {ratio_3f[i]:10.4f} {cop3:10.4f} {r3:8.3f}")

# %%
# =============================================================================
# Plot
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# --- Panel (a): 2f/1f ---
ax1.plot(M, ratio_2f, "o", markersize=5, color="C0", label="LDV pressure")
ax1.plot(M_piv, ratio_2f, "s", markersize=4, color="C1", label="PIV-calibrated")
ax1.plot(M_fine, coppens_2f, "--", linewidth=1, color="C3",
         label=f"$(1/4)\\beta Q_2 M$, $Q_2={Q_2F}$")

for i, vpp in enumerate(vpps):
    ax1.annotate(f"{vpp}", (M[i], ratio_2f[i]), fontsize=4,
                 textcoords="offset points", xytext=(3, 3))

ax1.set_xlabel(r"Mach number $M$")
ax1.set_ylabel(r"$P_{2f} / P_{1f}$")
ax1.set_title(r"(a) $2f/1f$")
ax1.legend(fontsize=5, frameon=False)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, None)
ax1.set_ylim(0, None)

# --- Panel (b): 3f/1f ---
ax2.plot(M, ratio_3f, "o", markersize=5, color="C0", label="LDV pressure")
ax2.plot(M_piv, ratio_3f, "s", markersize=4, color="C1", label="PIV-calibrated")
ax2.plot(M_fine, coppens_3f, "--", linewidth=1, color="C3",
         label=r"$(1/16)\beta^2 Q_2 Q_3 M^2$")

for i, vpp in enumerate(vpps):
    ax2.annotate(f"{vpp}", (M[i], ratio_3f[i]), fontsize=4,
                 textcoords="offset points", xytext=(3, 3))

ax2.set_xlabel(r"Mach number $M$")
ax2.set_ylabel(r"$P_{3f} / P_{1f}$")
ax2.set_title(r"(b) $3f/1f$")
ax2.legend(fontsize=5, frameon=False)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, None)
ax2.set_ylim(0, None)

plt.tight_layout()
out_path = OUT_DIR / "coppens_comparison.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
print("\n=== Done ===")
