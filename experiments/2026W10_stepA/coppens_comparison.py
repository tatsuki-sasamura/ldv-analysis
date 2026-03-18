# %%
"""Compare measured 2f/1f ratio with Coppens prediction.

Coppens (1968): P_2f / P_1f = (1/4) * beta * Q_2 * M
where beta = coefficient of nonlinearity, Q_2 = quality factor at 2f,
M = p_0 / (rho * c) is the Mach number of the fundamental.

Usage:
    python coppens_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    C_SOUND,
    FIG_DPI,
    RHO,
    figsize_for_layout,
    get_output_dir,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = get_output_dir(__file__)

# Coppens parameters
BETA = 3.5     # coefficient of nonlinearity for water
Q_2F = 100     # quality factor at the 2f frequency

# test12 voltage sweep data (6 MHz BW)
VPPS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
P0_1F = np.array([736, 1439, 2280, 2930, 3531, 4262, 5011, 5716, 6466]) * 1e3  # Pa
P0_2F = np.array([12, 58, 98, 168, 276, 353, 459, 620, 772]) * 1e3  # Pa

# %%
# =============================================================================
# Compute
# =============================================================================

ratio_measured = P0_2F / P0_1F
M = P0_1F / (RHO * C_SOUND**2)  # acoustic Mach number = u_0/c = p_0/(rho*c^2)

M_fine = np.linspace(0, M.max() * 1.15, 200)
coppens_pred = 0.25 * BETA * Q_2F * M_fine

print(f"Coppens: P_2f/P_1f = (1/4) * beta * Q_2 * M")
print(f"  beta = {BETA}, Q_2 = {Q_2F}")
print(f"  slope = (1/4)*beta*Q_2 = {0.25 * BETA * Q_2F:.1f}")
print()
print(f"{'Vpp':>5} {'M':>10} {'measured':>10} {'Coppens':>10} {'meas/pred':>10}")
print("-" * 50)
for i in range(len(VPPS)):
    cop = 0.25 * BETA * Q_2F * M[i]
    r = ratio_measured[i] / cop if cop > 0 else 0
    print(f"{VPPS[i]:5d} {M[i]:10.5f} {ratio_measured[i]:10.4f} {cop:10.4f} {r:10.3f}")

# %%
# =============================================================================
# Plot
# =============================================================================

# PIV-calibrated Mach number (PIV gives ~1.72x lower p0 at 5-10 Vpp)
PIV_LDV_RATIO = 1.72
M_piv = M / PIV_LDV_RATIO

fig, ax = plt.subplots(figsize=figsize_for_layout())

ax.plot(M, ratio_measured, "o", markersize=5, color="C0", label="LDV pressure")
ax.plot(M_piv, ratio_measured, "s", markersize=4, color="C1", label="PIV-calibrated pressure")
ax.plot(M_fine, coppens_pred, "--", linewidth=1, color="C3",
        label=f"Coppens: $(1/4)\\beta Q_2 M$, $Q_2={Q_2F}$")

for i, vpp in enumerate(VPPS):
    ax.annotate(f"{vpp}", (M[i], ratio_measured[i]), fontsize=4,
                textcoords="offset points", xytext=(3, 3))

ax.set_xlabel(r"Mach number $M = p_{0,\mathrm{1f}} / (\rho c^2)$")
ax.set_ylabel(r"$p_{0,\mathrm{2f}} / p_{0,\mathrm{1f}}$")
ax.set_title(r"2f/1f ratio vs Coppens prediction")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.tight_layout()
out_path = OUT_DIR / "coppens_comparison.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# Print comparison
print(f"\n{'Vpp':>5} {'M_LDV':>10} {'M_PIV':>10} {'meas/cop_LDV':>13} {'meas/cop_PIV':>13}")
print("-" * 55)
for i in range(len(VPPS)):
    cop_ldv = 0.25 * BETA * Q_2F * M[i]
    cop_piv = 0.25 * BETA * Q_2F * M_piv[i]
    print(f"{VPPS[i]:5d} {M[i]:10.5f} {M_piv[i]:10.5f} "
          f"{ratio_measured[i]/cop_ldv:13.3f} {ratio_measured[i]/cop_piv:13.3f}")

# %%
print("\n=== Done ===")
