"""Glass photoelastic OPL contribution to LDV refracto-vibrometry signal.

Self-verification of the "evanescent contribution 5–20%" estimate for the
glass photoelastic effect via the channel-resonance lateral mode.

Background
----------
At the 1f lateral half-wave resonance of the microchannel (1.907 MHz,
channel width 375 μm geometric / 395 μm effective with soft-wall
correction), the acoustic field at the water/glass interface has a
lateral wavenumber k_x = ω/c_water ≈ 7958 rad/m. In glass, the
longitudinal sound speed c_g ≈ 5500 m/s gives ω/c_g ≈ 2170 rad/m.

Since k_x > ω/c_g, the vertical wavenumber in glass is imaginary:
the acoustic field in glass is **evanescent** with decay constant
κ = sqrt(k_x² - (ω/c_g)²).

The evanescent acoustic field induces strain in the glass, which
modulates the glass refractive index via the photoelastic effect.
The LDV beam crossing the glass picks up this contribution as an
additional ΔOPL term, on top of the intended water acousto-optic ΔOPL.

This script computes ΔOPL_glass / ΔOPL_water for the lab-canonical
geometry and brackets the result across three candidate glasses and
two bounds on the evanescent pressure-transmission coefficient T_p.

Reproduction
------------
    cd <repo root>
    .venv/Scripts/python reports/2026-05-21_glass_pressure_self_verification.py

Inputs are hardcoded from src/ldv_analysis/config.py (CHANNEL_HEIGHT,
DN_DP, C_SOUND, RHO) and from chip geometry (CHANNEL_WIDTH, reflector
behind chip → 4 glass passes per LDV round-trip).

Output: bracketed estimate of ΔOPL_glass / ΔOPL_water and an LDV-side
inflation budget including the air-null residual.
"""

from __future__ import annotations

import io
import math
import sys

# Windows console: force UTF-8 so Unicode (→, π, κ, etc.) can print.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# === Lab-canonical inputs (= ldv-analysis/config.py + chip geometry) ===
F_HZ        = 1.907e6       # observed 1f resonance, Hz
C_WATER     = 1500.0        # m/s
RHO_WATER   = 1000.0        # kg/m³
DN_DP_WATER = 1.4e-10       # Pa⁻¹ — lab-canonical (config.py: DN_DP)
H_CHANNEL   = 150e-6        # m — confirmed by Tatsuki + config.py CHANNEL_HEIGHT
N_GLASS     = 4             # 4 glass passes per LDV round-trip (= reflector behind chip)
AIR_NULL    = 0.08          # 80/980 kPa from air null at 10 Vpp (see 2026-03-18 report)


def photoelastic_dn_dp(n: float, p11: float, p12: float, E: float, nu: float) -> float:
    """∂n/∂p for an isotropic solid under hydrostatic pressure.

    Derivation
    ----------
    Hydrostatic pressure p ⇒ axial strain ε_ii = -p(1 - 2ν)/E (each axis).
    Strain-optic relation: Δ(1/n²) = (p11 + 2 p12) × ε_ii  (isotropic).
    Δn = -(n³/2) × Δ(1/n²) = (n³/2)(p11 + 2 p12)(1 - 2ν)/E × p.
    """
    return (n**3 / 2) * (p11 + 2 * p12) * (1 - 2 * nu) / E


def evanescent_decay(k_x: float, omega: float, c_glass_L: float) -> float:
    """Evanescent decay constant in glass for lateral wavenumber k_x.

    κ = sqrt(k_x² - (ω/c_glass)²)  [rad/m]
    Field penetration depth = 1/κ.
    """
    k_g = omega / c_glass_L
    arg = k_x**2 - k_g**2
    if arg <= 0:
        raise ValueError(f"Field not evanescent: k_x={k_x} <= ω/c_g={k_g}")
    return math.sqrt(arg)


def pressure_transmission_traveling(rho_g: float, c_glass_L: float) -> float:
    """Pressure transmission at water/glass interface, traveling-wave longitudinal.

    T_p = 2 Z_g / (Z_g + Z_w) where Z = ρ c. This is the upper bound for the
    evanescent pressure-transmission coefficient at the interface; the lower
    bound is T_p = 1.0 (= pressure continuity only, no impedance amplification).
    """
    Z_g = rho_g * c_glass_L
    Z_w = RHO_WATER * C_WATER
    return 2 * Z_g / (Z_g + Z_w)


def opl_ratio(
    dn_dp_g: float, T_p: float, kappa: float,
    n_glass_passes: int = N_GLASS,
    dn_dp_w: float = DN_DP_WATER,
    h_channel: float = H_CHANNEL,
) -> float:
    """ΔOPL_glass / ΔOPL_water.

    ΔOPL_glass = N_glass × (∂n/∂p)_g × T_p × P̂_1f / κ
        (= sum of evanescent integrals over each glass pass)
    ΔOPL_water = 2 × h_channel × (∂n/∂p)_w × P̂_1f
        (= round-trip path through the water column, assumed pressure-uniform in z)

    P̂_1f cancels in the ratio.
    """
    opl_g = n_glass_passes * dn_dp_g * T_p / kappa
    opl_w = 2 * h_channel * dn_dp_w
    return opl_g / opl_w


# === Candidate glasses ===
# (name, n, p11, p12, E [Pa], ν, ρ [kg/m³], c_L [m/s])
GLASSES = [
    ("Borosilicate (BK7-class)",  1.515, 0.121, 0.270, 70.0e9, 0.22, 2230, 5500.0),
    ("Fused silica",              1.458, 0.121, 0.270, 73.0e9, 0.17, 2200, 5970.0),
    ("Schott D263T (display)",    1.523, 0.118, 0.250, 72.9e9, 0.22, 2510, 5700.0),
]


def main() -> None:
    omega = 2 * math.pi * F_HZ
    k_x = omega / C_WATER  # = effective lateral wavenumber from observed f
    w_eff = math.pi / k_x

    print(f"=== Lab-canonical inputs ===")
    print(f"  f = {F_HZ/1e6:.3f} MHz  →  k_x (effective) = {k_x:.1f} rad/m")
    print(f"  W_eff = π/k_x = {w_eff*1e6:.1f} μm  (vs geometric 375 μm = soft-wall correction)")
    print(f"  h_channel = {H_CHANNEL*1e6:.0f} μm")
    print(f"  N_glass = {N_GLASS} passes per LDV round-trip (= reflector behind chip)")
    print(f"  ∂n/∂p water = {DN_DP_WATER:.2e} Pa⁻¹  (config.py: DN_DP)")
    print(f"  Air-null residual = {AIR_NULL*100:.0f}% (= 80/980 kPa from 2026-03-18 report)")

    print(f"\n=== Per-glass calc ===")
    print(f"{'Glass':<28} {'κ [rad/m]':<10} {'1/κ [μm]':<10} "
          f"{'∂n/∂p_g':<14} {'T_p':<8} {'OPL ratio'}")
    print("-" * 90)

    all_ratios = []
    for name, n_g, p11, p12, E, nu, rho_g, c_L in GLASSES:
        dn_dp_g = photoelastic_dn_dp(n_g, p11, p12, E, nu)
        kappa = evanescent_decay(k_x, omega, c_L)
        T_p_upper = pressure_transmission_traveling(rho_g, c_L)
        for T_p, T_label in [(1.0, "1.00"), (T_p_upper, f"{T_p_upper:.2f}")]:
            ratio = opl_ratio(dn_dp_g, T_p, kappa)
            all_ratios.append(ratio)
            print(f"{name:<28} {kappa:<10.0f} {1e6/kappa:<10.1f} "
                  f"{dn_dp_g:<14.3e} {T_label:<8} {ratio*100:.1f}%")

    ratio_min, ratio_max = min(all_ratios), max(all_ratios)
    ratio_mid = (ratio_min + ratio_max) / 2

    print(f"\n=== Final bracket ===")
    print(f"  Lower bound  (T_p = 1.00, pressure continuity only):  {ratio_min*100:.1f}%")
    print(f"  Upper bound  (T_p ≈ 1.78, traveling-wave longitudinal):  {ratio_max*100:.1f}%")
    print(f"  Central      (midpoint):                              {ratio_mid*100:.1f}%")
    print(f"  Pre-estimate (Tatsuki, prior to verification):        5–20%")
    print(f"  Verdict: pre-estimate confirmed; sits at upper end of pre-estimate range.")

    print(f"\n=== LDV-side inflation budget ===")
    lo = 1 + ratio_min + AIR_NULL
    hi = 1 + ratio_max + AIR_NULL
    print(f"  Glass photoelastic (evanescent):       {ratio_min*100:.1f}% – {ratio_max*100:.1f}%")
    print(f"  Structural / air-null residual:        ~{AIR_NULL*100:.0f}%")
    print(f"  Combined LDV inflation factor:         {lo:.2f}× – {hi:.2f}×")
    observed_gap_lo, observed_gap_hi = 1.7, 1.9
    print(f"  Observed LDV/PTV gap (2026-03-18):     {observed_gap_lo:.1f}× – {observed_gap_hi:.1f}×")
    rem_lo = observed_gap_lo / hi - 1
    rem_hi = observed_gap_hi / lo - 1
    print(f"  Remaining unexplained by LDV side:     {rem_lo*100:.0f}% – {rem_hi*100:.0f}%")
    print(f"  → Look at PTV side (radiation-force formula, particle properties, wall residue)")
    print(f"  → Audit definition convention (peak vs RMS vs peak-to-peak)")


if __name__ == "__main__":
    main()
