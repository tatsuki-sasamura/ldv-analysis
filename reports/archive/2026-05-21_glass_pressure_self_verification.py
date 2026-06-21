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
DN_DP_WATER = 1.48e-10      # Pa⁻¹ — 633 nm visible-light value (2026-05-21
                            # adoption; see report §Addendum). Legacy
                            # 1.4e-10 gave OPL-ratio bracket 10.4-20.5%;
                            # current value gives 9.8-19.3%.
H_CHANNEL   = 150e-6        # m — confirmed by Tatsuki + config.py CHANNEL_HEIGHT
N_GLASS     = 4             # 4 glass passes per LDV round-trip (= reflector behind chip)
# The air-null structural residual (80/980 kPa magnitude from 2026-03-18)
# was previously combined here as a × 1.08 factor.  Dropped 2026-05-21
# after empirical R² = −4.56 on the W21 air-filled mode-fit showed that
# the air-filled signal lacks the |sin(πy/W)| mode shape — i.e. the
# mode-fit procedure that extracts LDV p_0 filters it out.  Any residual
# is captured by noise_rms_pressure in the LDV stat error budget rather
# than as a multiplicative inflation.  See
# reports/2026-05-21_ldv_ptv_uncertainty_budget.md §2.2 / §8 limit 3.


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
# Per 2026-05-22 source audit, the chip's top/bottom glass type cannot
# be confirmed from current documentation.  We evaluate all four
# plausible candidates (N-BK7, generic borosilicate / Borofloat33, fused
# silica, Schott D263T) and report the resulting OPL-ratio bracket as
# the worst-case glass-photoelastic uncertainty.  Sources for each row
# are documented in 2026-05-22_uncertainty_budget_source_audit.md
# items 3, 4, 5.
GLASSES = [
    # Schott N-BK7: SCHOTT N-BK7 official datasheet at 632.8 nm for
    # (n, E, ν, ρ); Krupych et al. 2011, Ukr. J. Phys. Opt. for (p11, p12).
    ("Schott N-BK7",        1.51509, 0.118, 0.226, 82.0e9, 0.206, 2510, 5640.0),
    # Generic borosilicate (Borofloat33 / Pyrex 7740-class): SCHOTT
    # Borofloat 33 datasheet for (n, E, ν, ρ, c_L); (p11, p12) assumed
    # borosilicate-family typical (close to N-BK7's Krupych values).
    ("Borofloat 33 / Pyrex", 1.4714, 0.118, 0.226, 64.0e9, 0.20,  2230, 5500.0),
    # Fused silica: Wikipedia / handbook values (Tatsuki-verified 2026-05-22
    # for non-photoelastic constants); (p11, p12) from Primak & Post 1959
    # *J. Appl. Phys.* 30 (standard reference for fused silica).
    ("Fused silica",        1.4585, 0.121, 0.270, 71.7e9, 0.17,  2203, 5960.0),
    # Schott D263T (cover-glass family): SCHOTT D 263 product page for
    # (n, E, ν, ρ); stress-optic K = 34.7 nm/cm/MPa = 3.47e-12 Pa^-1
    # (SCHOTT) combined with assumed p11 ≈ 0.118 (borosilicate-family)
    # back-solves to p12 ≈ 0.236.  p11/p12 individually NOT published
    # by SCHOTT — see audit item 5.
    ("Schott D263T",        1.5231, 0.118, 0.236, 72.9e9, 0.21,  2510, 5700.0),
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
    print(f"  Air-null residual: filtered by |sin(pi y/W)| mode-fit (R²=-4.56 on")
    print(f"     air-filled scan; see report §LDV-side inflation budget)")

    print(f"\n=== Per-glass calc ===")
    print(f"{'Glass':<28} {'κ [rad/m]':<10} {'1/κ [μm]':<10} "
          f"{'∂n/∂p_g':<14} {'T_p':<8} {'OPL ratio'}")
    print("-" * 90)

    all_ratios = []
    for name, n_g, p11, p12, E, nu, _rho_g, c_L in GLASSES:  # ρ_g unused under T_p=1
        dn_dp_g = photoelastic_dn_dp(n_g, p11, p12, E, nu)
        kappa = evanescent_decay(k_x, omega, c_L)
        # T_p = 1.0 only.  The previous (T_p=1.0, T_p=2 Z_g/(Z_g+Z_w))
        # bracket included the traveling-wave impedance formula as a
        # conservative "what-if" upper bound — but that formula derives
        # from p = ρc·v (real impedance), which holds for *propagating*
        # waves and breaks for the evanescent case at this geometry
        # (k_x > ω/c_glass).  For the evanescent field, pressure
        # continuity gives T_p ≈ 1 with no impedance amplification.
        # Dropped 2026-05-22 per user decision.
        ratio = opl_ratio(dn_dp_g, 1.0, kappa)
        all_ratios.append(ratio)
        print(f"{name:<28} {kappa:<10.0f} {1e6/kappa:<10.1f} "
              f"{dn_dp_g:<14.3e} {'1.00':<8} {ratio*100:.1f}%")

    ratio_min, ratio_max = min(all_ratios), max(all_ratios)
    ratio_mid = (ratio_min + ratio_max) / 2

    print(f"\n=== Final bracket (T_p = 1 model, 2026-05-22 onwards) ===")
    print(f"  Lower bound  (smallest-effect glass: N-BK7):          {ratio_min*100:.1f}%")
    print(f"  Upper bound  (largest-effect glass: fused silica):     {ratio_max*100:.1f}%")
    print(f"  Central      (midpoint across 4 candidate glasses):   {ratio_mid*100:.1f}%")
    print(f"  Bracket width comes from material-property spread")
    print(f"  across 4 candidate glasses, NOT from a T_p model bracket.")

    print(f"\n=== LDV-side inflation budget ===")
    # Glass photoelastic only.  Air-null structural residual is filtered
    # out by the |sin(πy/W)| mode-fit projection (R² = −4.56 on the W21
    # air-filled scan); see report §2.2.
    lo = 1 + ratio_min
    hi = 1 + ratio_max
    print(f"  Glass photoelastic (evanescent):       {ratio_min*100:.1f}% – {ratio_max*100:.1f}%")
    print(f"  Combined LDV inflation factor          {lo:.2f}× – {hi:.2f}×")
    print(f"    (glass only; air-null mode-fit-filtered, not added)")
    observed_gap_lo, observed_gap_hi = 1.7, 1.9
    print(f"  Observed LDV/PTV gap (2026-03-18):     {observed_gap_lo:.1f}× – {observed_gap_hi:.1f}×")
    rem_lo = observed_gap_lo / hi - 1
    rem_hi = observed_gap_hi / lo - 1
    print(f"  Remaining unexplained by LDV side:     {rem_lo*100:.0f}% – {rem_hi*100:.0f}%")
    print(f"  → Look at PTV side (radiation-force formula, particle properties, wall residue)")
    print(f"  → Audit definition convention (peak vs RMS vs peak-to-peak)")


if __name__ == "__main__":
    main()
