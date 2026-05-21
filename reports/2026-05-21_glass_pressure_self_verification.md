# Glass Photoelastic OPL Contribution — Self-Verification — 2026-05-21

## Purpose

Verify the "evanescent contribution 5–20%" estimate for the glass photoelastic effect on the LDV refracto-vibrometry signal. Establishes a bounded budget for the LDV-side contribution to the LDV/PTV pressure gap (see [`2026-03-18_ldv_piv_crossvalidation.md`](2026-03-18_ldv_piv_crossvalidation.md), ~1.7×–1.9× across 5–15 Vpp).

## TL;DR

| Quantity | Value |
|---|---|
| ΔOPL_glass / ΔOPL_water (at 633 nm, `DN_DP_water = 1.48e-10`) | **9.8% – 19.4%** (central ~14.6%) |
| Pre-estimate verdict | Consistent with the 5–20 % rule of thumb; lands at the upper end. **First-order bounded estimate, not a rigorous fluid-solid elastodynamic solution** (see §Robustness). |
| Combined LDV inflation (glass × air-null structural, multiplicative, phase-aligned upper-bound assumption) | **1.19× – 1.29×** |
| Observed LDV/PTV gap | 1.7× – 1.9× |
| **Remaining gap unexplained by LDV side** | **32% – 60%** |

> Numbers in §Result / §LDV-side inflation budget below have been re-stated
> at the updated `DN_DP_water = 1.48e-10 Pa⁻¹` (the 633 nm visible-light
> value, see §Addendum).  Legacy numbers using `1.4e-10` were 10.4–20.5%
> for the OPL ratio bracket; recomputing with 1.48e-10 narrows it to
> 9.8–19.3%.

→ Glass evanescent + structural together account for only about a third to half of the observed gap. The remainder must come from the PTV side (radiation-force formula, particle properties, wall residue) or from a definition mismatch (peak vs RMS vs peak-to-peak).

## Physical model

At the 1f lateral half-wave resonance of the microchannel (f = 1.907 MHz), the acoustic field at the water/glass interface has lateral wavenumber

$$k_x = \omega / c_\text{water} \approx 7988 \text{ rad/m}$$

(equivalent to W_eff = π/k_x ≈ 393 μm vs. geometric W = 375 μm — the difference is the soft-wall correction from the glass impedance mismatch).

In glass, the longitudinal acoustic wavenumber is ω/c_glass ≈ 2170 rad/m, much smaller than k_x. Therefore the vertical wavenumber in glass is imaginary:

$$\kappa = \sqrt{k_x^2 - (\omega/c_\text{glass})^2} \approx 7685 \text{ rad/m}$$

The acoustic field penetrates the glass as an evanescent decay with characteristic length **1/κ ≈ 130 μm**. Glass thickness ≥ 0.5 mm → e⁻⁴ at the far face → glass thickness is irrelevant for this calc.

The evanescent acoustic field induces strain in the glass, which modulates the glass refractive index via the photoelastic effect. The LDV beam crossing the glass picks up this contribution as an additional ΔOPL term, on top of the intended water acousto-optic ΔOPL.

## Derivation

### Photoelastic ∂n/∂p (isotropic solid, hydrostatic pressure)

Hydrostatic pressure p ⇒ axial strain ε_ii = -p(1-2ν)/E (each axis, isotropic Hooke's law). Strain-optic relation Δ(1/n²) = (p11 + 2 p12) ε_ii. Then:

$$\left(\frac{\partial n}{\partial p}\right)_\text{glass} = \frac{n^3}{2} \cdot (p_{11} + 2 p_{12}) \cdot \frac{1 - 2\nu}{E}$$

### OPL ratio

The LDV round-trip integrates Δn along the beam path. For the lateral mode, water pressure is uniform in z across the channel height (channel height h ≪ λ_water), so the water contribution is just h × Δn_water per pass, doubled for the round trip. For each glass pass, the evanescent integral collapses to (Δn_glass at interface) / κ.

$$\frac{\Delta \text{OPL}_\text{glass}}{\Delta \text{OPL}_\text{water}} = \frac{N_\text{glass} \cdot (\partial n/\partial p)_g \cdot T_p}{2 \cdot h_\text{channel} \cdot (\partial n/\partial p)_w \cdot \kappa}$$

where N_glass is the number of glass passes per LDV round-trip (= **4** for reflector-behind-chip geometry: top in + bottom in + bottom out + top out), and T_p is the pressure transmission coefficient at the water/glass interface.

### T_p bracket

T_p for the evanescent field is bounded between:
- **T_p = 1.00** — pressure continuity only, no impedance amplification (lower bound)
- **T_p = 1.78** — traveling-wave longitudinal-incidence pressure transmission 2Z_g/(Z_g + Z_w), with Z_g ≈ 8 Z_w (upper bound)

A proper evanescent-matching solution at the fluid/solid interface would tighten this, but it is the dominant remaining systematic.

## Inputs (lab-canonical, from `src/ldv_analysis/config.py` + chip)

| Parameter | Value | Source |
|---|---|---|
| f (observed 1f) | 1.907 MHz | observed across W10/W16/W21 campaigns |
| h_channel | 150 μm | `CHANNEL_HEIGHT` in config.py; confirmed by Tatsuki |
| W (geometric channel width) | 375 μm | `CHANNEL_WIDTH` in config.py |
| ∂n/∂p water | 1.4 × 10⁻¹⁰ Pa⁻¹ | `DN_DP` in config.py |
| c_water | 1500 m/s | `C_SOUND` |
| ρ_water | 1000 kg/m³ | `RHO` |
| N_glass (passes per LDV round-trip) | 4 | reflector behind chip |
| Air-null residual | ~8% | `2026-03-18_ldv_piv_crossvalidation.md` (80/980 kPa @ 10 Vpp) |

## Result

Result is bracketed across three candidate glasses (BK7-class, fused silica, Schott D263T) — variation between glasses is **~1 percentage point (~9 % relative)** for fixed T_p.

The table immediately below uses the **legacy** `DN_DP_water = 1.4 × 10⁻¹⁰
Pa⁻¹` (the value `config.py` had at the time the calculation was first run).
For the **updated** `DN_DP_water = 1.48 × 10⁻¹⁰ Pa⁻¹` (the 633 nm value
adopted on 2026-05-21; see addendum), every entry shrinks by the factor
1.4 / 1.48 = 0.946.

| Glass | κ [rad/m] | 1/κ [μm] | ∂n/∂p_g [Pa⁻¹] | T_p = 1.00 (legacy) | T_p ≈ 1.78 (legacy) |
|---|---|---|---|---|---|
| Borosilicate (BK7-class) | 7685 | 130 | 9.19 × 10⁻¹² | 11.4% | 20.3% |
| Fused silica | 7732 | 129 | 9.26 × 10⁻¹² | 11.4% | 20.5% |
| Schott D263T | 7706 | 130 | 8.39 × 10⁻¹² | 10.4% | 18.8% |

**Legacy bracket** (DN_DP_water = 1.4e-10): ΔOPL_glass / ΔOPL_water = 10.4% – 20.5%, central 15.4%.

**Final bracket** (DN_DP_water = 1.48e-10, as adopted in `config.py`):
**ΔOPL_glass / ΔOPL_water = 9.8% – 19.4%, central 14.5%.**

## LDV-side inflation budget

Combined multiplicatively, consistent with the multiplicative chain model
used in `reports/2026-05-21_ldv_ptv_uncertainty_budget.md` §1:

| Source | Contribution to LDV reading |
|---|---|
| Glass photoelastic (evanescent), bracket | × 1.10 – 1.20 |
| Structural / air-null residual (assumed phase-aligned, see below) | × 1.08 |
| **Combined LDV inflation factor** (multiplicative) | **× 1.19 – 1.29** |
| Observed LDV/PTV gap (2026-03-18) | × 1.7 – 1.9 |
| **Remaining gap unexplained by LDV side** | **31 % – 60 %** |

> **Caveat on air-null sign.** The 8 % is the air-filled-channel signal
> *magnitude*; treating it as a strictly positive multiplicative inflation
> assumes its phase is aligned with the water-filled signal — which has not
> been measured.  In the companion budget report
> (`2026-05-21_ldv_ptv_uncertainty_budget.md` §2.1) the air-null is moved
> to a symmetric ±8 % uncertainty for that reason.  The 1.19–1.29× quoted
> here is therefore an upper-bound estimate of the LDV inflation
> *assuming aligned phase*.

## Implications

1. **The 5–20 % pre-estimate is consistent with first-principles photoelastic + evanescent calc** — the central estimate ~15 % lands at the upper end of that range. This is a *first-order bounded estimate*, not a rigorous fluid-solid elastodynamic solution (see §Robustness).
2. **Glass photoelastic alone cannot explain the LDV/PTV gap.** Even at the worst-case upper bound (20.5 %), combined multiplicatively with the air-null residual (~8 %, phase-aligned assumption), the LDV reading is inflated by at most ×1.29. The observed gap is 1.7×–1.9×.
3. **The remaining ~32–61% must come from PTV side or definition mismatch.** Priority candidates:
   - PTV radiation-force formula (= particle radius, compressibility contrast, force-balance assumption)
   - PTV wall-correction residue after the size-vs-defocus height filter (Nikon Eclipse Ti2, 5 μm PS particles)
   - Convention mismatch between LDV and PTV (peak vs RMS vs peak-to-peak; standing-wave amplitude definition)

## Robustness

This calculation is a **first-order bounded estimate**, useful for an
order-of-magnitude budget and consistent with the prior 5–20 % rule of
thumb.  It is *not* a rigorous fluid-solid elastodynamic solution.  The
main approximations:

- **Hydrostatic-strain photoelastic form.**  The expression
  `(∂n/∂p)_g = (n³/2)(p11 + 2 p12)(1−2ν)/E` is derived for an isotropic
  glass under *hydrostatic* pressure.  At the water/glass interface the
  actual evanescent acoustic field has lateral wavenumber `k_x` larger
  than the bulk-glass wavenumber, so the elastic field in the glass has
  *shear* components in addition to dilatational.  A complete treatment
  would solve the coupled fluid-solid boundary-value problem, compute the
  full strain tensor in the glass, contract with the (anisotropic)
  photoelastic tensor, and project onto the LDV beam polarisation.  Doing
  so could change the result by an O(1) factor.  The hydrostatic
  approximation used here is justified only as the leading-order estimate.
- **Evanescent pressure-transmission coefficient T_p ∈ [1.00, 1.78].**  The
  upper bound is the 1D traveling-wave longitudinal-incidence formula
  `2 Z_g / (Z_g + Z_w)`, applied here to a *non-traveling* (evanescent)
  field — a useful heuristic but strictly speaking inappropriate.  The
  proper evanescent boundary-condition solution would yield a single
  T_p, not a bracket; computing it requires the elastodynamic boundary
  problem mentioned above.
- **Photoelastic constants for Schott D263T are estimates** (literature
  sparse for thin display glass).  BK7 and fused silica values are
  well-established.  Glass type contributes ~9 % relative spread across
  the three candidates at fixed T_p — a *minor* source of variation
  relative to the T_p bracket itself.
- **Soft-wall correction** is handled self-consistently by using `k_x`
  from the observed `f` (not from the geometric `W`).  No additional
  approximation here.

Net: treat the 10–20 % bracket as a Fermi-estimate envelope, not as a
calibration-grade number.  A rigorous fluid-solid solution is the
natural follow-up if this contribution turns out to be load-bearing for
a quantitative claim.

## Reproduction

```bash
cd <repo root>
.venv/Scripts/python reports/2026-05-21_glass_pressure_self_verification.py
```

## Action items

- Bring this calc to the **2026-05-22 13:00 CEST Thierry follow-up Zoom** as the LDV-side budget reference.
- Audit PTV-side analysis next (= `06_fitting.py` in the PTV codebase):
  - Verify radiation-force formula
  - Confirm wall-correction residue is small after height-distribution filter
  - Confirm amplitude convention matches LDV (peak)

## Addendum — `dn/dp_water` hypothesis ruled out (2026-05-21)

Briefly considered (and floated as a co-equal hypothesis in early
discussion) was the possibility that the canonical `dn/dp_water = 1.4e-10`
in `config.py` was a low-frequency limit and that the *true* value at
MHz acoustic frequencies might be ~1.7× larger, which alone would close
the LDV/PTV gap.

Literature review (GPT-5-Pro, 2026-05-21) rules this out:

| Probe λ | n_water (20 °C) | Adiabatic ∂n/∂p (Pa⁻¹) |
|---|---|---|
| 405 nm | 1.343 | 1.54 × 10⁻¹⁰ |
| 532 nm | 1.335 | 1.50 × 10⁻¹⁰ |
| 589 nm | 1.333 | 1.48 × 10⁻¹⁰ |
| **633 nm** (our laser) | **1.332** | **1.48 × 10⁻¹⁰** |
| 1064 nm | 1.324 | 1.43 × 10⁻¹⁰ |

Sources: IAPWS-95 + Lorentz–Lorenz; ultrasound calibration at
diposit.ub.edu uses C_po = 1.51 × 10⁻¹⁰ /Pa at 20 °C. Water
photoelastic coefficient is essentially **frequency-independent from
kHz to tens of MHz** — any apparent MHz-band variation comes from
optics (path-integral cancellation, attenuation, Bragg conditions),
not from the coefficient itself. The "fast sound" regime at GHz-THz
(inelastic X-ray scattering) is irrelevant at our 1.9 MHz drive.

**Consequence**: the correct value for our 633 nm hardware is
1.48 × 10⁻¹⁰ /Pa, only 5.7 % above the previous `config.py` value of
1.4 × 10⁻¹⁰. `config.py` was updated to 1.48 × 10⁻¹⁰ together with this
report's commit; every LDV-reported pressure drops by ~5.4 %.

This calc was written before the constant change, so the OPL ratios
above still use `DN_DP_WATER = 1.4e-10`. Recomputing with 1.48e-10:

| Glass | T_p = 1.00 | T_p ≈ 1.78 |
|---|---|---|
| Borosilicate (BK7-class) | 10.8 % | 19.2 % |
| Fused silica | 10.7 % | 19.3 % |
| Schott D263T | 9.8 % | 17.7 % |
| **Refined bracket** | **9.8 – 19.3 %** | central 14.5 % |

Qualitative conclusion is unchanged: glass photoelastic combined
*multiplicatively* with the phase-aligned air-null residual gives an
LDV-side inflation of **1.19 – 1.29×** (was 1.19 – 1.30× at the
legacy `DN_DP_water = 1.4e-10`).  The 1.7 – 1.9× LDV/PTV gap still has
32 – 60 % unexplained by LDV-side effects.

Caveat: the multiplicative combination assumes the air-null contribution
is *phase-aligned* with the water acousto-optic signal — see the
"Caveat on air-null sign" box in §LDV-side inflation budget above.  The
companion budget report
(`reports/2026-05-21_ldv_ptv_uncertainty_budget.md` §2.1) moves the
air-null contribution to a symmetric ±8 % uncertainty for the same
reason; both treatments are documented for the reader to choose.

The dn/dp candidate being ruled out **redirects the next investigation
to the PTV side** (Settnes-Bruus radiation-force formula, particle
properties, wall-correction residue) and to **convention checks** (peak
vs RMS vs peak-to-peak between the two codebases).
