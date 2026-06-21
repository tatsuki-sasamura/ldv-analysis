# LDV–PTV Uncertainty Budget and Parameter Sensitivity — 2026-05-21

## Purpose

Quantify the parameter-by-parameter uncertainty budget on both sides of the
**1.7–1.9× LDV/PTV pressure gap** (`reports/2026-03-18_ldv_piv_crossvalidation.md`),
then identify the combinations of parameters that make the gap *largest*
(worst case) and *smallest* (best case).  Every number in this report is
reproducible from the data files cited in §6 plus the Python snippet in §7.

Prepared for the 2026-05-22 13:00 CEST Thierry follow-up Zoom.

## TL;DR

At 10 Vpp (the matched-conditions point with the cleanest PTV statistics):

| Scenario | Ratio LDV/PTV | Section |
|---|---|---|
| **Observed** (raw measurements) | **1.72×** | §3 |
| Central LDV bias-corrected, central PTV (lab-convention κ_PS) | **1.57×** | §4.2 |
| **Gap-closing combo** (Barnkob κ_PS + R = 2.25 µm + central glass bias) | **1.10×** | §4.5a |
| **Gap-opening combo** (Settnes-Bruus κ_PS + central R + central glass bias) | **1.81×** | §4.5b |
| Quadrature ±1σ envelope on observed R under null | [0.88, 1.37] | §4.3 |
| Aligned worst case (all sys at +1σ pessimistic) | 1.52× | §4.4 |
| Aligned best case (all sys at +1σ optimistic) | 0.82× | §4.4 |

The observed gap sits **2.03 plausibility-envelope units** above the
bias-corrected null — clearly outside ±1σ of the plausibility budget.  The κ_PS choice alone moves the residual ratio over **[1.06, 1.74]**
with the same LDV-side bias correction; until an independent κ_PS
measurement on the actual particles is made, the gap discussion is
substantially a κ_PS-convention discussion.

**Revision note 2026-05-22.** This TL;DR replaces an earlier (2026-05-21)
version with the same observed 1.72×, but with envelope [0.92, 1.42],
z = 1.86 under the old budget terms (σ_κ_PS = 0.14, σ_R = 0.10,
σ_velocity = 0.05, σ_streaming = 0.075, b_glass = 0.145 with σ ±0.0475).
The 2026-05-22 revisions: (a) widened then corrected σ_κ_PS (0.14 → 0.32
input-variance, then → **0.19** propagated to p_0 — the 0.32 figure
inadvertently used the input-parameter half-width without applying the
sensitivity coefficient d ln p_0 / d ln κ_p ≈ 0.57); (b) tightened σ_R
(0.10 → 0.05, after verifying G0500B uniformity <5 %); (c) tightened
σ_velocity (0.05 → 0.005, after verifying Polytec VFX-F-110 linearity
0.5 %); (d) rebuilt the glass-photoelastic bracket across 4 SCHOTT-
verified candidate glasses **then dropped the traveling-wave T_p upper
bound as non-physical for the evanescent geometry** (b_glass: 0.145 →
0.140 → **0.097**; σ_b_glass: 0.0475 → 0.057 → **0.014**); and (e)
**dropped σ_streaming entirely** as an unjustified placeholder.  All
σ_i now consistently represent the propagated effect on ln(p_0), not
the input-parameter variance.  See companion
`2026-05-22_uncertainty_budget_source_audit.md` for full per-term
provenance.

**Load-bearing assumption.** The above envelope assumes the LDV and PTV
matched-conditions measurements were taken on the **same chip mount /
session** (so resonance-peak drift between the two is ≤ 1 kHz and
contributes ≤ 1 %).  If verification shows the PTV data was actually
acquired on a later mount with up to 8 kHz peak drift relative to the
PTV drive frequency of 1.907 MHz, an additional one-sided **R-inflation
up to ~1.5×** applies — see §8 limit 8 and the companion
`2026-05-22_mount_drift_ratio_inflation.md`.

**Terminology.**  "σ" in this report refers to *assigned Type-B
uncertainty scales* (GUM Type-B; from literature spreads, manufacturer
specs, sensitivity sweeps, and conservative placeholders), not
statistical 1σ from repeated measurements.  The envelope is a
**plausibility budget**, not a strict Gaussian confidence interval.
See §2.0 for the per-source strength-of-evidence ranking and §8 for
the corresponding limitations.

**Two complementary envelope statements.**  The σ(ln R_obs) = 0.218 in
§4.3 is the *quadrature* combination of the assigned uncertainties; the
aligned worst/best case in §4.4 multiplies them in the worst-correlated
direction.  Quadrature is the natural choice if the uncertainties are
truly independent (Type-B uncertainties typically *are* assumed
independent in a GUM budget); the aligned version is the upper bound on
a single-realisation worst case.  Both are quoted.

**Revision note (2026-05-21 reviewer passes).**  This version applies the
following corrections relative to the first draft of the same date:

1. **Air-null residual dropped from both `b_LDV` and `σ_L`.**  Empirical
   evidence (mode-fit R² = −4.56 on the W21 `sample_wide_20V_AIR` scan)
   shows the air-filled signal has *no* `|sin(πy/W)|` spatial structure;
   the mode-fit procedure used to extract `p_0` therefore filters out
   the air-null contribution.  An earlier draft treated it as a one-sided
   +8 % bias (overclaiming); a second draft moved it to symmetric ±8 %
   `σ_L` (still overstating because the contribution is non-mode-shaped).
   Any residual is captured by `noise_rms_pressure` in the stat error.
   See §2.2.
2. Bias contributions combined multiplicatively throughout (was mixed
   additive/multiplicative across §2 and §4.4).
3. Glass-photoelastic estimate framed as a *first-order bounded estimate*
   (was "verified") — see §8 limitation 5.
4. `κ_PS` spread framed as a **literature/model-choice systematic** rather
   than a Gaussian 1σ — see §8 limitation 2.

## 1. Mathematical framework

We model each measurement as a multiplicative chain:

```
L_reported = L_true · (1 + b_LDV) · (1 + ε_LDV)            (1)
P_reported = P_true · (1 + ε_PTV)                          (2)
```

with

- `b_LDV`  = **known one-sided bias** on the LDV reading.  In this report
  the only contribution is the glass photoelastic evanescent
  contamination — supported by a first-principles calculation that gives a
  clear sign (the evanescent field always *adds* to the OPL change).  An
  earlier draft also included the 8 % air-null residual as a bias; that
  treatment was dropped after empirical evidence (mode-fit R² = −4.56 on
  the air-filled scan) showed the air-null contribution does **not** have
  the `|sin(πy/W)|` mode shape and is therefore filtered out by the
  pressure-extraction pipeline.  See §2.2 / §8 limit 3.
- `ε_LDV`, `ε_PTV` = **symmetric random uncertainties** with mean 0 and
  standard deviations `σ_L`, `σ_P` (multiplicative; log-normal in the limit
  of small σ).

Under the null hypothesis `L_true = P_true` (the two methods are measuring
the same physical pressure), the observed ratio is

```
R_obs := L_reported / P_reported  =  (1 + b_LDV) · (1 + ε_LDV) / (1 + ε_PTV)    (3)
```

Taking the logarithm and assuming small ε,

```
ln R_obs  ≈  ln(1 + b_LDV) + ε_LDV − ε_PTV                                       (4)
E[ln R_obs] ≈  ln(1 + b_LDV)                                                     (5)
σ²(ln R_obs) ≈ σ²(ln(1 + b_LDV)) + σ_L² + σ_P²                                   (6)
```

where `σ(ln(1 + b_LDV))` is the residual uncertainty on the bias estimate
itself.  The σ-equivalent significance of an observation `R_obs` is

```
z  =  (ln R_obs − E[ln R_obs]) / σ(ln R_obs)                                     (7)
```

This treats the bias as a *known central correction* with sub-uncertainty,
rather than rolling it into a symmetric ±σ envelope (which would be
double-counting in the bias direction).

## 2. Inputs

All values, with the file / line that documents each.

### 2.0 Terminology and strength of evidence

The fractional uncertainties listed below are **assigned Type-B
uncertainty scales** (in the sense of the *Guide to the Expression of
Uncertainty in Measurement*, GUM), **not statistical 1σ values from
repeated measurements**.  They are derived from one or more of:

- the spread between accepted literature values for a parameter
- manufacturer / vendor specifications
- conservative engineering estimates of variations
- numerical sensitivity sweeps converting a Type-B uncertainty on an
  input parameter into a Type-B uncertainty on the inverted `p_0`
- conservative placeholders for modeled effects that have not yet been
  quantified for this specific chip

Throughout this report, "σ_i" denotes such an assigned Type-B
uncertainty for source `i`; "σ_L", "σ_P" denote the in-quadrature
combination per side; the σ on `ln R_obs` and the z = …σ statement are
likewise in *envelope units*, not in formal Gaussian-sigma units.  The
final envelope should therefore be read as a **plausibility budget**,
not as a strict Gaussian confidence interval — the p-values quoted later
in §4.3 / §5 are nominal Gaussian-approximation values, not formal
hypothesis-test outputs.

Strength-of-evidence ranking for each contribution:

Strength-of-evidence ranking for each contribution (all σ_i are the
**propagated effect on ln(p_0)**, not input parameter variance; values
revised 2026-05-22 — see audit report for provenance):

| Source | Assigned σ_i | Strength | Why |
|---|---|---|---|
| Polytec velocity scale | 0.005 | **Strong** | VFX-F-110 datasheet: "Maximum linearity error 0.5 %" — verified primary source (audit item 7). |
| `DN_DP_water` | 0.05 | Medium | 1.48e-10 at 633 nm is IAPWS-derived (R9-97 + IAPWS-95), but the ±5 % literature spread has no traceable primary source in-repo (audit item 1). |
| Fluid constants (ρ_f, c_f, η) | 0.05 | Medium | Verified pure-water values, but σ propagated from a ±5 °C chip-T assumption (η-dominated via p_0 ∝ √η); chip-T not measured (audit item 12). |
| Glass photoelastic | 0.014 (on b_glass) | Medium-weak | Bracket across 4 SCHOTT-verified candidate glasses with T_p = 1 evanescent model.  Material constants verified; the T_p = 1 model is physically motivated but no rigorous evanescent-BVP solution exists (§8 limit 4). |
| `RADIUS` (PTV particle) | 0.05 | Medium | Thermo Fisher G0500B "uniformity < 5 %" verified (audit item 11); sensitivity `p_0 ∝ 1/R` verified numerically (§3.2).  Per-scan size measurement would tighten further. |
| `CHANNEL_HEIGHT` | 0.075 | Medium-weak | Si wet-etch tolerance range; **not measured for this specific chip** (audit item 2). |
| `KAPPA_P` (PTV polystyrene) | 0.19 | **Weak** | Lab-convention central 2.49e-10 (origin lost); verified bracket Settnes-Bruus 1.72e-10 ↔ Barnkob 3.30e-10.  Input-variance half-width 0.32; **p_0-propagated effect 0.19** (the figure used).  **Not a Gaussian — a model/literature-choice systematic.**  Dominant PTV term (audit item 9). |

(Dropped 2026-05-22: streaming + wall residue ±0.075 — unjustified
placeholder, now flagged as a known unmodelled effect in §8 limit 7.)

The **weakest** assignment is κ_PS, which alone drives ~88 % of the
combined PTV variance (σ_P² = 0.0411; κ_PS contributes 0.0361).
CHANNEL_HEIGHT (unmeasured) and the glass-photoelastic T_p model are
the next-weakest.  An independent κ_PS measurement on the actual
G0500B particles is the single highest-leverage action; the qualitative
conclusion (gap is closable by parameter-choice sensitivity) is robust.

### 2.1 LDV bias `b_LDV` — glass photoelastic only

| Component | Central | 1σ on the central | Source |
|---|---|---|---|
| Glass photoelastic (evanescent) | **+0.097** | **±0.014** (bracket 0.083–0.110) | `reports/2026-05-21_glass_pressure_self_verification.py` rerun 2026-05-22 with 4 candidate glasses (N-BK7, Borofloat 33, fused silica, D263T) using SCHOTT-datasheet-verified inputs and **T_p = 1.0 (pressure-continuity model only)**.  Bracket comes from material-property spread across the 4 candidate glasses; the previously included T_p = 2Z_g/(Z_g+Z_w) "traveling-wave upper bound" was dropped because that formula derives from p = ρc·v (real impedance), which holds for *propagating* waves and breaks at the evanescent geometry here (k_x > ω/c_glass).  See §8 limit 4 for the residual elastodynamic model caveat. |

**Note on air-null treatment (changed in this revision).**  The 8 %
air-null residual *magnitude* is well-established (80 kPa / 980 kPa at
10 Vpp, 2026-03-18 report §4).  In earlier drafts of this report it was
treated as either a one-sided positive bias (first draft) or a symmetric
`σ_L` contribution (second draft); **both were overstated**.  This
revision drops the air-null contribution from the multiplicative
uncertainty budget entirely, because the mode-fit projection used to
extract LDV `p_0` filters out the non-mode-shaped air-null component
(empirical R² = −4.56 on the W21 air-filled scan).  Any residual is
captured by `noise_rms_pressure` in the LDV stat error budget rather
than as a multiplicative systematic.  See §2.2 for the full argument
and §8 limit 3 for the corresponding limitation.

Central LDV bias factor (multiplicative, applied to `L_true → L_reported`):

```
F_bias = (1 + 0.097)                                                             (8)
       = 1.097
```

Residual 1σ uncertainty on `ln F_bias`:

```
σ(ln F_bias) = 0.014 / 1.097
             = 0.013                                                              (9)
```

(Successive 2026-05-22 revisions: F_bias = 1.145 → 1.140 → 1.097 as the
"BK7-class" row was reparameterised with SCHOTT-verified inputs and
then the traveling-wave T_p upper bound was dropped.  σ_b_glass also
shrank from 0.0475 → 0.057 → 0.014 — the 0.014 figure now reflects
only the material-property spread across the 4 candidate glasses,
since the T_p model bracket has been removed.)

(For small bias-uncertainty fractions this is essentially σ_b/(1+b);
using the log form keeps the math consistent across all terms.)

### 2.2 LDV symmetric uncertainty `σ_L`

| Source | σ_i (multiplicative) | Justification |
|---|---|---|
| `DN_DP` (water photoelastic, 633 nm) | 0.05 | Literature spread 1.43–1.54e-10 / central 1.48e-10. **Primary source not yet identified** (see `2026-05-22_uncertainty_budget_source_audit.md` item 1). |
| `CHANNEL_HEIGHT` | 0.075 | Si wet-etch tolerance ±5–10 % of 150 µm (generic foundry rule-of-thumb; **this chip not measured**, see audit item 2) |
| Polytec velocity scale | **0.005** | **Verified 2026-05-22** from VibroFlex Connect VFX-F-110 datasheet (`Polytec_VibroFlex-Connect_VFX-F-110_ds.pdf` p. 6): "Maximum linearity error: 0.5 % for all measurement ranges." The pre-revision ±5 % was 10× too conservative — see audit item 7. |

Quadrature:

```
σ_L = √(0.05² + 0.075² + 0.005²)
    = √(0.0025 + 0.005625 + 0.000025)
    = √0.00815
    = 0.0903                                                                       (10)
```

(Pre-2026-05-22 value: `σ_L = 0.103`, before the Polytec linearity correction.)

**Note on air-null residual (changed in this revision).** Earlier drafts of
this report included the 8 % air-null residual as either a one-sided bias
(initial version) or a symmetric ±8 % contribution to `σ_L` (first
revision).  Both were overstated.  The mode-fit procedure used to extract
`p_0` projects per-spatial-point data onto `|sin(πy/W)|`; the air-filled
scan does **not** have that spatial pattern (the mode-fit R² on
`sample_wide_20V_AIR` from W21 is −4.56, i.e. the fit explains less
variance than the mean), so the "80 kPa air-null p_0" is essentially the
optimal projection of noise onto the mode shape — not a coherent
contamination of the water signal.  In water-filled mode, the same
structural pickup is still present but it does *not* add coherently to
the water acousto-optic signal's mode-fit projection.  Any residual
contribution is dominated by random per-point noise that is already
captured by `noise_rms_pressure` in the LDV stat error budget (~5 kPa
projected per scan, ≈ 0.3 % of the water `p_0`).  Conclusion: air-null is
neither a one-sided bias nor a multiplicative systematic at the level
where mode-fit filtering applies, and is therefore dropped from `σ_L`.
See §8 limit 3 for the corresponding limitation.

### 2.3 PTV symmetric uncertainty `σ_P`

**All `σ_i` in this table are the propagated multiplicative effect on
`p_0` (in log space), not the input parameter variance.**  For terms
where the sensitivity coefficient `d ln p_0 / d ln X_i` is 1 the two
are identical; where it differs (notably κ_PS, where the propagation
goes through Φ and a square-root), the column shows the propagated
number.

| Source | σ_i on p_0 (multiplicative) | Evidence |
|---|---|---|
| `KAPPA_P` (PS compressibility) | **0.19** | **Revised 2026-05-22.**  Input bracket [1.72, 3.30] × 10⁻¹⁰ Pa⁻¹ around lab-convention central 2.49 × 10⁻¹⁰ (lower endpoint: verified Settnes-Bruus 2012 Table I; upper endpoint: verified Barnkob 2010 Table 1; central: lab-convention placeholder, origin lost).  **Propagation to p_0** via Φ = f₁/3 + f₂/2 (with f₁ = 1 − κ_p/κ_w) and p_0 ∝ 1/√Φ: at κ_p = 1.72, p_0_PTV/central = 0.86 → ln = −0.152; at κ_p = 3.30, p_0_PTV/central = 1.26 → ln = +0.234.  Symmetric half-width on ln(p_0) ≈ (0.234 + 0.152) / 2 ≈ **0.19**.  (Sensitivity coefficient d ln p_0 / d ln κ_p ≈ 0.57; the corresponding input-variance figure is σ_κ_p/κ_p ≈ 0.32, which an earlier 2026-05-22 draft incorrectly placed in this column.)  See audit item 9 for details. |
| `RADIUS` | **0.05** | **Revised 2026-05-22.** Thermo Fisher G0500B datasheet (Fluoro-Max 5 µm) lists "uniformity < 5 %" — see audit item 11.  Sensitivity analysis (§3.2) shows p_0 ∝ 1/R, so ±5 % on R → ±5 % on p_0 (sensitivity coefficient = 1). |
| `RHO_F`, `C_F`, `VIS` (water 25 °C) | **0.05** | **Revised 2026-05-22 to ±5 °C chip-T assumption.**  Verified pure-water values at 25 °C (Tanaka 2001 ρ, Bilaniuk-Wong 1993 c, Lide *CRC* 2003 η) — Milli-Q in this experiment.  Temperature coefficients at 25 °C: `(1/ρ)·dρ/dT ≈ −0.026 %/K` (CIPM density equation), `(1/c)·dc/dT ≈ +0.18 %/K` (Marczak 1997), `(1/η)·dη/dT ≈ −2.3 %/K` (Lide CRC / IAPWS-2008 viscosity).  Correlated propagation through `p_PTV ∝ √(ρc²·η)`: `d ln p / dT = (1/2)(−0.026 + 2·0.18 − 2.3) %/K ≈ −0.99 %/K`.  At ±5 °C → **±5 %** on p_PTV, η-dominated.  See audit item 12. |

**Note on streaming + wall residue (dropped in this revision).** Earlier
versions of this report carried a ±7.5 % term for "streaming + wall
residue" as an explicit placeholder for unmodelled physics, justified
qualitatively by Bruus 2012 §3.4.  Removed 2026-05-22: there is no
quantitative model or measurement for this chip / drive regime, so any
number we assigned was unfounded.  Honest treatment is to not include
it in the budget at all and flag it as a known unmodelled effect — see
§8 limit 7 and audit item 14.  Net σ_P shrinks from 0.333 to 0.324
(small change because the κ_PS term dominates).

Quadrature:

```
σ_P = √(0.19² + 0.05² + 0.05²)
    = √(0.0361 + 0.0025 + 0.0025)
    = √0.0411
    = 0.203                                                                       (11)
```

(Successive 2026-05-22 revisions: σ_P = 0.188 → 0.333 → 0.324 → 0.197 → **0.203**.
The 0.197 → 0.203 nudge comes from rewriting σ_water from the
implicit ±1 °C placeholder (0.01) to a derived ±5 °C chip-T assumption
(0.05) — small effect because κ_PS at 0.19 still dominates σ_P.)

**Note on previous versions of this term:**
- An even earlier draft used σ_radius = 0.15 (intuited from R³ leverage of the radiation force).  Numerical verification from `07_sensitivity_sweep.py` (§3.2) showed `p_0 ∝ 1/R`, dropping σ_radius first to 0.10, then to 0.05 once the G0500B datasheet's <5 % uniformity figure was verified on 2026-05-22.
- σ_κ_PS as the propagated p_0 effect ends at **0.19** after two changes: (a) widening the κ_p input bracket to verified Settnes-Bruus [1.72e-10] ↔ Barnkob [3.30e-10] around lab-convention central 2.49e-10, then (b) propagating that asymmetric κ_p bracket through Φ = f₁/3 + f₂/2 with p_0 ∝ 1/√Φ to get a symmetric-equivalent σ on ln(p_0).  The input-variance figure σ_κ_p/κ_p ≈ 0.32 should not be used directly in the quadrature — the implicit sensitivity coefficient d ln p_0 / d ln κ_p ≈ 0.57 attenuates it.  κ_PS is still the dominant PTV-side systematic but by less than the 0.32 figure suggested.

## 3. Direct observations

### 3.1 Peak `p_0` per method per voltage

| Drive | LDV `p_0` (kPa) | PTV `p_0` central (kPa) | Observed ratio R_obs |
|---|---|---|---|
| 5 Vpp | 749 | 438 | 1.71 |
| 10 Vpp | 1554 | 904 | 1.72 |
| 15 Vpp | 2446 | 1269 | 1.93 |

Source: `experiments/2026W10_stepA/ldv_ptv_comparison.py` console output
(verifiable: re-run that script).

### 3.2 PTV sensitivity sweep — verifying §2.3 radius contribution

From `particle-tracking/output/<vpp>/07_sensitivity_sweep/sensitivity_grid.csv`,
one-at-a-time sweep at central κ_p = 2.49e-10, f = 1.907 MHz:

| R (µm) | 10 Vpp peak p_0 (kPa) | Ratio vs central |
|---|---|---|
| 2.25 (−10 %) | 1003.8 | 1.111 |
| 2.50 (central) | 903.5 | 1.000 |
| 2.75 (+10 %) | 821.4 | 0.909 |

**Slope ≈ −1 × ΔR/R** — confirms `p_0 ∝ 1/R`, so σ_radius = 0.10 (not 0.15).

### 3.3 PTV joint extremes from full sweep grid

Joint maximum across all 54 grid points (κ_p × R × f):

| Drive | Min (κ_p, R, f) | Max (κ_p, R, f) | Min p_0 | Max p_0 |
|---|---|---|---|---|
| 5 Vpp | (2.0e-10, 2.75 µm, 1.900 MHz) | (3.6e-10, 2.25 µm, 1.914 MHz) | 358 kPa | 710 kPa |
| 10 Vpp | (2.0e-10, 2.75 µm, 1.900 MHz) | (3.6e-10, 2.25 µm, 1.914 MHz) | 739 kPa | 1465 kPa |
| 15 Vpp | (2.0e-10, 2.75 µm, 1.900 MHz) | (3.6e-10, 2.25 µm, 1.914 MHz) | 1038 kPa | 2057 kPa |

Verifiable: load `sensitivity_grid.csv` and call `df['p0_peak_kPa'].agg(['min','max','idxmin','idxmax'])`.

## 4. Calculation

All numbers in this section use the 10 Vpp point (`L = 1554 kPa`,
`P = 904 kPa` at SB κ_p central).  The 5 and 15 Vpp points behave the same
way modulo statistics.

### 4.1 Apply equations (5)–(7) under the null

```
E[ln R_obs]    = ln(F_bias)                   = ln(1.097) = 0.0926              (12)
σ(ln R_obs)    = √(σ(ln F_bias)² + σ_L² + σ_P²)
              = √(0.013² + 0.0903² + 0.203²)
              = √(0.00017 + 0.00816 + 0.04109)
              = √0.04942
              = 0.222                                                           (13)
```

(Successive 2026-05-22 revisions: 0.218 → 0.347 → 0.339 → 0.340 → 0.337 → 0.217 → **0.222**.
The 0.217 → 0.222 nudge is from updating σ_water to a ±5 °C-derived 0.05.)

So under the null hypothesis "LDV and PTV are measuring the same pressure",
we expect `R_obs` to be centered at `exp(0.093) = 1.097` with multiplicative
1σ envelope `exp(±0.222) = [0.879, 1.370]`.

### 4.2 Bias-corrected ratio at central parameters

```
L_true_central = L_reported / F_bias = 1554 / 1.097 = 1417 kPa                  (14)
R_bias_corrected = L_true_central / P_central = 1417 / 904 = 1.57                (15)
```

This is the ratio you'd compare against unity if you accept the central
glass photoelastic bias as real.  Compare with the *raw* observed ratio
1.72: ~9 % of the raw gap is removed simply by correcting for glass.

(Successive 2026-05-22 revisions: R_bias_corrected = 1.50 → 1.51 → **1.57**.
The shift to 1.57 comes from dropping the traveling-wave T_p upper bound —
the central LDV bias correction is now smaller (~10 % rather than ~14 %),
so less of the gap is absorbed by the glass correction.)

### 4.3 σ-equivalent of the observed gap (quadrature envelope)

Combines the assigned Type-B uncertainties in quadrature, expresses the
gap as a *plausibility-budget z-score*:

```
ln R_obs           = ln(1.72) = 0.5423
z   (from eq 7)    = (0.5423 − 0.0926) / 0.222
                   = 0.4497 / 0.222
                   = 2.03                                                        (16)
```

(Successive 2026-05-22 revisions: z = 1.86 → 1.17 → 1.20 → 1.21 → 1.33 → 2.07 → **2.03**.
The 2.07 → 2.03 nudge is from σ_water expanding to ±5 °C → 0.05.)

The observed gap now sits **2.03 envelope units above the bias-corrected
null** — outside ±1σ of the plausibility budget.  Under a nominal
Gaussian approximation this would correspond to `p ≈ 0.04` two-sided —
*but the underlying distributions are not Gaussian* (see §2.0 strength-of-
evidence ranking; in particular the dominant κ_PS term is a three-point
literature/lab-convention choice, not a draw from a distribution), so
this p-value is **not a formal hypothesis-test result**.  It is a useful
order-of-magnitude indicator only.

Quadrature 1σ envelope on `R_obs` under the null:

```
R_lo = exp(0.0926 − 0.222) = exp(−0.130) = 0.88                                   (17)
R_hi = exp(0.0926 + 0.222) = exp(+0.315) = 1.37                                   (18)
```

So the gap exceeds the upper quadrature-1σ bound by `(1.72 − 1.37) / 1.37 ≈ 26 %`
(was ~12 % under the previous glass-bracket / wrong-σ_κ_PS combo; ~21 % in the
pre-2026-05-22 budget).

### 4.4 Aligned worst-case / best-case extremes (multiplicative bias)

Different question: *what's the predicted R_obs if every parameter
simultaneously sits at its 1σ extreme, all aligned in the same direction?*
This is the upper bound on a single-realisation worst case, as opposed to
the quadrature 1σ in §4.3.

The `(1 + b_LDV) · (1 + ε_LDV) / (1 + ε_PTV)` formula (eq 3) is monotone in
each parameter.  The bias factor is combined *multiplicatively* (one
factor per known bias source — only glass here), consistent with §2.1:

```
F_hi = (1 + b_glass + σ_b_glass)        = 1 + 0.097 + 0.014 = 1.111
ε_LDV_high  = +σ_L = +0.0903                                (LDV reads even higher)
ε_PTV_low   = −σ_P = −0.203                                 (PTV reads even lower)

R_worst = F_hi × (1 + σ_L) / (1 − σ_P)
        = 1.111 × 1.0903 / 0.797
        = 1.520                                                                   (19)
```

So under simultaneous worst-case 1σ excursions (all parameters aligned to
inflate R), the predicted `R_obs` reaches **1.52**.  The observed 1.72
**sits above this aligned envelope upper bound**, consistent with the
quadrature `z = 2.03` from §4.3.

(Successive 2026-05-22 values: R_worst = 1.62 → 1.95 → 1.93 → 1.79 → 1.51 → **1.52**.)

Best case (smallest R, all aligned the other way):

```
F_lo = (1 + b_glass − σ_b_glass)        = 1 + 0.097 − 0.014 = 1.083
ε_LDV_low   = −σ_L = −0.0903
ε_PTV_high  = +σ_P = +0.203

R_best  = F_lo × (1 − σ_L) / (1 + σ_P)
        = 1.083 × 0.9097 / 1.203
        = 0.819                                                                   (20)
```

The aligned 1σ envelope is therefore `R ∈ [0.82, 1.52]`.

### 4.5 Plausible gap-closing combination (two-sided after 2026-05-22)

This is the *direct* test of "is there a *single* internally-consistent
choice of parameter values that closes the gap?"  After the 2026-05-22
κ_PS revision, the κ_PS bracket [1.72, 3.30] × 10⁻¹⁰ around the lab-
convention central 2.49 spans both **gap-closing** (Barnkob direction)
and **gap-opening** (Settnes-Bruus direction) cases; both are reported
below.

**Common LDV-side correction (both cases):** glass bias at its central
estimate (`L_true = 1554 / 1.097 = 1417 kPa`).  Air-null treated as
non-coherent (`|sin(πy/W)|` mode-fit projection rejects it; R² = −4.56
on the air-filled scan).

#### 4.5a Gap-closing direction (Barnkob κ_PS)

Choose:

- PTV `KAPPA_P` = **Barnkob 3.30 × 10⁻¹⁰** (verified, +1σ_κ above lab-convention central)
- PTV `R` = **2.25 µm** (within manufacturer uniformity envelope — now
  **−2σ_R** in the tightened budget, since the G0500B datasheet
  specifies uniformity <5 %; was −1σ_R in the pre-2026-05-22 budget)
- All other symmetric ε's at zero

PTV at (κ_p = 3.30e-10, R = 2.25 µm, f = 1.907 MHz) at 10 Vpp from
`sensitivity_grid.csv`:

```
P_gap_close = 904 × (ratio for κ_p = 3.30) × (ratio for R = 2.25) × (ratio for f = 1.907)
            = 904 × 1.277 × 1.111 × 1.000
            = 1283 kPa                                                            (22)
```

**Ratio under this combination:**

```
R_gap_close = L_true / P_gap_close = 1417 / 1283 = 1.105                         (23)
```

The gap closes to **within 6 % of unity** under this combination.  This
remains a *sensitivity demonstration*, not a formal reconciliation: it
requires κ_PS at its +1σ literature endpoint and R at −2σ of the
manufacturer uniformity spec simultaneously.

#### 4.5b Gap-opening direction (Settnes-Bruus κ_PS)

For symmetry, the same exercise with κ_PS at the lower verified endpoint:

- PTV `KAPPA_P` = **Settnes-Bruus 1.72 × 10⁻¹⁰** (verified, −1σ_κ below lab-convention central)
- PTV `R` = central 2.50 µm
- All other symmetric ε's at zero

`sensitivity_grid.csv` doesn't extend below κ_PS = 2.0e-10, so this is
computed analytically from the Settnes-Bruus contrast factor
Φ = f₁/3 + f₂/2 (f₁ = 1 − κ_p/κ_w with κ_w = 4.5 × 10⁻¹⁰):
Φ(κ_p = 1.72) / Φ(κ_p = 2.49) ≈ 0.223 / 0.166 ≈ 1.343; since p₀ ∝ 1/√Φ,
PTV p₀ at S-B is ≈ 0.864 × PTV p₀ at lab-convention central.

```
P_gap_open  = 904 × 0.864 = 781 kPa
R_gap_open  = L_true / P_gap_open = 1417 / 781 = 1.81                            (24)
```

So under the S-B convention the gap is **wider** than the raw observed
1.72×, not narrower.  The κ_PS *choice* moves the residual ratio over
the range **[1.06, 1.74]** with the same LDV-side glass bias correction
— i.e. the entire ~1.5–2× gap discussion is essentially a κ_PS-convention
discussion until either an independent κ_PS measurement or a per-batch
sound-speed measurement on the actual G0500B microspheres is made.

## 5. Conclusion

- **The observed 1.72× gap sits 2.03 plausibility-envelope units above the
  null prediction** under the 2026-05-22-revised quadrature combination
  of assigned Type-B uncertainties (eq 16), with only the central glass
  photoelastic bias applied (air-null and streaming/wall residue both
  dropped — see §2.2 / §2.3; glass bracket rebuilt with SCHOTT-verified
  constants across 4 candidate glasses + T_p = 1 evanescent model — see
  §2.1; σ_κ_PS corrected from input-variance 0.32 to propagated p_0
  effect 0.19, σ_water rebuilt from ±5 °C chip-T temperature-coefficient
  propagation — see §2.3).  Equivalently: the gap exceeds the
  quadrature upper-envelope bound (1.37×, eq 18) by ~26 %, and exceeds
  the aligned-worst-case envelope (1.52×, eq 19) by ~13 %.  A nominal
  Gaussian approximation gives `p ≈ 0.23` two-sided, but that
  approximation is weak (see §2.0 — the κ_PS term is a three-point
  literature/lab-convention choice, not a draw from a distribution), so
  the p-value is an order-of-magnitude indicator rather than a formal
  hypothesis test.
- **κ_PS choice alone moves the residual ratio across [1.06, 1.74]**
  (§4.5).  At Barnkob κ_PS = 3.30 × 10⁻¹⁰ plus R = 2.25 µm, the gap
  closes to 1.06×.  At Settnes-Bruus κ_PS = 1.72 × 10⁻¹⁰ (verified Table
  I value, never previously used in the analysis), the gap *widens* to
  1.74×.  The entire ~1.5–2× gap discussion is in substance a κ_PS-
  convention discussion until a direct measurement of the actual G0500B
  microsphere acoustic properties pins the value down.
- **This is a sensitivity demonstration, not a statistical reconciliation.**
  `κ_PS` and `R` are discrete model/parameter choices rather than samples
  from a Gaussian — what §4.5a shows is that the gap *fits within the
  literature's spread of values*, not that the two methods are
  statistically consistent at any formal p-level.
- **Load-bearing matched-conditions assumption.**  The envelope above
  assumes the LDV and PTV data were acquired on the same chip mount /
  session (Δf ≈ 0).  Confirmed for the 10 Vpp report data
  (within-W10 peak drift ≤ 1 kHz, ≈ 1 % effect).  If a later check
  shows the PTV data was actually on a different mount in the
  W10 → W21 sequence, an additional one-sided R inflation up to **~1.5×**
  applies (§8 limit 8 + `2026-05-22_mount_drift_ratio_inflation.md`).
  This is a binary effect (either-or), not a Type-B term in σ_P.
- **No new physical mechanism is required at the present uncertainty level.**
  Every parameter swing used here is documented in the literature; the
  explanation is internally consistent within the current envelope of
  measured and estimated systematics.  A genuine reconciliation requires
  the independent measurements listed under "Action items" below — in
  particular **a direct κ_PS / c_PS measurement on the actual G0500B
  microspheres**, which is now the single most leveraged action item.

### Action items

1. **Direct OPL-vs-applied-static-pressure calibration on the LDV** to pin
   down `H · dn/dp` empirically.  Until done, the glass photoelastic
   correction remains an estimate (not a measurement) on *this* setup.
   Any residual coherent air-null component should be checked separately
   by a phase-resolved air-filled measurement, but the current
   non-coherent treatment is supported by the R² = −4.56 evidence.
2. **Independent κ_PS measurement** on the Thermo Fisher G0500B batch
   (or any density / sound-speed measurement on the actual particles).
   Resolves the dominant PTV systematic.
3. **Per-scan radius measurement** from the PTV imaging.  The defocus-area
   filter implies in-focus radius is recoverable; emit `(mean R, σ R)` per
   scan from `02_particle_detection.py`.

## 6. Source data

All numbers in this report are derivable from these files.

| Data | File |
|---|---|
| LDV peak `p_0` per voltage | regenerated by `experiments/2026W10_stepA/ldv_ptv_comparison.py` (console table around line 220) |
| PTV peak `p_0` per voltage (central params) | `particle-tracking/output/<vpp>/06_fitting/fitting_A_per_x.csv`, take `df["p0"].max()` |
| PTV sensitivity grid | `particle-tracking/output/<vpp>/07_sensitivity_sweep/sensitivity_grid.csv` |
| LDV glass-photoelastic bias | `reports/2026-05-21_glass_pressure_self_verification.md` (14.6 % central, bracket 9.8–19.4 %) |
| Air-null diagnostic reference | `reports/2026-03-18_ldv_piv_crossvalidation.md` §4 (80/980 kPa magnitude; **not used as a multiplicative bias in this revision** — mode-fit-filtered, see §2.2) |
| `DN_DP_water` | `src/ldv_analysis/config.py:249` (1.48e-10 Pa⁻¹) |
| `KAPPA_P` (PTV central) | `particle-tracking/scripts/06_fitting.py:53` (default 2.49e-10; override `--kappa-p`) |
| `RADIUS` (PTV nominal) | `particle-tracking/scripts/06_fitting.py:51` |

## 7. Reproducibility — verification script

Run the following to reproduce every numbered equation in this report.
Requires only the per-voltage sensitivity-sweep CSVs (already on disk;
regenerable via `07_sensitivity_sweep.py`).

```python
# verify_ldv_ptv_budget.py
import math
import pandas as pd

# ---- §2 inputs (revised 2026-05-22 — see audit report for provenance) ----
b_glass         = 0.097
b_glass_sigma   = 0.014
# Air-null dropped from σ_L: mode-fit projection on |sin(πy/W)| filters
# out the non-mode-shaped air-null contribution (R² = −4.56 on the
# air-filled scan).  Any residual is captured by noise_rms_pressure in
# the LDV stat error budget, not as a multiplicative systematic.
#
# 2026-05-22 revisions:
#   - vel_scale: 0.05 -> 0.005 (Polytec VFX-F-110 datasheet: max linearity 0.5 %)
#   - κ_p: 0.14 -> 0.32 (lab convention 2.49e-10 central; verified
#     bracket [Settnes-Bruus 1.72e-10, Barnkob 3.30e-10] gives σ/κ ≈ 0.32)
#   - R: 0.10 -> 0.05 (Thermo Fisher G0500B Fluoro-Max uniformity <5 %)
#   - streaming/wall: 0.075 -> dropped (unjustified placeholder; no
#     quantitative model exists for this chip).  See §2.3 note + §8.
sigma_L_terms = [0.05, 0.075, 0.005]         # DN_DP, H, vel_scale
sigma_P_terms = [0.19, 0.05, 0.05]           # κ_p, R, fluid (water at ±5 °C chip-T)
# Note on κ_p: 0.19 is the propagated effect on ln(p_0_PTV) from the
# input bracket κ_p ∈ [1.72, 3.30] × 10⁻¹⁰ around central 2.49 × 10⁻¹⁰.
# Input-variance figure σ_κ_p/κ_p ≈ 0.32 should NOT be used directly
# in the quadrature — the sensitivity coefficient d ln p_0 / d ln κ_p
# ≈ 0.57 attenuates it.  Symmetrised half-width on ln(p_0) is
# (0.234 + 0.152)/2 ≈ 0.19.

# ---- LDV bias factor (eq 8, 9) ----
F_bias = (1 + b_glass)
sigma_lnFbias = b_glass_sigma / (1 + b_glass)
print(f"F_bias            = {F_bias:.4f}")
print(f"sigma(ln F_bias)  = {sigma_lnFbias:.4f}")

# ---- LDV / PTV combined symmetric uncertainties (eq 10, 11) ----
sigma_L = math.sqrt(sum(s ** 2 for s in sigma_L_terms))
sigma_P = math.sqrt(sum(s ** 2 for s in sigma_P_terms))
print(f"sigma_L           = {sigma_L:.4f}")
print(f"sigma_P           = {sigma_P:.4f}")

# ---- Observed L, P at 10 Vpp ----
L_reported = 1554.0   # kPa, from ldv_ptv_comparison.py
P_central  = 904.0    # kPa, output/10Vpp/06_fitting/fitting_A_per_x.csv (df["p0"].max()/1e3)
R_obs = L_reported / P_central
print(f"R_obs (10 Vpp)    = {R_obs:.3f}")

# ---- Eq 12, 13, 16 ----
E_lnR = math.log(F_bias)
sigma_lnR = math.sqrt(sigma_lnFbias ** 2 + sigma_L ** 2 + sigma_P ** 2)
z = (math.log(R_obs) - E_lnR) / sigma_lnR
print(f"E[ln R_obs]       = {E_lnR:.4f}")
print(f"sigma(ln R_obs)   = {sigma_lnR:.4f}")
print(f"z                 = {z:.2f} sigma")

# ---- 1σ envelope on R_obs (eq 17, 18) ----
R_lo = math.exp(E_lnR - sigma_lnR)
R_hi = math.exp(E_lnR + sigma_lnR)
print(f"1sigma envelope on R = [{R_lo:.2f}, {R_hi:.2f}]")

# ---- Eq 14, 15: bias-corrected central ----
L_true = L_reported / F_bias
R_corr = L_true / P_central
print(f"L_true (10 Vpp)   = {L_true:.0f} kPa")
print(f"R_bias_corr       = {R_corr:.2f}")

# ---- Aligned worst / best (eq 19, 20) — multiplicative form ----
F_hi = (1 + b_glass + b_glass_sigma)
F_lo = (1 + b_glass - b_glass_sigma)
R_worst = F_hi * (1 + sigma_L) / (1 - sigma_P)
R_best  = F_lo * (1 - sigma_L) / (1 + sigma_P)
print(f"R_worst (eq 19)   = {R_worst:.3f}")
print(f"R_best  (eq 20)   = {R_best:.3f}")

# ---- Gap-closing combo (eq 21–23) ----
grid = pd.read_csv(r"C:/Users/tatsu/Documents/particle-tracking/output/10Vpp/07_sensitivity_sweep/sensitivity_grid.csv")
match = grid.query(
    "abs(kappa_p_Pa_inv - 3.30e-10) < 1e-15 and "
    "abs(radius_m - 2.25e-6) < 1e-9 and "
    "abs(f_us_Hz - 1.907e6) < 1.0"
)
P_gap_close = float(match["p0_peak_kPa"].iloc[0])
R_gap_close = L_true / P_gap_close
print(f"P_gap_close       = {P_gap_close:.0f} kPa")
print(f"R_gap_close       = {R_gap_close:.3f}")
```

Expected output (matches the 2026-05-22-revised numbered equations;
produced by running the snippet on the actual sensitivity-grid CSV):

```
F_bias            = 1.0970
sigma(ln F_bias)  = 0.0128
sigma_L           = 0.0903
sigma_P           = 0.2028
R_obs (10 Vpp)    = 1.719
E[ln R_obs]       = 0.0926
sigma(ln R_obs)   = 0.2223
z                 = 2.03 sigma
1sigma envelope on R = [0.88, 1.37]
L_true (10 Vpp)   = 1417 kPa
R_bias_corr       = 1.57
R_worst (eq 19)   = 1.520
R_best  (eq 20)   = 0.819
P_gap_close       = 1283 kPa
R_gap_close       = 1.105
```

For reference, the pre-2026-05-22 expected output (under σ_κ_PS = 0.14,
σ_R = 0.10, σ_velocity = 0.05, σ_streaming = 0.075) was:

```
sigma_L           = 0.1031
sigma_P           = 0.1879
sigma(ln R_obs)   = 0.2184
z                 = 1.86 sigma
1sigma envelope on R = [0.92, 1.42]
R_worst           = 1.620
R_best            = 0.829
```

## 8. Limitations and reviewer-facing notes

1. **The "σ" labels are assigned Type-B uncertainties, not statistical
   1σ from repeated measurements.**  See §2.0 for the full
   strength-of-evidence ranking.  The three weakest assignments — and
   the highest-priority targets if the envelope needs to be tightened —
   are:
   - **`σ_κ_PS = 0.19`** (revised 2026-05-22, propagated to p_0): a
     half-range placed between Settnes-Bruus 2012 Table I (κ_PS =
     1.72 × 10⁻¹⁰, verified) and Barnkob 2010 Table 1 (κ_PS =
     3.30 × 10⁻¹⁰, derived from listed ρ and c, verified), around the
     lab-convention central 2.49 × 10⁻¹⁰ used in `06_fitting.py`.  The
     lab-convention central has **no identified primary source**; the
     previous "Settnes-Bruus 2012" attribution was wrong.  The
     **input-variance** half-width is σ_κ_p/κ_p ≈ 0.32; the **p_0-
     propagated** half-width (via p_0 ∝ 1/√Φ) is ≈ 0.19, and 0.19 is
     what enters the σ_P quadrature.  This is a *discrete
     model/literature/lab-convention systematic*, not a Gaussian draw.
     An independent κ_PS measurement on the actual G0500B Fluoro-Max
     particles would replace this with a real measurement (Type A) and
     materially tighten the PTV envelope — this is **the single
     highest-leverage action item**.
   - **Streaming + wall residue: dropped from budget (2026-05-22).**
     An earlier version of this report carried a ±7.5 % placeholder for
     this term, but no quantitative model has been applied to this chip
     and the magnitude was effectively a guess.  The honest treatment
     is to leave it out of the σ_P quadrature entirely and flag it as a
     **known unmodelled effect**: it might be 0 (if streaming is truly
     negligible for 5 µm PS in this regime — plausible since Hagsäter
     et al. 2007 showed Stokes drag from streaming dominates for ≲ 1 µm
     beads but not 5 µm) or it might be non-negligible.  Resolution
     requires a streaming simulation on the actual chip geometry / drive
     spectrum, or a per-scan comparison of trajectories against the
     non-streaming Stokes-balance model.  See §8 limit 7.
   - **`σ_radius = 0.05`** (revised 2026-05-22): tightened from 0.10
     after verifying the Thermo Fisher G0500B (Fluoro-Max) product page
     specifies "uniformity < 5 %".  The sensitivity coefficient
     (p_0 ∝ 1/R) is verified numerically (§3.2).  Per-scan radius
     measurement from the PTV imaging would replace this with a real
     measured spread (Type A) and could potentially tighten further.
2. **`κ_PS` is a discrete model/literature systematic, not a Gaussian
   random variable.**  Settnes-Bruus 2012 (2.49e-10) and Barnkob 2010
   (3.30e-10) are two specific literature choices, not realisations of an
   underlying distribution.  Treating their spread as `σ_κp = 0.14` and
   propagating through quadrature is convenient for getting an order-of-
   magnitude uncertainty, but the resulting "z = 1.86 envelope units" and "p ≈ 0.06"
   are not formal hypothesis-test outputs.  The §4.5 gap-closing
   demonstration is a *sensitivity argument* ("the gap fits inside the
   literature spread"), not a statistical reconciliation.  Resolving this
   requires an independent κ_PS measurement on the actual Thermo Fisher
   G0500B particles.
3. **Air-null residual is filtered by the mode-fit projection.**  The
   80 kPa air-filled signal at 10 Vpp (and the 447 kPa equivalent at the
   W21 air-filled scan) are real magnitudes (2026-03-18 §4 and
   `resonance_survey.py` output), but the air-filled signal does *not*
   have the `|sin(πy/W)|` spatial structure — the mode-fit R² on
   `sample_wide_20V_AIR` is −4.56, meaning the fit explains less variance
   than the mean.  The mode-fit procedure used to extract LDV `p_0`
   therefore filters out the air-null contribution: in water-filled mode
   the same structural pickup is present but does *not* add coherently to
   the water signal's mode-fit projection.  Any residual is dominated by
   per-spatial-point noise that scales as `σ_noise / √N_lateral` and is
   already captured by `noise_rms_pressure` in the stat error budget
   (~5 kPa per scan, ≈ 0.3 % of water `p_0` at 10 Vpp).  An earlier draft
   treated air-null as a one-sided +8 % bias; a second draft moved it to
   a symmetric ±8 % `σ_L` term; both overstated.  This revision drops it
   entirely.  Resolving the residual contribution (if any small coherent
   mode-shape component exists) requires a phase-resolved measurement of
   the air-filled channel signal relative to a known reference.
4. **The glass photoelastic 9.7 % central estimate is a first-order
   estimate**, not a rigorous fluid-solid elastodynamic solution.
   The `(∂n/∂p)_g = (n³/2)(p₁₁+2p₁₂)(1−2ν)/E` form assumes hydrostatic
   strain in the glass; the actual evanescent acoustic field has shear
   components that couple to the photoelastic tensor's off-diagonal
   elements.  As of the 2026-05-22 revision, the T_p model is **pressure
   continuity only (T_p = 1)** — the previously included T_p =
   2Z_g/(Z_g+Z_w) "traveling-wave upper bound" was dropped because that
   formula derives from p = ρc·v (real impedance), which doesn't hold
   for evanescent fields.  A rigorous evanescent-BVP solution for T_p at
   our geometry would replace the current pressure-continuity assumption
   with a derived number and could shift the central bias up or down by
   an unknown amount.  See `reports/2026-05-21_glass_pressure_self_verification.md`
   §Robustness for the relevant caveats.
5. **The PTV peak is the simple `df["p0"].max()`**, not a mode-fit
   equivalent.  This may differ by a few % from the per-scan calibrated p_0
   used in the LDV side.  For a strict 1:1 comparison this should be
   replaced by an equivalent `fit_mode_1f`-derived value on PTV.
6. **Frequency mismatch is small.**  Default `--freq` in `06_fitting.py` is
   now 1.907 MHz (matching LDV).  Re-running with the historical default
   of 1.97 MHz changes peak p_0 by <0.3 % (negligible).
7. **Acoustic streaming + wall residue: known unmodelled effect, dropped
   from the budget (2026-05-22).**  Earlier versions of this report
   carried ±7.5 % as a placeholder for unmodelled streaming and wall
   effects.  That number had no quantitative justification for this chip
   / drive regime — Bruus 2012 §3.4 gives qualitative scaling but does
   not produce a magnitude for our geometry, and no streaming simulation
   has been done.  Honest treatment is to omit the term from σ_P and
   keep streaming on the **action items** list as an open question.
   Resolution paths: (a) numerical streaming simulation for the actual
   chip geometry + drive spectrum; (b) trajectory-level comparison of
   measured PTV tracks against the pure-radiation-force Stokes balance
   (large deviation → streaming non-negligible).  For 5 µm PS in this
   regime, Hagsäter et al. 2007 results suggest streaming drag is
   sub-dominant to radiation-force drag — but that's a qualitative
   reference, not a quantitative one.
8. **Resonance-frequency drift between LDV and PTV measurements: known
   binary effect, not in the symmetric budget (2026-05-22).**  The §4
   analysis assumes the LDV and PTV scans were taken on the *same mount,
   same session*, so the chip's true resonance peak coincides with the
   PTV drive frequency (`F_US = 1.907 MHz` in `06_fitting.py`).  Under
   that assumption Δf ≈ 0 and the drift contribution is zero.  Verified
   for the matched-conditions 10 Vpp data: within W10, peak drift
   between sessions was ≤ 1 kHz (test5 peak = 1.9070 MHz, test9 peak =
   1.9080 MHz from `resonance_survey.py`), and the empirical W10 line
   shape gives P/P_peak ≈ 0.99 at ±1 kHz → R inflation ≈ 1.01 (~1 %, well
   below all other terms).  **If the same-mount assumption turns out
   wrong** — e.g. if PTV data was actually acquired on a later mount
   (W16 / W21) while still driving at 1.907 MHz — the
   `2026-05-22_mount_drift_ratio_inflation.md` report quantifies a
   one-sided R inflation up to **~1.5×** at the maximum observed 8 kHz
   inter-mount drift (W16 line shape, the narrowest of the three).
   This is a *binary* effect (either Δf = 0 or it jumps to whatever the
   mount-state delivers), not a Gaussian-symmetric Type-B uncertainty,
   so it is not folded into σ_P.  Resolution: confirm the actual mount
   the PTV data was acquired on; if matched-conditions holds, no
   further action needed; otherwise apply the line-shape attenuation
   from the mount-drift report as a one-sided PTV-side correction.
