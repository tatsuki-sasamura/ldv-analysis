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
| Central LDV bias-corrected, central PTV | **1.50×** | §4.2 |
| **Plausible gap-closing combo** (Barnkob κ_PS + R = 2.25 µm + central glass bias removed) | **1.06×** | §4.5 |
| Quadrature ±1σ envelope on observed R under null | [0.92, 1.42] | §4.3 |
| Aligned worst case (all sys at +1σ pessimistic) | 1.62× | §4.4 |
| Aligned best case (all sys at +1σ optimistic) | 0.83× | §4.4 |

The observed gap is **1.86 σ-equivalent** above the bias-corrected null —
significant but not overwhelming.  A *single* combination of parameter
choices (Barnkob κ_PS + 10 % smaller mean particle radius) on top of the
central glass-photoelastic bias correction closes it to within 6 % of
unity, **demonstrating that the gap is closable by parameter-choice
sensitivity** without invoking new physics.

**Two complementary 1σ statements.**  The σ(ln R_obs) = 0.218 in §4.3 is
the *quadrature* combination of independent uncertainties; the aligned
worst/best case in §4.4 multiplies them in the worst-correlated direction.
Quadrature is the right answer if the errors are truly independent and
Gaussian; the aligned version is the upper bound on a single-realisation
worst case.  Both are quoted.

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

### 2.1 LDV bias `b_LDV` — glass photoelastic only

| Component | Central | 1σ on the central | Source |
|---|---|---|---|
| Glass photoelastic (evanescent) | +0.145 | ±0.0475 (bracket 0.098–0.193) | `reports/2026-05-21_glass_pressure_self_verification.md`, table "Final bracket" |

**Note on air-null treatment (changed in this revision).**  The 8 %
air-null residual *magnitude* is well-established (80 kPa / 980 kPa at
10 Vpp, 2026-03-18 report §4).  But whether that residual *adds to* or
*subtracts from* the water-filled signal depends on a phase relationship
between the structural / glass pickup and the water acousto-optic signal
— a phase that has not been independently measured.  Treating it as a
strictly positive bias was overclaiming.  In this revision, the air-null
contribution is moved to the symmetric uncertainty `σ_L` in §2.2.

Central LDV bias factor (multiplicative, applied to `L_true → L_reported`):

```
F_bias = (1 + 0.145)                                                             (8)
       = 1.145
```

Residual 1σ uncertainty on `ln F_bias`:

```
σ(ln F_bias) = 0.0475 / 1.145
             = 0.0415                                                             (9)
```

(For small bias-uncertainty fractions this is essentially σ_b/(1+b);
using the log form keeps the math consistent across all terms.)

### 2.2 LDV symmetric uncertainty `σ_L`

| Source | σ_i (multiplicative) | Justification |
|---|---|---|
| `DN_DP` (water photoelastic, 633 nm) | 0.05 | Literature spread 1.43–1.54e-10 / central 1.48e-10. See `src/ldv_analysis/config.py` + `2026-05-21_glass_pressure_self_verification.md` addendum. |
| `CHANNEL_HEIGHT` | 0.075 | Si wet-etch tolerance ±5–10 % of 150 µm (manufacturer spec) |
| Polytec velocity scale | 0.05 | Vendor spec (Polytec decoder ±5 %) |

Quadrature:

```
σ_L = √(0.05² + 0.075² + 0.05²)
    = √(0.0025 + 0.005625 + 0.0025)
    = √0.010625
    = 0.103                                                                       (10)
```

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

| Source | σ_i (multiplicative) | Evidence |
|---|---|---|
| `KAPPA_P` (PS compressibility) | 0.14 | Settnes-Bruus 2.49e-10 ↔ Barnkob 3.30e-10; ratio 1.33, central placed at SB, 1σ ≈ half-range = ±0.14 on p_0 |
| `RADIUS` | 0.10 | ±10 % nominal R → ±10 % p_0 (verified numerically: §3.2) |
| Streaming + wall residue (unmodelled) | 0.075 | Conservative ±7.5 % guess for unmodelled physics (Bruus 2012 §3.4) |
| `RHO_F`, `C_F`, `VIS` (water 25 °C) | 0.01 | NIST tables, all <0.5 % combined |

Quadrature:

```
σ_P = √(0.14² + 0.10² + 0.075² + 0.01²)
    = √(0.0196 + 0.0100 + 0.005625 + 0.0001)
    = √0.035325
    = 0.188                                                                       (11)
```

**Note on previous version:** an earlier draft of this report used σ_radius
= 0.15 (intuited from the R³ leverage of the radiation force).  Numerical
verification from `07_sensitivity_sweep.py` (§3.2) shows the inverted p_0
scales as **1/R, not R³**, because the R² in `E_ac = 3η|A|/(2(π/W)R²Φ)`
cancels most of the R³ in the force formula.  Updated.

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
E[ln R_obs]    = ln(F_bias)                   = ln(1.145) = 0.1354              (12)
σ(ln R_obs)    = √(σ(ln F_bias)² + σ_L² + σ_P²)
              = √(0.0415² + 0.103² + 0.188²)
              = √(0.00172 + 0.01063 + 0.03534)
              = √0.04769
              = 0.2184                                                          (13)
```

So under the null hypothesis "LDV and PTV are measuring the same pressure",
we expect `R_obs` to be centered at `exp(0.135) = 1.145` with multiplicative
1σ envelope `exp(±0.218) = [0.920, 1.424]`.

### 4.2 Bias-corrected ratio at central parameters

```
L_true_central = L_reported / F_bias = 1554 / 1.145 = 1357 kPa                  (14)
R_bias_corrected = L_true_central / P_central = 1357 / 904 = 1.50                (15)
```

This is the ratio you'd compare against unity if you accept the central
glass photoelastic bias as real.  Compare with the *raw* observed ratio
1.72: ~13 % of the raw gap is removed simply by correcting for glass.

### 4.3 σ-equivalent of the observed gap (quadrature 1σ)

Treats the symmetric uncertainties as independent Gaussians, combines in
quadrature, expresses the gap as a z-score:

```
ln R_obs           = ln(1.72) = 0.5423
z   (from eq 7)    = (0.5423 − 0.1354) / 0.2184
                   = 0.4069 / 0.2184
                   = 1.86                                                        (16)
```

The observed gap is **1.86 σ above the null expectation**.  Significant
(p ≈ 0.06 two-sided under a Gaussian assumption) but not at the level
where we should reject the hypothesis that the two methods agree once
the glass bias and symmetric uncertainties are accounted for.

Quadrature 1σ envelope on `R_obs` under the null:

```
R_lo = exp(0.1354 − 0.2184) = exp(−0.083) = 0.92                                  (17)
R_hi = exp(0.1354 + 0.2184) = exp(+0.354) = 1.42                                  (18)
```

So the gap exceeds the upper quadrature-1σ bound by `(1.72 − 1.42) / 1.42 ≈ 21 %`.

### 4.4 Aligned worst-case / best-case extremes (multiplicative bias)

Different question: *what's the predicted R_obs if every parameter
simultaneously sits at its 1σ extreme, all aligned in the same direction?*
This is the upper bound on a single-realisation worst case, as opposed to
the quadrature 1σ in §4.3.

The `(1 + b_LDV) · (1 + ε_LDV) / (1 + ε_PTV)` formula (eq 3) is monotone in
each parameter.  The bias factor is combined *multiplicatively* (one
factor per known bias source — only glass here), consistent with §2.1:

```
F_hi = (1 + b_glass + σ_b_glass)        = 1 + 0.145 + 0.0475 = 1.1925
ε_LDV_high  = +σ_L = +0.103                                 (LDV reads even higher)
ε_PTV_low   = −σ_P = −0.188                                 (PTV reads even lower)

R_worst = F_hi × (1 + σ_L) / (1 − σ_P)
        = 1.1925 × 1.1031 / 0.8121
        = 1.620                                                                   (19)
```

So under simultaneous worst-case 1σ excursions (all parameters aligned to
inflate R), the predicted `R_obs` reaches **1.62**.  The observed 1.72
lies *above* this aligned envelope upper bound — consistent with the
quadrature `z = 1.86` from §4.3.

Best case (smallest R, all aligned the other way):

```
F_lo = (1 + b_glass − σ_b_glass)        = 1 + 0.145 − 0.0475 = 1.0975
ε_LDV_low   = −σ_L = −0.103
ε_PTV_high  = +σ_P = +0.188

R_best  = F_lo × (1 − σ_L) / (1 + σ_P)
        = 1.0975 × 0.8969 / 1.1879
        = 0.829                                                                   (20)
```

The aligned 1σ envelope is therefore `R ∈ [0.83, 1.62]`.

### 4.5 Plausible gap-closing combination

This is the *direct* test of "is there a *single* internally-consistent
choice of parameter values that closes the gap?"

Choose:

- LDV glass bias at its **central** estimate (no rolling of the dice on
  the glass photoelastic correction; just trust the central evanescent
  calculation)
- LDV air-null contribution treated as **non-coherent** with the water
  signal — i.e. removed by mode-fit projection onto `|sin(πy/W)|`, as
  evidenced by R² = −4.56 on the air-filled scan (see §2.2 note).  No
  additional correction or uncertainty applied beyond `noise_rms_pressure`
  in the stat error
- PTV `KAPPA_P` = **Barnkob 3.30e-10** (a documented alternative literature
  value, not a Gaussian draw)
- PTV `R` = **2.25 µm** (consistent with a 10 % under-estimate of the actual
  mean radius)
- All other symmetric ε's at zero

LDV corrected for the glass bias only:

```
L_true = L_reported / F_bias = 1554 / 1.145 = 1357 kPa                           (21)
```

PTV at (κ_p = 3.30e-10, R = 2.25 µm, f = 1.907 MHz) at 10 Vpp from
`sensitivity_grid.csv`:

```
P_gap_close = 904 × (ratio for κ_p = 3.30) × (ratio for R = 2.25) × (ratio for f = 1.907)
            = 904 × 1.277 × 1.111 × 1.000
            = 1283 kPa                                                            (22)
```

(Cross-check: directly query `sensitivity_grid.csv` for that row.  Both
methods agree to within ~0.1 % rounding.)

**Ratio under this combination:**

```
R_gap_close = L_true / P_gap_close = 1357 / 1283 = 1.058                         (23)
```

The gap closes to **within 6 % of unity** under literature-defensible
parameter choices.  Equivalently: the residual ratio 1.06 sits comfortably
inside the quadrature 1σ envelope [0.91, 1.44].  No new physics required —
but this is best understood as a *sensitivity demonstration*, not a
statistical test: `κ_PS` is a discrete literature/model-choice systematic,
not a Gaussian random variable, and the radius shift is at the magnitude
of the manufacturer tolerance.

## 5. Conclusion

- **The observed 1.72× gap is 1.86 σ above the null prediction** under
  the quadrature uncertainty propagation (eq 16) with only the central
  glass photoelastic bias applied (air-null dropped — its non-mode-shaped
  contribution is filtered by the `|sin(πy/W)|` projection; see §2.2).
  Equivalently: the gap exceeds the quadrature 1σ upper bound (1.42×, eq
  18) by ~21 %, and exceeds the aligned-worst-case envelope (1.62×, eq
  19) by ~6 %.  Either way, the gap is **significant but not damning**
  (`p ≈ 0.06` two-sided under Gaussian assumption — and that assumption
  itself is weak for the dominant `κ_PS` term).
- **PTV-side parameter choice alone can plausibly close the gap to 1.06×**
  (eq 23): pick PTV `κ_PS` = Barnkob 3.30e-10 (a documented literature
  alternative to Settnes-Bruus), `R` = 2.25 µm (within manufacturer
  tolerance), and apply only the central glass photoelastic correction on
  the LDV side.
- **This is a sensitivity demonstration, not a statistical reconciliation.**
  `κ_PS` and `R` are discrete model/parameter choices rather than samples
  from a Gaussian — what equation 23 shows is that the gap *fits within
  the literature's spread of values*, not that the two methods are
  statistically consistent at any formal p-level.
- **No new physical mechanism is required at the present uncertainty level.**
  Every parameter swing used here is documented in the literature; the
  explanation is internally consistent within the current envelope of
  measured and estimated systematics.  A genuine reconciliation requires
  the independent measurements listed under "Action items" below.

### Action items

1. **Direct OPL-vs-applied-static-pressure calibration on the LDV** to pin
   down `H · dn/dp` empirically.  Until done, the glass and air-null biases
   remain estimates, not measurements on *this* setup.
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
| LDV bias contributions | `reports/2026-05-21_glass_pressure_self_verification.md` (glass: 14.5 % central, bracket 9.8–19.3 %); `reports/2026-03-18_ldv_piv_crossvalidation.md` §4 (air-null: 8 %) |
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

# ---- §2 inputs ----
b_glass         = 0.145
b_glass_sigma   = 0.0475
# Air-null dropped from σ_L: mode-fit projection on |sin(πy/W)| filters
# out the non-mode-shaped air-null contribution (R² = −4.56 on the
# air-filled scan).  Any residual is captured by noise_rms_pressure in
# the LDV stat error budget, not as a multiplicative systematic.
sigma_L_terms = [0.05, 0.075, 0.05]          # DN_DP, H, vel_scale
sigma_P_terms = [0.14, 0.10, 0.075, 0.01]    # κ_p, R, streaming, fluid

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

Expected output (matches the report's numbered equations; produced by
running the snippet on the actual sensitivity-grid CSV):

```
F_bias            = 1.1450
sigma(ln F_bias)  = 0.0415
sigma_L           = 0.1031
sigma_P           = 0.1879
R_obs (10 Vpp)    = 1.719
E[ln R_obs]       = 0.1354
sigma(ln R_obs)   = 0.2184
z                 = 1.86 sigma
1sigma envelope on R = [0.92, 1.42]
L_true (10 Vpp)   = 1357 kPa
R_bias_corr       = 1.50
R_worst (eq 19)   = 1.620
R_best  (eq 20)   = 0.829
P_gap_close       = 1282 kPa
R_gap_close       = 1.058
```

## 8. Limitations and reviewer-facing notes

1. **The "1σ" labels are reasonable estimates, not statistical 1σ from
   measured ensembles.**  In particular:
   - `σ_radius = 0.10` is the per-scan inversion sensitivity verified
     numerically (§3.2), *not* the spread of measured particle sizes within
     a single PTV scan.  Replace with measured σ once it's available.
   - `σ(streaming + wall residue) = 0.075` is a conservative placeholder.
     Bruus 2012 §3.4 is the qualitative justification, but no quantitative
     model has been applied to this chip.
2. **`κ_PS` is a discrete model/literature systematic, not a Gaussian
   random variable.**  Settnes-Bruus 2012 (2.49e-10) and Barnkob 2010
   (3.30e-10) are two specific literature choices, not realisations of an
   underlying distribution.  Treating their spread as `σ_κp = 0.14` and
   propagating through quadrature is convenient for getting an order-of-
   magnitude uncertainty, but the resulting "z = 1.75σ" and "p ≈ 0.08"
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
4. **The glass photoelastic 14.5 % central estimate is a first-order
   bounded estimate**, not a rigorous fluid-solid elastodynamic solution.
   The `(∂n/∂p)_g = (n³/2)(p11+2p12)(1−2ν)/E` form assumes hydrostatic
   strain in the glass; the actual evanescent acoustic field has shear
   components that couple to the photoelastic tensor's off-diagonal
   elements.  The `T_p ∈ [1.00, 1.78]` bracket applies a 1D traveling-wave
   pressure-transmission formula to an evanescent field.  See
   `reports/2026-05-21_glass_pressure_self_verification.md` §Robustness
   for the relevant caveats.
5. **The PTV peak is the simple `df["p0"].max()`**, not a mode-fit
   equivalent.  This may differ by a few % from the per-scan calibrated p_0
   used in the LDV side.  For a strict 1:1 comparison this should be
   replaced by an equivalent `fit_mode_1f`-derived value on PTV.
6. **Frequency mismatch is small.**  Default `--freq` in `06_fitting.py` is
   now 1.907 MHz (matching LDV).  Re-running with the historical default
   of 1.97 MHz changes peak p_0 by <0.3 % (negligible).
