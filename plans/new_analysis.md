# Plan: New analysis scripts

## 1. Electro-acoustic energy plot

**Script:** `experiments/2026W10_stepA/electroacoustic.py`

**Input data:** Fine frequency sweep (test5) — has V, I, phase, p₀ at 56
frequencies.

**Compute:**
- P_elec = ½ V I cos(φ) per frequency (electrical input power, W)
- E_ac = p₀² / (2 ρ c²) per frequency (peak acoustic energy density, J/m³)

**Output:** 2-panel or dual-axis plot of P_elec(f) and E_ac(f). Optionally
a scatter of E_ac vs P_elec to show the relationship directly.

**Constants:** ρ = 1004 kg/m³, c = 1508 m/s (already in codebase).

**Note:** E_ac here is the peak potential energy density at the best axial
position, not total stored energy (would need volume integration over the
full 2D map, which mixes axial mode structure). Keep it simple — plot the
directly measured quantities.

---

## 2. BVD equivalent circuit extraction

**Script:** `experiments/2026W10_stepA/bvd_fit.py`

**Input data:** Fine frequency sweep (test5) — V, I, phase at 56
frequencies. Optionally also test3 (coarse, 11 frequencies) and test9
(25 Vpp, 13 frequencies) for cross-validation.

**Compute:**
- Z(f) = V / I, θ(f) = phase_vi → complex impedance Z = |Z| e^(jθ)
- Fit multi-branch BVD model:
  ```
  Z(f) = 1/(jωC₀) + Σₙ [ Rₙ + jωLₙ + 1/(jωCₙ) ]⁻¹  (parallel sum)
  ```
  Actually: the standard BVD has C₀ in parallel with series RLC branches:
  ```
  Y(f) = jωC₀ + Σₙ 1/[ Rₙ + j(ωLₙ - 1/(ωCₙ)) ]
  Z(f) = 1 / Y(f)
  ```
- Start with 2 motional branches (the two visible resonance peaks at
  ~1.910 and ~1.970 MHz) + C₀.
- Fit parameters: C₀, R₁, L₁, C₁, R₂, L₂, C₂ (7 parameters).
- Use `scipy.optimize.least_squares` on the complex residual
  Z_model(f) - Z_data(f).

**Output:**
- Plot: measured |Z|(f) and θ(f) overlaid with BVD fit
- Table: C₀, and per-branch fₙ, Qₙ = ωₙLₙ/Rₙ, Rₙ
- Derived: Q from BVD vs Q from transient (cross-validation)

**Potential issue:** The test5 sweep has non-uniform frequency spacing (1 kHz
near peaks, 5 kHz in valleys). This is fine for least-squares fitting but
the plot should use markers, not interpolated lines.
