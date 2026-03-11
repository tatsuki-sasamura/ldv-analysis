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

---

## 3. Background noise and SNR assessment

**Goal:** Quantify per-point signal-to-noise ratio to enable data quality
filtering and uncertainty estimation.

### 3a. Noise extraction in `fft_cache.py`

**Approach:** RMS of the post-burst segment (signal-free tail after burst
ends). Only applicable to burst-mode datasets, which is sufficient — we
don't need SNR on every dataset.

**Changes to `_compute()`:**
- After burst detection (`ss_end`), the remaining samples `wf2[:, ss_end:]`
  contain only noise. Compute per-point RMS of that segment.
- Convert to velocity and pressure using the same calibration factors.
- Store `noise_rms_velocity` and `noise_rms_pressure` in the cache.
- For continuous-mode files (no burst), store NaN arrays.

**Cache fields added:** `noise_rms_velocity`, `noise_rms_pressure` (1-D,
per point, NaN for continuous mode).

### 3b. SNR computation and assessment

**Per-point SNR:** `SNR = pressure_1f / pressure_noise` (linear ratio), or
`SNR_dB = 20 log10(pressure_1f / pressure_noise)`.

**Integration into existing scripts:**
- `pressure_map_2d.py`: add SNR grid panel to the output (spatial map of SNR)
- `voltage_sweep.py`: report median SNR per voltage level
- `freq_sweep_*.py`: report median SNR per frequency

**Standalone assessment script** (optional):
`experiments/2026W10_stepA/snr_assessment.py`
- Histogram of per-point SNR across a representative dataset
- SNR vs voltage (from voltage sweep files)
- SNR vs frequency (from freq sweep files)
- Spatial map of SNR (low-SNR points correlate with channel edges / wall scatter)

### Notes

- Post-burst segment is typically ~34,000 samples (burst-mode files), giving
  a well-averaged noise estimate.
- Expected SNR at 25 Vpp: >40 dB at antinode, <10 dB at channel edges.
- This enables principled quality filtering: replace ad-hoc RSSI thresholds
  with SNR-based masks where appropriate.
- Continuous-mode datasets (no burst) get NaN — acceptable since SNR
  assessment is not needed universally.
