# `ldv-analysis` implementation plan — PRL figure pipeline (W21)

**Scope:** W21 voltage-cascade measurement, 1f / 2f frequency sweeps, complex
modal response, electrical input power, and the per-figure data export
consumed by the PRL.

**Authority:** This plan implements the **figure-centered** contract in
`nonlinearphysics-manuscript/prl/analysis_contract.md`. Where this plan and
the contract disagree, the contract wins.

**Naming convention:** All work packages and outputs are named after
manuscript figures (Fig. 1, Fig. 2a, Fig. 2b, Fig. 2c, Fig. 3, Fig. S1,
Fig. S2, Fig. S3). The previous "Gate N" naming is fully superseded.

---

## 1. Purpose

This pipeline extends `ldv-analysis` from "harmonic-peak figure scripts" to
a reproducible measurement pipeline that emits, for every PRL figure and
supplemental figure, exactly the data required to regenerate that figure.

Concretely, it produces:

1. **Complex pressure phasors** `P_nf^complex(x, y; run)` with an explicit
   phasor convention, for every cascade scan and every external `nf`
   frequency sweep.
2. **Modal amplitudes** `\hat P_nf` from a fixed-projection complex fit,
   with real/imag covariance.
3. **Effective external response** `G_n^ext(f)` from one- or two-pole
   complex fits to the `nf` frequency sweep, used as the Fig. S2 mechanism
   guide input.
4. **Direct input power** `P_in = ⟨v(t) i(t)⟩` from simultaneously
   acquired voltage/current waveforms.
5. **Acoustic energy densities** `E_n = \hat P_nf^2 / (4 ρ_0 c_0^2)` using
   pre-registered water constants.
6. **Data-driven low-drive window** selection by the contract criterion
   `|\hat P_{2f}/\hat P_{1f}| < 0.1`.
7. **Effective Coppens slope** `K_exp` from the low-drive harmonic ratio,
   reported as the experimental effective perturbative coupling — not from
   `G_2^ext` alone.
8. **Per-figure data files** `figN.npz` and pre-rendered `figN.pdf` /
   `figN.png` for every PRL figure listed in the contract Tag Map.

---

## 2. Non-goals

This repository explicitly does **not**:

- Solve the high-drive coupled-mode equations. The contract demotes the
  1f–2f model to an idealized mechanism guide; that lives in
  `harmonic_model`, which is forbidden from importing `ldv-analysis`.
- Refit any coupling coefficient on the high-drive window (50–120 Vpp).
- Assume `G_2^ext` is the response to the internal `1f × 1f` source.
- Compute or propagate the viscous Settnes–Bruus contrast-factor band.
  Main-text Fig. 3 uses inviscid Gor'kov, `Φ_n/Φ_1 = 1`, so
  particle-dependent uncertainties cancel from the ratio.
- Export a generic "measurement bundle" for `harmonic_model`. The public
  data products are the per-figure `figN.npz` files.

---

## 3. Pre-registered analysis conventions

These conventions are copied from the contract's "Analysis conventions and
cross-figure definitions" table. They must be reflected in
`config.toml` and asserted in code; they may not be changed once a figure
has been generated against them.

| Convention | Value / source |
|---|---|
| Canonical run set | `sample_101x21_fsweep_peak_*_*Vpp_20260525_*` over the measured W21 chip drive range. |
| Drive label | PZT terminal `V_pp`, post-amplifier and calibration-corrected per `CALIBRATION_NOTE.md`. AFG-side `V_pp` is **not** the drive label. |
| Common harmonic frequency / position | Per run: `f` chosen as the 1f axial-peak resonance; same spatial functional used for all `nf` in that run. |
| Low-drive window | Defined from the data by `|\hat P_{2f}/\hat P_{1f}| < 0.1`. Voltage values are **not** hard-coded. If fewer than 3 points satisfy the condition, the explicit exponent claim is dropped (Fig. 2a fallback wording). |
| `P_in` | On-state mean of `⟨v(t) i(t)⟩` over the steady-state portion of the burst (excludes turn-on transient). Direct time-domain product is primary; `(1/2) Re(V_h I_h^*)` is a diagnostic. |
| FFT cache | Cache primary complex quantities (`P_nf^complex`, `V_1f^complex`, `I_1f^complex`, steady-window indices, DFT-bin noise). **Do not** cache the time-origin-invariant `arg P_nf - n arg V_1f` ("phase-invariant phase") as a primary quantity; compute it from the cached primaries when needed. |
| Noise-bias correction | Per-bin DFT noise floor subtracted in quadrature; applied only above the 3σ threshold. |
| 4f / 5f adoption | Include in main force reconstruction iff SNR > 5 **and** `E_n/E_1` exceeds its 1σ uncertainty band. Otherwise exclude from the reconstruction and record the non-adoption in figure metadata. |
| Water properties | Logged temperature if available; otherwise nominal `T_0 = 298.15 K`: `ρ_0 = 997.047 kg/m³`, `c_0 = 1496.70 m/s`, `η = 8.90 × 10⁻⁴ Pa·s` (supplemental absolute-time only). `B/A = 5.0`, `β = 1 + B/(2A) = 3.5`. |
| Radiation-force approximation | Inviscid Gor'kov in main text: `Φ_n/Φ_1 = 1`. Viscous / thermoviscous corrections are optional supplemental sensitivity checks only. |
| Uncertainty propagation | Analytic primary where available. Monte Carlo as a check for nonlinear ratios, thresholding, covariance, and fit-selection effects. |
| Central performance metric | `R_κ ≡ κ_foc / κ_lin = E_foc / E_lin`. No absolute trajectory endpoints. |
| Local-linearity thresholds | Primary `δ_lin = 5%`; sensitivity check `10%`. |
| Drive-purity threshold (Fig. 1) | AFG-side `V_nf / V_1f < 0.5%` for `n ≥ 2`, and at least an order of magnitude below the observed `\hat P_nf / \hat P_1f`. |

Three constants are still **unresolved** and must be settled by the user
before the pipeline emits release-quality complex figure data:

- `dn/dp(water) @ 633 nm`: `pra/main.tex:472` has `1.4 × 10⁻¹⁰ Pa⁻¹`,
  `ldv-analysis/src/ldv_analysis/config.py` has `1.48 × 10⁻¹⁰ Pa⁻¹`.
  5% systematic into `E_1`. See
  `pra/NOTES_pending_citations.md` item 2.
- Pressure-calibration audit (complex `C_p←d`) — see §11 D1.
- `K_exp` slope: see §4 Fig. 2b notes; the contract permits the
  measured value to stand without a theoretical reference.

---

## 4. Tag map (figure ↔ analysis ↔ output)

This table is the single source of truth for what each `[HYP-F...]` tag
in `prl/paragraph_draft.md` depends on. Decision criteria and fallback
wordings live in the contract; this table just routes them to code.

| Figure | Draft tag | Owner WP | Output |
|---|---|---|---|
| Fig. 1 (spatial emergence) | `[HYP-F1]` | F1 | `fig1.{pdf,png,npz}` |
| Fig. 2a (perturbative scaling) | `[HYP-F2a]` | F2a | `fig2.npz` (+ `fig2.{pdf,png}` shared with 2b, 2c) |
| Fig. 2b (Coppens slope / `K_exp`) | (none in current contract — see §11 R0) | F2b | `fig2.npz` (subset) |
| Fig. 2c (input-power response) | `[HYP-F2c]` | F2c | `fig2.npz` (subset) |
| Fig. 3 (force reconstruction + reshaping) | `[HYP-F3]` | F3 | `fig3.{pdf,png,npz}` |
| Fig. S1 (rule-out tests) | implicit Fig. 1 support | S1 | `figS1.{pdf,png,npz}` |
| Fig. S2 (`Q_n`, `G_n^ext`, model guide) | `[HYP-FS2]` | S2 | `figS2.{pdf,png,npz}` |
| Fig. S3 (validity domain — drives `R_κ` interpretation) | `[HYP-FS3]` | (HM-side; LDV exports inputs) | inputs in `fig3.npz` |

---

## 5. Target architecture

```text
raw TDMS / HDF5
      │
      ▼
complex waveform cache  (WP-L1)
  ├─ V_1f, I_1f phasors
  ├─ LDV decoder phasors
  ├─ calibrated pressure phasors P_nf^complex
  ├─ direct ⟨vi⟩
  └─ per-bin noise statistics
      │
      ▼
fixed spatial projection  (WP-L2)
  ├─ complex modal amplitudes \hat P_nf
  ├─ real/imag covariance
  └─ projection definition + provenance
      │
      ├──────────────────┐
      ▼                  ▼
cascade analysis     external response fitting (WP-L5)
  (WP-L4)             ├─ G_1^ext one-pole
  ├─ Fig. 1           └─ G_2^ext one- or two-pole
  ├─ Fig. 2a              │
  ├─ Fig. 2b              │
  └─ Fig. 2c              │
      │                   │
      ▼                   ▼
per-figure data export  (WP-L7)
  ├─ fig1.{pdf,png,npz}
  ├─ fig2.{pdf,png,npz}
  ├─ fig3.{pdf,png,npz}
  ├─ figS1.{pdf,png,npz}
  └─ figS2.{pdf,png,npz}
      │
      ▼
harmonic_model  (separate repo)
  ├─ reads fig2.npz, fig3.npz, figS2.npz
  └─ produces figS3.{pdf,png,npz}, model-guide overlays
```

---

## 6. Implementation packages

The packages keep the original WP-L0..L8 numbering for continuity, but
their contents are re-scoped to figures.

### WP-L0 — Pipeline config, manifest, single entry point

#### New files

```text
experiments/2026W21/prl_pipeline/
├── config.toml
├── manifest.py
├── run.py
├── stages.py
└── README.md
```

#### Fixed in `config.toml`

- canonical run glob and explicit run IDs
- channel width `W = 375 μm`, height `H = 150 μm`
- water constants: `ρ_0 = 997.047`, `c_0 = 1496.70`, `η = 8.90e-4`, `T_0 = 298.15 K`
- nonlinear-acoustics: `B_over_A = 5.0`, `beta = 3.5`
- low-drive selector: `low_drive_p2_over_p1_threshold = 0.1`
- minimum low-drive count: `min_low_drive_points = 3`
- adoption thresholds: `harmonic_snr_min = 5.0`, `harmonic_e_ratio_sigma_min = 1.0`
- drive-purity threshold: `drive_purity_threshold = 0.005`
- noise-bias rule: `noise_sigma_min = 3.0`
- steady-state window rule (burst protocol-specific)
- operating-frequency rule (per-run 1f axial-peak resonance)
- fixed spatial projection rule (reference scan, signed mode shape, axial weight)
- fit-model order: 1f one pole, 2f two poles primary
- random seed
- output schema version
- whether to record Git commit and input-file hash

#### CLI

```bash
python experiments/2026W21/prl_pipeline/run.py --stage cache
python experiments/2026W21/prl_pipeline/run.py --stage project
python experiments/2026W21/prl_pipeline/run.py --stage resonance
python experiments/2026W21/prl_pipeline/run.py --stage figures
python experiments/2026W21/prl_pipeline/run.py --stage all
```

Each stage carries input/output manifests, and even when reusing prior
artefacts checks the schema and source hash.

#### Acceptance

- The 12 runs and low-drive selector are uniquely reproducible from
  `config.toml`.
- No other low-drive threshold survives anywhere in the pipeline scripts.
- Run logs record config, commit, and cache version.

---

### WP-L1 — Complex phasor cache + direct ⟨v i⟩

#### Modified

```text
src/ldv_analysis/fft_cache.py
src/ldv_analysis/config.py
src/ldv_analysis/io_utils.py
src/ldv_analysis/__init__.py
```

#### New

```text
src/ldv_analysis/phasor.py
src/ldv_analysis/electrical.py
tests/test_phasor.py
tests/test_electrical.py
```

#### Cache version

Bump `_CACHE_VERSION`. Old magnitude/phase fields remain only as legacy
read-only fields. The new pipeline uses complex fields as the sole source
of truth.

#### New cache fields

```text
voltage_1f_complex_v
current_1f_complex_a
ldv_decoder_1f_complex_v ... ldv_decoder_5f_complex_v
pressure_1f_complex_pa  ... pressure_5f_complex_pa
pressure_1f_noise_var_pa2 ... pressure_5f_noise_var_pa2
pin_mean_w
pin_window_se_w
phasor_convention
pressure_calibration_id
pressure_calibration_complex_pa_per_v
amplitude_convention
```

**Not** cached as a primary quantity: `arg P_nf - n arg V_1f` ("phase
invariant"). It is computed from the cached primaries on demand. This
matches the contract's FFT/cache policy.

#### Power calculation

For every scan point on the same steady-state window:

`P_in = (1/N) Σ [v(t_m) - v_0] [i(t_m) - i_0]`

with the offset rule fixed in `config.toml`. Primary value is the
time-domain direct mean. Diagnostic: `P_spec = Σ_h (1/2) Re(V_h I_h^*)`,
its agreement with `P_in` recorded as a sanity check.

#### Optional displacement role

If displacement decoder waveforms exist, expose them as `ROLE_LDV_DISPLACEMENT`
in `io_utils.py`. Files without it must still load.

#### Acceptance

- Synthetic tone: amplitude and phase recovered to within numerical
  precision.
- Time-origin invariance: shifting `t → t + Δt` leaves `\bar P_n =
  P_n exp(-i n arg V_1)` unchanged.
- Pure sinusoid: `⟨v i⟩` and `(1/2) Re(V I^*)` agree to numerical
  precision.
- Synthetic waveform with electrical harmonics: time-domain ⟨v i⟩
  recovers the total harmonic power.
- Decoder cross-check (velocity vs displacement, or an alternative phase
  test) passes; otherwise §11 D1 gating applies.

---

### WP-L2 — Complex modal projection + fixed spatial functional

#### Modified

```text
src/ldv_analysis/mode_fit.py
src/ldv_analysis/sweep_fit.py
src/ldv_analysis/grid_utils.py
```

#### New

```text
src/ldv_analysis/spatial_projection.py
tests/test_complex_mode_fit.py
tests/test_spatial_projection.py
```

#### Data structures

```python
@dataclass(frozen=True)
class ProjectionDefinition:
    projection_id: str
    channel_center_model: tuple[float, float]
    channel_width_m: float
    axial_coordinates_m: np.ndarray
    axial_weights_complex: np.ndarray
    transverse_harmonic: int
    transverse_weights_real: np.ndarray
    reference_run_id: str
    reference_frequency_hz: float

@dataclass(frozen=True)
class ComplexProjectionResult:
    amplitude_complex_pa: complex
    covariance_re_im_pa2: np.ndarray  # (2, 2)
    r2_complex: float
    n_used: int
    kept_mask: np.ndarray
    projection_id: str
```

#### Complex fit

For mode shape `u_n(y)`, solve

`\hat P_n = arg min_a Σ_j w_j |p_j - a u_n(y_j)|^2`.

- Do not magnitude-fit real and imaginary parts independently.
- Derive a `(2, 2)` real covariance from the complex residual.
- Sigma-clipping (if used) stores its mask and threshold.
- Report both magnitude-only `R^2` and complex `R^2`.

#### Fixed axial projection

MVP: a single axial column locked at the 1f reference. Recommended
implementation: a normalized complex weight `w_x` from the reference
field, with a linear projection `P_n = Σ_x w_x^* a_n(x)`.

- The weight is fixed across all frequencies in a sweep.
- The weight is **never** constructed from validation data.
- The legacy `fit_axial()` "per-frequency maximum column" path may
  remain for legacy plots, but is forbidden for response fits.

#### 2f two-pole extension

Minimum requirement: complex two-pole fit on the locked axial projection.
If data quality supports it, fit multiple axial projections jointly,
sharing pole frequencies and linewidths and letting only the complex
residues vary per projection.

#### Acceptance

- Synthetic complex modes: amplitude, phase, covariance recovered.
- Frequency reordering: projection definition invariant.
- No frequency-dependent reselection of `x_best`.
- Legacy magnitude output and the new complex magnitude agree in the
  low-noise limit.

---

### WP-L3 — Noise bias, SNR, uncertainty

#### New

```text
src/ldv_analysis/uncertainty.py
tests/test_noise_bias.py
```

#### Policy

The complex coefficient is **not** subjected to an arbitrary "complex
noise vector" subtraction. Magnitude and energy use the variance-based
correction:

`|P_n|_corr = sqrt(max(|P_n|_raw^2 - σ_{P,n}^2, 0))`

- Below 3σ: censored / NaN. Do not represent as zero.
- Phase output requires a separately satisfied phase-SNR threshold.
- For a "corrected complex phasor", preserve the raw phase and scale
  only the magnitude.
- Store separately: fit SE, DFT noise, pressure calibration, spatial
  variability.
- Figures combine the necessary components explicitly.

#### Acceptance

- Injected complex Gaussian noise: energy bias removed.
- Threshold neighbourhood: no negative energies or spurious phases.
- Uncertainty components traceable from any export artefact.

---

### WP-L4 — Cascade operating point and per-figure analyses

#### New

```text
experiments/2026W21/prl_pipeline/cascade.py
experiments/2026W21/prl_pipeline/fig1_spatial.py
experiments/2026W21/prl_pipeline/fig2a_scaling.py
experiments/2026W21/prl_pipeline/fig2b_coppens.py
experiments/2026W21/prl_pipeline/fig2c_power.py
experiments/2026W21/prl_pipeline/fig3_force.py
experiments/2026W21/prl_pipeline/figS1_ruleouts.py
```

Legacy scripts (`harmonic_ladder.py`, `harmonic_scaling.py`,
`harmonic_mode_shapes.py`, `figure_data.py`) migrate to call the shared
APIs. No figure script keeps its own extraction convention.

#### Per-run operating point

1. Determine `f_op` from the 1f response complex fit or pre-registered
   peak rule.
2. Select that frequency file. If interpolating, interpolate the complex
   amplitude — never the magnitude peak alone.
3. Lock the spatial functional from the 1f reference.
4. Extract `P_1f` … `P_5f` with the same functional.
5. Form `\bar P_n = P_n exp(-i n arg V_1)` (drive-referenced
   phase-invariant complex amplitude).

#### Fig. 1 (formerly Gate 1)

- 120 Vpp 1f–3f complex / magnitude maps.
- Signed mode-shape fit and `R^2`.
- Axial variability.
- AFG/PZT drive purity bound, threaded through `drive_purity_threshold`
  in config.
- Decoder linearity and air-filled null (the latter shown in Fig. S1).

#### Fig. 2a (formerly Gate 2)

Low-drive window from `|\hat P_{2f}/\hat P_{1f}| < 0.1`. Weighted fit

`log|P_n| = log C_n + p_n log V_pp`.

- Slope `p_n` covariance and bootstrap CI saved.
- Diagnose both `|P_2/P_1^2|`, `|P_3/P_1^3|` and the complex ratios.
- Extended window saved only as sensitivity output, under a separate
  name.

#### Fig. 2b (Coppens slope / `K_exp`)

`K_exp` is the perturbative-window slope of `\hat P_{2f}/\hat P_{1f}`
versus `M = \hat P_{1f}/(ρ_0 c_0^2)`:

`\hat P_{2f}/\hat P_{1f} = (β/4) M K_exp`.

- Origin-through fit on the same low-drive window used by Fig. 2a.
- Report `K_exp` with covariance and bootstrap CI.
- Show the single-pole reference `Q_2 cos θ` from Fig. S2 only as a
  guide, never as a quantitative prediction.

#### Fig. 2c (formerly Gate 3)

`E_1 = |\hat P_{1f}|^2 / (4 ρ_0 c_0^2)`,
`E_1^lin = s P_in` with `s` from the low-drive window through-origin fit.

- Combined drive-side and acoustic-side uncertainty in `E_1/E_1^lin`.
- Highest-drive deviation and sigma significance recorded.

#### Fig. 3 (formerly Gate 4)

For every drive:

- `E_n = |\hat P_nf|^2 / (4 ρ_0 c_0^2)` for adopted harmonics.
- Inviscid Gor'kov assumption: `Φ_n / Φ_1 = 1`. Particle-dependent
  prefactors cancel.
- `\tilde F(y) = Σ_n (-1)^n n (E_n / E_1) sin(2 n k y)` with
  `k = π/W`.
- `E_foc / E_1 = 1 - 4 (E_2/E_1) + 9 (E_3/E_1) - 16 (E_4/E_1) +
  25 (E_5/E_1) - ⋯` with uncertainty.
- Analytic propagation primary; Monte Carlo as a check.

#### Fig. S1 (rule-out tests)

- AFG-side `V_nf / V_1f` against drive level.
- Velocity / displacement decoder cross-check.
- Air-filled null measurement.
- Per-bin DFT noise floor versus fitted `\hat P_nf`.

Decision criterion: rule-out signals at least one order of magnitude
below the observed pressure-harmonic ratios, or otherwise explicitly
bounded in the figure caption.

#### Outputs

```text
output/prl_pipeline/fig1/   fig1.{pdf,png,npz}, fig1_decision.json
output/prl_pipeline/fig2/   fig2.{pdf,png,npz}, fig2_decisions.json (a, b, c keyed)
output/prl_pipeline/fig3/   fig3.{pdf,png,npz}, fig3_decision.json
output/prl_pipeline/figS1/  figS1.{pdf,png,npz}
```

Each `*_decision.json` carries: contract decision-criterion values, the
satisfied / not-satisfied flag, and the chosen contract fallback wording
ID.

#### Acceptance

- A single `config.toml` and a single `run.py --stage figures` reproduce
  every Fig. 1–3, Fig. S1 figure and its decision JSON from the canonical
  12 cascade runs.
- Legacy figure scripts produce numerically identical figures via the
  shared APIs (or a documented diff is recorded).
- Decision JSON entries reference contract fallback wording IDs.

---

### WP-L5 — Complex resonance response and `G_n^ext`

#### New

```text
src/ldv_analysis/resonance_fit.py
tests/test_resonance_fit.py
experiments/2026W21/prl_pipeline/resonance.py
experiments/2026W21/prl_pipeline/figS2_response.py
```

#### Quantity

Primary: physical-units transfer function

`H_n^ext(f) [Pa/V]`.

Dimensionless `G_n^ext` is derived as `G_n^ext = H_n^ext V_ref / P_ref`
with `V_ref`, `P_ref` recorded in export metadata. This prevents implicit
normalization.

#### One-pole model (1f primary)

`H(f) = b_0 + b_1 (f - f_c) + r / [1 + i Q (f/f_0 - f_0/f)]`.

#### Two-pole model (2f primary)

`H(f) = b_0 + b_1 (f - f_c) + Σ_{j=1,2} r_j / [1 + i Q_j (f/f_j - f_j/f)]`.

- `f_j, Q_j` real; residues `r_j` and background complex.
- Joint real / imaginary residual fit.
- Per-frequency covariance whitening when available.
- Magnitude-only Lorentzian only as FWHM sanity check.
- Build-up transient `Q` only as independent diagnostic.

#### Outputs

- pole frequencies, `Q_j`, complex residues, parameter covariance
- measured and fitted complex locus
- `|H|`, phase, real / imaginary residual
- FWHM-`Q` vs transient-`Q` comparison
- complex `H_n^ext` at the operating point
- effective `Q_n` to be consumed by `harmonic_model` mechanism-guide

These go into `figS2.{pdf,png,npz}` along with the model-guide curve
overlays (see §7).

#### Acceptance

- Synthetic one- and two-pole responses recovered within nominal
  uncertainty.
- Pole-frequency ordering enforced (no label-swap between 2f poles).
- Fixed-projection vs moving-maximum diagnostic figure available.
- Visible residual structure prevents collapse to a single `Q cos θ`
  reference.

---

### WP-L6 — Effective internal coupling (`K_exp`), back-coupling demoted

#### New

```text
src/ldv_analysis/coupling_fit.py
tests/test_coupling_fit.py
experiments/2026W21/prl_pipeline/coupling.py
```

#### Primary deliverable: `K_exp`

On the low-drive window only, fit

`\bar P_2 = (β/4) M K_exp \bar P_1 + ε_2`

(equivalently `\bar P_2 \bar P_1^{-1} = (β/4) M K_exp + ε`) using
complex weighted regression. Origin-through is primary; intercept is a
leakage / background diagnostic.

External response comparison `K_exp` versus `(β/4) Re G_2^ext(2 f)` is a
systematic check, not an identity. Disagreement is reported per the
contract's Fig. 2b fallback wording, **not** treated as failure.

#### Back-coupling `B_121^eff` — diagnostic only

The previous plan made `B_121^eff` a primary calibration target. The new
contract demotes it: the harmonic-ladder mechanism guide does not need
it. We compute it as a diagnostic only:

`P_1 - P_{1, lin} = B_121^eff P_2 P_1^* + ε_1`,

with full identifiability triage (`status ∈ {identified, weakly_identified,
not_identified}`). If `not_identified`, the figure is not generated and a
note is added to the diagnostics table. We **never** fit `B_121^eff` on
high-drive data.

#### Acceptance

- Synthetic complex data: `K_exp`, `B_121^eff` recovered.
- Zero back-coupling fixture: `not_identified` is reached without false
  identification.
- High-drive rows handed to the calibration API raise an exception
  (mask-typed).

---

### WP-L7 — Per-figure data export

#### New

```text
src/ldv_analysis/figure_export.py
tests/test_figure_export.py
```

#### Output structure

Replaces the previous "versioned bundle". Per-figure trio:

```text
experiments/2026W21/output/prl_figures/
├── fig1/
│   ├── fig1.pdf
│   ├── fig1.png
│   └── fig1.npz
├── fig2/
│   ├── fig2.pdf
│   ├── fig2.png
│   └── fig2.npz
├── fig3/
│   ├── fig3.pdf
│   ├── fig3.png
│   └── fig3.npz
├── figS1/
│   └── figS1.{pdf,png,npz}
└── figS2/
    └── figS2.{pdf,png,npz}
```

Each `figN.npz`:

- contains exactly the arrays needed to regenerate `figN.{pdf,png}` and
  to feed any downstream consumer (e.g., `harmonic_model` consumes
  `fig2.npz` and `figS2.npz`).
- is loaded with `np.load(..., allow_pickle=False)`.
- carries a sibling `figN.json` with metadata: schema version, contract
  hash, source repository commit, projection IDs, calibration IDs,
  adopted harmonic flags, decision wording IDs.
- excludes pickle and object-dtype arrays.

`fig2.npz` contents at minimum:

```text
drive_vpp_pzt                       (N,)
f_op_hz                              (N,)
p_in_w, p_in_std_w                  (N,)
harmonic_number                     (H,)
p_mode_complex_pa                   (N, H)
p_mode_cov_reim_pa2                 (N, H, 2, 2)
p_mode_mag_corrected_pa             (N, H)
p_mode_snr                          (N, H)
harmonic_adopted                    (N, H)
low_drive_mask                      (N,)
power_law_p_n                       (H,)
power_law_p_n_cov                   (H, H)
coppens_K_exp                       scalar (complex)
coppens_K_exp_cov                   (2, 2)
e1_lin_slope_pa_per_w               scalar
e1_lin_slope_se                     scalar
e1_over_e1_lin                      (N,)
e1_over_e1_lin_sigma                (N,)
```

`fig3.npz`:

```text
drive_vpp_pzt                       (N,)
e_n_pa2                              (N, H)
e_n_pa2_cov_reim                    (N, H, 2, 2)  # ratio uncertainty primary
e_n_over_e1                         (N, H)
f_shape_y                           (Y,)   # dimensionless y grid
f_shape_tilde_F                     (N, Y)  # \tilde F(y)
e_foc_over_e1                       (N,)
e_foc_over_e1_sigma                 (N,)
r_kappa                             (N,)
r_kappa_sigma                       (N,)
```

`figS2.npz`:

```text
f_sweep_1f                          (F1,)
h1_ext_complex_pa_per_v             (F1,)
one_pole_fit_params                 dict-like with field-name list
f_sweep_2f                          (F2,)
h2_ext_complex_pa_per_v             (F2,)
two_pole_fit_params                 ...
q_n_effective                       (H,)
q_n_effective_se                    (H,)
```

#### Schema validation

- shape, dtype, finite-value, units, mask disjointness
- low-drive selection invariant: `low_drive_mask = (|p2/p1| < 0.1)`
- phasor convention is a known ID
- `projection_id` constant across all response points
- complex covariance positive semi-definite (or symmetrised, reported)
- a small synthetic fixture is shared with `harmonic_model` (see §9)

#### Acceptance

- Fresh Python process can re-render every figure from `figN.npz`
  alone.
- Going from SI to dimensionless and back is metadata-driven.
- Source files, settings, code revision are traceable from a `figN.npz`
  alone.

---

### WP-L8 — Figure report and regression protection

#### New

```text
experiments/2026W21/prl_pipeline/report.py
experiments/2026W21/prl_pipeline/decisions.py
tests/test_figure_decisions.py
```

#### Report content

- conventions and the values actually used (against the contract table)
- per-figure decision JSON, satisfied / not-satisfied flags, fallback
  wording IDs (contract-pinned)
- diagnostic deltas vs the legacy figure scripts
- unresolved systematic issues (κ_P bracket no longer applies; dn/dp
  pending; pressure-calibration audit status)

#### CI

- Synthetic / small fixture tests run in CI without raw W21 data.
- Full-data pipeline is a manifest-only dry run and a schema check in
  CI; integration is a target for environments with real data.

---

## 7. Mechanism-guide overlay (data-only contract for `harmonic_model`)

`harmonic_model` is a separate repository (see §9) and is forbidden from
importing `ldv-analysis`. The minimal data it needs to produce its
mechanism-guide curves and the Fig. S3 outputs:

- `figS2.npz`: measured effective `Q_n`, complex `G_n^ext` at operating
  point, two-pole parameters when present.
- `fig2.npz`: low-drive normalization (the `K_exp` slope and the
  perturbative window definition), the cascade `\bar P_nf` for the
  overlay comparison.
- `fig3.npz`: `E_n / E_1` and `\tilde F(y)` for the topology /
  central-stiffness / trajectory-domain checks.

Anything `harmonic_model` needs that is not in those files is a request
for an LDV-side figure-export schema bump.

---

## 8. Recommended PR split

| PR | Content | Depends on |
|---|---|---|
| LDV-1 | Phasor convention, complex cache, direct ⟨v i⟩ | — |
| LDV-2 | Complex mode fit, fixed projection | LDV-1 |
| LDV-3 | Uncertainty / noise-bias API | LDV-1, LDV-2 |
| LDV-4 | Complex one- and two-pole response fit | LDV-2, LDV-3 |
| LDV-5 | W21 canonical cascade + Fig. 1, 2a, 2c, S1 | LDV-1..3 |
| LDV-6 | Fig. 2b `K_exp` + diagnostic `B_121^eff` | LDV-4, LDV-5 |
| LDV-7 | `fig{1,2,3,S1,S2}.npz` schema and renderers | LDV-4..6 |
| LDV-8 | Fig. 3 reconstruction script | LDV-7 |
| LDV-9 | Figure report, legacy migration, documentation | LDV-5..8 |

Each PR adds synthetic tests **before** the real-data acceptance work,
so figure pixel diffs are not the only acceptance criterion.

---

## 9. Shared synthetic fixture with `harmonic_model`

Distributed at the same content to both repositories:

```text
ldv-analysis/tests/fixtures/figure_npz_minimal_v1/
  ├─ fig2_minimal.npz
  ├─ fig3_minimal.npz
  └─ figS2_minimal.npz
harmonic_model/tests/data/figure_npz_minimal_v1/
  ├─ fig2_minimal.npz
  ├─ fig3_minimal.npz
  └─ figS2_minimal.npz
```

Ground truth in the fixture:

- 4 low-drive + 2 high-drive points
- known complex `H_1^ext`, `H_2^ext` (one-pole + two-pole)
- known `K_exp`
- known `E_n / E_1` profile, `\tilde F(y)`, `κ_foc`, equilibria, `y_{5%}`

Integration test confirms:

1. Both repositories read identical numbers.
2. `K_exp` is recovered on the LDV side from the cascade `\bar P_nf`.
3. `harmonic_model` mechanism-guide curves overlay the cascade points
   within the expected mechanism-only tolerance.
4. Fig. 3 force, central stiffness, validity-domain primitives are
   computed identically (LDV computes them in `fig3.{npz}`, HM in
   `figS3.{npz}`).
5. A deliberately mis-versioned phasor-convention fixture is rejected.

The fixture generator is described in a spec shared between repositories,
not committed to only one of them.

---

## 10. Key design decisions and stopping conditions

### D1 — Pressure phasor calibration

If the complex `C_{p←d}(f)` cannot be confirmed by velocity / displacement
decoder cross-check or an alternative phase test, magnitude-only Fig. 1
and Fig. 2a stay viable, but complex `G_n^ext`, `K_exp`, and `B_121^eff`
release artefacts are halted. This is a **hard schedule risk**.

### D2 — External 2f response vs internal source

External two-pole response is acceptable for pole locations and
linewidths. Internal source residue / overlap is calibrated separately
from the low-drive cascade. Disagreement between external and internal is
**not** a failure; it is the documented reason for treating `K_exp` as an
effective coupling.

### D3 — Back-coupling

Old plan made `B_121^eff` a primary target. New plan: diagnostic only,
identifiability-status reported, never fit on high-drive data.

### D4 — Moving maximum

The "per-frequency maximum column" path is forbidden for any
transfer-function or coupling-coefficient extraction. It may stay for
legacy plots.

### D5 — Higher harmonics

4f / 5f are adopted only when SNR and energy threshold are met. Excluded
harmonics are censored, not zeroed. Adoption status is recorded in
`figN.json`.

---

## 11. Open conventions still to settle (user decisions)

| ID | Issue | Required action | Owner |
|---|---|---|---|
| R0 | `[HYP-F2b]` tag missing in contract Tag Map and `paragraph_draft.md` | Add `[HYP-F2b]` or document Fig. 2b as decisional-only | user |
| R1 | `dn/dp(water) @ 633 nm` 1.4 vs 1.48 × 10⁻¹⁰ cross-repo | Pick a primary source, unify both repos, add citation to contract | user |
| R2 | Pressure-calibration audit (complex `C_{p←d}`) | Pass velocity / displacement cross-check or alternative phase test before complex release | user |
| R3 | 4f / 5f adoption sanity check on W21 SNR | Inspect actual SNR and `E_n/E_1` uncertainty at top drives | both |
| R4 | Logged water temperature | Use logged value if available; otherwise contract nominal `T_0 = 298.15 K` | both |

Items R0–R3 each have a Task in the user's task list.

---

## 12. Definition of Done

`ldv-analysis` side of the PRL pipeline is complete when:

- [ ] Canonical 12 runs and the low-drive selector are fixed by
      `config.toml`.
- [ ] Complex pressure phasor convention and calibration are tested.
- [ ] `\bar P_n = P_n exp(-i n arg V_1)` is computed from cached
      primaries (not cached as primary).
- [ ] `P_in` from direct `⟨v i⟩` matches `(1/2) Re(V I^*)` to within
      the documented tolerance for pure sinusoids.
- [ ] Cascade harmonics share the same spatial functional per run.
- [ ] External `G_n^ext` from fixed-projection complex one-/two-pole
      fit.
- [ ] `K_exp` from the low-drive data-driven window; `B_121^eff` reported
      as a diagnostic only.
- [ ] Per-figure `fig{1,2,3,S1,S2}.{pdf,png,npz}` and the matching
      `*_decision.json` are reproducible from a single command.
- [ ] `harmonic_model` integration fixture and schema round-trip test
      pass on both repos.
- [ ] Figures, CSVs, decisions, and provenance share a common run ID.
- [ ] Contract fallback wording IDs are routed through `decision.json`
      entries.
