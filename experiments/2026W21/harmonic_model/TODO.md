# harmonic_model — TODO

Confrontation suite between the 1f–2f coupled-mode model and the W21 cascade dataset, used to clear gates 3–6 of the PRL analysis contract (`nonlinearphysics-manuscript/prl/analysis_contract.md`). Distinct from the exploratory scripts in `experiments/2026W21/*.py`, which generated the cascade data this folder consumes.

This folder owns the analyses that *can change the wording* of the PRL — every output here is referenced by a `[HYP-Gn]` tag in `paragraph_draft.md`.

**Important convention update (κ-revised Gate 6).** The primary throughput-relevant scalar is $\kappa_\mathrm{foc}$, defined as the central force-field curvature; the throughput connection is the *analytic* centered-contraction-criterion identity $|y_\mathrm{out}|/|y_\mathrm{in}| = \exp[-\kappa_\mathrm{foc} L /(\zeta u_x)]$, giving $q_{v,\max}^{(c)} \propto \kappa_\mathrm{foc}$ exactly for any fixed contraction ratio. No trajectory cutoffs appear in the main metric. Trajectory analysis is repositioned to a *descriptive validity-domain map* in SM §S3, not a pass/fail competitor.

## Cross-references

- PRL draft: `nonlinearphysics-manuscript/prl/paragraph_draft.md`
- Analysis contract (gates, pass criteria, fallbacks): `nonlinearphysics-manuscript/prl/analysis_contract.md`
- Data root: defined via `DATA_ROOT` in `experiments/2026W21/vpp_vs_pressure.py`
- Cache root: `get_cache_dir(...)` per `ldv_analysis.config`
- Upstream exploratory scripts (already present): `harmonic_ladder.py`, `harmonic_cascade.py`, `q1_linewidth.py`, `peak_freq_drift_b2.py`, `drive_purity_b1.py`, `prl_draft.py` (Fig 1, 2a precursors)

## Hard rules (matched to analysis contract Gate 0)

These are calibration conventions that must NOT be changed once an analysis below has been run. If a calibration changes, every downstream output must be re-run.

- `V_drive_label`: PZT terminal Vpp (post-amp, gain-corrected per `CALIBRATION_NOTE.md`). Do not use AFG-side Vpp.
- `f_canonical(run)`: pick the 1f axial-peak frequency per run from the 1f sweep; use the SAME `f_canonical` for all `nf` extraction within that run.
- `(x, y)_canonical(run)`: pick the axial position where `|P_1f|` peaks per run; use the SAME `(x, y)` for all `nf` extraction.
- Low-drive window: 10, 20, 30, 40 Vpp PZT.
- `P_in` window: steady-state portion of the burst (exclude turn-on transient; protocol-specific window in `hardware.yaml`).
- Central performance metric: `R_kappa = kappa_foc / kappa_lin = E_foc / E_lin`. No trajectory endpoints enter its definition.
- Local-linearity thresholds: primary `delta_lin = 5%`; sensitivity check `10%`. Define `y_delta` as the largest centered interval on which `|F_rad / (-kappa_foc * y) - 1| <= delta_lin`.
- Trajectory-domain check: descriptive 2D map only; not a pass/fail at any single `(y_i, y_f)`.

## Files to create (grouped by gate)

### `g3_input_power_sublinearity.py` — Gate 3
**Purpose.** Determine whether `E_1(P_in)` falls below its linear extrapolation, i.e., the input-power-side sublinearity claim that underwrites the "fundamental sublinearity" language in the PRL.

**Inputs.**
- Canonical 12 scans (V = 10–120 Vpp).
- Pico CH A (voltage), CH D (current) raw waveforms per scan.
- Pico CH B (LDV velocity) → `hat_P_1f` via existing pipeline.

**Steps.**
1. For each scan, compute `P_in = mean(v(t) * i(t))` over the steady-state portion of the burst.
2. For each scan, extract `hat_P_1f` at `(f_canonical, x_canonical, y_canonical)`.
3. Compute `E_1 = hat_P_1f**2 / (4 * rho_0 * c_0**2)` per scan.
4. Linear fit `E_1 = a * P_in` on the low-drive window (10–40 Vpp); record `a`, `sigma_a`.
5. Define `E_lin(P_in) = a * P_in` extrapolated to all drives.
6. Compute ratio `E_1 / E_lin` per scan with combined uncertainty (fit-SE on `hat_P_1f`, electrical-noise SE on `P_in`, slope uncertainty `sigma_a` propagated).

**Outputs.**
- `output/g3_input_power_sublinearity.npz`: `V_drive`, `P_in`, `hat_P_1f`, `E_1`, `E_lin`, `E_1_over_E_lin`, `sigma_*`.
- `output/g3_input_power_sublinearity.png`: log-log `E_1` and `E_lin` vs `P_in` (left); ratio `E_1/E_lin` vs `P_in` with 3σ band and "1" reference line (right).

**Pass criterion.** `E_1/E_lin` at the highest drive falls below 1 by more than 3σ.

**Fallback action.** If criterion fails (i.e., $E_1(P_\mathrm{in})$ is consistent with linear), edit `paragraph_draft.md`: drop "fundamental sublinearity" language; rewrite Sec. 3 P2 crossover bullet.

---

### `g4_harmonic_reshaping.py` — Gate 4
**Purpose.** Quantify `E_foc / E_1` and propagate uncertainties to test the "harmonic reshaping" factor in the two-mechanism decomposition.

**Inputs.**
- `hat_P_nf` for `n = 1, 2, 3` (and 4, 5 if Gate 0 SNR thresholds met) per scan, from existing harmonic_ladder cache.
- Particle parameters: `rho_p`, `kappa_p` for 5 μm polystyrene; nominal `Phi_n` from Settnes-Bruus 2012 with `delta_n / a` from `delta_n = sqrt(2 * eta / (rho_0 * omega_n))`.

**Steps.**
1. For each scan, compute `E_n = hat_P_nf**2 / (4 * rho_0 * c_0**2)` for n = 1, 2, 3.
2. Compute `Phi_n` per harmonic via Settnes-Bruus formula at `omega_n = 2 * pi * n * f_canonical`.
3. Compute `E_foc / E_1 = 1 - sum_{n>=2} (-1)**(n+1) * n**2 * (Phi_n / Phi_1) * (E_n / E_1)`.
4. Propagate uncertainties:
   - Fit-SE on `hat_P_nf`.
   - Noise-bias correction (subtract noise floor in quadrature).
   - `Phi_n / Phi_1` uncertainty band from particle-radius uncertainty (±10% nominal, sensitivity in SM).
5. Plot `E_foc / E_1` vs `V_drive` with uncertainty band.
6. Plot per-mode contribution `(-1)**(n+1) * n**2 * (Phi_n/Phi_1) * (E_n/E_1)` vs `V_drive` to show odd/even competition.

**Outputs.**
- `output/g4_harmonic_reshaping.npz`: per-scan `E_n`, `Phi_n`, `E_foc_over_E_1`, per-mode contributions, all `sigma_*`.
- `output/g4_harmonic_reshaping.png`: top: `E_foc/E_1` vs `V_drive` with 1σ and 3σ bands; bottom: stacked per-mode contributions.

**Pass criterion.** `1 - E_foc/E_1` exceeds 3σ above zero at the highest drive AND monotonic decrease.

**Fallback action.** If non-monotonic, reframe Sec. 3 P3 around odd/even competition; if within uncertainty for all drives, remove Fig. 3(d) and the throughput interpretation entirely.

---

### `g5_susceptibility.py` — Gate 5 prerequisite, also feeds SM §S2
**Purpose.** Extract complex `G_1(omega)`, `G_2(omega)` from the 1f and 2f frequency sweeps, including two-pole fits when warranted. Provides inputs for both `g5_coupled_mode_model.py` and the SM tables.

**Inputs.**
- 1f narrow-band sweep runs around 1.9 MHz.
- 2f sweep runs around 3.8 MHz (check existing data inventory; if missing, flag as a data gap).

**Steps.**
1. For each sweep, extract complex `hat_P_nf(f_drive)` at `(x_canonical, y_canonical)`.
2. Fit to one-Lorentzian and two-Lorentzian models; choose by AIC/BIC.
3. Report fit parameters (poles, widths, residues) + uncertainties.
4. Compute `Q_n` from `|G_n(omega)|**2` FWHM as a sanity-check vs the transient `Q_n`.
5. *Compare two routes for `K_the`*: (i) `(beta/4) * Re G_2` from the susceptibility directly; (ii) perturbative-slope calibration from the Coppens fit (`g4_*` low-drive subset). Difference is reported as a systematic uncertainty (relevant to EM2).

**Outputs.**
- `output/g5_susceptibility.npz`: complex `G_1(omega)`, `G_2(omega)`, fit params, model selection, both `K_the` routes.
- `output/g5_susceptibility.png`: real, imag, magnitude of G_n with single- and two-pole fits.

**Data-availability check.** Verify that 2f frequency sweeps exist in the cascade data inventory. If only 1f sweeps + 2f point measurements exist, the model needs a separate dedicated 2f sweep run before Gate 5 can be cleared.

---

### `g5_coupled_mode_model.py` — Gate 5
**Purpose.** Run the 1f–2f coupled-mode model with low-drive-calibrated couplings, no high-drive refit. Confront with the cascade data.

**Inputs.**
- Measured `G_1(omega)`, `G_2(omega)` complex-valued from `g5_susceptibility.py`.
- Low-drive `c_{2,11}` calibration (Coppens slope from V = 10–40 Vpp).
- Low-drive `c_{1,21}` calibration (1f–2f phase relation from V = 10–40 Vpp).

**Steps.**
1. Implement the coupled-mode iteration per EM1:
   `A_1 = G_1 * (V_drive + c_{1,21} * A_2 * conj(A_1))`
   `A_2 = G_2 * c_{2,11} * A_1**2`
2. Fixed-point iteration with `alpha_relax = 0.5`, convergence `max(|delta A|/|A|) < 1e-4`.
3. Run for V_drive = 10–120 Vpp using `f_canonical`-evaluated `G_1, G_2`.
4. Compare predicted `hat_P_1f(V_drive)`, `hat_P_2f(V_drive)`, `hat_P_2f / hat_P_1f` vs `M` to measured.

**Outputs.**
- `output/g5_coupled_mode_model.npz`: model `|A_1|`, `|A_2|` per drive; measurement comparison residuals.
- `output/g5_coupled_mode_model.png`: overlay model vs data for the three comparison plots; residual panel.

**Pass criterion.** Onset of departure within 30% V_drive; sign of `hat_P_1f` correction matches; magnitude within factor 2 at highest drive.

**Fallback action.** Per analysis contract Gate 5 — model demoted to SM if total failure; main-text language softened to "captures the leading mechanism" if partial.

**Dependency.** `g5_susceptibility.py`.

---

### Gate 6 — validity domain of the central-stiffness interpretation

**Substantial restructuring from the prior version.** Gate 6 no longer tests trajectory-vs-stiffness agreement at one selected `(y_i, y_f)` pair; the primary κ-based metric is decoupled from trajectory choices entirely. Gate 6 now verifies that κ_foc itself is well-defined, the central equilibrium is stable, and the κ-throughput interpretation has a meaningful local-linear region. The four scripts below replace the prior single `g6_trajectory_vs_stiffness.py`.

#### `g6a_force_topology.py`
**Purpose.** Confirm the central equilibrium at `y = 0` remains stable and that no off-center stable equilibria appear inside the centered focusing basin across the measured drive range.

**Inputs.** Reconstructed `F_rad(y)` per scan from existing mode fits (`hat_P_nf` + per-mode `Phi_n`, `sin(2nky)` factors).

**Steps.**
1. For each scan, evaluate `F_rad(y)` on a dense `y` grid spanning `[-W/2, W/2]`.
2. Locate all zeros of `F_rad(y)`. Classify each by sign of `dF/dy` (stable / unstable).
3. Verify `y = 0` remains a stable zero across all drives.
4. Track migration of any additional equilibria with `V_drive`.

**Outputs.**
- `output/g6a_force_topology.npz`: zero locations + stability per scan.
- `output/g6a_force_topology.png`: zero-tracking diagram (zero positions vs `V_drive`, colored by stability).

**Pass criterion.** Central equilibrium stable for all measured drives; no additional stable equilibria appear in the central focusing basin.

**Fallback action.** If off-center stable equilibria appear at any measured drive, retire scalar-throughput language; restructure Sec. 3 P3 around full force-field topology.

#### `g6b_kappa_robustness.py`
**Purpose.** Verify κ_foc obtained from the mode-resolved analytic formula agrees with κ from a direct local odd-polynomial fit to the reconstructed `F_rad(y)` near `y = 0`, and that the answer is robust to the fit-window width.

**Inputs.** Same as `g6a`.

**Steps.**
1. **Mode-sum κ**: compute `kappa_mode = 8 pi a**3 k**2 * sum_n (-1)**(n+1) * n**2 * Phi_n * E_n` per scan from harmonic fit amplitudes.
2. **Local-fit κ**: fit `F_rad(y) = -kappa * y + c_3 * y**3 + c_5 * y**5` on `|y| <= y_fit` for several `y_fit` values (e.g., `W/20, W/10, W/5`); record `kappa_fit(y_fit)`.
3. Compare `kappa_mode` and `kappa_fit` at each scan; report agreement.
4. Test fit-window sensitivity: `kappa_fit(y_fit)` should converge as `y_fit -> 0`.

**Outputs.**
- `output/g6b_kappa_robustness.npz`: `kappa_mode`, `kappa_fit(y_fit)` per scan.
- `output/g6b_kappa_robustness.png`: `kappa_mode` vs `kappa_fit` scatter per scan; window-sensitivity overlay.

**Pass criterion.** `|kappa_mode - kappa_fit(y_fit -> 0)| / kappa_mode < combined_uncertainty` for all drives.

**Fallback action.** If mode-sum and local-fit disagree beyond uncertainty, use the local-fit κ with enlarged uncertainty for `R_kappa`; do not interpret ideal-mode `E_foc` as a precise quantity.

#### `g6c_linearity_domain.py`
**Purpose.** Compute the central-linearity interval `y_delta` per scan for `delta_lin = 5%` (primary) and `10%` (sensitivity), where `Delta_lin(y) = |F_rad(y) / (-kappa_foc * y) - 1|`.

**Inputs.** Same as `g6a`.

**Steps.**
1. For each scan, compute `Delta_lin(y) = |F_rad(y) / (-kappa_foc * y) - 1|` on a dense `y` grid.
2. Find the largest centered interval on which `Delta_lin(y) <= delta_lin` for `delta_lin = 5%` and `10%`.
3. Track `y_{5%}, y_{10%}` vs `V_drive`.

**Outputs.**
- `output/g6c_linearity_domain.npz`: `y_{5%}, y_{10%}` per scan.
- `output/g6c_linearity_domain.png`: `Delta_lin(y)` profile at three representative drives (low, mid, high); `y_{5%}, y_{10%}` vs `V_drive`.

**Pass criterion / interpretation.**
- If `y_{5%} / W > 0.1` across all drives → centered-throughput interpretation has appreciable validity; retain "centered-throughput scaling" language.
- If `y_{5%} / W <= 0.05` → central linear region is narrow; fall back to "local focusing stiffness" language in main text. (See Gate 6 fallback wordings.)

#### `g6d_trajectory_domain_map.py` — SM §S3 figure only
**Purpose.** For Supplemental Material §S3 only: descriptive map of `R_t(y_i, y_f) / R_kappa - 1` over a 2D endpoint grid, illustrating where the κ-throughput interpretation is exact. Not a pass/fail test.

**Inputs.** Same as `g6a`. Representative PS particle (5 μm, water) for absolute-time conversion only.

**Steps.**
1. For each scan, numerically integrate Stokes trajectories from `y_i` to `y_f` on a 2D grid (e.g., `y_i / W in [0.05, 0.45]`, `y_f / W in [0.005, 0.2]`).
2. Compute `R_t(y_i, y_f) = t_lin(y_i, y_f) / t_measured(y_i, y_f)`.
3. Map `|R_t(y_i, y_f) / R_kappa - 1|` and overlay `y_{5%}, y_{10%}` contours from `g6c`.

**Outputs.**
- `output/g6d_trajectory_domain_map.npz`: per-scan `R_t(y_i, y_f)` grid + `R_kappa`.
- `output/g6d_trajectory_domain_map.png`: 2D heatmap of `|R_t / R_kappa - 1|` at representative drives, with linearity-domain contours.

**No pass/fail.** This is descriptive material for SM §S3; it documents the validity domain rather than testing it.

---

### `make_fig3.py` — only runs after gates 4 and 6 pass
**Purpose.** Combine the gate-cleared outputs into the actual PRL Fig. 3 (a–d).

**Inputs.**
- `g4_harmonic_reshaping.npz` (for Fig. 3(d) decomposition curves).
- `g3_input_power_sublinearity.npz` (for the `E_1/E_lin` curve in Fig. 3(d)).
- Force reconstruction per scan (low / intermediate / high drive picks, pre-registered as 10, 60, 120 Vpp).
- `g6a_force_topology.png` confirmation for Sec. 3 P3 "central attractor remains stable" claim.

**Steps.**
- 3-panel (a, b, c) force-profile evolution at pre-registered drives.
- Panel (d): three measurement-constrained curves `E_1/E_lin`, `E_foc/E_1`, `R_kappa = E_foc/E_lin` vs `P_in` with uncertainty bands.

**Pre-condition.** Do not run unless Gates 4 and 6a, 6b, 6c are cleared. If Gate 6a fails (off-center equilibria), Fig. 3(d) is suppressed and the figure becomes (a)–(c) plus a force-topology panel.

## Order of operations

1. `g5_susceptibility.py` — data inventory check first, may surface need for additional sweeps.
2. `g3_input_power_sublinearity.py` — fastest gate; clears or kills the fundamental-sublinearity story in one run.
3. `g4_harmonic_reshaping.py` — central gate for the manuscript center; most consequential single output.
4. `g6a_force_topology.py` — confirms primary metric is well-defined.
5. `g6b_kappa_robustness.py` — confirms κ_foc agrees between mode-sum and local-fit routes.
6. `g6c_linearity_domain.py` — sets manuscript language (centered-throughput vs local-stiffness).
7. `g5_coupled_mode_model.py` — depends on `g5_susceptibility.py`; affects "model" language but not paper centrality.
8. `g6d_trajectory_domain_map.py` — SM §S3 figure, last because most expensive.
9. `make_fig3.py` — only after 4 and 6a-c.

## Reporting

Each script outputs:
- `output/g{N}_*.npz` (raw arrays, all uncertainties, pass/fail flags where applicable)
- `output/g{N}_*.png` (the figure shell for review)
- `output/g{N}_*.report.md` (short markdown reporting Pass/Fail per the contract criterion; this drives whether `paragraph_draft.md` keeps or drops the corresponding `[HYP-Gn]` claim)

When all gates have been run, generate a one-page `output/contract_status.md` summarizing pass/fail per gate, with pointers to each report and the manuscript edits triggered.

## Not in scope here

- Q-loss decomposition by mechanism (wall viscous, PZT, bonding) — flagged out of scope by analysis contract.
- Off-center inlet distributions — flagged out of scope (but `g6d` partially documents the implication).
- 1f sweep complex fit — handled in `q1_linewidth.py` and reused.
- $G_n$ extraction beyond $n = 2$ — data inventory does not support it, by Gate 0 conventions.
- Single-`(y_i, y_f)` trajectory pass/fail test — *intentionally removed*; the κ-throughput interpretation is established analytically from the centered contraction criterion, not numerically from one chosen endpoint pair.
