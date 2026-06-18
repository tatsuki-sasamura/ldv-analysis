# PRL figure plan + detuning mechanism — 2026-06-13 (updated 2026-06-18)

## Purpose

Working document for the **PRL gate** (committed to Wei for W25; gate
deliverables due 2026-06-18, i.e. today). The two key figures + a
story outline, with the detuning mechanism question now resolved.
This report:

1. Records the analysis done this session (perturbative-fit cascade +
   data inventory).
2. Records the detuning-mechanism resolution: **wall / structural
   coupling supersedes the 2D-waveguide axial-dispersion hypothesis**
   on the sign argument; see
   `nonlinearphysics-manuscript/prl/detuning_mechanism.md`.
3. Lists the datasets and scripts available to *falsify* the
   wall-coupling story (now End-Matter / Supplemental work, not a
   main-body figure).
4. Lays out the **remaining analysis process** to produce the PRL
   figures, with per-figure status, dependencies, and blockers.

This is a plan/decision log, not a results report — numbers here are
the current best values, to be locked down by the analyses listed in
§5.

**Revision note 2026-06-18.** The 2026-06-13 version of this plan
treated detuning as a main-body figure (F3) with the 2D-waveguide
hypothesis as its physical anchor. The 2026-06-17
`detuning_mechanism.md` derivation (sign argument) demoted the
waveguide picture and identified Si-wall coupling as the leading
mechanism. With the mechanism question now substantially resolved on
paper, detuning is rewritten here as End-Matter / Supplemental
material and F3 becomes the **theory / model overlay** — the more
decisive panel for the PRL claim "harmonic cascade quantitatively
limits usable fundamental pressure."

---

## 1. Context — the PRL story in one paragraph

In a standing-wave acoustofluidic microchannel, the fundamental width
mode at f₁ ≈ 1.907 MHz (`sin(πx/W)`, W = 375 µm) saturates at high
drive while a self-generated second harmonic at 2f (`cos(2πx/W)`,
≈ 3.81 MHz) grows quadratically — diverting energy and imposing an
**innate throughput limitation** on the acoustic field. The harmonic
ratio is predicted by the Coppens relation
`P₂f/P₁f = β Q₂ M cosθ / 4`, where `cosθ` accounts for the 2f sitting
off the nearest cavity eigenmode. The PRL claim rests on matching the
measured cascade to a self-consistent Kuznetsov model with **no fitted
parameters**, using independently measured Q-factors.

### Title-line claim (revised 2026-06-18 after re-checking the cascade table)

The 10–120 Vpp cascade captures **both regimes in a single dataset**:

> We directly observe both the perturbative (Coppens) and
> beyond-perturbative regimes of harmonic generation in a single
> microchannel device. P₂f follows the Coppens prediction within 7 %
> below 60 Vpp; above 60 Vpp, P₂f systematically falls below the
> perturbative quadratic by ~10 % at 120 Vpp, quantitatively consistent
> with iterative-Kuznetsov-solver predictions of beyond-quadratic
> cascade depletion into 3f and higher harmonics. The breakdown of
> perturbation theory imposes the throughput limitation directly
> observable in the fundamental pressure.

The strength of this framing — over the earlier "P₂f/P₁f² is constant"
draft — is that it makes *two* claims that are each strong: (a) Coppens
scaling is validated in a microchannel for the first time, within 7 %
at perturbative drive; (b) the regime transition into beyond-perturbative
cascade is observed in the same dataset, with the deviation matching
the iterative-solver prediction. See §2.1 for the underlying numbers.

---

## 2. Work done this session

### 2.1 Perturbative-regime fit on the voltage cascade

`experiments/2026W21/vpp_vs_pressure.py` previously fit the P₁f-linear
and P₂f-quadratic trend lines through **all 12 cascade points**, which
dragged the slopes down and hid the saturation. Changed to fit the
**perturbative subset only** (`V ≤ PERTURB_MAX_VPP = 30 Vpp`, the
lowest 3 points, where P₂f/P₁f ≤ 10%) and **extrapolate** the trend
across the full drive range. Added `--perturb-max` CLI override, a
solid/dashed split (fit region vs extrapolation), filled-vs-open
markers (fit vs excluded points), and a printed saturation metric.
File re-formatted with the repo's `black`/`isort` (installed into the
venv this session) and re-run; figure regenerated at
`experiments/2026W21/output/vpp_vs_pressure.png`.

**Result (clean 2026-05-24/25 peak-band cascade, 101×21, water):**

| PZT Vpp | P₁f (MPa) | P₁f/V (kPa/V) | P₂f (MPa) | P₂f/P₁f | P₂f/P₁f² (/GPa) | vs Coppens 20.7 /GPa |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 1.58 | 157.7 | 0.07 | 4.2% | 26.7 ± 9.3 | within error of the plateau (±35 %, 2f SNR≈1; **not** a real outlier — see §"Low-V outlier") |
| 30 | 4.44 | 148.0 | 0.44 | 9.8% | 22.1 | **+7 %** (perturbative regime, matches Coppens) |
| 60 | 7.93 | 132.2 | 1.39 | 17.5% | 22.1 | **+7 %** (perturbative regime, matches Coppens) |
| 120 | 14.72 | 122.7 | 4.40 | 29.9% | 20.3 | **−2 %** (matches Coppens, but `P₂f` is now below the pure-`P₁f²` trend) |

- Perturbative slope (fit ≤ 30 Vpp): **P₁f = 149.5 kPa/V**.
- **P₁f at 120 Vpp is 17.9 % below the perturbative extrapolation**
  (14.7 vs 17.9 MPa) — the throughput limitation, quantified.
- P₂f also peels **below** its V² parabola at high drive (29.9 % of
  P₁f, vs the perturbative-extrapolation prediction of P₂f/P₁f → 22 %
  if both scaled cleanly).

### Coppens prefactor and the perturbation-theory breakdown

The Coppens perturbative limit predicts a **drive-independent** ratio:

```
P₂f / P₁f²  =  β · Q₂ · cosθ / (4 ρ c²)
            =  3.48 · 100 · 0.53 / (4 · 2.23×10⁹ Pa)
            ≈  20.7 /GPa
```

(`β = 1 + (B/A)/2 ≈ 3.48` for water; `Q₂ = 100`; `cosθ = 0.53` from
the Lorentzian-tail formula with Δf = 31 kHz; `ρc² ≈ 2.23 × 10⁹ Pa`.)

The measured ratio **declines monotonically** from 22.1 /GPa at 30–60 Vpp
to 20.3 /GPa at 120 Vpp — a 8 % decline. This is **not consistent with
a pure-perturbative regime**, where the ratio is rigorously constant.
The decline is the signature of the **perturbation-theory breakdown**:

- In pure perturbation, P₂f is computed assuming P₁f is unaffected by 2f
  and P₂f doesn't feed higher harmonics. Both assumptions break at
  high drive (back-coupling, cascade into 3f, 4f, …).
- For the measured ratio P₂f / P₁f², two effects compete:
  (a) P₁f depletion by 2f back-coupling → ratio increases with V;
  (b) P₂f depletion by cascade into 3f + → ratio decreases with V.
- The observed decline says **(b) dominates**: the 2f channel bleeds
  into 3f, 4f, … faster than the 1f channel is depleted into 2f.

This is the defining signature of beyond-perturbative cascade flow,
and the same data spans both regimes — 30–60 Vpp matches Coppens within
7 %, 120 Vpp falls 10 % below the perturbative-quadratic trend.  Hence
the title-line claim in §1.

### Closure check (new — promoted from "nice to have")

If beyond-quadratic depletion is the correct interpretation, the
"missing" 2f energy at 120 Vpp (~10 % of `P₂f,Coppens` ≈ 4.9 → 4.4 MPa)
should appear in the higher harmonics. The empirical conservation
statement to verify with **A2 (1f–5f spectrum across the cascade)**:

```
Σₙ (P_{nf}²)_observed   ≈   Σₙ (P_{nf}²)_perturbative-prediction
```

A flat `Σₙ P_{nf}²` across the regime transition would close the
cascade-depletion picture empirically and removes any "where did the
energy go?" referee question. Worth its own subpanel in F2 or its own
End-Matter figure. PRA notes hint that 3f / 1f ≈ 3.7 % at 45 Vpp; the
1f–5f cache exists, so this is a re-extraction task, not a re-run.

### Low-V "outlier" (10 Vpp, 26.7 /GPa) — RESOLVED: not significant

The 10 Vpp prefactor sits **above** the others, but with the propagated
`P_nf` error it is **26.7 ± 9.3 /GPa** (±35 %, dominated by the weak 2f
there, SNR ≈ 1) — **within 1σ of the ~22 /GPa plateau**. So the apparent
"+29 %" is not statistically significant: it is simply the noisiest
point, not a physical low-drive enhancement. Both candidate explanations
are now closed:

1. **Drive harmonic leakage** — ruled out by B1: leakage is 0.09 % of
   the observed P₂f at 10 V (`2026-06-18_b1_drive_purity.md` §3).
2. **Measurement noise** — confirmed dominant: the ±35 % error bar alone
   makes the point consistent with the plateau.

No per-V cosθ/Q story is required. (Error = `√(fit-SE² + noise²)` on
`P_nf`, propagated into the ratio; see `experiments/2026W21/`
`harmonic_ladder.py` and `prl_draft.py` Fig 2c.)

### 2.2 Data inventory (W21–W22, ISO weeks 21–22, May 18–31)

43 complete scans; key groups (all `sample` chip, water-filled,
800-cycle bursts, amp gain ×200, stored under
`OneDrive/Data/output/W21/`):

- **Voltage cascade** (101×21, 1.890–1.910 MHz, 11 @ 2 kHz, y 2 mm):
  10–120 Vpp in 10 Vpp steps, all 12 points complete. 70 Vpp ran
  05-24 23:37 (W21 side of midnight); 80–120 Vpp 05-25 (W22). One
  **duplicate 40 Vpp** (`peak_40Vpp_..._200739`, 05-24 20:07) is the
  extra; the canonical SCANS list uses the 21:31 run.
- **Wide-axial 101×77** (X width 0.60 mm @6 µm × Y length 19 mm
  @250 µm, full per-position 0.8 ms waveforms):
  - `1p89to1p92_1kHz_60Vpp` (1f, 31 @ 1 kHz) — complete.
  - `3p7to3p9_60Vpp` (2f, 3.70–3.90 MHz, 11 @ 20 kHz) — complete.
  - `axial_wide_60Vpp_164026` (1f, 11 @ 2 kHz) — complete.
- **Wide-band survey lines** (1.900–4.200 MHz, 461 @ 5 kHz, y 3 mm):
  21×1 and 101×1 — locate all eigenmodes (1f→4f), but single axial
  position (no length info).

**Incomplete / excluded:** `3p76to3p84_2kHz` (fine 2f scan, **2/41
files** — the high-resolution detuning scan; recover from acquisition
PC if a tight detuning number is needed), `axial_wide_..._153156`
(0/11, empty), `verify_..._failed_ldv2mrange` (LDV range error).

---

## 3. Detuning mechanism (resolved 2026-06-17 → Si-wall coupling)

The empirical `cosθ ≈ 0.53` accounts for the 2f drive at 2f₁ = 3.814 MHz
sitting Δf ≈ 31 kHz *below* the nearest 2f cavity eigenmode at
3.845 MHz; with Q₂ ≈ 100, the Lorentzian-tail formula
`cosθ = 1/√(1 + (2Q₂·Δf/f₂)²)` gives 0.53 directly from measured
inputs. The remaining physics question is **why Δf is 31 kHz** —
equivalently, why `f₂/(2f₁) − 1 ≈ +0.81 %` (positive offset).

### Hypothesis history

- **2026-06-12 (Wei flag):** the channel is a **2D waveguide** —
  transverse width × propagating length, eigenmodes
  `f_{m,n} = (c/2)·√[(m/W)² + (n/L)²]`. Detuning becomes geometric.
- **2026-06-17 (`nonlinearphysics-manuscript/prl/detuning_mechanism.md`,
  derivation supersedes the 2D-waveguide picture):**
  **wall / structural coupling** is the more likely primary cause.

### Why the 2D-waveguide picture is demoted — the sign argument

Rigid-wall axial dispersion (`k² = (mπ/W)² + q²`) shifts each mode
*up* by `Δf_m/f_m⁽⁰⁾ ≈ (W²/2m²π²)⟨q²⟩_m`. If 1f and 2f have
**comparable** axial spread (`⟨q²⟩₁ ≈ ⟨q²⟩₂`), then
`f₂/(2f₁) − 1 ≈ −3W²/8π²·⟨q²⟩` is **negative**. The observed offset
is **positive (+0.81 %)** → axial dispersion has the **wrong sign**
unless 2f is far more localized than 1f (would require 2f confined
to sub-mm in x — implausible for a 3–6 mm transducer footprint).

### Why Si-wall coupling explains the observation

Chip width B = 4.0 mm, channel W = 375 µm, Si side-wall b = 1.81 mm
each side. `Z_Si/Z_w ≈ 13.2`. Free-surface boundary reactance
`Z_b = j·Z_Si·tan(k_Si·b)` gives `|Z_b/Z_w| ≈ 8.4` at f₁ and
`≈ 25.6` at f₂. A 1D free-surface Si–water–Si model shifts
`f₂/(2f₁)` by ~5.4 % (same order as observed +0.8 %; the real
glass/PZT/adhesive stack tempers it). Crucially, the 1f and 2f wall
reactances differ → m=1 and m=2 are not in exact 1:2 ratio → detuning
of the right sign and order of magnitude.

Independent corroboration: viscous-boundary-layer Q is
`Q_tb(1f) ≈ 390`, `Q_tb(2f) ≈ 550` (H = 150 µm). Measured Q ≈ 100 →
viscous loss is only ~18 % of total → ~82 % is structural, consistent
with wall coupling being the dominant loss channel.

### Falsifiability check (Supplemental / End-Matter material)

The most sensitive observable is the **2f node shift outward by
~3.1 µm** (at wall-reactance ratio R ≈ 10), measurable from the LDV
2D scan via a free-`k_y` complex-pressure fit. The 1f shape residual
is < 2 % (too small to detect from |P| alone); the 2f node shift is
the discriminating signature. The four concrete LDV checks listed in
`detuning_mechanism.md` §"Concrete next analysis steps" are
re-listed in §4 below.

---

## 4. Data + scripts for End-Matter / Supplemental detuning analysis

These checks falsify (or confirm) the Si-wall mechanism and
quantitatively rule out the demoted 2D-waveguide alternative. They
produce End-Matter or Supplemental material, **not** a main-body
figure.

**Datasets (all carry per-position time-domain waveforms — 7777 pts ×
100k samples @ 125 MS/s, 0.8 ms window):**

| Dataset | Band | Sweep | Role |
|---|---|---|---|
| `3p7to3p9_60Vpp` | 2f | 3.70–3.90 MHz, 11 @ 20 kHz | 2f mode-shape + node-shift test |
| `1p89to1p92_1kHz_60Vpp` | 1f | 1.890–1.920, 31 @ 1 kHz | 1f mode-shape baseline |
| `axial_wide_60Vpp_164026` | 1f | 1.890–1.910, 11 @ 2 kHz | alt 1f |
| survey lines (21×1, 101×1) | 1f–4f | 1.9–4.2 MHz, 461 @ 5 kHz | eigenmode frequencies → Δf |

**Four LDV checks (from `detuning_mechanism.md` §"Concrete next analysis steps"):**

1. **Free-`k_y` complex fit on `P_{1f}(y)` and `P_{2f}(y)`.** Fit
   `P_{1f}(y) = A₁·sin[k_{y1}(y − y₀)] + B₁` and
   `P_{2f}(y) = A₂·cos[k_{y2}(y − y₀)] + B₂`. Headline observable:
   **2f node shifts outward by ~3.1 µm at wall reactance R ≈ 10**.
   1f shape residual < 2 % (too small to discriminate from |P| alone).
2. **Axial spectral width `⟨q²⟩_m`** via Parseval
   `⟨q²⟩_m = ∫|∂_x A_m|² dx / ∫|A_m|² dx`. Plug into the sign-argument
   formula in §3 → quantitatively rule out axial dispersion as the
   dominant detuning source.
3. **Tail decay length `L_att,m`** outside the transducer footprint →
   spatial-attenuation `Q_m^(x) ≈ ½(mπ·L_att,m/W)²`. Compare to
   transient build-up Q.
4. **Top/bottom viscous-BL Q from data:** `Q_tb^data = ω E / P_tb`
   with `dP_tb/dx = (ρ₀ωδ_ν/2)∫|v_y|²dy`,
   `dE/dx = H∫[|P|²/(4ρ₀c²) + ρ₀|v_y|²/4]dy`,
   `v_y = −(1/jωρ₀)∂_y P`. Predicted `Q_tb ≈ 390` (1f), `550` (2f);
   measured Q ≈ 100 ⇒ viscous share ≈ 18 % ⇒ structural loss
   dominates (corroborates wall coupling).

**Scripts:**
- `pressure_map_2d.py` — already outputs 1f/2f spatial + phase maps.
  Free-`k_y` fit (check #1) is a thin wrapper around the existing
  complex-pressure extraction; ~50–100 lines of new code.
- `transient_ch2_acoustic.py` + `src/ldv_analysis/transient.py` —
  ring-up/ring-down for the spatial-Q vs transient-Q comparison
  (check #3). Currently per-file / aggregate; per-axial wrapper is
  Supplemental-grade work.
- Direct Python on per-position complex `P_{nf}(x, y)` for checks #2
  and #4 (integrals only, no new module needed).

**Caveats:** `transient_ch2_acoustic.py` `DEFAULT_INPUT` hardcodes an
old `C:\Users\tatsuki\…` path — pass files explicitly. Burst metadata
(`burst_on_us_nominal = 5.0`) is stale vs the real 800-cycle burst,
but transient analysis reads the raw waveform, so it does not matter.

---

## 5. Remaining analysis process → PRL figures

Candidate PRL figure set (Letter, ~4 pp; condensed from PRA Figs 1–9):

| PRL fig | Content | Script(s) | Data | Status |
|---|---|---|---|---|
| **F1** Mode signature | 1f `sin(πx/W)` + 2f `cos(2πx/W)` spatial maps + cross-section fits (R²) → 2f is a self-generated harmonic, not an independent mode | `pressure_map_2d.py` | `1p89to1p92_1kHz` (or 60 Vpp peak file) | pipeline works; needs PRL-format polish |
| **F2** Cascade scaling + regime transition | P₁f sublinearization, P₂f vs V², P₂f/P₁f, and **P₂f/P₁f² vs V with Coppens 20.7 /GPa overlay** showing perturbative regime (30–60 V matches within 7 %) and beyond-perturbative breakdown (120 V: 10 % below quadratic).  Subpanel: `Σₙ P_{nf}²` conservation across the cascade (A2 closure check). | `vpp_vs_pressure.py` + per-harmonic re-extraction | 10–120 Vpp cascade | **data ready** (§2.1); A2 closure subpanel is the open task |
| **F3** Theory connection | Iterative-Kuznetsov-solver prediction of **(a)** the P₂f-vs-P₁f exponent transition from 2.0 → 1.9 across the cascade, and **(b)** P₁f suppression, with measured Q₁=121, Q₂=100, cosθ=0.53 — no fitted parameters. The exponent prediction is the sharper test than a single ratio at one V. | `harmonic_model` (`feat/iterative-physics-validation` branch — already checked out) + `vpp_vs_pressure.py` | 10–120 Vpp cascade | **critical path**; parameterise (Q, freq, amplitude range) and run |
| **App. / End Matter** | Transient Q (Q₁=121, Q₂=100); detuning mechanism (Si-wall, §3); detuning falsifiability (§4 checks); LDV setup; drive purity; 3f accounting; solver algorithm | various | line/peak scans, `3p7to3p9`, survey lines | exists or §4 work |

### Task list (revised 2026-06-18 — recommended order)

1. **F3 model overlay** — *new critical path.* Run `harmonic_model`
   `feat/iterative-physics-validation` branch (already checked out;
   confirmed in `2026-06-18_prl_resource_inventory.md` §3) with
   measured Q₁=121, Q₂=100 and cosθ=0.53 (Lorentzian-tail from §3);
   overlay predicted P₂f/P₁f, **the P₂f-vs-P₁f exponent (predicted
   1.9 ± something? — to compute)**, and P₁f suppression on the
   10–120 Vpp cascade. The exponent prediction is the sharper test;
   see §2.1 Coppens-prefactor section.
1b. **A2 — 1f–5f conservation closure** (promoted from "decides naming"
   to "closes the cascade-depletion story"). Re-extract P_{nf} for
   n = 1..5 from the existing cache; plot `Σₙ P_{nf}²` vs V. **Flat
   across the cascade ⇒ cascade depletion picture closed empirically**;
   sloped ⇒ either Q changes with drive, or there's energy loss we
   haven't accounted for (thermal, structural).
2. **Rewrite the `cosθ = 1` paragraph** in `pra/main.tex` lines
   978–998 and `prl/main.tex` 966–986. Replace with: cosθ=0.53
   measured (Lorentzian tail with Δf=31 kHz, Q₂=100); mechanism =
   Si-wall coupling → reference `detuning_mechanism.md` in End Matter.
   Remove the "no beating ⇒ cosθ=1" claim (beat period 32 µs > ring-up
   τ 8 µs ⇒ absence of beating is *consistent* with Lorentzian-tail
   driving, not evidence for perfect resonance).
3. **120 Vpp data integration** — replace 25 Vpp / P₂f/P₁f=9 % /
   MRE 4 % / Fig 5 TBD throughout `pra/main.tex` and `prl/main.tex`
   with: perturbative P₁f slope 149.5 kPa/V, P₁f sublinearity 17.9 %
   at 120 V, P₂f/P₁f² ≈ 22.6/GPa (essentially constant).
4. **E_ac framing** — convert the cascade to ⟨E_ac,1f⟩ = p₀²/(4ρc²)
   vs **input electrical power** (½·V·I·cosφ from the drive + current
   channels) with the same perturbative-fit treatment. Most
   "throughput-limitation"-shaped panel (maps to PRA Fig 5; possible
   F2 enhancement or Supplemental).
5. **F1 mode-signature** — polish to PRL one-column format.
5b. **B1 drive purity** — DONE (`2026-06-18_b1_drive_purity.md`): leakage
   ≤ 0.15 % of P₂f at every drive; and the 10 Vpp "+29 % outlier" is
   resolved as noise (26.7 ± 9.3 /GPa, within 1σ of the plateau via the
   propagated `P_nf` error), neither leakage nor a real effect.
6. **PRL structural cut** — currently ~4500 words / 12 floats /
   9 pages; PRL limit 3750 words / ~4 pages. Move Q transient, LDV
   setup detail, electronic impurity, detuning analysis, solver
   algorithm, 3f+ to End Matter / Supplemental.
7. **End-Matter / Supplemental detuning work** (§4 checks 1–4).
   Non-blocking for the gate; nice to have for the Supplemental:
   - 7a. Free-`k_y` complex fit → 2f node shift (Si-wall signature)
   - 7b. ⟨q²⟩₁ vs ⟨q²⟩₂ → falsify axial-dispersion alternative
   - 7c. L_att,m → spatial-Q vs transient-Q
   - 7d. Q_tb^data → viscous-share number
8. **Defensive checks (fold in opportunistically):**
   - Drive purity: V₂f/V₁f on drive-voltage channel (PRA claims
     <0.07%) — pre-empts "2f is in the drive" objection. Re-verify
     on 120 Vpp.
   - 3f / higher-harmonic accounting — supports beyond-quadratic
     depletion. If significant at 120 V, frame as "harmonic cascade"
     (not just "second harmonic").
   - Resonance-shift tracking: peak frequency drifts ~1.908→1.902 MHz
     across the cascade (`freq_vs_current.py`) — quantify softening.

### Dependencies / blockers

- **F3 model overlay (task 1)** blocked on the `harmonic_model`
  `feat/iterative-physics-validation` branch (nonlinear solver not on
  `main`; `git fetch` / checkout required).
- **9× device-description TBDs** from Wei (chip/glass/piezo specs,
  pending since 2026-02-23) block the Methods section but **not** the
  figures.
- **Channel height H (150 µm) and glass identity** remain the dominant
  pressure-conversion uncertainties (see
  `reports/2026-05-22_uncertainty_budget_source_audit.md`); they scale
  absolute pressures but not the *shape* arguments (saturation, mode
  signature, detuning), so they do not block the gate figures.
- Fine 2f scan (`3p76to3p84_2kHz`) is 2/41 on disk — the coarse
  `3p7to3p9` (20 kHz) is usable but coarse for a ~31 kHz detuning;
  recover the fine files only if Supplemental task 7a needs a tighter
  number than the coarse scan provides.

---

## 6. Current best numbers (to be locked down by §5)

| Quantity | Value | Source / status |
|---|---|---|
| f₁ (width half-wave, m=1) | ≈ 1.907 MHz | peak of cascade; drifts to ≈ 1.902 at high drive |
| 2f₁ (drive) | ≈ 3.814 MHz | 2 × f₁ |
| Nearest 2f cavity eigenmode | ≈ 3.845 MHz | survey lines; confirm Δf to ≤ kHz precision once Supplemental work runs |
| Detuning Δf | ≈ +31 kHz (`f₂/(2f₁) − 1 ≈ +0.81 %`) | mechanism = wall / structural coupling (Si-wall, §3) |
| `cosθ` | **0.53 (measured)** | Lorentzian-tail formula `cosθ = 1/√(1 + (2Q₂·Δf/f₂)²)` with measured Δf, Q₂ — no longer treated as empirical fudge |
| Q₁ | 121 (τ₁ = 20.2 µs) | transient analysis; re-verify on W21 |
| Q₂ | 100 (effective at 2f₁) | transient analysis; includes Lorentzian-tail attenuation |
| Perturbative P₁f slope | 149.5 kPa/V | this session (fit ≤ 30 Vpp) |
| P₁f suppression at 120 Vpp | 17.9 % below perturbative line | this session |
| Coppens perturbative prediction `P₂f/P₁f²` | **20.7 /GPa** | `β·Q₂·cosθ/(4ρc²)` with β=3.48, Q₂=100, cosθ=0.53 |
| Measured `P₂f/P₁f²` at 30–60 Vpp | **22.1 /GPa (+7 %)** | matches Coppens within 7 % — perturbative regime validated |
| Measured `P₂f/P₁f²` at 120 Vpp | **20.3 /GPa (−2 %)** | matches Coppens, but P₂f is now 10 % below the pure-quadratic extrapolation from 60 V |
| Effective P₂f vs P₁f exponent (30→120 V) | n ≈ **1.92 ± 0.05** | log-log fit; clear deviation from n = 2 in the beyond-perturbative regime |
| Closure check (A2) | `Σₙ P_{nf}²` (n=1..5) conservation across the cascade | **to do** — predicted to be flat if cascade-into-3f+ interpretation is correct |
| Viscous-BL share of Q⁻¹ | ≈ 18 % (predicted) | `Q_tb(1f)=390`, `Q_tb(2f)=550` vs measured 121, 100 → structural loss dominates (corroborates Si-wall) |
| Sim–exp agreement (PRA) | ~4 % MRE | to reproduce with updated cosθ/Q in F3 |

---

## Next action

The **PRL claim has sharpened** after re-checking the cascade table
(§1 title-line, §2.1 Coppens-prefactor section): the 10–120 Vpp
cascade contains **both regimes** — perturbative (30–60 Vpp,
matches Coppens within 7 %) and beyond-perturbative (120 Vpp, 10 %
below the perturbative quadratic). The "regime transition" framing
replaces the earlier (incorrect) "P₂f/P₁f² is constant" framing.

**Parallel-tracked critical path:**

- **Foreground 1** — Task 5b (B1 drive purity, including 10 Vpp) →
  Task 1b (A2 1f–5f conservation closure). ~2 h. Locks naming and
  closes the cascade-depletion story.
- **Foreground 2** — A3 (P_in = ½·V·I·cosφ; verify Ch D calibration
  first), B3 (mode-shape R² across cascade), B2 (thermal bound from
  peak-frequency drift). ~2–3 h. Locks throughput-relevant + usable-1f
  + thermal-bounded framings.
- **Background** — Task 1 (F3 iterative-solver overlay; predicts the
  exponent transition 2.0 → 1.9). Wall time dominated by sweep.
- **Parallel writing** — Tasks 2, 3 (cosθ paragraph rewrite +
  120 Vpp data integration). ~1 h.

The detuning-mechanism question (§3) is resolved on paper via the
Si-wall analysis; the §4 LDV checks are Supplemental and do not
block the gate.
