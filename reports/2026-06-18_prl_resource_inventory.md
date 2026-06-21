# PRL resource inventory — data, acquisition, simulation — 2026-06-18

Inventory of what is on disk and what each tool can produce, scoped to
the PRL gate analyses listed in
`reports/2026-06-13_prl_figure_plan_and_waveguide_detuning.md`.

No HDF5 contents were opened to build this; only directory listings
and YAML protocols.

---

## 1. W21 LDV data (`D:\OneDrive - Lund University\Data\output\W21`)

Three main data families plus assorted diagnostics. All are 800-cycle
bursts, water-filled, BNC splitter removed, drive_voltage_vpp attribute
gives true AFG output that × amp_gain (~×200) → PZT V.

### 1.1 Voltage cascade — peak band (101×21 area, 1.890–1.910 MHz @ 2 kHz, 11 freqs)

The headline PRL dataset. 12 cascade points from 10 to 120 Vpp at PZT.

| Drive (Vpp) | Run | Status |
|---:|---|---|
| 10 | `sample_101x21_fsweep_peak_10Vpp_20260524_182813` | ✓ |
| 20 | `sample_101x21_fsweep_peak_20Vpp_20260524_185622_failed_air` | ❌ |
| 20 | `sample_101x21_fsweep_peak_20Vpp_20260524_203544` | ✓ canonical |
| 30 | `sample_101x21_fsweep_peak_30Vpp_20260524_192431_failed_air` | ❌ |
| 30 | `sample_101x21_fsweep_peak_30Vpp_20260524_210338` | ✓ |
| 40 | `sample_101x21_fsweep_peak_40Vpp_20260524_195225_failed_air` | ❌ |
| 40 | `sample_101x21_fsweep_peak_40Vpp_20260524_200739` | ✓ extra |
| 40 | `sample_101x21_fsweep_peak_40Vpp_20260524_213135` | ✓ canonical |
| 50 | `sample_101x21_fsweep_peak_50Vpp_20260524_215928` | ✓ |
| 60 | `sample_101x21_fsweep_peak_60Vpp_20260524_222721_failed_bubble` | ❌ |
| 60 | `sample_101x21_fsweep_peak_60Vpp_20260524_223919` | ✓ canonical |
| 60 | `sample_101x21_fsweep_peak_60Vpp_20260526_155045` | ✓ extra |
| 60 | `sample_101x21_fsweep_peak_60Vpp_20260526_182949` | ✓ extra |
| 70 | `sample_101x21_fsweep_peak_70Vpp_20260524_233721` | ✓ |
| 80 | `sample_101x21_fsweep_peak_80Vpp_20260525_000520` | ✓ |
| 90 | `sample_101x21_fsweep_peak_90Vpp_20260525_003315` | ✓ |
| 100 | `sample_101x21_fsweep_peak_100Vpp_20260525_010113` | ✓ |
| 110 | `sample_101x21_fsweep_peak_110Vpp_20260525_012912` | ✓ |
| 120 | `sample_101x21_fsweep_peak_120Vpp_20260525_020136` | ✓ |

Per scan: 101 × 21 × 11 = 23,331 per-position waveforms. Channels A
drive_voltage, B ldv_output, C ldv_displacement, D current
(100k samples @ 125 MS/s, 0.8 ms window per capture).

### 1.2 Wide axial scans (101 × 77, 60 Vpp) — Si-wall / detuning data

Full per-position waveforms over 0.60 mm × 19 mm; the only datasets
that resolve axial structure long enough for the mechanism tests.

| Run | Band | Sweep | Coverage |
|---|---|---|---|
| `sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237` | **1f** | 1.890–1.920 MHz, 31 @ 1 kHz | 1f mode-shape + axial ⟨q²⟩₁ |
| `sample_101x77_fsweep_3p7to3p9_60Vpp_20260530_013206` | **2f** | 3.70–3.90 MHz, 11 @ 20 kHz | 2f mode-shape + node-shift + ⟨q²⟩₂ |
| `sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_20260530_072344` | **2f fine** | 3.76–3.84 MHz, 41 @ 2 kHz | tight Δf measurement; **may be incomplete** (2/41 noted in plan §4) |
| `sample_101x77_fsweep_axial_wide_60Vpp_20260526_153156` | 1f | 1.890–1.910, 11 @ 2 kHz | empty per plan §4 |
| `sample_101x77_fsweep_axial_wide_60Vpp_20260526_164026` | 1f | 1.890–1.910, 11 @ 2 kHz | ✓ usable alt 1f |

### 1.3 Wide-band survey lines — eigenmode location (Δf, cosθ inputs)

Single Y row, broad freq coverage, **single axial position so no length info**.

| Run | Grid | Sweep | Use |
|---|---|---|---|
| `sample_21x1_fsweep_survey_60Vpp_20260530_001834` | 21 × 1 | 1.9–4.2 MHz, 461 @ 5 kHz | pin all 1f/2f/3f/4f eigenmodes |
| `sample_101x1_fsweep_survey_60Vpp_20260530_004439` | 101 × 1 | 1.9–4.2 MHz, 461 @ 5 kHz | finer-x version |

### 1.4 Verify / sanity-check lines (single-Y reproductions)

| Run | Use |
|---|---|
| `verify_peak60_line_y3_20260526_153834` | rig sanity: middle row of the 101×21 peak grid at Y=3.0 mm, 60 Vpp |
| `verify_peak60_line_y3_20260530_000443` | repeat of above |
| `verify_peak60_line_y3_20260529_231455_failed_ldv2mrange` | ❌ |

### 1.5 Older series (W21 first night, May 18 — superseded but cached)

These are the datasets the old `resonance_survey.py` consumes for the
W10/W16/W21 cross-mount drift comparison. **They were taken WITH the
BNC splitter** (×2 mislabel) and are not used for the cascade story.

- Wide ladder 10–50 Vpp (`sample_101x1_fsweep_<N>Vpp_20260518_*`, 141 freqs 1.860–2.000 MHz)
- Narrow ladder 10–60 Vpp (`sample_101x1_fsweep_narrow_<N>Vpp_20260518_*`, 41 freqs 1.880–1.920 MHz)
- One pre-flush failed run (`*_failed_lowH2O`)
- 11×1 line, single 30 Vpp 101×21 scan, smoke tests, 11×11 0.5×6 mm grid

### 1.6 Misc

- `sample_101x21_fsweep_narrow_10Vpp_20260524_*` — two 101×21 narrow-band runs (re-mount era pre-cascade)
- `sample_101x51_fsweep_coarse_10Vpp_20260524_132528` — 101×51 coarse survey
- `sample_101x1_fsweep_coarse_10Vpp_20260524_130731` — line coarse survey
- `RUN_COMMAND.txt` — operator note: every scan was run with the same
  `run_protocol.py … --require-rssi --afg-preflight-policy first_point_each_condition`
  CLI from `C:\Users\tatsuki\Documents\ldv-daq`.

---

## 2. Acquisition repo (`C:\Users\tatsu\Documents\ldv-daq`)

Measurement-side acquisition. **Produces HDF5 v2** that ldv-analysis
consumes unchanged. Hardware drivers for AFG (Tek AFG3022B), Pico
(5442D), XY stage (GSC02C), Z focus (Thorlabs MLJ150), RSSI (NI
USB-6009). 40+ protocol YAMLs in `protocols/` covering every dataset
in §1.

### Standard scan recipe (peak-band cascade as canonical example)

From `protocols/sample_101x21_fsweep_peak_60Vpp.yaml`:

```yaml
scan: pattern=raster, nx=101, ny=21, x_step_um=5, y_step_um=100, settle_s=0.05
conditions: 11 freqs 1.890-1.910 MHz @ 2 kHz, voltage_vpp=0.30 (= 60 Vpp at PZT via gain x200)
burst: 800 cycles, trigger=BUS, burst_on/off=5/525 us nominal, duty=1%
pico: 125 MS/s, 100k samples (0.8 ms), trigger=external, threshold 1.6 V
channel_map: A=drive_voltage, B=ldv_output, C=ldv_displacement, D=current
```

### Capability matrix — what new acquisitions are *possible* on demand

| Capability | Existing protocol(s) | Repurpose for PRL analysis |
|---|---|---|
| Cascade voltage step (peak band, 101×21) | 12 yamls 10–120 Vpp | already complete |
| Fine 2f scan (101×77 @ 2 kHz) | `sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp` | recover fine 2f for tight Δf (§1.2 noted incomplete) |
| Wide-band survey | `sample_21x1_fsweep_survey_60Vpp.yaml` etc. | rescan if anything looks off |
| Single-Y sanity | `verify_peak60_line_y3.yaml` | rapid re-check after rig touches |
| Narrow ladders | 7 yamls (narrow_10–80 Vpp) | post-splitter calibration; already done |

### Scripts available out of the box

`scripts/run_protocol.py` (main acquisition entry), plus smoke tests
for each device (`smoke_afg3022b.py`, `smoke_pico5442d.py`,
`smoke_stage_gsc02c.py`, `smoke_rssi_niusb6009.py`,
`smoke_zstage_thorlabs.py`, `smoke_one_point_hdf5.py`,
`smoke_install_check.py`).

---

## 3. Simulation repo (`/home/tatsuki/harmonic_model` in WSL)

1D acoustophoresis simulation library. Currently on
**`feat/iterative-physics-validation`** branch (exactly the branch the
2026-06-13 plan needs for F3 model overlay; no `git fetch / checkout`
needed — already there).

### 3.1 Capability summary

| Capability | Status | Where |
|---|---|---|
| 1D linear Kuznetsov solver (Q-factor damping) | ✓ main / v0.3.x | `acoustics/linear/` |
| **1D nonlinear iterative Kuznetsov solver** | ✓ this branch | `solver/kuznetsov.py` → `Kuznetsov1DSolver`, `KuznetsovSolution` |
| Multi-modal phasor field | ✓ | `field/multimodal.py`, `field/phasor.py` |
| Multi-harmonic decomposition | ✓ | `update_multiharmonic_golden.py` golden tests; iterative example dumps each harmonic per iteration |
| Acoustophoresis trajectory integration (Gorkov + Stokes) | ✓ | `acoustophoresis/{force,velocity,trajectory,processor,properties,constants}` (incl. PS particle preset) |
| Verifiers (wave eq, Euler, continuity, BC) | ✓ | `verification/` |
| 2D / 3D solver | ✗ planned v0.6 | — |
| Impedance / absorbing BC | ✗ velocity BC only | — |
| Broadband / pulse | ✗ steady-state only | — |

### 3.2 Runnable nonlinear examples

In `example/nonlinear/`:

| Example | Script | Purpose |
|---|---|---|
| **iterative_physics_validation** | `iterative_physics_validation.py` | Sweeps boundary velocity across `MAX_ENERGY_TARGET = 2000 J/m³` (Mach number range); compares perturbation (n_iter=1) vs self-consistent (n_iter>1); reports P₂/P₁, P₃/P₁; benchmarks against Gorkov stability limit P₂/P₁ = βQM/4 = 0.5 |
| kuznetsov_validation | `kuznetsov_validation.py` | Pure Kuznetsov-equation validation |
| iterative_convergence_optimization | `convergence_optimization.py` + parallel utils | Performance/convergence analysis as n_iter increases |

Default constants in `iterative_physics_validation.py`:
- W = 375 µm ✓ (matches chip)
- FREQ = 2 MHz (vs measured 1.907 — easy to swap)
- Q = 400 (must be replaced with **W21-internal** Q — old Q₁ = 121, Q₂ = 100 from W10 burst transients are stale and not applicable here, see `2026-06-18_f2_eigenmode_pin.md` §3)
- B/A = `WATER_NONLINEAR_PARAM` (canonical water value)
- N_AMPLITUDES = 100, MAX_ENERGY_TARGET = 2000 J/m³

Configurable `X_AXIS_PARAMETER` ∈ {`prescribed_energy`, `true_energy`,
`piv_energy`, `boundary_velocity`} — directly maps to the PRL framing
choices (energy vs Mach vs P_1f).

### 3.3 Manuscript-figure generator

`scripts/generate_manuscript_figures.py` exists — needs reading to see
which figures it already produces vs which need adding for the PRL.

### 3.4 Open analysis plan

`iterative_solver_analysis_plan.md` is on disk in the repo root:
convergence metrics (relative field change, harmonic amplitude vs
iteration), spectral analysis, modal structure, energy cascade, B/A
and Q parameter studies. **Aligns with the PRL F3 needs.**

---

## 4. PRL analyses × resource availability

Tying the inventory back to the PRL gate analyses (per plan §5 + the
A/B/C analysis list from the prior session synthesis):

| PRL analysis | Data needed | Sim needed | Available now? | Blocker |
|---|---|---|---|---|
| **F1 mode signature** (sin πx/W vs cos 2πx/W, R²) | 1 peak-band scan (any V) | – | ✓ (1.1) | format-polish only |
| **F2 cascade scaling** (P₁f sublinear + P₂f/P₁f² const) | full 12-point cascade (1.1) | – | ✓ already plotted (`vpp_vs_pressure.py`) | – |
| **F3 model overlay** (Coppens / iterative-Kuznetsov, no fitted params) | F2 data; **W21-internal Q₁, Q₂** (W10 values Q₁=121, Q₂=100 are stale — different mount); cosθ from W21-pinned f_2f = 3.794 MHz | `iterative_physics_validation.py` with `Q ← W21-internal`, `FREQ ← 1.907 MHz`, scan over measured amplitude range | ✓ branch already checked out; **blocked on W21-internal Q** | parameterise + run; overlay onto F2 axes |
| **A2** 1f–5f conservation closure (`Σₙ P_{nf}²` flat across cascade ⇒ cascade depletion picture closed) | peak-band cascade (1.1), per-harmonic | sim optional | ✓ (cached `pressure_{n}f` n=1..5) | re-port 1f–5f extension onto `sweep_fit`; **promoted from "naming check" to "closure check" — see `2026-06-13_prl_…` §2.1** |
| **A4** Q vs drive level | per-Vpp ring-up envelopes from cascade (1.1) | – | ✓ raw data in cache | wrapper around `transient.sliding_dft_envelope` |
| **A5** all eigenmodes 1f–4f | survey lines (1.3) | – | ✓ | run `sweep_fit.sweep_peaks` over wide-band lines |
| **B1** drive purity at 120 V | cascade Ch A waveforms (1.1) | – | ✓ raw data | one-line FFT on Ch1 per scan |
| **B2** peak-freq drift across cascade | `freq_vs_current.py` outputs | – | ✓ | aggregate the per-Vpp peak-freq numbers |
| **B3** 1f mode-shape R² across cascade | sweep_fit per-Vpp outputs | – | ✓ | aggregate R² over the existing fits |
| **B4** P₂f tail above f_cavity | survey lines (1.3) | – | ✓ (falls out of A5) | – |
| **C1** free-`k_y` complex fit → 2f node shift | wide-axial 2f (1.2 `3p7to3p9` or fine) | – | ✓ | new wrapper around complex P_{nf}(x,y) |
| **C2** ⟨q²⟩₁ vs ⟨q²⟩₂ via Parseval | wide-axial 1f + 2f (1.2) | – | ✓ | closed-form integral |
| **C3** L_att,m axial tail | wide-axial 1f + 2f (1.2) | – | ✓ | exponential-tail fit on existing data |
| **C4** Q_tb^data viscous share | wide-axial complex P + ∂_y P (1.2) | – | ✓ | closed-form integral; no new module |
| **E_ac vs P_in,electrical** | cascade Ch A (drive_voltage) + Ch D (current) | – | ✓ Ch A & Ch D both in cache (`voltage_1f`, `current_1f`, `phase_vi`) | compute ½·V·I·cosφ; not currently in `vpp_vs_pressure.py` |

**Net: every PRL analysis is data-available *right now*.** Nothing is
blocked on new acquisition. The simulation branch is also already
checked out.

---

## 5. Critical-path summary (for the gate, ~hours window)

In order of decreasing tone/structure leverage per hour invested:

1. **B1** (drive purity at 120 V) — half-hour, decisive defense of "self-generated cascade"
2. **A2** (3f+ at 120 V) — 1–2 h, decides "harmonic cascade" vs "second harmonic" naming throughout
3. **F3** (`iterative_physics_validation.py` re-parameterised with measured Q, freq, amplitude range; overlay onto cascade) — variable, the gate-critical "no fitted parameters" figure
4. **A4** (Q vs V) — opportunistic; supports F3's "Q as fixed input" assumption
5. **Cosθ paragraph rewrite** in `pra/main.tex` 978–998 / `prl/main.tex` 966–986 (replace cosθ=1 with cosθ=0.53 measured + Si-wall mechanism → End Matter)
6. **C1, C2** (free-`k_y` fit, ⟨q²⟩ falsification) — End Matter / Supplemental, post-gate

All five upstream items use data already on disk in §1 and tools
already in §2 / §3.

---

## 6. Notes for future use

- The wide-axial 60 Vpp datasets (§1.2) are the **only** ones that
  carry per-position waveforms across the full 19 mm channel length.
  Without them, the entire detuning-mechanism falsifiability work
  (C1–C4) is blocked.
- The cascade has multiple "extra" runs (40 V × 2, 60 V × 3): useful
  for repeatability checks, not required for the gate figures.
- The `failed_*` scans are intentionally renamed and **should not** be
  glob-included in any cascade analysis; the canonical list is the 12
  voltage points table in §1.1.
- The simulation FREQ default of 2 MHz vs the chip's 1.907 MHz is a
  small (~5 %) detuning relative to the modeled cavity — for the PRL
  overlay it should be set to 1.907 MHz exactly so cosθ stays
  self-consistent.
