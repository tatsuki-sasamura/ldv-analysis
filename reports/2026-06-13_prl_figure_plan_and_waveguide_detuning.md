# PRL figure plan + waveguide-detuning hypothesis — 2026-06-13

## Purpose

Working document for the **PRL gate** (committed to Wei for W25; gate
deliverables due ~2026-06-18): the two key figures + a story outline,
plus the **detuning check** Wei flagged on 2026-06-12. This report:

1. Records the analysis done this session (perturbative-fit cascade +
   data inventory).
2. States the **2D-waveguide detuning hypothesis** and how it changes
   the model story.
3. Lists the datasets and scripts available to test it.
4. Lays out the **remaining analysis process** to produce the PRL
   figures, with per-figure status, dependencies, and blockers.

This is a plan/decision log, not a results report — numbers here are
the current best values, to be locked down by the analyses listed in
§5.

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

| PZT Vpp | P₁f (MPa) | P₁f/V (kPa/V) | P₂f (MPa) | P₂f/P₁f | P₂f/P₁f² (/GPa) |
|--:|--:|--:|--:|--:|--:|
| 10 | 1.58 | 157.7 | 0.07 | 4.2% | 26.7 |
| 30 | 4.44 | 148.0 | 0.44 | 9.8% | 22.1 |
| 60 | 7.93 | 132.2 | 1.39 | 17.5% | 22.1 |
| 120 | 14.72 | 122.7 | 4.40 | 29.9% | 20.3 |

- Perturbative slope (fit ≤30 Vpp): **P₁f = 149.5 kPa/V**.
- **P₁f at 120 Vpp is 17.9% below the perturbative extrapolation**
  (14.7 vs 17.9 MPa) — the throughput limitation, quantified.
- P₂f also peels **below** its V² parabola at high drive, and the
  Coppens prefactor P₂f/P₁f² declines 26.7 → 20.3 /GPa from 10→120 Vpp.
  Consistent with P₂f ∝ P₁f² (and P₁f saturating) plus beyond-quadratic
  cascade depletion into 3f (3f maps reach ~2.8 MPa at 60 Vpp).

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

## 3. The 2D-waveguide detuning hypothesis (NEW — 2026-06-13)

**Current model:** treats the channel as a 1D width resonator; the 2f
detuning from exactly 2×f₁ is absorbed into an empirical `cosθ ≈ 0.5`
with no independent basis. A referee will challenge exactly this.

**Hypothesis (user, 2026-06-13):** the channel is not a 1D width
resonator but a **2D waveguide** — transverse width (the `sin/cos`
profile across W) × propagating length, with the height dimension
ignored. Eigenmodes are then indexed by a transverse order *m* **and**
an axial order *n* along the length:

> f_{m,n} = (c/2)·√[(m/W)² + (n/L)²]

The 2f transverse mode (m=2) therefore carries its own axial
wavenumber and does **not** land at exactly 2f₁. The "detuning"
becomes **geometric**, not empirical — which would replace the
`cosθ ≈ 0.5` fudge with a measured/derived quantity and materially
strengthen the PRL.

**Falsifiable predictions:**
- *Steady state:* a propagating guided mode shows a **phase ramp along
  the length (Y)**; a pure standing resonance shows piecewise-constant
  phase with π jumps. The 2f axial structure should differ from 1f.
- *Transient:* a guided mode rings up by **propagating from the source
  with a finite (near-cutoff, slow) group velocity** → ring-up
  onset/arrival time varies with axial position. A pure standing mode
  rings up in phase everywhere.

---

## 4. Data + scripts to test the hypothesis

**Datasets (all carry per-position time-domain waveforms — 7777 pts ×
100k samples @ 125 MS/s, 0.8 ms window):**

| Dataset | Band | Sweep | Role |
|---|---|---|---|
| `3p7to3p9_60Vpp` | 2f | 3.70–3.90 MHz, 11 @ 20 kHz | **primary** — 2f guided/detuned mode |
| `1p89to1p92_1kHz_60Vpp` | 1f | 1.890–1.920, 31 @ 1 kHz | 1f baseline (standing vs guided) |
| `axial_wide_60Vpp_164026` | 1f | 1.890–1.910, 11 @ 2 kHz | alt 1f |
| survey lines (21×1, 101×1) | 1f–4f | 1.9–4.2 MHz, 461 @ 5 kHz | pin eigenmode frequencies only (single y) |

**Scripts:**
- `pressure_map_2d.py` — already outputs 1f/2f **phase maps**. Fastest
  first test (no new code): 2f phase vs axial Y → ramp = propagating.
- `transient_ch2_acoustic.py` + `src/ldv_analysis/transient.py`
  (`sliding_dft_envelope`, `detect_burst`, `load_point_waveforms`,
  `tau_to_Q`) — ring-up/ring-down machinery, currently **per-file /
  aggregate**. Needs a thin wrapper to extract ring-up onset **per
  axial position** and fit a group velocity.

**Caveats:** `transient_ch2_acoustic.py` `DEFAULT_INPUT` hardcodes an
old `C:\Users\tatsuki\…` path — pass files explicitly. Burst metadata
(`burst_on_us_nominal = 5.0`) is stale vs the real 800-cycle burst, but
transient analysis reads the raw waveform, so it does not matter.

**Planned two steps:**
1. 2f phase-vs-Y map from `3p7to3p9` (existing script) — standing vs
   propagating in one look.
2. Per-axial-position ring-up timing wrapper on the same data —
   measure group velocity / arrival delay (direct transient signature).

---

## 5. Remaining analysis process → PRL figures

Candidate PRL figure set (Letter, ~4 pp; condensed from PRA Figs 1–9):

| PRL fig | Content | Script(s) | Data | Status |
|---|---|---|---|---|
| **F1** Mode signature | 1f `sin(πx/W)` + 2f `cos(2πx/W)` spatial maps + cross-section fits (R²) → 2f is a self-generated harmonic, not an independent mode | `pressure_map_2d.py` | `1p89to1p92_1kHz` (or 60 Vpp peak file) | **pipeline works**; needs PRL-format polish |
| **F2** Cascade + model | P₁f saturation (perturbative fit, extrapolated) + P₂f growth + ratio, with self-consistent model overlay | `vpp_vs_pressure.py` + `harmonic_model` | 10–120 Vpp cascade | **data side ready** (§2.1); model overlay pending (depends on detuning + Q) |
| **F3** Detuning / waveguide | Axial mode structure + transient propagation (if hypothesis holds) → geometric origin of `cosθ` | `pressure_map_2d.py` (phase) + new transient wrapper | `3p7to3p9`, `1p89to1p92_1kHz`, survey lines | **new work** (§3–4) |
| **App.** Transient Q | Ring-up/ring-down envelopes → Q₁=121, Q₂=100 | `transient_ch2_acoustic.py` | line/peak scans | exists; re-verify on W21 data |

### Task list (recommended order)

1. **Detuning / waveguide check (§4)** — *critical path, Wei-flagged.*
   - 1a. 2f phase-vs-Y map (existing script).
   - 1b. Per-axial ring-up timing wrapper → group velocity.
   - 1c. Locate all eigenmodes from survey lines (1f–4f); compute the
     2f₁-to-eigenmode detuning and the implied `cosθ`; compare to the
     empirical 0.5. If the 2D-waveguide picture holds, replace `cosθ`
     with the geometric value.
2. **Self-consistent model overlay (F2)** — run the `harmonic_model`
   Kuznetsov iterative solver with measured Q₁=121, Q₂=100 and the
   `cosθ` from step 1; overlay predicted P₂f/P₁f and E_ac roll-off on
   the cascade. *Depends on step 1.* Requires the
   `feat/iterative-physics-validation` branch of `harmonic_model`
   (nonlinear solver is not on `main`; local `main` is 40 commits
   behind — `git fetch`/checkout first).
3. **E_ac framing** — convert the cascade to ⟨E_ac,1f⟩ = p₀²/(4ρc²) vs
   **input electrical power** (½·V·I·cosφ from the drive + current
   channels) with the same perturbative-fit treatment. This is the
   most "throughput-limitation"-shaped panel (maps to PRA Fig 5).
4. **Mode-signature figure (F1)** — polish to PRL format.
5. **Defensive checks (fold in opportunistically):**
   - Drive purity: V₂f/V₁f on the drive-voltage channel (PRA claims
     <0.07%) — pre-empts "2f is in the drive" objection.
   - 3f / higher-harmonic accounting vs drive — supports the
     beyond-quadratic depletion observation.
   - Resonance-shift tracking: peak freq already drifts 1.908→1.902 MHz
     across the cascade (`freq_vs_current.py`) — quantify softening.

### Dependencies / blockers

- **Model overlay (step 2)** blocked on **step 1** (detuning) and on
  the `harmonic_model` nonlinear branch.
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
  recover the fine files if a tight number is required.

---

## 6. Current best numbers (to be locked down by §5)

| Quantity | Value | Source / status |
|---|---|---|
| f₁ (width half-wave) | ≈ 1.907 MHz | peak of cascade; drifts to 1.902 at high drive |
| 2f₁ | ≈ 3.814 MHz | 2×f₁ |
| Nearest cavity eigenmode | ≈ 3.845 MHz | to confirm from survey lines |
| Detuning | ≈ 31 kHz | to confirm; hypothesis: geometric (axial mode) |
| `cosθ` | ≈ 0.5 (empirical) | **to replace** with measured/geometric value |
| Q₁ | 121 (τ₁ = 20.2 µs) | transient analysis; re-verify on W21 |
| Q₂ | 100 (effective at 2f₁) | transient analysis; includes detuning loss |
| Perturbative P₁f slope | 149.5 kPa/V | this session (fit ≤30 Vpp) |
| P₁f suppression at 120 Vpp | 17.9% below perturbative line | this session |
| Sim–exp agreement (PRA) | ~4% MRE | to reproduce with updated cosθ/Q |

---

## Next action

On user signal (W21 data finishing local download): start **step 1a**
— 2f phase-vs-Y map from `3p7to3p9` — then the per-axial ring-up
wrapper (1b). The wrapper can be drafted before the data lands.
