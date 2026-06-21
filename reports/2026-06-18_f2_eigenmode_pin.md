# f_2f eigenmode pinning + Coppens recalibration — 2026-06-18

## Purpose

B1's coarse 2f-band measurement (20 kHz resolution) and the survey
lines disagreed by 45 kHz on the location of the 2f cavity eigenmode
(3.800 vs 3.845 MHz), which propagates to a factor of 2.7 uncertainty
in the Coppens prefactor `P_2f/P_1f² = β·Q_2·cos(θ)/(4ρc²)`.

The fine `sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_*` scan was
recovered from the acquisition PC on 2026-06-18 (41 files, 2 kHz steps
covering 3.760-3.840 MHz). This report uses it to pin the eigenmode.

Script: `experiments/2026W21/f2_eigenmode_scan.py`. Reuses
`AxialFit.p1_n2_mag` (cos(2πx/W) projection of `pressure_1f`) plus a
parabolic-peak interpolation and a Lorentzian fit on `|P|²` over a
±25 kHz window around the peak.

## Findings

### 1. f_2f = 3.794 MHz (pinned to ~1 kHz)

| Method | f_2f (MHz) | Notes |
|---|---|---|
| Parabolic-peak on T(f) | **3.7938** | three-point log-parabolic |
| Lorentzian fit on T² | **3.7942** | ±25 kHz window |

The Lorentzian FWHM is 11.4 kHz, giving Q_field = 332 from
`f/FWHM(|P|²)`. R² of the cos(2πx/W) projection is ≥ 0.9 across
3.78-3.83 MHz — clean n = 2 transverse mode.

**The single-Y survey-line value of 3.845 MHz is wrong**: a 21×1 or
101×1 line at a single axial position projects onto a mixed axial
structure (whatever ⟨q²⟩₂ happens to be at that y), so its peak does
not correspond to a single transverse eigenmode. The 101×77 area scan
explicitly projects onto cos(2πx/W) and is mode-selective. Going
forward, **f_2f = 3.794 MHz is the trusted value**, not 3.845 MHz.

### 2. There is a second 2f cavity mode at ~3.817 MHz

The full T(f) curve shows two peaks:

- Main: **3.794 MHz**, T = 12.8 kPa/V
- Secondary: **~3.817 MHz**, T ≈ 10 kPa/V

Both project cleanly onto cos(2πx/W) (R² > 0.9), so they are both
n = 2 transverse modes — likely two distinct axial structures
differing by `Δq_y` (the n = 2 ladder, `f² = (2c/W)²/4 +
(q_y c/2π)²`, gives spacing of order ~10–30 kHz for axial nodes in
the 19 mm channel). A two-mode cavity in the 2f band changes the
Coppens picture qualitatively (see §4).

The cascade fundamental drives at 2·f_1f = 2·1.902 ≈ **3.804 MHz**,
which falls **between** the two modes (~10 kHz above the main mode,
~13 kHz below the secondary). Both modes contribute to the cascade-
generated P_2f response, summed as complex Lorentzian amplitudes.

### 3. Q discrepancy: spectral Q_2 = 332 vs transient Q_2 = 100

From the FWHM Lorentzian fit on `|P_cos2|²(f)`, Q_field = f_0/FWHM =
332. From the W10 burst-mode ring-up transient measurement (used in
the PRA/PRL drafts and the figure plan), Q_2 = 100. The two should
agree to within a factor of 2 at most.

3.3× is a real disagreement. Most likely:

- **Transient Q is mode-mixed.** The ring-down envelope is whatever
  combination of the 3.794 + 3.817 modes ends up oscillating; their
  beating at Δf = 23 kHz looks like a faster decay in the envelope.
  The Coppens-relevant Q is the mode-selective spectral one — but
  only for the mode actually driven.
- **Spectral Q is single-mode-overstated.** Fitting a single
  Lorentzian to a two-mode system in the wings tightens the apparent
  FWHM (the actual lineshape has a saddle at 3.804 MHz that the
  single-Lorentzian curve doesn't reproduce).

Either way, the assumption "Q_2 = 100 enters Coppens unchanged" needs
re-examining. A mode-projected ring-down (cos(2πx/W)-projected complex
amplitude vs time on a single high-drive file) would settle this.

### 4. Single-Lorentzian Coppens with mixed-vintage inputs over-predicts the cascade by 36% — but the input mix is illegitimate

**⚠ The Q_2 = 100 below is W10 burst-derived, mixed into a W21-pinned
eigenmode geometry — illegitimate.** Recompute with W21-internal Q
before reading anything into the resulting agreement / disagreement.
For traceability of how earlier "within 7 %" claims arose, the
mixed-input number is shown:

Plugging f_2f = 3.794 MHz **(W21)**, Q_2 = 100 **(STALE: W10 burst,
do not use)**, f_1f = 1.902 MHz **(W21 cascade plateau)** into the
Lorentzian-tail cos(θ) formula:

| input | value | source |
|---|---|---|
| Δf = f_2f − 2·f_1f | **−9.8 kHz** | (3.794 − 3.804) MHz, W21 |
| cosθ | **0.888** (with stale Q) | 1/√(1+(2·100·Δf/f_2f)²) — Q is W10 stale |
| Coppens prefactor | **34.6 /GPa** (with stale Q) | β·Q_2·cosθ/(4ρc²) with β=3.48, Q_2=100 W10 |
| measured plateau (30-110 V) | **~22 /GPa** | vpp_vs_pressure cascade (W21) |

If instead we use the W21 spectral Q_field = 332 from §3 (subject to
the two-mode caveat), Coppens predicts ~115 /GPa — **5× over** measurement.
**Neither mixed-input number should be quoted as an absolute agreement
or disagreement against the cascade**; the right path is a W21-internal
mode-projected Q (action item).

**Coppens over-predicts by 36 % (with stale Q)**, not under-predicts by 7 %. The
"within 7 %" claim in the 2026-06-13 figure plan §2.1 used the
incorrect 3.845 MHz survey-line value, which gave cosθ = 0.42 and
prediction = 14 /GPa — happening to land near measurement by
coincidence (the survey-line error and the Q-or-two-mode-cavity issues
roughly cancelled).

Inverting the data for an effective `Q·cosθ` product:
`22 = β·Q·cosθ·1e9/(4ρc²) = 3.48·Q·cosθ/8.92 = 0.39·Q·cosθ` →
`Q·cosθ = 56.4`. With single-Lorentzian cosθ = 0.888 fixed by Δf, this
gives `Q_eff = 64`. (The earlier "`Q_eff = 64` vs `Q = 100`" framing
implicitly compared against the stale W10 transient value; the meaningful
comparison is against a W21-internal Q, which is pending.)

### 5. The "regime transition" shape claim survives

The cascade plateau in `P_2f/P_1f²` is flat at ~22 /GPa from 30 V to
110 V and falls to ~20 /GPa at 120 V (see B1 report). That **shape**
(flat in the perturbative regime, falling at high drive) is what makes
the regime-transition story work. The absolute calibration vs Coppens
is a separate question:

- If we believe single-mode Coppens with **the stale W10 Q_2 = 100** + cosθ
  from Δf = 10 kHz, measured is **36 % below** prediction.  Not a meaningful
  comparison until Q is W21-internal.
- If we attribute the gap to a **two-mode cavity** that interferes
  destructively at 3.804 MHz (between the two peaks), the
  single-mode Coppens is the wrong functional form; need a sum of two
  complex Lorentzian responses.

Either way, the PRL "no fitted parameters" claim now depends on
resolving (4) — two-mode treatment or Q recalibration. F3 (iterative
Kuznetsov overlay) inherits this dependency: the simulation also
assumes a single-mode 2f cavity.

## Implications and immediate next moves

1. **Update the 2026-06-13 figure plan**: replace f_2f = 3.845 MHz and
   cosθ = 0.53 with the new pinned f_2f = 3.794 MHz; flag the two-mode
   structure; soften the "within 7 % of Coppens" headline until the
   two-mode treatment is in.
2. **Two-mode Coppens** — short calculation, no new data. Fit two
   Lorentzian amplitudes to T(f), compute complex sum at 2 f_drive.
3. **Mode-projected transient Q_2** — settle the 3.3× spectral-vs-
   transient gap by extracting the cos(2πx/W)-projected ring-down
   envelope from any one high-drive cascade file.
4. **10 V noise/SNR re-frame** — DONE: propagating the `P_nf` error
   gives `P_2f/P_1f² = 26.7 ± 9.3 /GPa` at 10 V, within 1σ of the
   ~22 /GPa plateau (±35 %, 2f SNR ≈ 1). The "10 V is +29 % above
   Coppens" is noise, not physics — confirming the positive-bias
   suspicion. (`harmonic_ladder.py` p_std / `prl_draft.py` Fig 2c.)

Outputs (gitignored, regenerable from shared cache):
`experiments/2026W21/output/f2_eigenmode_scan/{*.png,*.csv}`.
