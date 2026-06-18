# B1 drive purity — 2026-06-18

## Purpose

PRL gate analysis B1 (per
`reports/2026-06-13_prl_figure_plan_and_waveguide_detuning.md` §5
task 5b and `reports/2026-06-18_prl_resource_inventory.md` §4): quantify
the 2f harmonic content on the electrical drive (Ch A) and current
(Ch D) across the 10–120 Vpp cascade, and bound the contribution of
that drive impurity to the observed acoustic P_2f.

Two questions to settle:

1. Is the PRA-cited bound "V_2f/V_1f < 0.07 % at 25 V" still correct on
   the W21 splitter-removed cascade?
2. Is the +29 % `P_2f/P_1f²` outlier at 10 Vpp (vs Coppens 20.7 /GPa,
   see 2026-06-13 plan §2.1) a drive-leakage artefact?

## Method

Two scripts, both under `experiments/2026W21/`:

- `drive_purity_b1.py` — exact-frequency DFT of Ch A and Ch D at
  n·f_drive for n = 1..5 on the 5 central scan points of every f*.h5
  in each cascade Vpp directory. Median across position × frequency
  per Vpp. Reuses `fft_cache.detect_burst_window` and
  `find_drive_frequency`. Output: `V_nf(Vpp)` and `I_nf(Vpp)` in
  physical units (V after the ×10 attenuation, A after the 0.2 A/V
  scale) plus their ratios to the fundamental.

- `drive_purity_b1_transfer.py` — refinement that bounds the leakage
  contribution to acoustic P_2f using the **measured** electroacoustic
  transfer at 2f-band drives. The
  `sample_101x77_fsweep_3p7to3p9_60Vpp_20260530_013206` scan drives the
  chip at 3.70–3.90 MHz; we project its `pressure_1f` onto the
  `cos(2πx/W)` cavity mode (via `AxialFit.p1_n2_mag`) and divide by
  the measured PZT voltage to get `T(f) = |P_cos2|/V` [Pa/V] across
  the 2f band. The bound on leakage at cascade conditions is then
  `P_2f,leak(Vpp) = T(2·f_peak_1f(Vpp)) · V_2f_PZT(Vpp)`, with
  `V_2f_PZT(Vpp)` from the first script and `f_peak_1f(Vpp)` from the
  cascade `vpp_vs_pressure.py` per-Vpp peak.

Outputs (gitignored, under
`experiments/2026W21/output/drive_purity_b1/`):
`drive_purity_b1.{png,csv}` and `drive_purity_b1_transfer.{png,csv}`.

## Results

### 1. Drive harmonic spectrum (Ch A, Ch D, n = 1..5)

Median across 11 sweep frequencies × 5 centre points per Vpp:

| PZT Vpp | V_1f (V) | V_2f/V_1f | V_3f/V_1f | V_4f/V_1f | V_5f/V_1f |
|--:|--:|--:|--:|--:|--:|
| 10 | 4.61 | 0.140 % | 0.099 % | 0.034 % | 0.035 % |
| 20 | 9.21 | 0.243 % | 0.099 % | 0.015 % | 0.026 % |
| 30 | 13.83 | 0.356 % | 0.097 % | 0.021 % | 0.044 % |
| 60 | 27.63 | 0.697 % | 0.067 % | 0.025 % | 0.036 % |
| 90 | 41.47 | 0.979 % | 0.044 % | 0.070 % | 0.035 % |
| 120 | 55.36 | **1.198 %** | 0.083 % | 0.153 % | 0.033 % |

Current (Ch D) tracks voltage closely:

| PZT Vpp | I_1f (mA) | I_2f/I_1f | I_3f/I_1f |
|--:|--:|--:|--:|
| 10 | 12.3 | 0.155 % | 0.220 % |
| 30 | 38.9 | 0.427 % | 0.167 % |
| 60 | 78.2 | 0.816 % | 0.132 % |
| 120 | 157.7 | **1.402 %** | 0.304 % |

**V_2f scales as V²** (i.e. V_2f/V_1f scales as V_1f). Numerically
V_2f/V_1f² ≈ 0.00025 /V, stable to ±20 % across the cascade — the
signature of amplifier-stage quadratic nonlinearity. The AFG itself
would put a fixed V_2f/V_1f independent of amplitude knob; the linear
growth of the ratio with V proves the **200× wideband amp is the
2f source**, not the synthesiser.

I_2f/I_1f tracks V_2f/V_1f within ~10 % at every Vpp — the PZT
load just transmits the drive harmonic with its capacitive
`Z(2f) = Z(1f)/2`.

**The PRA "<0.07 % at 25 V" bound is wrong.** Measured V_2f/V_1f at
20–30 Vpp is 0.24–0.36 %, ~5× the previously cited bound. Likely the
old number was from an AFG-output measurement that bypassed the
amplifier. The manuscript needs updating with the actual amp-coupled
numbers; cleanest statement is "drive 2f content scales as V²,
reaching V_2f/V_1f = 1.2 % at 120 Vpp."

### 2. Acoustic-leakage bound via the measured 2f-band transfer

Measured T(f) at 36 V_PZT drive in the 2f band (cos(2πx/W) projection
of `pressure_1f` from the 3p7to3p9 scan):

| f (MHz) | 3.70 | 3.74 | **3.80** | 3.82 | 3.84 | 3.86 |
|---|---|---|---|---|---|---|
| T (kPa/V) | 3.5 | 3.7 | **10.2** | 9.0 | 5.9 | 8.2 |
| R²(cos2) | -9 | 0.47 | **0.99** | 0.99 | 0.85 | 0.95 |

The cavity peak in this band is at **3.80 MHz**, with an R² = 0.99
cos(2πx/W) projection — a clean n = 2 transverse mode (see §
"Side findings" below for why this differs from the inventory value).
At the cascade 2f drive frequencies (3.80–3.82 MHz),
`T ≈ 9.2–10.0 kPa/V`. This is **~16× weaker** than the 1f transfer at
resonance (~150 kPa/V from the cascade slope) — the chip is a strong
resonator at f_1, a weak one at 2f_1.

Leakage bound across the cascade:

| Vpp | V_2f (V) | 2f_drive (MHz) | T (kPa/V) | P_2f,leak (kPa) | P_2f,obs (kPa) | **leak/obs** |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.0065 | 3.816 | 9.24 | 0.06 | 66.5 | **0.09 %** |
| 30 | 0.049 | 3.816 | 9.24 | 0.46 | 435 | **0.10 %** |
| 60 | 0.193 | 3.812 | 9.48 | 1.82 | 1391 | **0.13 %** |
| 90 | 0.406 | 3.808 | 9.72 | 3.95 | 2730 | **0.14 %** |
| 120 | 0.663 | 3.804 | 9.96 | 6.61 | 4398 | **0.15 %** |

**Leakage contributes 0.09–0.15 % of the observed P_2f at every drive
level.** The acoustic 2f field is ≥ 99.85 % cascade-generated.

This is a much tighter bound than the naive "use the 1f acoustic gain"
estimate (~4 % across the cascade), which over-counted leakage 30×
because it ignored cavity selectivity. The proper bound uses the
measured 2f-band transfer.

### 3. The 10 Vpp Coppens outlier — not leakage, and not significant

At 10 Vpp the leakage contribution is 0.06 kPa to an observed P_2f of
66.5 kPa (0.09 %). Subtracting changes `P_2f/P_1f² = 26.7 /GPa` by
< 0.1 %, so it is not drive leakage.

**Update (per-harmonic error propagation).** Folding the measured `P_nf`
uncertainty `√(fit-SE² + noise²)` into the prefactor gives
`P_2f/P_1f² = 26.7 ± 9.3 /GPa` at 10 V (±35 %, dominated by the weak 2f
there, SNR ≈ 1). That is **within 1σ of the ~22 /GPa plateau**, so the
"+29 %" is not statistically significant — it is the noisiest point, not
a real low-drive enhancement, and no per-V cosθ/Q story is needed.
(See `experiments/2026W21/prl_draft.py` Fig 2c and the `p_std` added to
`harmonic_ladder.py`.)

## Side findings

These are not B1's core question but came out of the transfer-function
measurement and are worth flagging.

### 2f cavity peak sits at ~3.80 MHz, NOT 3.845 MHz

The transfer T(f) measurement resolves a clean cos(2πx/W) peak
(R² = 0.99) at **3.80 MHz**, with a secondary structure around 3.86 MHz.
The PRA / `nonlinearphysics-manuscript/prl/detuning_mechanism.md`
analysis used a 2f cavity eigenmode at 3.845 MHz (from the survey
lines) and Δf = 3.845 − 2·1.907 = +31 kHz to compute
cosθ = 0.53 from the Lorentzian tail.

If the actual eigenmode is at 3.80 MHz, then Δf = 3.80 − 3.814
≈ **−14 kHz** — the drive sits *above* the eigenmode, not below. The
magnitude of cosθ from the Lorentzian-tail formula stays similar
(symmetric in Δf²) so the Coppens prediction barely moves, but the
**sign of the detuning is opposite** to what the Si-wall argument in
`detuning_mechanism.md` predicted. Worth re-deriving the Si-wall
expectation with this constraint, and cross-checking against the
survey-line dataset to understand why the 21×1 / 101×1 survey reported
3.845 MHz while the 101×77 area scan reports 3.80 MHz (different
axial location? Different mode shape weighting? Single-Y survey
projecting onto a mixed transverse mode?).

### Amplifier gain rolls off above ~3 MHz

The `sample_101x77_fsweep_3p7to3p9_60Vpp_*` scan is labelled 60 Vpp but
the cached `voltage_1f` reports **35–36 V on the PZT** across
3.70–3.90 MHz drive. The same labelled-60-Vpp cascade scan at 1.9 MHz
puts 28 V on the PZT (`drive_purity_b1.csv` row 6, V_1f). So the
amp+load combination delivers more at 3.8 MHz than at 1.9 MHz, which
means **the folder-name Vpp labels are nominal-AFG × assumed-gain, not
measured PZT voltage**. Cross-frequency comparisons should always use
the cached `voltage_1f`, not the folder label. Doesn't affect the
leakage bound here (T was computed from the actual V), but it does
mean any drive-amplitude figure should plot the measured PZT voltage,
not the label.

## Conclusions

1. **Drive leakage contributes ≤ 0.15 % to observed P_2f at every
   drive level.** The cascade story is unconditionally safe from a
   drive-purity referee objection.
2. **The PRA "<0.07 %" drive-purity claim is out of date**; the
   actual cascade-coupled number is 0.14–1.2 % on V_2f/V_1f,
   amplifier-dominated.
3. **The 10 Vpp Coppens outlier is resolved** — not drive leakage
   (≤ 0.1 %) and not statistically significant: 26.7 ± 9.3 /GPa, within
   1σ of the ~22 /GPa plateau (propagated `P_nf` error; the noisiest
   point, 2f SNR ≈ 1). Not a real low-drive enhancement.
4. **Two side findings** worth chasing in the detuning End-Matter:
   (a) the 2f cavity peak in this band is at 3.80 MHz, not 3.845 MHz —
   flipping the sign of Δf; (b) PZT voltage at 3.8 MHz drive is
   ~30 % higher than at 1.9 MHz drive at the same AFG setting.

## Next analysis

Per the 2026-06-13 plan §5 task order:

- **B2 — peak-frequency drift across cascade**: thermal-rise bound +
  per-V cosθ. (No longer needed to explain the 10 V outlier — resolved
  as noise, see §3 update — but still gives the thermal bound.)
- **A2 — 1f–5f conservation closure**: re-extract Σₙ P_{nf}² across
  the cascade from the existing cache (n=1..5 are already in
  `fft_cache.MAX_HARMONIC = 5`); flat ⇒ cascade-depletion picture
  closed.
- **F3 — iterative-Kuznetsov overlay**: critical-path PRL figure,
  the simulation branch is already checked out.
