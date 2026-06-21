# Mount-drift detuning → LDV/PTV ratio inflation — 2026-05-21

## Question

If PTV's drive frequency was held fixed (`F_US = 1.907 MHz` in
`particle-tracking/scripts/06_fitting.py`) while the chip's actual 1f
resonance had drifted between mounts (up to ~8 kHz across W10 → W21),
how much does the detuning inflate the observed LDV/PTV pressure ratio?

## Method

Empirical, not Lorentzian-fit. For each of the 12 trimmed
resonance-survey sweeps:

1. Find measured `f_peak = argmax P_1f(f)`.
2. Linearly interpolate the sweep at `f_peak ± 8 kHz`.
3. `mean = (P(−8) + P(+8)) / 2` (one-sided where the sweep is
   asymmetric around peak).
4. Ratio inflation = `1 / mean`.

Source: `experiments/2026W21_freq_sweep/output/resonance_survey/summaries/<id>.npz`,
built by `resonance_survey.py` from the v8 (Hann + 4× zero-pad) FFT
caches.

## Per-dataset results at Δf = 8 kHz

| Dataset | f_peak (MHz) | P(−8) | P(+8) | mean | inflation |
|---|---:|---:|---:|---:|---:|
| W10 test5 fine 10 Vpp     | 1.9070 | —* | 0.82 | 0.82 | 1.22 |
| W10 test9 narrow 25 Vpp   | 1.9080 | —* | 0.88 | 0.88 | 1.13 |
| W16 test2 20 Vpp          | 1.9040 | 0.66 | 0.64 | **0.65** | **1.54** |
| W21 wide 20 Vpp           | 1.9000 | 0.82 | 0.77 | 0.80 | 1.25 |
| W21 wide 40 Vpp           | 1.9010 | 0.80 | 0.70 | 0.75 | 1.33 |
| W21 wide 60 Vpp           | 1.9030 | 0.86 | 0.73 | 0.79 | 1.26 |
| W21 wide 80 Vpp           | 1.9040 | 0.77 | 0.76 | 0.76 | 1.31 |
| W21 narrow 80 Vpp         | 1.9040 | 0.76 | 0.79 | 0.77 | 1.30 |
| W21 narrow 60 Vpp         | 1.9040 | 0.72 | 0.75 | 0.73 | 1.36 |
| W21 narrow 40 Vpp         | 1.9050 | 0.76 | 0.69 | 0.72 | 1.38 |
| W21 narrow 20 Vpp (split) | 1.9050 | 0.71 | 0.63 | 0.67 | 1.49 |
| W21 narrow 20 Vpp (cal)   | 1.9050 | 0.73 | 0.63 | 0.68 | 1.47 |

(*) W10 test5/test9 summaries don't extend far enough below their peak;
the −8 kHz interpolation falls outside the stored frequency range, so
+ side only.

## Per-campaign roll-up

| Campaign | n sweeps | mean P/P_peak at ±8 kHz | mean ratio inflation |
|---|---:|---:|---:|
| W10 | 2 | 0.85 | **1.18×** |
| W16 | 1 | 0.65 | **1.54×** |
| W21 | 9 | 0.74 | **1.36×** |

## Key takeaways

- **Line shapes are mount-dependent** even though Q from Lorentzian fits
  looks similar (90–110). W16 has a genuinely narrower / steeper
  measured response than W10 or W21.
- **Worst case at the 8 kHz maximum measured drift**: ~1.5× inflation
  (using W16's narrow line shape).
- **Typical W21 case**: ~1.3× inflation per 8 kHz detuning.
- **Asymmetry**: most high-drive W21 cases drop a few-% faster on the
  +Δf side than −Δf side — consistent with mild softening nonlinearity
  that leans the peak low.

## Plots

- `experiments/2026W21_freq_sweep/output/resonance_survey/cross_campaign_summary.png`
  — peak frequency per dataset, three rows (one per mount).
- `experiments/2026W21_freq_sweep/output/resonance_survey/lineshape_centered.png`
  — all 12 sweeps overlaid, peak-centred and normalised.

## Implication for the 1.7–1.9× LDV/PTV gap

If the PTV scan was on the W10 mount (matched conditions with the LDV
cross-val, Δf ≈ 0), drift contributes nothing.

If the PTV scan was on a later mount (W16 or W21) while still being
inverted with `F_US = 1.907 MHz`, up to **~1.5× of the observed
1.7–1.9× gap is absorbable** by detuning alone, before invoking any
other LDV-side bias or PTV-side parameter sensitivity.

The remaining open question is which mount the PTV data was actually
acquired on.
