# Manuscript figure generation (Figs 5–8)

## Context

The manuscript `main.tex` (nonlinear harmonic limitation of acoustophoresis,
target: Physical Review Applied) needs experimental figures 5–8. Simulation
figures (1–4) already exist. The experimental data is in ldv-analysis but no
manuscript-ready figure scripts exist yet.

## Output

`E:/OneDrive - Lund University/Publications/nonlinearphysics/manuscript/figures/`
Save as .eps + .png. Naming: `Fig5.eps`, `Fig6.eps`, etc.

## Script

`experiments/2026W10_stepA/manuscript_figures.py` — single script, all figures.

## Available data vs missing

| Figure | Data | Script | Blocker |
|--------|------|--------|---------|
| Fig 5 (E_ac vs P_in) | ✅ DONE | ✅ DONE | PIV overlay = separate codebase |
| Fig 6 (harmonics vs V) | ✅ DONE | ✅ DONE | — |
| Fig 7 (waveform distortion) | test10 raw waveforms in TDMS | write | — |
| Fig 8 (spatial modes) | test9 mode shapes ready | write | — |

## Fig 5 — E_ac vs V²

- Source: test10 voltage sweep (5 files: 5, 10, 15, 20, 25 Vpp)
- E_ac = p_1f² / (4 * 1004 * 1508²) from fitted p0_1f (same as voltage_sweep.py)
- Plot: E_ac vs Vpp² with linear fit through origin
- Reuse: `voltage_sweep.py` processing pattern (vel_correction, fit_columns, make_quality_mask)

## Fig 6 — Drive-resolved harmonics

Three panels (a–c), shared x-axis (Vpp):
- (a) p_1f and p_2f vs Vpp + fits (linear, quadratic)
- (b) p_2f/p_1f ratio vs Vpp
- (c) Ch1 drive voltage 2f/1f ratio — DFT at f and 2f on Ch1 steady-state window

## Fig 7 — Waveform distortion

- Source: test10 at 5 Vpp and 25 Vpp
- Load raw Ch2 waveform at strongest scan point via `load_point_waveforms`
- Convert to pressure, show ~5 cycles at steady state
- Two panels: (a) 5 Vpp clean sinusoid, (b) 25 Vpp distorted

## Fig 8 — Spatial mode profiles (1f + 2f)

- Source: test9 at 1907 kHz (best 1f resonance, 25 Vpp)
- 2×2: (a) 1f amplitude + sin fit, (b) 1f phase + model, (c) 2f amplitude + cos fit, (d) 2f phase + model
- Complex fitting via `fit_mode_1f`, `fit_mode_2f`
- Reuse: `freq_sweep_25vpp.py` mode shape pattern

