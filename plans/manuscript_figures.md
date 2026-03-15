# Manuscript figure generation (Figs 5–10)

## Context

The manuscript `main.tex` (nonlinear harmonic limitation of acoustophoresis,
target: Physical Review Applied) needs experimental figures 5–10 and an
overlay figure 10. Simulation figures (1–4) already exist. The experimental
data is in ldv-analysis; simulation code is in
`/home/tatsuki/harmonic_model/` (WSL).

## Output

`D:/OneDrive - Lund University/Publications/nonlinearphysics/manuscript/figures/`
Save as .eps + .png. Naming: `Fig5.eps`, `Fig6.eps`, etc.

## Script

`experiments/2026W10_stepA/manuscript_figures.py` — single script, all figures.

## Plotting data cache

Both experiment and simulation scripts export plotting-ready data to NPZ
files in the manuscript figures directory. This enables:
1. Cross-referencing between codebases (Fig 10 overlay)
2. Faster figure regeneration (skip recomputation)

### Experiment cache (`experiment_data.npz`)

Saved by `manuscript_figures.py` after the voltage-sweep processing loop:

| Key | Shape | Unit | Description |
|-----|-------|------|-------------|
| `Vpp` | (5,) | V | Drive voltage |
| `p0_1f` | (5,) | Pa | Fitted 1f pressure amplitude |
| `p0_2f` | (5,) | Pa | Fitted 2f pressure amplitude |
| `E_ac` | (5,) | J/m³ | Acoustic energy density |
| `P_in` | (5,) | W | Electrical input power |
| `ratio` | (5,) | — | p₂f/p₁f |
| `ch1_2f_1f` | (5,) | — | PZT drive voltage 2f/1f ratio |

### Simulation cache (`simulation_data.npz`)

Saved by `generate_manuscript_figures.py` after the v₀ sweep:

| Key | Shape | Unit | Description |
|-----|-------|------|-------------|
| `v0` | (N,) | m/s | Boundary velocity |
| `p1f_sc` | (N,) | Pa | Self-consistent 1f amplitude |
| `p2f_sc` | (N,) | Pa | Self-consistent 2f amplitude |
| `p1f_pert` | (N,) | Pa | Perturbation 1f amplitude |
| `p2f_pert` | (N,) | Pa | Perturbation 2f amplitude |
| `E_ac_sc` | (N,) | J/m³ | Self-consistent E_ac |
| `E_ac_pert_first` | (N,) | J/m³ | Perturbation linear E_ac |
| `E_ac_pert_total` | (N,) | J/m³ | Perturbation total E_ac |
| `ratio_sc` | (N,) | — | Self-consistent p₂f/p₁f |
| `ratio_pert` | (N,) | — | Perturbation p₂f/p₁f |

## Figure summary

| Figure | Description | Status |
|--------|-------------|--------|
| Fig 5 | E_ac vs P_in | Done |
| Fig 6 | Harmonics vs V (2 panels) | Done |
| Fig 7 | Transient ring-up envelopes (3 panels) | Done |
| Fig 8 | Waveform distortion (2×3 grid) | Done |
| Fig 9 | Spatial mode profiles (2×2) | Done |
| Fig 10 | Simulation vs experiment overlay | Done (needs sim cache) |

## Fig 5 — E_ac vs P_in

- Source: test10 voltage sweep (5 files: 5, 10, 15, 20, 25 Vpp)
- E_ac = p_1f² / (4 × 1004 × 1508²) from fitted p0_1f
- Plot: E_ac vs P_in with linear fit through origin
- Single-column (3.375 × 2.5 in)

## Fig 6 — Drive-resolved harmonics

Two panels (a–b), shared x-axis (Vpp), single-column (3.375 × 4.2 in):
- (a) p_1f and p_2f vs Vpp + fits (linear, quadratic)
- (b) p_2f/p_1f ratio vs Vpp

## Fig 7 — Transient ring-up envelopes

Three panels (a–c), 25 Vpp, double-column (7.0 × 2.2 in):
- (a) ⟨P₁f/P_ss⟩ ring-up with exponential fit → Q₁f
- (b) ⟨P₂f/P_ss,2f⟩ ring-up with driven-resonator fit (source ∝ p₁²) → Q₂f
- (c) ⟨I⟩ driving current envelope (raw, no fit) — reaches steady state
  much faster than pressure, justifying step-input assumption for
  acoustic transient model

All panels show ±1σ shaded band from point-to-point averaging.
Self-contained computation: envelopes computed directly from TDMS in the
averaging loop (no dependency on transient_ch2/ch4 scripts).

## Fig 8 — Waveform distortion

- Source: test10 at 10 and 25 Vpp, 3 width positions (y=0, +W/4, +W/2)
- 2×3 grid (7.0 × 3.0 in), raw pressure + 1f+2f reconstruction overlay
- Local DFT over display window for phase-consistent reconstruction
- ±3 MPa ylim, 1 µs window

## Fig 9 — Spatial mode profiles (1f + 2f)

- Source: test10 voltage sweep at 1907 kHz
- 2×2 (7.0 × 3.0 in): (a) P_1f cross-section, (b) P_2f cross-section,
  (c) 2D P_1f map 25 Vpp, (d) 2D P_2f map 25 Vpp
- Complex LSQ fitting via `fit_mode_1f`, `fit_mode_2f` (centre=0)
- 2f ylim capped at 1.2× max fit amplitude (outlier clipping)
- 2D maps: viridis, percentile clipping, red dashed antinode marker

## Fig 10 — Simulation vs experiment overlay

- Common x-axis: **E_ac [J/m³]** (only quantity both codebases can compute)
- y-axis: p₂f/p₁f ratio
- Plot: experiment markers + simulation curve (self-consistent)
- Single-column (3.375 × 2.5 in)
- Reads `experiment_data.npz` + `simulation_data.npz` from figures directory
- Shows qualitative agreement (both grow) with quantitative discrepancy
  → detuning/Q₂ discussion in Sec. V
