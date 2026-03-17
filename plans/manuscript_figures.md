# Manuscript figure generation (Figs 5–9, A1)

## Context

The manuscript `main.tex` (nonlinear harmonic limitation of acoustophoresis,
target: Physical Review Applied) needs experimental figures 5–9 and an
appendix figure A1. Simulation figures (1–4) already exist. The experimental
data is in ldv-analysis; simulation code is in
`/home/tatsuki/harmonic_model/` (WSL).

## Output

`$MANUSCRIPT_DIR/figures/` (set `MANUSCRIPT_DIR` in `.env` per device).
Save as .eps + .png. Naming: `Fig5.eps`, `Fig6.eps`, etc.

## Script

`experiments/2026W10_stepA/manuscript_figures.py` — single script, all figures.

## Plotting data cache

Both experiment and simulation scripts export plotting-ready data to NPZ
files in the manuscript figures directory. This enables:
1. Cross-referencing between codebases (Fig 9 overlay)
2. Faster figure regeneration (skip recomputation)

### Experiment caches (per-figure NPZ)

Saved by `manuscript_figures.py`. Cache names match figure names:
`Fig5.npz`, `Fig6.npz`, `Fig7.npz`, `Fig8.npz`, `FigA1.npz`.

Key shared data (in `Fig8.npz`, formerly fig6):

| Key | Shape | Unit | Description |
|-----|-------|------|-------------|
| `Vpp` | (5,) | V | Drive voltage |
| `p0_1f` | (5,) | Pa | Fitted 1f pressure amplitude |
| `p0_2f` | (5,) | Pa | Fitted 2f pressure amplitude |
| `E_ac` | (5,) | J/m³ | Acoustic energy density (1f only) |
| `P_in` | (5,) | W | Electrical input power |
| `ratio` | (5,) | — | p₂f/p₁f |
| `ch1_2f_1f` | (5,) | — | PZT drive voltage 2f/1f ratio |

### Simulation cache (`fig3.npz`)

Saved by `generate_manuscript_figures.py` after the v₀ sweep:

| Key | Shape | Unit | Description |
|-----|-------|------|-------------|
| `v0` | (N,) | m/s | Boundary velocity |
| `p1f_sc` | (N,) | Pa | Self-consistent 1f amplitude |
| `p2f_sc` | (N,) | Pa | Self-consistent 2f amplitude |
| `p1f_pert` | (N,) | Pa | Perturbation 1f amplitude |
| `p2f_pert` | (N,) | Pa | Perturbation 2f amplitude |
| `E_ac_sc` | (N,) | J/m³ | Self-consistent total E_ac |
| `E_ac_pert_first` | (N,) | J/m³ | Perturbation linear E_ac |
| `E_ac_pert_total` | (N,) | J/m³ | Perturbation total E_ac |
| `ratio_sc` | (N,) | — | Self-consistent p₂f/p₁f |
| `ratio_pert` | (N,) | — | Perturbation p₂f/p₁f |

## Figure summary

| Figure | Description | Status |
|--------|-------------|--------|
| Fig 5 | ⟨E_{ac,1f}⟩ vs P_in | Done |
| Fig 6 | Waveform distortion (2×3 grid) | Done |
| Fig 7 | Spatial mode profiles (2×2) | Done |
| Fig 8 | Harmonics vs V_drive (2 panels) + fit R² | Done |
| Fig 9 | Simulation vs experiment overlay (MRE=4.1%) | Done |
| Fig A1 | Transient ring-up envelopes (3 panels, appendix) | Done |

## Notation conventions

- **P_{nf}**: peak pressure amplitude (scalar, from mode-shape fit)
- **p_{nf}(y)**: spatial pressure distribution (1D cross-section)
- **p_{nf}(x,y)**: spatial pressure distribution (2D map)
- **E_{ac,1f}**: 1f-only acoustic energy density = p₀²/(4ρc²)
- **V_drive**: drive voltage (x-axis symbol), unit: V_pp

## Fig 5 — ⟨E_{ac,1f}⟩ vs P_in

- Source: test10 voltage sweep (5 files: 5, 10, 15, 20, 25 Vpp)
- E_{ac,1f} = ⟨p₀²⟩ / (4 × 1004 × 1508²) from column-wise mode fits
- Plot: E_{ac,1f} vs P_in with linear fit through origin
- Single-column (3.375 × 2.5 in)

## Fig 6 — Waveform distortion

- Source: test10 at 10 and 25 Vpp, 3 width positions (y=0, +W/4, +W/2)
- 2×3 grid (7.0 × 3.0 in), raw pressure + 1f+2f reconstruction overlay
- Local DFT over display window for phase-consistent reconstruction
- ±3 MPa ylim, 1 µs window

## Fig 7 — Spatial mode profiles (1f + 2f)

- Source: test10 voltage sweep at 1907 kHz
- 2×2 (7.0 × 3.0 in): (a) p_{1f}(y) cross-section, (b) p_{2f}(y) cross-section,
  (c) p_{1f}(x,y) 2D map 25 Vpp, (d) p_{2f}(x,y) 2D map 25 Vpp
- Complex LSQ fitting via `fit_mode_1f`, `fit_mode_2f` (centre=0)
- 2f ylim capped at 1.2× max fit amplitude (outlier clipping)
- 2D maps: viridis, percentile clipping, red dashed antinode marker

## Fig 8 — Drive-resolved harmonics

Two panels (a–b), shared x-axis (V_drive [V_pp]), single-column (3.375 × 4.2 in):
- (a) P_{1f} and P_{2f} vs V_drive + fits through origin
  - P_{1f} = 163.5 kPa/V (R²=0.999)
  - P_{2f} = 0.63 kPa/V² (R²=0.995)
- (b) P_{2f}/P_{1f} ratio vs V_drive
  - slope = 0.0039/V (R²=0.997)

## Fig 9 — Simulation vs experiment overlay

- Common x-axis: **⟨E_{ac,1f}⟩ [J/m³]** (1f-only, both codebases)
- y-axis: P_{2f}/P_{1f} ratio
- Simulation E_{ac,1f} computed from p1f_sc²/(4×1000×1500²) — NOT E_ac_sc
  (which includes 2f energy). This matches the experiment convention.
- Plot: experiment markers + simulation curve (self-consistent)
- Mean relative error: MRE = 4.1% (interpolated at matching E_{ac,1f})
- Single-column (3.375 × 2.5 in)

## Fig A1 — Transient ring-up envelopes (appendix)

Three panels (a–c), 25 Vpp, single-column (3.375 × 5.5 in):
- (a) ⟨P₁f/P_ss⟩ ring-up with exponential fit → Q₁f = 121
- (b) ⟨P₂f/P_ss,2f⟩ ring-up with driven-resonator fit (source ∝ p₁²) → Q₂f = 100
- (c) ⟨I⟩ driving current envelope (raw, no fit) — reaches steady state
  much faster than pressure, justifying step-input assumption for
  acoustic transient model

All panels show ±1σ shaded band from point-to-point averaging.
Self-contained computation: envelopes computed directly from TDMS in the
averaging loop (no dependency on transient_ch2/ch4 scripts).
