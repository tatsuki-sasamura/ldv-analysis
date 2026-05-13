# 2026 Week 10 — Step A Resonance Characterization

Burst-mode refracto-vibrometry analysis of the acoustofluidic chip.

Data lives on OneDrive (not in the repo). Set `LDV_DATA_ROOT` and
`MANUSCRIPT_DIR` in `.env`; scripts resolve them via
`config.get_data_dir()` and `config.MANUSCRIPT_DIR`.

> **Looking for an old analysis?** This folder was pruned on 2026-05-13
> to the 12 actively-used scripts. The full 31-script working state
> lives at git tag `v1.0`:
> ```bash
> git worktree add ../ldv-v1 v1.0
> cd ../ldv-v1
> .venv/Scripts/python experiments/2026W10_stepA/<old_script>.py
> ```

---

## Quick start: reproduce the manuscript figures

```bash
# AF2026 abstract (TDMS-self-sufficient via figure_data library)
.venv/Scripts/python experiments/2026W10_stepA/af2026_figures.py --fresh

# PRA paper (regenerates Fig 5-9 + A1)
.venv/Scripts/python experiments/2026W10_stepA/manuscript_figures.py --fresh
```

`.eps` / `.png` go to `$MANUSCRIPT_DIR/{pra,acoustofluidics}/figures/`.
Subsequent runs use NPZ caches (~seconds).

---

## Scripts

### A. Manuscript figures

| Script | Output | Notes |
|---|---|---|
| `manuscript_figures.py` | PRA Fig 5, 6, 7, 8, 9, A1 | uses `figure_data` lib + own Fig5/6/9/A1 code |
| `af2026_figures.py` | AF2026 Fig 1, 2 | self-sufficient from TDMS via `figure_data` lib |

### B. Cross-validation & physics

| Script | Output |
|---|---|
| `ldv_ptv_comparison.py` | LDV vs PTV: p₀ vs voltage, scatter, axial profile, E_ac maps |
| `harmonics_vs_voltage.py` | P_{1f,2f,3f} vs V_drive (log-log), Mach number plots |
| `coppens_comparison.py` | Coppens cascade overlay |
| `transient_ch2_acoustic.py` | 1f / 2f τ → Q from ring-up envelope |

### C. Per-file inspection

| Script | Output |
|---|---|
| `pressure_map_2d.py <tdms> [--harmonics]` | 2D maps, mode-shape fits, R² |
| `transient_animation.py [tdms]` | MP4 of instantaneous p(x, y, t) |
| `transient_animation_dft.py --harmonic N` | MP4 of \|p_{nf}(x, y, t)\| envelope |

### D. Calibration & diagnostics

| Script | Purpose |
|---|---|
| `calibrate_geometry.py <tdms_1> [tdms_2 ...]` | Channel center + tilt from RSSI; saves JSON |
| `sanity_check.py` | Drift, electrical THD, missed bursts, RSSI |
| `fft_sanity_check.py` | Pipeline math verification on one data point |

---

## Library helpers (`src/ldv_analysis/`)

| Module | Purpose |
|---|---|
| `config.py` | Constants (RHO, C_SOUND, CHANNEL_WIDTH, SENSITIVITY), `.env` resolution, `velocity_to_pressure` |
| `io_utils.py` | TDMS metadata + waveform reading |
| `fft_cache.py` | Per-TDMS burst-mode FFT pipeline, cached as `_fft_cache_*.npz` |
| `filters.py` | Data-quality masks (voltage, RSSI, burst timing) |
| `mode_fit.py` | Sinusoidal mode-shape LSQ with iterative sigma clipping |
| `grid_utils.py` | Scan-point → (length, width) grid mapping |
| `transient.py` | Sliding DFT, τ→Q, rise/fall fit models |
| `figure_data.py` | Shared TDMS-to-NPZ extraction for PRA/AF figures |

Tests live in `tests/` (81 currently); run `pytest -q`.

---

## Output structure

```
output/
  cache/                              FFT caches (_fft_cache_*.npz)
  manuscript_figures/                 PRA fig outputs (also copied to MANUSCRIPT_DIR)
  af2026_figures/                     AF outputs (also copied to MANUSCRIPT_DIR)
  ldv_ptv_comparison/                 cross-validation plots
  harmonics_vs_voltage/               drive-scaling plots
  coppens_comparison/                 theory overlay
  transient_ch2_acoustic/             1f / 2f tau/Q fits
  transient_animation/                MP4 outputs
  pressure_map_2d/                    per-TDMS 2D maps
  sanity_check/, fft_sanity_check/    diagnostics
```

Cached `.npz` files in `output/cache/` are keyed by TDMS stem; deleting
them forces recomputation.

---

## Conventions

- **Coordinates**: `x` = channel length, `y` = channel width. `W = 375 µm`, `H = 150 µm`.
- **Pressure sign**: `p = -v_apparent / (2π f H dn/dp)`. The minus sign comes from `+p → +n → +OPL → -v_LDV`.
- **Water**: ρ = 1000 kg/m³, c = 1500 m/s, β = 3.5. Constants live in `config.py`.
- **1f mode**: `sin(πy/W)` — node at center, anti-nodes at walls.
- **2f mode**: `cos(2πy/W)` — anti-nodes at center and walls.
- **All scripts** apply `make_valid_mask` + `make_burst_timing_mask`.

## See also

- `plans/manuscript_figures.md`, `plans/af2026_figures.md` — figure specs
- `calibration/ptv_stage_alignment/README.md` — PTV-LDV coordinate calibration
- Project root `README.md` — overall project structure
