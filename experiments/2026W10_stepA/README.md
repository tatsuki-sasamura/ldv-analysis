# 2026 Week 10 — Step A Resonance Characterisation

Burst-mode refracto-vibrometry analysis of the acoustofluidic chip.
Mode shapes, frequency sweeps, voltage scaling, transient ring-up/down,
and cross-validation against PTV.

Data lives on OneDrive (not in the repo). Set `LDV_DATA_ROOT` in `.env`
to your local mirror; scripts resolve it via `config.get_data_dir()`.

---

## Quick start: reproduce the manuscript figures

```bash
# 1. Build all FFT caches first (a few mins per TDMS, ~5-30 GB each)
.venv/Scripts/python experiments/2026W10_stepA/manuscript_figures.py --fresh

# 2. Subsequent runs use the cached .npz files (~seconds)
.venv/Scripts/python experiments/2026W10_stepA/manuscript_figures.py
.venv/Scripts/python experiments/2026W10_stepA/af2026_figures.py
```

Set `MANUSCRIPT_DIR` in `.env` to control where `.eps`/`.png` are written
(default: `$MANUSCRIPT_DIR/pra/figures/` and `$MANUSCRIPT_DIR/acoustofluidics/figures/`).

---

## Scripts by purpose

### A. Manuscript figures (canonical end-products)

| Script | Output | Source data |
|---|---|---|
| `manuscript_figures.py` | PRA Fig 5-9, A1 | test10 voltage sweep + transient |
| `af2026_figures.py` | AF2026 Fig 1, 2 | Fig7.npz, Fig8.npz, fig3.npz caches |
| `ldv_ptv_comparison.py` | LDV-PTV cross-validation (4+3 panels) | test10 + PTV `output/{5,10,15}Vpp/` |

### B. Spatial pressure analysis (mode shapes, 2D maps)

| Script | Output | Use case |
|---|---|---|
| `pressure_map_2d.py` | 2D velocity/pressure/phase maps + 1f/2f/3f mode-shape fits | Any 2D area-scan TDMS |
| `harmonics_profile.py` | Per-harmonic width profile from one TDMS | Quick look at a single file |
| `single_mode_shape.py` | 1f mode shape, waveform/spectrum, repeatability | Single line-scan file |
| `axial_p0_distribution.py` | p₀(x) along channel length, multi-voltage overlay | Compare axial profile across drives |
| `cross_section_profile.py` | Boundary-quality cross sections | Diagnostic for edge artifacts |
| `harmonics_vs_voltage.py` | P_{1f,2f,3f} vs V_drive (log-log), Mach number plots | Drive-scaling tests |

### C. Frequency sweeps (one script per dataset family)

| Script | Targets | Notes |
|---|---|---|
| `freq_sweep.py` | `test14_*.tdms` | General-purpose CLI; specify dir + glob |
| `freq_sweep_coarse.py` | `stepA_sweep_*.tdms` (Mar 3, 1920-2020 kHz) | |
| `freq_sweep_fine.py` | `test5_*.tdms` (Mar 6, fine sweep around 1f) | |
| `freq_sweep_2f.py` | `test7_*.tdms` (Mar 6, 2f sweep 3.7-3.9 MHz) | Finds f₂ ≈ 3.845 MHz |
| `freq_sweep_25vpp.py` | `test9_*.tdms` (Mar 6, 25 Vpp) | 1f and 2f mode-shape fits |
| `freq_axial_sweep.py` | `test3_*.tdms` (Mar 6, freq × axial) | p₀(f, x) heatmap |

### D. Drive-amplitude scaling

| Script | Targets |
|---|---|
| `voltage_sweep.py` | test10 2D area scans across Vpp |
| `voltage_sweep_1d.py` | test12 line scans at y=8.9 mm |
| `harmonics_vs_voltage.py` | also under B — voltage scaling with harmonics |

### E. Transient / Q-factor

| Script | Output | Notes |
|---|---|---|
| `transient_ch2_acoustic.py` | 1f and 2f τ/Q from ring-up envelope | Driven-resonator + beat fits |
| `transient_ch4_current.py` | PZT electrical resonance detuning | From current envelope |
| `transient_animation.py` | MP4 of instantaneous p(x, y, t) | FFT integration + 6 MHz LPF |
| `transient_animation_dft.py` | MP4 of \|p_{nf}(x, y, t)\| envelope | Sliding DFT at any harmonic |
| `pressure_buildup.py` | Pcolormesh of p(scan_point, time) | Mode-shape evolving during burst |

### F. Theory comparisons

| Script | Compares against |
|---|---|
| `coppens_comparison.py` | Coppens cascade: P_{2f}/P_{1f}, P_{3f}/P_{1f} |
| `electroacoustic.py` | P_elec(f) vs E_ac(f) Baasch-style |
| `bvd_fit.py` | BVD multi-branch impedance model |

### G. Calibration

| Script | Purpose |
|---|---|
| `calibrate_geometry.py` | Channel centre + tilt from RSSI (jointly across files) |

### H. Diagnostics & sanity

| Script | Checks |
|---|---|
| `sanity_check.py` | Drift, electrical THD, missed bursts, burst timing, RSSI |
| `fft_sanity_check.py` | Verifies the FFT-to-pressure conversion math on one point |
| `snr_assessment.py` | SNR histograms and spatial maps |
| `thermal_drift_check.py` | Ch1/Ch4 stability across scan |
| `ldv_z_sweep.py` | LDV head z-position effect (chip-vs-optical drift) |

---

## Output structure

```
output/
  cache/                              FFT caches (_fft_cache_*.npz)
  manuscript_figures/                 PRA fig data (Fig5-9, A1) -- normally also copied to MANUSCRIPT_DIR
  af2026_figures/                     AF2026 fig data (Fig1, 2)
  ldv_ptv_comparison/                 LDV-vs-PTV plots
  harmonics_vs_voltage/               drive-scaling plots
  pressure_map_2d/                    2D maps + mode shapes (one set per TDMS)
  freq_sweep_*/                       4-panel sweeps + mode_shapes/
  transient_ch2_acoustic/             1f/2f tau/Q fits
  transient_animation/                MP4 outputs
  sanity_check/, snr_assessment/, …   diagnostics
```

Cached `.npz` files in `output/cache/` are shared across all scripts and
keyed by TDMS stem; deleting them forces recomputation.

---

## Conventions

- **Coordinates**: `x` = channel length (axial), `y` = channel width (transverse). Channel width `W = 375 µm`, height `H = 150 µm`.
- **Pressure sign**: `p = -v_apparent / (2π·f·H·dn/dp)` — minus sign because +p → +n → +OPL → -v_LDV.
- **Water**: ρ = 1000 kg/m³, c = 1500 m/s, β = 3.5. Set in `config.py`.
- **1f mode**: node at y=0, anti-nodes at walls → `sin(πy/W)`.
- **2f mode**: anti-nodes at centre and walls → `cos(2πy/W)`.
- **Filters**: all scripts use `make_valid_mask` + `make_burst_timing_mask` from `ldv_analysis.filters` for consistency.

## See also

- `plans/manuscript_figures.md` — PRA figure layout and notation
- `plans/af2026_figures.md` — AF2026 figure spec
- `calibration/ptv_stage_alignment/README.md` — PTV–LDV coordinate calibration
- Project root `README.md` — overall project structure
