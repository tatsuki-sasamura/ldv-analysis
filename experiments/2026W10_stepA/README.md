# 2026 Week 10 — Step A Resonance Characterisation

Burst-mode refracto-vibrometry analysis of the acoustofluidic chip.
Identifies half-wavelength resonance, maps mode shapes, and estimates Q factors.

## Data sources

Data lives on Google Drive, not in the repository.

### March 3 — `G:/My Drive/20260303experimentA/`

| Files | Grid | Description |
|-------|------|-------------|
| `stepA1967.tdms` | 101×3 | Single-frequency scan at 1967 kHz (y = 2 mm) |
| `stepA1967_where_is_the_best_x_position.tdms` | 101×101 | 2D area scan at 1967 kHz |
| `stepA_sweep_1920.tdms` … `stepA_sweep_2020.tdms` | 101×2 | 29 files, coarse frequency sweep (1920–2020 kHz) |

### March 6 — `G:/My Drive/20260306experimentA/`

| Files | Grid | Description |
|-------|------|-------------|
| `test3_1900.tdms` … `test3_2000.tdms` | 21×11 | 11 files, coarse freq × axial position sweep |
| `test5_1900.tdms` … `test5_2015.tdms` | 101×2 | 56 files, fine frequency sweep at x = 9 mm |

## Scripts → datasets

| Script | Dataset | What it does |
|--------|---------|--------------|
| `single_mode_shape.py` | `stepA1967.tdms` (or any single file) | 1f mode shape, waveform/spectrum, 3-line repeatability |
| `freq_sweep_coarse.py` | `stepA_sweep_*.tdms` (Mar 3) | Coarse frequency sweep, sinusoidal mode-shape fit at each freq |
| `freq_sweep_fine.py` | `test5_*.tdms` (Mar 6) | Fine frequency sweep around resonance peaks |
| `freq_axial_sweep.py` | `test3_*.tdms` (Mar 6) | Frequency × axial position heatmap of p0(f, x) |
| `pressure_map_2d.py` | `stepA1967_…_best_x_position.tdms` (or any area scan) | 2D pressure/velocity/phase maps with boundary detection |
| `pressure_buildup.py` | `stepA_sweep_1970.tdms` (or any single file) | Time-resolved mode-shape evolution during burst |
| `thermal_drift_check.py` | `stepA1967.tdms` (or any single file) | Ch1/Ch4 electrical stability across scan |
| `transient_ringup_fit.py` | `stepA1967.tdms` (or any single file) | Ring-up/ring-down envelope fit for Q estimation |

Single-file scripts accept a TDMS path as a command-line argument; the defaults above are used if none is given.

## Output structure

```
output/
  cache/                     Shared FFT caches (_fft_cache_*.npz)
  single_mode_shape/         Mode shape, waveform, repeatability plots
  freq_sweep_coarse/         4-panel sweep + mode_shapes/ subdir
  freq_sweep_fine/           4-panel sweep + mode_shapes_test5/ subdir
  freq_axial_sweep/          Heatmaps + mode_shapes_test3/ subdir
  pressure_map_2d/           2D pcolormesh maps
  pressure_buildup/          Pcolormesh, snapshots, ring-up curve
  thermal_drift_check/       Electrical stability plots
  transient_ringup_fit/      Envelope fit plots
```
