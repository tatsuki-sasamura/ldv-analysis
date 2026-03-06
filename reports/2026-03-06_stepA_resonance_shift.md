# Step A Resonance Shift Investigation — 2026-03-06

## Purpose

Investigate the acoustic resonance after a 3-day gap (March 3 → 6). Check whether the channel conditions changed, find the new resonance frequency and axial antinode, and attempt 2f external excitation.

## What happened

### Channel degradation and recovery

Pre-flush measurements (test1930, test1970 at y = 2 mm, 101×2 grid) showed:
- RSSI degraded: median 1.53 V (vs 1.68 on Mar 3), 42/202 points below 1.0 V
- Pressure collapsed: p0 at 1970 kHz dropped from 877 → 153 kPa; 1930 kHz from 651 → 38 kPa
- Mode shape lost — nearly flat, no sinusoidal structure

After flushing the channel:
- RSSI recovered to 1.96 V median, <1.0 V points back to normal (~5–10/202)
- Voltage stable at ~5.0 V

### Resonance frequency shifted

Wider scans (test2, 101×11 grid, y = 2–12 mm) revealed:
- y = 2 mm (the March 3 scan position) is now an **axial node** — near-zero pressure at all frequencies
- Pressure concentrates at y = 7–12 mm

A coarse frequency sweep (test3, 21×11 grid, 1.900–2.000 MHz in 10 kHz steps) mapped p0 across frequency and axial position:

| Feature | March 3 | March 6 |
|---------|---------|---------|
| Lower pressure peak | 1.930 MHz (651 kPa) | **1.910 MHz (1335 kPa)** |
| Upper pressure peak | 1.970 MHz (877 kPa) | 1.970 MHz (766 kPa) |
| Electrical resonance (current max) | 1.970 MHz | 1.970 MHz |
| Electrical anti-resonance (current min) | ~1.995 MHz | ~1.990 MHz |
| Best axial position | y = 2 mm | **y = 8–9 mm** |

The lower acoustic mode shifted from 1.930 → 1.910 MHz and became the stronger peak (1335 vs 766 kPa). The electrical resonance at 1.970 MHz stayed — the PZT didn't change, only the fluid acoustics (speed of sound shift after flushing, consistent with temperature or dissolved gas change).

### BVD multi-mode structure preserved

The frequency sweep shows the same two-peak + anti-resonance structure as March 3, confirming BVD behaviour with multiple motional branches. The V-I phase sweeps from -70° at 1.900 to -10° at 1.980 MHz without crossing zero in the measured range.

## New scripts

| Script | Purpose |
|--------|---------|
| `scripts/2026W10stepA/pressure_buildup.py` | Time-resolved pressure field evolution during burst. Sliding short-time DFT, pcolormesh, mode-shape snapshots, and p0 + current ring-up dual-axis plot |
| `scripts/2026W10stepA/freq_x_sweep.py` | Frequency × axial position sweep (test3 data). Mode-shape fits at each (f, x), heatmaps (p0, RSSI, R²), and 4-panel freq sweep at best x |

## Data files

### Pre-flush (degraded)
- `stepA_1f_1930.tdms`, `stepA_1f_1970.tdms` — 1f reference, 101×2 grid at y = 2 mm

### Post-flush, wide scans
- `test2_1920.tdms`, `test2_1930.tdms`, `test2_1970.tdms` — 101×11 grid (y = 2–12 mm)

### Coarse frequency sweep
- `test3_1900.tdms` … `test3_2000.tdms` — 11 files, 21×11 grid, 10 kHz steps

### 2f excitation (not yet analysed)
- `stepA_2f_3860.tdms` … `stepA_2f_3940.tdms` — 5 files at 2× frequency

All in `G:/My Drive/20260306experimentA/`.

## Output files

All in `output/2026W10stepA/`:

- `freq_x_p0_heatmap.png` — p0(f, x) heatmap showing resonance at 1.910 MHz, x = 9 mm
- `freq_x_rssi_heatmap.png` — RSSI median, confirming optical signal is healthy
- `freq_x_r2_heatmap.png` — mode-shape fit quality
- `freq_sweep_test3.png` — 4-panel sweep (p0, phase, current, voltage) at x = 9 mm
- `mode_shapes_test3/` — 121 individual mode-shape plots
- `compare_mar3_mar6_1f.png` — Mar 3 vs Mar 6 mode shape at y = 2
- `recovery_check_test2_{1920,1930,1970}.png` — 2D maps + mode shape comparison
- `pressure_buildup_stepA_sweep_{1930,1970}.png` — time-resolved pressure field (Mar 3 data)
- `pressure_buildup_slices_stepA_sweep_{1930,1970}.png` — mode shape snapshots
- `pressure_ringup_stepA_sweep_{1930,1970}.png` — p0 + current ring-up

## Pressure build-up analysis (March 3 data)

Time-resolved short-time DFT (10 µs window, 5 µs step) on March 3 sweep files:

**1.970 MHz (main resonance):** Current reaches steady state by ~30 µs; pressure takes ~120 µs. The mode shape is purely half-wavelength throughout — same shape, just scaling up.

**1.930 MHz (secondary peak):** Current overshoots at ~10 µs then decays with oscillations (~100 µs to settle). Pressure rises more smoothly, steady by ~80 µs. The current overshoot + ringing is characteristic of off-resonance transient excitation.

## Next steps

- [ ] Analyse 2f excitation data (stepA_2f_3860–3940)
- [ ] Fine frequency sweep around new resonance (1.905–1.915 MHz) to pinpoint peak
- [ ] Check if axial pattern is stable over time or drifts
- [ ] Proceed to Step B at updated resonance frequency and axial position

## Equipment

Same as March 3 (see `reports/2026-03-03_stepA_freq_sweep.md`).
