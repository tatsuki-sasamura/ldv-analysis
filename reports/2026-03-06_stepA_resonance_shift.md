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

### Fine frequency sweep (test5)

After a second channel degradation (test4 data unusable — RSSI ~1.90, p0 ~75–88 kPa with no mode shape), another flush restored the channel with improved coupling (RSSI 2.17, p0 27% higher than test3 at same conditions).

56 frequencies at x = 9 mm (101×2 grid), non-uniform steps:
- 1.900–1.920 MHz in 1 kHz (fine, lower peak)
- 1.920–1.960 MHz in 5 kHz (coarse, valley)
- 1.960–1.980 MHz in 1 kHz (fine, upper peak)
- 1.980–2.020 MHz in 5 kHz (coarse, anti-resonance + tail)

| Feature | test3 (coarse) | test5 (fine) |
|---------|---------------|-------------|
| Lower acoustic peak | 1.910 MHz, 1335 kPa | **1.907 MHz, 1244 kPa** |
| Upper acoustic peak | 1.970 MHz, 766 kPa | **1.974 MHz, 595 kPa** |
| Valley minimum | — | 1.950 MHz, 134 kPa |
| Electrical resonance (I max) | 1.970 MHz | **1.966 MHz, 24.0 mA** |
| Electrical anti-resonance (I min) | ~1.990 MHz | **1.995 MHz, 4.6 mA** |
| V-I phase zero crossing | — | ~1.990–1.995 MHz |

The lower mode is 2× stronger than the upper at this axial position. The upper mode has persistently negative R² at x = 9 mm, suggesting its axial antinode is at a different position.

Pressure persists at the electrical anti-resonance (~500 kPa at 1.990–1.995 MHz despite I = 4.6 mA) because the high impedance means larger voltage across the PZT.

### 2f harmonic content

The 2f component (evaluated at 2×f_drive) is ~2.5% of the 1f amplitude at resonance (31 kPa vs 1244 kPa at 1.909 MHz). It tracks the 1f resonance shape, indicating nonlinear distortion of the fundamental rather than an independent 2f acoustic mode.

## New scripts

| Script | Purpose |
|--------|---------|
| `experiments/2026W10_stepA/pressure_buildup.py` | Time-resolved pressure field evolution during burst. Sliding short-time DFT, pcolormesh, mode-shape snapshots, and p0 + current ring-up dual-axis plot |
| `experiments/2026W10_stepA/freq_axial_sweep.py` | Frequency × axial position sweep (test3 data). Mode-shape fits at each (f, x), heatmaps (p0, RSSI, R²), and 4-panel freq sweep at best x |
| `experiments/2026W10_stepA/freq_sweep_fine.py` | Fine frequency sweep (test5 data). 4-panel sweep, individual + overview mode-shape plots |

## Data files

### Pre-flush (degraded)
- `stepA_1f_1930.tdms`, `stepA_1f_1970.tdms` — 1f reference, 101×2 grid at y = 2 mm

### Post-flush, wide scans
- `test2_1920.tdms`, `test2_1930.tdms`, `test2_1970.tdms` — 101×11 grid (y = 2–12 mm)

### Coarse frequency sweep
- `test3_1900.tdms` … `test3_2000.tdms` — 11 files, 21×11 grid, 10 kHz steps

### Failed fine sweep (dead channel)
- `test4_1900.tdms` … `test4_1913.tdms` — 14 files, unusable (flat mode shape, p0 ~75–88 kPa)

### Fine frequency sweep (post second flush)
- `test5_1900.tdms` … `test5_2015.tdms` — 56 files, 101×2 grid at x = 9 mm, non-uniform frequency steps

### 2f excitation (not yet analysed)
- `stepA_2f_3860.tdms` … `stepA_2f_3940.tdms` — 5 files at 2× frequency

All in `C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260306experimentA/`.

## Output files

All in `experiments/2026W10_stepA/output/`:

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
- `freq_sweep_test5.png` — 56-point fine frequency sweep (4-panel)
- `mode_shapes_test5/` — 56 individual mode-shape plots
- `mode_shapes_test5_overview.png` — grid overview of all mode shapes
- `harmonics_1f_2f_test5.png` — 1f vs 2f harmonic comparison

## Pressure build-up analysis (March 3 data)

Time-resolved short-time DFT (10 µs window, 5 µs step) on March 3 sweep files:

**1.970 MHz (main resonance):** Current reaches steady state by ~30 µs; pressure takes ~120 µs. The mode shape is purely half-wavelength throughout — same shape, just scaling up.

**1.930 MHz (secondary peak):** Current overshoots at ~10 µs then decays with oscillations (~100 µs to settle). Pressure rises more smoothly, steady by ~80 µs. The current overshoot + ringing is characteristic of off-resonance transient excitation.

## Transient ring-up/down analysis (test5)

Improved `transient_ringup_fit.py` with sliding single-frequency DFT (replacing noisy Hilbert envelope), multi-point p_ss-weighted averaging across all valid scan points, and a 5 µs skip of the initial flat region (physical acoustic propagation delay). All 57 test5 files processed successfully.

### Two distinct Q factors

| Quantity | Method | Value |
|----------|--------|-------|
| Q_chip (acoustic cavity) | Ch2 rise τ at 1.91 MHz | **~100** |
| Q_PZT (motional branch) | Ch4 rise τ at 1.967 MHz | **~120–130** |

At 1.91 MHz (chip resonance), Ch2 rise/fall give Q ≈ 100 with negligible asymmetry — the acoustic cavity dominates. At 1.967 MHz (PZT resonance), rise Q < fall Q because the rise is a coupled system (PZT ramping + cavity filling) while the fall is free cavity decay.

The ~5 µs flat region at burst transitions was proven physical (not a window artifact) by testing multiple DFT window sizes — the duration is independent of window width, consistent with the time for a standing wave to establish (~12 cycles at 2 MHz).

## 2D pressure map at 1f chip resonance (test6_1907)

101×101 area scan at 1.907 MHz. Clean half-wavelength mode shape $p(y) = p_0 |\sin(\pi y/W)|$:

| Quantity | Value |
|----------|-------|
| Peak fitted $p_0$ | **1562 kPa** at x = 8.30 mm |
| $p_0$ range along channel | 253–1562 kPa |
| Axial node | x ≈ 3.5 mm |
| Axial antinode | x ≈ 8.1 mm |

From the node-antinode spacing: $\lambda_\text{axial} = 4 \times (8.1 - 3.5) = 18.4$ mm. Combined with the width mode $k_y = \pi/W$:

$$k = \sqrt{k_x^2 + k_y^2} \Rightarrow \lambda = 0.749 \text{ mm} \Rightarrow c = f\lambda = 1429 \text{ m/s}$$

Slightly below $c_\text{water} \approx 1500$ m/s — the ~5% discrepancy is consistent with non-rigid wall boundary conditions (effective acoustic width ~393 µm > physical 375 µm).

## 2D pressure map at PZT resonance (test6_1974)

101×101 area scan at 1.974 MHz. **Chaotic mode shape** — no clean sinusoidal structure. The axial profile shows rapid, irregular oscillations (p_0 = 160–684 kPa) with no coherent envelope. This confirms 1.974 MHz is not a chip acoustic resonance; the PZT drives hard but the cavity doesn't support a clean standing wave.

## 2f chip resonance search (test7, continuous excitation)

Frequency sweep 3.700–3.900 MHz in 5 kHz steps, 21×2 line scans. Mode shape fit: $p(y) = p_0 |\cos(2\pi y/W)|$ (full-wavelength mode with antinode at centre, nodes at ±W/4).

| Quantity | Value |
|----------|-------|
| 2f resonance frequency | **3.845 MHz** |
| Peak $p_0$ | 90 kPa |
| Best R² | 0.88 at 3.795 MHz |
| $f_\text{2f} / f_\text{1f}$ | 3.845 / 1.907 = **2.016** (≈ 2×, as expected) |

Multiple peaks in the sweep (3.785, 3.845, 3.885 MHz) — likely the same 2f width mode at different axial orders. Each peak spans ~30–40 kHz, broader than the expected FWHM of ~19 kHz for Q = 200, possibly due to overlapping axial modes in the 1D scan.

### 2f burst-mode transient (test7appendix_burst_3845)

| Quantity | Value |
|----------|-------|
| $p_\text{ss}$ | 92 kPa (consistent with continuous: 90 kPa) |
| Ch2 rise τ | 19.2 µs → **Q = 232** |
| Ch2 fall τ | 22.2 µs → **Q = 268** |
| Ch4 motional | τ_mot = 0.1 µs, Q = 1 (PZT is a pure capacitor at 3.8 MHz) |

Q_2f ≈ 230–270 is significantly higher than Q_1f ≈ 100. Physical explanation: the viscous boundary layer thickness scales as $\sim 1/\sqrt{f}$, so wall losses decrease at higher frequency relative to stored energy.

### 2f harmonic in 1f data — spontaneous mode excitation

Extracting the 2f DFT component from the test6_1907 (1.907 MHz drive) area scan:

- **2f mean pressure = 23 kPa** (~5% of 1f)
- **Spatial pattern follows the 2f mode** — pressure concentrated at channel centre (y = 0), exactly where |cos(2πy/W)| has its antinode, while the 1f mode has its node there
- 2×1.907 = 3.814 MHz is only 30 kHz below the 2f resonance at 3.845 MHz

This demonstrates nonlinear harmonic generation from the 1f standing wave, selectively amplified by the nearby 2f cavity resonance. The spatial pattern confirms it is a real acoustic 2f mode, not measurement artifact.

### 2f at PZT resonance (1.974 MHz) — no coherent mode

Extracting the 2f component from the test6_1974 (1.974 MHz drive) area scan:

| Quantity | 1.907 MHz (channel res.) | 1.974 MHz (PZT res.) |
|----------|-------------------------|---------------------|
| 1f $p_0$ | 1562 kPa | 684 kPa |
| 2f $p_0$ | 84 kPa | 32 kPa |
| 2f/1f | 5.4% | 4.7% |
| 2f spatial pattern | Clean $|\cos(2\pi y/W)|$ | No coherent structure |
| $2f_\text{drive}$ | 3.814 MHz (in 2f band) | 3.948 MHz (outside 2f band) |

The 2f/1f **ratio** is similar (~5%), but the 2f spatial pattern at 1.974 MHz is incoherent — no standing wave forms. Two effects are confounded:

1. **Off-resonance at 2f**: $2 \times 1.974 = 3.948$ MHz is well outside the 2f channel resonance band (test7 shows 2f modes at 3.775–3.900 MHz), so there is no cavity Q-enhancement.
2. **Incoherent 1f source**: the 1f mode at 1.974 MHz is itself chaotic (no clean standing wave), so the nonlinear driving term is spatially disordered.

These cannot be separated from the available data. However, the key conclusion stands: **a coherent 2f standing wave requires both a strong 1f mode and a resonant 2f channel mode at $2f_\text{drive}$**.

## 2f area scans (test8)

Mode-shape fit: $p(y) = p_0 |\cos(2\pi y/W)|$. All ~10× weaker than the 1f resonance (1562 kPa at 1.907 MHz).

### test8_3785 (3.785 MHz)

| Quantity | Value |
|----------|-------|
| Peak fitted $p_0$ | **104 kPa** at x = 11.4 mm |
| $p_0$ range along channel | 23–104 kPa |

Two axial antinodes (x ≈ 4.4 and 11 mm) with a node at x ≈ 6–7 mm.

### test8_3845 (3.845 MHz — main 2f resonance)

| Quantity | Value |
|----------|-------|
| Peak fitted $p_0$ | **97 kPa** at x = 8.2 mm |
| $p_0$ range along channel | 10–97 kPa |

Many closely-spaced axial antinodes (x ≈ 3.9, 6.2, 8.2, 9.8 mm) — a higher axial order than 3.785 MHz.

### test8_3885 (3.885 MHz)

| Quantity | Value |
|----------|-------|
| Peak fitted $p_0$ | **116 kPa** at x = 6.6 mm |
| $p_0$ range along channel | 13–116 kPa |

~4 axial antinodes concentrated at x = 6–10 mm.

### test8_3814 (3.814 MHz = 2 × 1.907 MHz)

The exact frequency of the nonlinear 2f harmonic generated by 1f drive.

| Quantity | Value |
|----------|-------|
| Peak fitted $p_0$ | **122 kPa** at x = 8.8 mm |
| $p_0$ range along channel | 16–122 kPa |

Broad axial antinodes at x ≈ 4.5 and 8.8 mm, similar to 3.785 MHz (low axial order). Comparable p₀ to the three resonance peaks confirms 3.814 MHz is well within the 2f resonance bandwidth (Δf = 31 kHz from 3.845 MHz peak, vs FWHM ≈ 3845/250 ≈ 15 kHz). Note: all test8 2f scans used 25 Vpp (vs 10 Vpp for 1f scans).

### Comparison

| Freq (MHz) | Peak p₀ (kPa) | Best x (mm) | Axial character |
|------------|---------------|-------------|-----------------|
| 3.785 | 104 | 11.4 | 2 broad lobes |
| 3.814 | 122 | 8.8 | 2 broad lobes (low axial order) |
| 3.845 | 97 | 8.2 | ~5 closely-spaced antinodes |
| 3.885 | 116 | 6.6 | ~4 antinodes |

All four give comparable p₀ (~100–120 kPa at 25 Vpp), despite the 1D sweep (test7) showing 3.845 MHz as the dominant peak. This confirms the four frequencies are different axial orders of the same 2f width mode; the 1D sweep was biased by the scan line position.

### Serpentine scan artifact

The 2f area scans (especially test8_3785) show horizontal stripes in both pressure and phase maps. Cause: the Polytec scanner uses serpentine (bidirectional) y-scanning, and stage position hysteresis between forward and reverse passes creates a systematic offset (~few µm). For the 2f mode, the steep spatial gradient of |cos(2πy/W)| amplifies this into visible alternating bright/dark rows. The forward-reverse phase offset is 26° at 3.785 MHz vs 8° at 3.845 MHz. The fitted p₀ values (projected across the full width profile) are unaffected; only the 2D visualizations are degraded. Future 2f area scans should use unidirectional scanning.

## New scripts

| Script | Purpose |
|--------|---------|
| `experiments/2026W10_stepA/freq_sweep_2f.py` | 2f frequency sweep (test7 data). |cos| mode-shape fit, 4-panel sweep, individual + overview mode-shape plots |
| `experiments/2026W10_stepA/pressure_map_2d.py` | 2D pcolormesh maps (velocity, pressure, phase, RSSI) + mode-shape fit (auto-detects 1f/2f from drive freq) + p₀(x) axial profile. `--harmonics` flag extracts 2f from raw waveforms and generates stacked 1f/2f comparison maps |

## Updated data files

### 2D area scans
- `test6_1907.tdms` — 101×101 grid, burst, 1f chip resonance
- `test6_1974.tdms` — 101×101 grid, burst, PZT resonance

### 2f coarse frequency sweep (continuous excitation)
- `test7_3700.tdms` … `test7_3900.tdms` — 41 files, 21×2 grid, 5 kHz steps

### 2f burst appendix
- `test7appendix_burst_3845.tdms` — 101×2 grid, burst, 2f chip resonance

### 2f area scans
- `test8_3785.tdms` — 51×51 grid, burst, 2f at 3.785 MHz
- `test8_3845.tdms` — 51×51 grid, burst, 2f at 3.845 MHz (main 2f resonance)
- `test8_3885.tdms` — 51×51 grid, burst, 2f at 3.885 MHz
- `test8_3814.tdms` — 51×51 grid, burst, 2f at 3.814 MHz (= 2×1.907 MHz)

All in `C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260306experimentA/`.

## 25 Vpp frequency sweep (test9, 1.900–1.912 MHz)

Quick check that the 1f resonance frequency doesn't shift at higher drive voltage. 101×2 line scans at x = 9 mm, continuous excitation, 1 kHz steps. Also extracted the 2f harmonic (DFT at 2×f_drive) and fitted |cos(2πy/W)| mode shapes.

### 1f resonance unchanged

| Quantity | 10 Vpp (test5) | 25 Vpp (test9) | Ratio |
|----------|---------------|---------------|-------|
| Peak $p_0^{1f}$ | 1244 kPa at 1.907 MHz | **3151 kPa at 1.905 MHz** | 2.52× |
| Expected ratio (25/10) | — | — | 2.5× |

$p_0 \propto V$ confirmed — no saturation, no resonance shift from PZT self-heating at 25 Vpp. The 1f resonance broadens into a flat plateau (~3050–3150 kPa from 1.905–1.913 MHz) compared to the sharper peak at 10 Vpp, suggesting nonlinear broadening or thermal effects at higher drive.

Note: raw Ch2 waveforms occasionally hit the ±2 m/s decoder ceiling, but these are momentary signal dropouts (brief RSSI dips within a burst), not velocity saturation. The DFT-extracted 1f amplitude never exceeds 1.21 m/s at any good-RSSI point, and the fitted $p_0$ is unaffected.

### 2f harmonic scales faster than $V^2$

| Quantity | 10 Vpp (test5) | 25 Vpp (test9) | Ratio |
|----------|---------------|---------------|-------|
| Peak $p_0^{2f}$ | 31 kPa | **292 kPa at 1.909 MHz** | 9.3× |
| Expected ratio $(25/10)^2$ | — | — | 6.25× |
| $p_0^{2f}/p_0^{1f}$ | 2.5% | **9.4%** | 3.7× |
| Expected ratio (25/10) | — | — | 2.5× |

The 2f harmonic scales faster than the naive $p_{2f} \propto V^2$ prediction. The full voltage sweep (5 steps) will determine the actual exponent.

### Non-monotonic 2f/1f ratio vs frequency

The 2f/1f ratio peaks at 1.909 MHz (9.4%) and drops sharply at 1.912 MHz (6.8%), despite the single-resonance detuning factor $\cos\theta = 1/\sqrt{1 + (2Q_2\Delta f/f_2)^2}$ monotonically increasing as $2f_\text{drive}$ approaches the 2f resonance at 3.845 MHz:

| $f_\text{drive}$ (MHz) | $2f$ (MHz) | $\Delta f$ (kHz) | $\cos\theta$ | Measured ratio |
|----------|---------|----------|-----------|---------------|
| 1.900 | 3.800 | +45 | 0.17 | 4.2% |
| 1.905 | 3.810 | +35 | 0.22 | 8.6% |
| 1.909 | 3.818 | +27 | 0.27 | **9.4%** |
| 1.912 | 3.824 | +21 | 0.34 | 6.8% |

The drop at 1.912 MHz cannot be explained by detuning alone. The 2f area scans (test8) revealed that the "2f resonance" is not a single mode — it comprises multiple axial orders with completely different spatial profiles:

- 3.814 MHz: 2 broad axial lobes (low order)
- 3.845 MHz: ~5 closely-spaced antinodes (high order)

The effective 2f transfer function at $x = 9$ mm is a superposition of multiple Lorentzians:

$$\frac{p_{2f}}{p_{1f}} \propto \sum_n \frac{A_n(x)}{1 + (2Q_n \Delta f_n / f_n)^2}$$

where each axial mode $n$ has its own spatial weight $A_n(x)$ at the measurement position. At 1.909 MHz ($2f = 3.818$), the harmonic sits on a local maximum of the low-order 3.814-type mode at $x = 9$ mm. By 1.912 MHz ($2f = 3.824$), it falls into a valley between axial orders before reaching the 3.845-type mode.

### New scripts

| Script | Purpose |
|--------|---------|
| `experiments/2026W10_stepA/freq_sweep_25vpp.py` | 25 Vpp frequency sweep with 1f + 2f harmonic analysis, 5-panel sweep plot, dual mode-shape fits |

### Data files

- `test9_1900.tdms` … `test9_1912.tdms` — 13 files (so far), 101×2 grid, continuous, 25 Vpp, x = 9 mm

All in `C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260306experimentA/`.

## Next steps

- [x] ~~Fine frequency sweep around new resonance~~ → done (test5)
- [x] ~~101×101 2D map at 1.907 MHz~~ → done (test6_1907)
- [x] ~~2D map at PZT resonance for comparison~~ → done (test6_1974, chaotic)
- [x] ~~2f coarse frequency search~~ → done (test7, peak at 3.845 MHz)
- [x] ~~2f burst transient for Q~~ → done (Q ≈ 230–270)
- [x] ~~2D maps at three 2f peaks (3.785, 3.845, 3.885 MHz)~~ — done (p₀ ≈ 100 kPa at all three; different axial orders)
- [x] ~~2D map at 3.814 MHz (= 2×1.907 MHz)~~ — done (p₀=122 kPa, within 2f resonance bandwidth)
- [ ] Voltage sweep: V = 2–24 Vpp in 2V steps, 11 frequencies around 1.907 MHz, to measure V–p₀ linearity and track f_res shift
- [ ] Proceed to Step B at updated resonance frequency and axial position

## Equipment

Same as March 3 (see `reports/2026-03-03_stepA_freq_sweep.md`).
