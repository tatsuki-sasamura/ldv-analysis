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

## Scripts

All in `experiments/2026W10_stepA/`:

| Script | Data | Purpose |
|--------|------|---------|
| `freq_sweep_coarse.py` | test3 | Coarse freq sweep (1.900–2.000 MHz, 10 kHz steps). 4-panel sweep + mode shapes |
| `freq_sweep_fine.py` | test5 | Fine freq sweep (56 frequencies, non-uniform steps). 4-panel sweep + mode shapes |
| `freq_sweep_25vpp.py` | test9 | 25 Vpp freq sweep with 1f + 2f harmonic analysis. 5-panel sweep + dual mode-shape fits |
| `freq_sweep_2f.py` | test7 | 2f freq sweep (3.700–3.900 MHz). \|cos\| mode-shape fit, 4-panel sweep |
| `freq_axial_sweep.py` | test3 | Frequency × axial position sweep. Heatmaps (p₀, RSSI, R²), 4-panel at best x |
| `pressure_map_2d.py` | test6, test8 | 2D pcolormesh maps + mode-shape fit. Auto-detects 1f/2f. `--harmonics` for 2f extraction |
| `pressure_buildup.py` | Mar 3 data | Time-resolved pressure field during burst. Sliding DFT, snapshots, ring-up plot |
| `transient_ringup_fit.py` | test5 | Ring-up/down τ fitting via sliding DFT envelope. Q estimation |
| `single_mode_shape.py` | test5 | Single-frequency mode-shape analysis |
| `thermal_drift_check.py` | test5 | Thermal drift check across repeated measurements |
| `voltage_sweep.py` | test10 | Voltage sweep (5–25 Vpp). 1f + 2f p₀ vs V, linear/quadratic fits |

Output goes to per-script subdirectories under `experiments/2026W10_stepA/output/` (gitignored). FFT caches in `output/cache/`.

## Data files

All in dataset `20260306experimentA` (resolved via `get_data_dir()`).

| Files | Grid | Description |
|-------|------|-------------|
| `stepA_1f_{1930,1970}.tdms` | 101×2 | Pre-flush 1f reference at y = 2 mm (degraded) |
| `test2_{1920,1930,1970}.tdms` | 101×11 | Post-flush wide scans, y = 2–12 mm |
| `test3_1900…2000.tdms` (11) | 21×11 | Coarse freq sweep, 10 kHz steps |
| `test4_1900…1913.tdms` (14) | — | Failed sweep (dead channel, unusable) |
| `test5_1900…2015.tdms` (56) | 101×2 | Fine freq sweep at x = 9 mm |
| `test6_1907.tdms` | 101×101 | 1f 2D map at chip resonance |
| `test6_1974.tdms` | 101×101 | 1f 2D map at PZT resonance |
| `test7_3700…3900.tdms` (41) | 21×2 | 2f coarse freq sweep (continuous) |
| `test7_3905…4000.tdms` (20) | 21×2 | 2f extended sweep |
| `test7appendix_burst_{3785,3814,3845,3885}.tdms` | 101×2 | 2f burst transients |
| `test8_{3785,3814,3845,3885}.tdms` | 51×51 | 2f area scans |
| `test9_1900…1912.tdms` (13) | 101×2 | 25 Vpp freq sweep at x = 9 mm |

Step B data in dataset `20260307experimentB`:

| Files | Grid | Description |
|-------|------|-------------|
| `test10_1907_{5,10,15,20,25}Vpp_*.tdms` (5) | 2D | Voltage sweep at 1.907 MHz |

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

The 2f/1f **ratio** is similar (~5%), but the 2f spatial pattern at 1.974 MHz is incoherent — no standing wave forms. Two possible explanations:

1. **Incoherent 1f source**: the 1f mode at 1.974 MHz is itself chaotic (no clean standing wave), so the nonlinear driving term is spatially disordered — even if a 2f channel mode existed at 3.948 MHz, it would be poorly driven.
2. **No 2f channel mode at 3.948 MHz**: possible but unconfirmed — test7 only covered 3.700–3.900 MHz. A sweep up to ~4.0 MHz would be needed to rule this out.

These cannot be separated from the available data. The incoherent 1f source alone is sufficient to explain the lack of 2f structure — no claim about 2f channel response at 3.948 MHz can be made without measurement.

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


## 25 Vpp frequency sweep (test9, 1.900–1.912 MHz)

Quick check that the 1f resonance frequency doesn't shift at higher drive voltage. 101×2 line scans at x = 9 mm, 1 kHz steps. Also extracted the 2f harmonic (DFT at 2×f_drive) and fitted |cos(2πy/W)| mode shapes.

**Note**: test9 used 2000 burst cycles (2000/1.907 MHz ≈ 1049 µs), which exceeds the 800 µs capture window. The falling edge is not captured, so the fft_cache treats the data as continuous. Steady-state FFT is unaffected, but **ring-down transients cannot be extracted from test9**. At ~2 MHz, use 1000 cycles (~524 µs) to fit burst + ring-down within 800 µs. At ~4 MHz, 2000 cycles (~500 µs) is fine.

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


## Completed tasks

- Fine frequency sweep (test5), 2D maps (test6), 2f search (test7), 2f area scans (test8), 25 Vpp sweep (test9)
- Step B voltage sweep (test10): p₀_1f = 752–4031 kPa (5–25 Vpp), linear 162.6 kPa/V; p₀_2f quadratic 0.622 kPa/V²
- Burst detection bug fix: multi-probe reference selection in `fft_cache.py`
- Data quality filtering: edge-margin + RSSI in `pressure_map_2d.py` and `voltage_sweep.py`

## Open tasks

- [ ] **Transient envelope ringing biases τ estimation**: The sliding DFT envelope shows oscillatory ringing (underdamped beat between closely-spaced modes), visible in all 2f burst data and at the PZT resonance. Current fit model `exp(-t/τ)` ignores this ringing, biasing τ — especially during fall where the envelope crosses zero and `|env|` creates cusps. Evidence: rise/fall Q discrepancy correlates with ringing severity (3814 kHz: Q_rise=155 vs Q_fall=342, factor 2.2×; 1907 kHz with minimal ringing: Q_rise=102 vs Q_fall=105, nearly equal). Fix: fit `exp(-t/τ)·cos(2π·Δf·t + φ)` to the complex envelope (preserving phase), or fit magnitude with `exp(-t/τ)·|cos(2π·b·t + φ)|` model. The beat frequency b encodes the mode splitting, which is physically meaningful.

## Equipment

Same as March 3 (see `reports/2026-03-03_stepA_freq_sweep.md`).
