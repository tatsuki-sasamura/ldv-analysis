# Step A Frequency Sweep — 2026-03-03

## Purpose

Identify the half-wavelength resonance of the acoustofluidic chip by sweeping the drive frequency and measuring the 1f acoustic pressure amplitude via refracto-vibrometry.

## What's new since 2026-03-01

### New scripts (`experiments/2026W10_stepA/`)

Six analysis scripts were added for the Week 10 Step A experiment:

| Script | Purpose |
|--------|---------|
| `fft_cache.py` | Shared FFT cache — burst detection, exact-frequency DFT, per-point 1f extraction (velocity, pressure, phase, voltage, current, impedance), cached to `.npz` for fast re-runs |
| `freq_sweep.py` | Batch-processes all sweep files, fits sinusoidal mode shape at each frequency, plots 4-panel resonance curve |
| `analyze_single.py` | Single-file mode shape, waveform/spectrum, repeatability (3 y-lines) |
| `map_2d.py` | 2D pcolormesh maps with channel boundary detection and centred coordinates |
| `thermal_drift_check.py` | Ch1/Ch4 electrical stability across scan (running median, missed-burst flagging) |
| `transient_fit.py` | Ring-up/ring-down envelope fitting for Q estimation |

All scripts share `fft_cache.py` to avoid duplicated burst detection and FFT code. The cache stores pre-computed 1f quantities per scan point; subsequent runs load from `.npz` directly.

### Signal processing

- **Exact-frequency DFT** instead of nearest-bin FFT — evaluates the DFT at the precise drive frequency via dot product ($X = w_{ss} \cdot e^{-2j\pi f t}$), eliminating spectral leakage / scalloping loss from the rectangular window.
- **Sub-bin drive frequency detection** — parabolic interpolation on the log-magnitude spectrum of the full burst record for sub-Hz accuracy.
- **Data quality filtering:**
  - RSSI > 1.0 V (from distribution analysis: bimodal gap separates good LDV lock from failed points)
  - $V_{1f} > 0.5 \times \text{median}(V_{1f})$ to exclude missed burst triggers

### Mode-shape fitting

At each frequency, fit $p(y) = p_0 |\sin(\pi y / W)|$ by least-squares projection, with channel centre $x_c$ optimised by brute-force search. R² reported per frequency (0.4–0.8 range; limited by SNR and edge artifacts, not the fitting method).

### Burst trigger test report

`reports/2026-03-03_burst_trigger_test.md` — verified that LabVIEW/PicoScope acquisition is Ch1-triggered, burst timing captured correctly, ~490 µs usable steady state.

### Minor changes

- `src/ldv_analysis/config.py`: added Polytec decoder impedance matching notes (explains VELOCITY_SCALE = 1.0)
- `pyproject.toml`: added `scienceplots` to dependencies
- `.vscode/settings.json`: Windows Python path

## Data

- **29 sweep files:** `stepA_sweep_1920.tdms` … `stepA_sweep_2020.tdms`
  - 5 kHz steps from 1920–1960 kHz, 1–2 kHz steps near resonance (1960–1975 kHz), 5 kHz steps to 2020 kHz
- **Scan grid per file:** 101 x-points × 2 y-lines (x = channel width, y = channel length)
- **Scan position:** y ≈ 2.000 mm along channel, chosen from 2D area scan as an axial antinode
- **Drive:** burst mode, ~5 Vpp, 100 µs ring-up margin before FFT window

## Results

| Quantity | Value |
|----------|-------|
| Resonance frequency | **1.970 MHz** |
| Peak $p_0$ | **877 kPa** |
| V-I phase zero crossing | Near 1.970 MHz (series resonance) |
| Drive voltage | 4.96–5.08 V (stable, ~2% variation from amplifier loading) |
| Transient Q estimates | $Q_\text{Ch2} \approx 116$ ($\tau \approx 19$ µs), $Q_\text{Ch4} \approx 197$ ($\tau \approx 32$ µs) |

The full resonance curve is captured: $p_0$ rises from ~190 kPa at 1.920 MHz, peaks sharply at 1.970 MHz, falls to ~65 kPa at 1.995–2.005 MHz, then rises slightly to ~145 kPa at 2.015–2.020 MHz (onset of another mode).

### Multi-mode structure

The sweep reveals structure beyond a single resonance:

- **Secondary pressure peak at ~1.930 MHz:** $p_0 \approx 650$ kPa, visible as a distinct local maximum before the main peak. Origin unclear — could be a second acoustic mode (separate BVD branch) or a structural resonance of the chip/PZT assembly coupling pressure into the channel.
- **Electrical anti-resonance at ~1.945–1.950 MHz:** current dips to a local minimum (~13.5 mA), typical BVD behaviour (impedance maximum between series resonances).
- **Main resonance at 1.970 MHz:** V-I phase zero crossing, current peak, pressure maximum.

The anti-resonance between the two features is characteristic of a BVD (Butterworth-Van Dyke) equivalent circuit with multiple motional branches. The line shape is therefore not a simple Lorentzian — fitting $p_0(f)$ requires a multi-branch BVD model or restricting the fit to the main peak only.

### Comparison with literature

Barnkob et al. 2010 (similar chip, $w = 377$ µm) reported $Q \approx 200\text{--}577$, $p_a \approx 0.08\text{--}0.66$ MPa — our results are consistent in order of magnitude.

## Output files

All in `output/2026W10stepA/`:

- `freq_sweep.png` — 4-panel resonance curve ($p_0$, V-I phase, current, voltage vs frequency)
- `mode_shapes/mode_shape_*kHz.png` — 29 individual mode-shape plots with sinusoidal fit and R²
- `map2d_{pressure,velocity,phase,rssi}_1f_*.png` — 2D spatial maps from area scans
- `map2d_p0_vs_y_*.png` — fitted $p_0$ along channel length
- `waveform_stepA1967.png` — representative burst waveform and spectrum
- `repeatability_stepA1967.png` — 3 y-lines overlaid
- `thermal_drift_stepA1967.png` — electrical stability across scan
- `transient_fit_stepA1967.png` — ring-up/ring-down envelope fits

## Next steps

- [ ] Normalise $p_0$ by drive voltage for a proper transfer function
- [ ] Extract Q — simple Lorentzian won't work due to anti-resonance; consider BVD multi-branch fit or half-power bandwidth on the main peak only
- [ ] Compare Q from resonance width with Q from transient ring-up
- [ ] Proceed to Step B: high-voltage pulsed drive at resonance frequency

## Equipment

| Item | Model | Notes |
|------|-------|-------|
| Function generator | Agilent 33250A (80 MHz) | Burst mode, CW within burst |
| Amplifier | AG1020 | 25 W RF, 0.15–400 MHz |
| ADC | PicoScope 5442D (4-ch, 14-bit, 125 MS/s) | |
| LDV | Polytec VibroFlex (VFX-F-110 + VIB-A-511) | Velocity decoder, 2 m/s/V @ 50 Ω |
| Current probe | Tektronix CT-1 (5 mV/mA) | |
| Chip | Glass-silicon, $w = 375$ µm, $H = 150$ µm | |
