# Burst Trigger Test — 2026-03-03

## Data

- **File:** `bursttest1.tdms` (+ `bursttest1.tdms_index`)
- **Scan grid:** 101 x-points × 2 y-lines = 202 points
- **Sampling:** 125 MHz (dt = 8 ns), 100,000 samples/record → 0.8 ms record duration
- **4 channels:** Ch1 (drive voltage), Ch2 (LDV velocity), Ch3 (LDV displacement), Ch4 (current)
- **Drive:** ~1.970 MHz, burst mode (CW, low voltage)

## Purpose

Verify that LabVIEW/PicoScope burst-mode acquisition is Ch1-triggered and timing is known, as a prerequisite for using pulsed drive at high voltages in Step B. See [FIRST-REPORT experiment plan W10](file:///C:/Users/Tatsuki%20Sasamura/Documents/Research/Projects/Paper/FIRST-REPORT%20experiment%20plan%20W10.md).

## Results

### Ch1 (drive voltage) — burst envelope confirmed

- RMS envelope analysis (1000-sample chunks) shows:
  - **ON:** 0–504 µs, RMS ≈ 0.155 V (pk-pk ≈ 0.49 V)
  - **OFF:** 512–800 µs, RMS ≈ 0.016 V (noise floor)
- Sharp turn-off at ~505 µs — **Ch1-triggered acquisition confirmed**

### Ch2 (velocity) — acoustic signal detected

- Ch2 is **not flat** — clear ON/OFF contrast at points inside the channel
- **88 / 202 scan points** have ON/OFF RMS ratio > 1.5
- Strongest response in channel center (points ~80–180, x ≈ 30.2–30.5 mm): up to **ratio 8.4×**
- Edge points (0–20, 195–201): ratio ~1.0 (noise only, as expected)

### Waveform structure (point 120, ON/OFF ratio = 4.7×)

| Region | Time window | Observation |
|--------|------------|-------------|
| Turn-ON transient | 0–10 µs | Ch1 starts immediately; Ch2 builds up over ~5–10 µs (acoustic ring-up) |
| Steady state | ~10–500 µs | Both channels periodic and stable; ~490 µs usable = ~980 cycles at 2 MHz |
| Turn-OFF transient | 505–515 µs | Ch1 drops abruptly; Ch2 rings down over ~5–10 µs |

![Burst waveform visualization](../output/bursttest1_waveform.png)

## Conclusions

1. **Trigger test passes.** Acquisition is Ch1-triggered; burst timing is captured correctly. No need to consult Ola about LabVIEW trigger settings.
2. **Burst drive produces detectable acoustic signal** with clear spatial structure across the channel.
3. **Usable steady-state window:** ~490 µs after ~10 µs ring-up. Sufficient for FFT (980 cycles, df ≈ 2 kHz).
4. **Ring-up/ring-down (~5–10 µs)** can independently estimate Q (τ = Q / πf₁).
5. **Pulsed drive is viable for Step B** — no issues with acquisition or signal quality.

## Equipment

| Item | Model | Notes |
|------|-------|-------|
| Function generator | Agilent 33250A (80 MHz) | Burst mode, free-running |
| Amplifier | EVAL-ADA4870 (low voltage) | Used for this test; AG1020 or AR 75A250A for Step B |
| ADC | PicoScope 5442D (4-ch, 14-bit, 125 MS/s) | |
| LDV | Polytec VibroFlex (VFX-F-110 + VIB-A-511) | |
| Current probe | Tektronix CT-1 (5 mV/mA, 500 mA RMS max) | |
