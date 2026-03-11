# SNR Assessment — 2026-03-11

## Method

Signal-to-noise ratio was assessed using the post-burst segment of burst-mode LDV recordings. In burst excitation, the piezo drives for ~528 µs then shuts off; the remaining ~170 µs of the 800 µs record contains no acoustic signal and represents the LDV decoder noise floor.

**Noise extraction:** Per-point RMS of the Ch2 (velocity) waveform from 100 µs after burst-off to the end of the record (~21,500 samples at 125 MHz). The 100 µs skip allows the acoustic ring-down to decay. The velocity RMS is converted to equivalent pressure noise using the same refracto-vibrometry calibration: `p_noise = v_rms / (2πf × H × dn/dp)`.

**SNR definition:** `SNR = 20 log10(pressure_1f / noise_rms_pressure)` per scan point. The `pressure_1f` is the DFT amplitude at the drive frequency during the steady-state burst window.

## Dataset

Voltage sweep at 1.907 MHz (test10), 101×101 area scan, 5 files from 5–25 Vpp. The 5 Vpp file was recorded in continuous mode (no burst), so no noise data is available.

## Results

| Vpp | Median SNR inside (dB) | Median SNR outside (dB) | Noise floor (kPa) |
|-----|------------------------|-------------------------|--------------------|
| 10  | 21.6                   | −10.0                   | 48                 |
| 15  | 25.9                   | −6.0                    | 43                 |
| 20  | 26.0                   | −3.3                    | 52                 |
| 25  | 24.5                   | −4.5                    | 60                 |

- **Inside-channel SNR** is 22–26 dB (median), corresponding to amplitude ratios of 12–20×.
- **Outside-channel SNR** is negative: the 1f signal outside the channel is below the noise floor, as expected (no acoustic mode outside the walls).
- **Noise floor** is ~50 kPa equivalent pressure, increasing with voltage because higher drive amplitudes require a larger LDV decoder range (higher velocity range = more decoder noise).

## Interpretation

The ~50 kPa noise floor sets a lower bound on resolvable pressure. At the antinode of the 25 Vpp dataset (p0 ≈ 4 MPa), SNR exceeds 30 dB. Near the channel walls where the 1f mode has a node, SNR drops below 0 dB and pressure values are noise-dominated.

For mode-shape fitting, the edge points (within ~1 grid step of the wall) contribute mostly noise. The existing `EDGE_MARGIN = 1` quality filter in `make_quality_mask()` addresses this.
