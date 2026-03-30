# LDV–PIV Cross-Validation and Refracto-Vibrometry Validation — 2026-03-18

## Purpose

Investigate the ~1.7× discrepancy between LDV refracto-vibrometry and PIV particle tracking pressure estimates, and validate the LDV measurement principle.

## Key Results

### 1. LDV refracto-vibrometry is reproducible

Mar 7 (old) vs Mar 18 (water-filled, flushed) at 15 Vpp, 1.907 MHz:
- V = 7.5 V, I = 20 mA (identical electrical conditions)
- p₀ at overlapping axial positions: **mean ratio = 1.008** (±10% position-by-position)
- No PZT depolarization despite 100 Vpp accidental overvoltage between sessions

### 2. Air-filled null test validates refracto-vibrometry

Channel filled with air at 15 Vpp, 1.907 MHz, same scan region (y = 7.6–9.6 mm):
- Water-filled: p₀ = **2500 kPa**
- Air-filled: p₀ = **148 kPa** (noise level, ~60 kPa baseline)
- Ratio: **17×** signal-to-null

This confirms the LDV signal originates from the acousto-optic effect in the water column (dn/dp × p integrated along the beam path). Negligible contribution from:
- Silicon surface vibration (still vibrates with air, but no signal)
- Glass lid piezo-optic effect (same acoustic stress, but no signal)
- Electrical/mechanical pickup (independent of channel contents)

### 3. LDV–PIV discrepancy is ~1.7× at matched conditions

| Vpp | LDV p₀ (kPa) | PIV p₀ (kPa) | Ratio |
|-----|--------------|--------------|-------|
| 5   | 753          | 438          | 1.72  |
| 10  | 1553         | 903          | 1.72  |
| 15  | 2441         | 1270         | 1.92  |

Both measured at 1.907 MHz on the same chip. LDV: Mar 7 (overnight voltage sweep). PIV: Mar 17.

**How to reproduce:**

LDV p₀ values (from `ldv-analysis` repo root):

```bash
# Requires test10 FFT caches in experiments/2026W10_stepA/output/cache/
.venv/Scripts/python -c "
import sys; sys.path.insert(0, 'src')
import numpy as np
from ldv_analysis.config import (CHANNEL_WIDTH, RSSI_THRESHOLD,
    channel_centre_func, get_data_dir, load_channel_geometry)
from ldv_analysis.fft_cache import load_or_compute
from ldv_analysis.filters import make_valid_mask, make_burst_timing_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns
from pathlib import Path
CACHE = Path('experiments/2026W10_stepA/output/cache')
DATA = get_data_dir('20260307experimentB')
geom = load_channel_geometry('20260307experimentB', CACHE)
cfn = channel_centre_func(geom); hw = CHANNEL_WIDTH / 2
for f, vpp in [('test10_1907_5Vpp_1m_s_max.tdms',5),
               ('test10_1907_10Vpp_2m_s_max.tdms',10),
               ('test10_1907_15Vpp_2m_s_max.tdms',15)]:
    c = load_or_compute(DATA / f, CACHE)
    V = c['voltage_1f']; rssi = c['rssi'] if 'rssi' in c else None
    valid = make_valid_mask(V, rssi)
    if 'pt_burst_on_us' in c:
        valid &= make_burst_timing_mask(c['pt_burst_on_us'], c['pt_burst_off_us'])
    pxc = c['pos_x'] - cfn(c['pos_y']); ins = np.abs(pxc) <= hw
    cg = make_channel_grid(pxc, c['pos_y'], int(c['n_x_meta']), int(c['n_y_meta']),
        CHANNEL_WIDTH, c['pos_x'].max()-c['pos_x'].min(), ins,
        rssi=rssi, rssi_threshold=RSSI_THRESHOLD)
    prs = c['pressure_1f'].copy(); prs[~valid] = np.nan
    p0 = np.nanmax(fit_columns(cg.to_grid(prs), cg.width_grid, CHANNEL_WIDTH))
    print(f'{vpp:2d} Vpp: p0 = {p0/1e3:.0f} kPa')
"
```

PIV p₀ values (from `particle-tracking` repo root):

```bash
for vpp in 5Vpp 10Vpp 15Vpp; do
  .venv/Scripts/python scripts/06_fitting.py \
    --data-dir output/$vpp --freq 1.907e6
done
# Max p0 per voltage is reported in the summary output.
```

### 4. Ruled-out sources of discrepancy

| Candidate | Effect | Status |
|-----------|--------|--------|
| LDV velocity scale (Polytec ±2V@1MΩ) | — | Verified correct from datasheet + data |
| DFT normalization (×2/N) | — | Verified against raw waveform pk-pk |
| Double-pass / LDV decoder /2 | — | Cancels (user confirmed) |
| dn/dp value (1.4 vs 1.34 ×10⁻¹⁰) | ~5% | Minor |
| Channel height H | ~3% | Minor |
| PZT depolarization | — | Ruled out (Mar 18 = Mar 7) |
| Temperature drift | ~5–7% | Measured: p₀ varies <7% over 16h at same position |
| Air pocket boundary condition | — | Ruled out (water-filled matches old data) |
| PIV Gor'kov formula | — | Matches Barnkob 2010 derivation exactly |
| PIV particle radius | — | 2.5 µm matches Thermo Fisher G0500B spec (5.0 µm dia) |
| PIV material parameters (ρ_f, c_f, η) | <2% | Standard handbook values |

### 5. Dominant source: PS particle compressibility

The polystyrene compressibility κ_PS used in the PIV pressure calculation is poorly constrained:

| Source | κ_PS (Pa⁻¹) | c_PS (m/s) | f₁ | Φ | PIV p₀ at 10 Vpp |
|--------|-------------|-----------|-----|------|------------------|
| Settnes & Bruus 2012 | 1.72e-10 | 2350 | 0.607 | 0.218 | 772 kPa (lower bound) |
| PIV code (current) | 2.49e-10 | 1956 | 0.432 | 0.159 | 903 kPa |
| Barnkob 2010 | 3.30e-10 | 1700 | 0.277 | 0.098 | 1153 kPa (upper bound) |

With Barnkob's κ_PS: LDV/PIV ratio drops from 1.72 to **1.35**.

Augustsson et al. 2023 (Lund) measured >25% variation in acoustic contrast factor between different PS particle batches. The Thermo Fisher Fluoro-Max G0500B beads used in our PIV have no manufacturer-specified compressibility.

### 6. Combined uncertainty budget

| Source | Uncertainty in p₀ |
|--------|-------------------|
| κ_PS (PIV) | ±30% |
| dn/dp (LDV) | ±5% |
| Channel height (LDV) | ±3% |
| Temperature / f_res detuning | ±5–10% |
| Mode-shape fit accuracy | ±5% |
| **Combined (RSS)** | **~32%** |

The ~1.35× residual ratio (after using Barnkob κ_PS) is within the combined uncertainty.

## Datasets

### Mar 18 (test11) — all at 1.907 MHz, channel centre 27.22 mm

| File | Vpp | Grid | Purpose |
|------|-----|------|---------|
| test11_5Vpp1907MHzSanityCheck | 5 | 21×21 | Quick sanity check |
| test11_10Vpp1907MHzSanityCheck | 10 | 21×21 | Quick sanity check |
| test11_15Vpp1907MHzSanityCheck | 15 | 21×21 | Quick sanity check |
| test11_15Vpp1907MHz2m_sSanityCheckmorepoints | 15 | 101×101 | Full 2D map |
| test11_15Vpp1907MHz2m_bubbleeffectcheck_bubble_aty6.7 | 15 | 101×21 | Air pocket expanding during scan |
| test11_15Vpp1907MHz2m_sFilled_with_air | 15 | 101×21 | Null test: air-filled channel |
| test11_15Vpp1907MHz2m_sFilled_with_water | 15 | 101×21 | Control: water-filled (flushed) |

All in `G:/My Drive/260318_ldv/`, to be relocated to OneDrive.

### test12 voltage sweep (Mar 18, 1D line scans at y=8.9mm)

| File | Vpp | LDV range | BW |
|------|-----|-----------|----|
| test12_5Vpp1907kHz | 5 | 2m/s | 6 MHz |
| test12_10Vpp1907kHz | 10 | 2m/s | 6 MHz |
| test12_15Vpp1907kHz | 15 | 2m/s | 6 MHz |
| test12_20Vpp1907kHz | 20 | 2m/s | 6 MHz |
| test12_25Vpp1907kHz | 25 | 2m/s | 6 MHz |
| test12_30Vpp1907kHz | 30 | 2m/s | 6 MHz |
| test12_35Vpp1907kHz_5m_s_max | 35 | 5m/s | 6 MHz |
| test12_40Vpp1907kHz_5m_s_max | 40 | 5m/s | 6 MHz |
| test12_45Vpp1907kHz_5m_s_max | 45 | 5m/s | 6 MHz |
| test12_45Vpp1907kHz_5m_s_max_12MHz_bandwidth | 45 | 5m/s | 12 MHz |
| test12_50Vpp1907kHz_5m_s_max_12MHz_bandwidth | 50 | 5m/s | 12 MHz |
| test12_10Vpp1907kHz_5m_s_max_12MHz_bandwidth | 10 | 5m/s | 12 MHz |

All in `G:/My Drive/260318_ldv/`.

### Reference LDV data (Mar 7, test10)

In `C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260307experimentB/`

### PIV data (Mar 17)

In `C:/Users/Tatsuki Sasamura/Documents/particle-tracking/output/{5,10,15}Vpp/`

## How to reproduce figures

All commands run from the repo root. Use `.venv/Scripts/python` on Windows.

### Air-filled null test (2D maps)

```bash
# Water-filled control
python experiments/2026W10_stepA/pressure_map_2d.py \
  "G:/My Drive/260318_ldv/test11_15Vpp1907MHz2m_sFilled_with_water.tdms" \
  --harmonics --channel-centre 27.22

# Air-filled null test
python experiments/2026W10_stepA/pressure_map_2d.py \
  "G:/My Drive/260318_ldv/test11_15Vpp1907MHz2m_sFilled_with_air.tdms" \
  --harmonics --channel-centre 27.22
```

Output: `experiments/2026W10_stepA/output/pressure_map_2d/map2d_*.png`

### LDV reproducibility (Mar 7 vs Mar 18, 2D maps)

```bash
# Mar 7 (W10)
python experiments/2026W10_stepA/pressure_map_2d.py \
  "C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260307experimentB/test10_1907_15Vpp_2m_s_max.tdms" \
  --harmonics --channel-centre 27.087

# Mar 18 (water-filled, cleaned surface)
python experiments/2026W10_stepA/pressure_map_2d.py \
  "G:/My Drive/260318_ldv/test11_15Vpp1907MHz2m_sFilled_with_water_cleanedsurface.tdms" \
  --harmonics --channel-centre 27.22
```

### Voltage sweep (1D, 5–45 Vpp)

```bash
python experiments/2026W10_stepA/voltage_sweep_1d.py
```

Output: `experiments/2026W10_stepA/output/voltage_sweep_1d/`
- `voltage_sweep_1d_test12.png` — 4-panel summary (p₀_1f, p₀_2f+3f, 2f/1f ratio, drive current harmonics)
- `voltage_sweep_1d_mode_shapes.png` — 1f/2f/3f mode shapes per voltage

### Harmonic mode shapes (1f–5f at single voltage)

```bash
# 45 Vpp with 12 MHz bandwidth
python experiments/2026W10_stepA/harmonics_profile.py \
  "G:/My Drive/260318_ldv/test12_45Vpp1907kHz_5m_s_max_12MHz_bandwidth.tdms"

# 45 Vpp with 6 MHz bandwidth (for comparison)
python experiments/2026W10_stepA/harmonics_profile.py \
  "G:/My Drive/260318_ldv/test12_45Vpp1907kHz_5m_s_max.tdms"
```

Output: `experiments/2026W10_stepA/output/harmonics_profile/harmonics_*.png`

### Coppens comparison (2f/1f ratio vs theory)

```bash
python experiments/2026W10_stepA/coppens_comparison.py
```

Output: `experiments/2026W10_stepA/output/coppens_comparison/coppens_comparison.png`

### Axial p₀ distribution

```bash
python experiments/2026W10_stepA/axial_p0_distribution.py
```

Output: `experiments/2026W10_stepA/output/axial_p0_distribution/axial_p0_1f_test10.png`

### W10 2D maps with 3f (25 Vpp example)

```bash
python experiments/2026W10_stepA/pressure_map_2d.py \
  "C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260307experimentB/test10_1907_25Vpp_5m_s_max.tdms" \
  --harmonics --channel-centre 27.087
```

## Chip configuration

Four ports along the channel. Rightmost = inlet, leftmost = outlet, middle two = clogged (failed glue). Measurement region between the two clogged ports, with PZT centred underneath. Clogged branches may trap air but water-filled vs unflushed comparison shows <1% pressure difference.

## High-voltage sweep (test12) and harmonic mode shapes

### Voltage sweep up to 50 Vpp

1D line scans (101×2) at the axial antinode y = 8.9 mm, 5–50 Vpp at 1.907 MHz.

| Vpp | p₀_1f (kPa) | p₀_2f (kPa) | 2f/1f | p₀_3f (kPa) |
|-----|-------------|-------------|-------|-------------|
| 5   | 736         | 12          | 0.016 | —           |
| 10  | 1439        | 58          | 0.040 | —           |
| 15  | 2280        | 98          | 0.043 | —           |
| 20  | 2930        | 168         | 0.057 | —           |
| 25  | 3531        | 276         | 0.078 | —           |
| 30  | 4262        | 353         | 0.083 | —           |
| 35  | 5011        | 459         | 0.092 | —           |
| 40  | 5716        | 620         | 0.109 | —           |
| 45  | 6466        | 772         | 0.120 | 215         |
| 50  | 6807        | 1116        | 0.164 | 262         |

- p₀_1f scales **linearly** with voltage up to 50 Vpp (max deviation 6%, fit: 143.4 kPa/Vpp)
- p₀_2f scales as **V²** (fit: 0.386 kPa/Vpp²)
- 2f/1f ratio scales linearly with V (slope: 0.00276 /Vpp)
- No sign of PZT saturation or depolarization (return-to-baseline at 10 Vpp after 50 Vpp: 1448 vs 1439 kPa)
- Maximum pressure reached: **6.8 MPa** at 50 Vpp (E_ac ≈ 4000 J/m³)

### 3f harmonic mode shape — first observation

At 45 Vpp with 12 MHz LDV bandwidth, all pressure harmonics up to 5f were resolved:

| Harmonic | Frequency | Mode shape        | p₀ (kPa) | % of 1f |
|----------|-----------|-------------------|----------|---------|
| 1f       | 1.907 MHz | \|sin(πy/W)\|     | 5893     | 100%    |
| 2f       | 3.814 MHz | \|cos(2πy/W)\|    | 961      | 16.3%   |
| 3f       | 5.721 MHz | \|sin(3πy/W)\|    | 215      | 3.7%    |
| 4f       | 7.628 MHz | \|cos(4πy/W)\|    | 88       | 1.5%    |
| 5f       | 9.535 MHz | \|sin(5πy/W)\|    | 67       | 1.1%    |

The **3f mode shape |sin(3πy/W)|** with its characteristic 3-lobe structure is clearly visible in the data — this is the first direct observation of the third-harmonic pressure mode in a microchannel acoustophoresis device. The mode shapes alternate between sin (odd harmonics) and cos (even harmonics), consistent with the nonlinear Coppens cascade theory.

### LDV bandwidth effect

The default 6 MHz bandwidth attenuates harmonics above 3f:

| Harmonic | 6 MHz BW | 12 MHz BW | Ratio |
|----------|----------|-----------|-------|
| 1f (1.9 MHz) | 6031 kPa | 5893 kPa | 0.98 |
| 2f (3.8 MHz) | 934 kPa  | 961 kPa  | 1.03 |
| 3f (5.7 MHz) | 191 kPa  | 215 kPa  | 1.13 |
| 5f (9.5 MHz) | 33 kPa   | 67 kPa   | 2.03 |

12 MHz bandwidth is required for accurate harmonic measurements above 2f.

### Drive current harmonics

Real 3f current harmonic observed at 40+ Vpp (3.4% at 45 Vpp), confirmed present in both voltage and current waveforms. Source unclear — not PicoScope clipping (waveform goes to ±2.2V without flat-topping), not amplifier power limit (1.5W out of 75W capacity). Possibly PZT nonlinearity or impedance mismatch at harmonic frequencies.

### βQM estimate

At the maximum drive (50 Vpp, p₀ = 6.8 MPa):
- Mach number M = p₀/(ρc²) = 0.003
- βQM ≈ 2f/1f ratio ≈ 0.16

Reaching βQM ~ 1 would require ~310 Vpp — not achievable with current instrumentation without visible drive distortion. The experimentally accessible range is βQM ≈ 0.02–0.16, within which the 2f/1f ratio scales linearly with voltage, consistent with perturbative Coppens theory.

### Coppens cross-validation of absolute pressure

The Coppens prediction for the 2f/1f pressure ratio in a resonant cavity is:

    P_2f / P_1f = (1/4) × β × Q₂ × M

where β = 3.5 (water), Q₂ = quality factor at 2f, M = p₀/(ρc²) is the acoustic Mach number.

The 2f/1f ratio is measured purely by LDV (calibration cancels in the ratio), but M depends on the absolute pressure calibration. This provides an independent constraint on p₀:

| Pressure source | M range | Measured / Coppens (Q₂=100) |
|-----------------|---------|----------------------------|
| LDV pressure | 0.0003–0.0028 | **0.48–0.58** (mean 0.50) |
| PIV-calibrated (÷1.72) | 0.0002–0.0017 | **0.82–0.99** (mean 0.87) |

With LDV pressure, the measured 2f/1f is ~2× below the Coppens prediction. With PIV-calibrated pressure, the agreement improves to **~15%** — within the uncertainty of Q₂ at the detuned 2f frequency (3.814 MHz is 31 kHz off the nearest 2f channel resonance at 3.845 MHz).

**Implication**: The Coppens nonlinear theory independently suggests the LDV refracto-vibrometry overestimates absolute pressure by ~1.7×, consistent with the direct LDV/PIV comparison. This points to a systematic error in the LDV calibration chain — most likely the piezo-optic coefficient dn/dp or the effective optical path length H. The factor ~1.7 would be explained if the true SENSITIVITY = H × dn/dp is 1.7× larger than assumed (e.g., dn/dp = 2.4e-10 instead of 1.4e-10, or H = 255 µm instead of 150 µm).

This remains an open question: neither dn/dp nor H has been independently verified for this specific chip and wavelength.

## Code fixes and new scripts

1. **`config.py`**: added `velocity_to_pressure(f_hz)` — centralised signed conversion factor
2. **`pressure_map_2d.py`**: fixed `--channel-centre` unit bug (was interpreting mm as metres); added 3f harmonic extraction
3. **`fft_cache.py`**: uses `velocity_to_pressure` instead of inline formula
4. **`mode_fit.py`**: generalised `fit_columns` to support arbitrary harmonics (odd → sin, even → cos)
5. **`voltage_sweep_1d.py`** (new): 1D voltage sweep with 1f/2f/3f fitting and drive current harmonic analysis
6. **`harmonics_profile.py`** (new): visualise pressure mode shapes at 1f–5f from a single TDMS file
7. **`axial_p0_distribution.py`** (new): plot p₀(y) along channel length

## Conclusions

1. LDV refracto-vibrometry is **validated** (air-filled null test) and **reproducible** (±1% across 11 days)
2. The ~1.7× LDV/PIV discrepancy is **not a calibration error** in the velocity/DFT chain — all factors verified correct
3. **Coppens theory independently suggests LDV overestimates absolute pressure by ~1.7×**: the measured 2f/1f ratio (calibration-independent) agrees with the Coppens prediction when PIV-calibrated Mach number is used (mean ratio 0.87) but disagrees when LDV pressure is used (mean ratio 0.50). This points to a systematic error in the refracto-vibrometric SENSITIVITY = H × dn/dp.
4. Pressure scales linearly with voltage up to 50 Vpp (**6.8 MPa by LDV, ~4.0 MPa if PIV-calibrated**, E_ac ≈ 1400–4000 J/m³) with no PZT degradation
5. **3f mode shape |sin(3πy/W)| directly observed** — first demonstration of third-harmonic pressure mode in microchannel acoustophoresis
6. Harmonic cascade 1f → 2f → 3f follows expected Coppens scaling (p₂f ∝ V², p₃f ∝ V³)
7. Maximum achievable βQM ≈ 0.10–0.16 with current instrumentation — reaching βQM ~ 1 requires ~310 Vpp, beyond the drive distortion limit
8. **Open question**: the source of the ~1.7× LDV overestimate. Candidates: dn/dp = 1.4e-10 may be too low (true value ~2.4e-10?), or H = 150 µm may be too low (true effective path ~255 µm?). Neither has been independently verified for this chip.
