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

### Reference LDV data (Mar 7, test10)

In `C:/Users/Tatsuki Sasamura/OneDrive - Lund University/Data/20260307experimentB/`

### PIV data (Mar 17)

In `C:/Users/Tatsuki Sasamura/Documents/particle-tracking/output/{5,10,15}Vpp/`

## Chip configuration

Four ports along the channel. Rightmost = inlet, leftmost = outlet, middle two = clogged (failed glue). Measurement region between the two clogged ports, with PZT centred underneath. Clogged branches may trap air but water-filled vs unflushed comparison shows <1% pressure difference.

## Code fixes during this investigation

1. **`config.py`**: added `velocity_to_pressure(f_hz)` — centralised signed conversion factor
2. **`pressure_map_2d.py`**: fixed `--channel-centre` unit bug (was interpreting mm as metres)
3. **`fft_cache.py`**: uses `velocity_to_pressure` instead of inline formula

## Conclusions

1. LDV refracto-vibrometry is **validated** (air-filled null test) and **reproducible** (±1% across 11 days)
2. The ~1.7× LDV/PIV discrepancy is **not a calibration error** — it is dominated by uncertainty in the PS particle compressibility used in the PIV analysis
3. Both methods likely agree within their combined systematic uncertainties (~32%)
4. To close the gap: independently measure κ_PS for the Thermo Fisher G0500B beads, or perform simultaneous LDV+PIV on the same chip
