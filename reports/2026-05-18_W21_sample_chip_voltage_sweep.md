# W21 sample-chip voltage sweep + v2 DAQ cross-validation — 2026-05-18

## Purpose

First production-scale use of the new v2 HDF5 DAQ (``ldv-daq`` repo)
on the ``sample`` chip.  Three goals:

1. End-to-end validation that ``ldv-analysis`` (this repo) reads,
   validates, and processes real v2 HDF5 output correctly.
2. Map the chip's electrical and acoustic frequency response around
   ~1.9 MHz.
3. Sweep drive voltage from 10 to 60 Vpp (labelled) to map P_1f and
   P_2f as a function of drive.

## Datasets

All in ``C:\Users\tatsuki\OneDrive - Lund University\Data\output\``.

### Sanity scans (early-day)
- ``sample_11x11_0p5x6_10Vpp_*`` — 11x11 grid, 1.907 MHz, 10 Vpp
- ``sample_101x21_30Vpp_20260518_003801`` — 101x21, 1.907 MHz, 30 Vpp
  (first 2D map; pressure pattern matches ``|sin(pi y/W)|`` shape
  with channel walls at x ≈ 10.24 and 10.62 mm and channel center
  at x ≈ 10.43 mm)

### 1f frequency sweep, 101x1 line scan, water-filled

All on ``sample`` chip, 1 kHz steps.  "Wide" = 1.900–2.000 MHz
(101 freqs).  "Narrow" = 1.880–1.920 MHz (41 freqs).  6 m/s LDV
range from the 50 Vpp scan onward; earlier scans were on 1 m/s
or 5 m/s range as noted.

| Label | Source dir | Sweep | LDV range | Peak P_1f | Peak f (P_1f) | Peak P_2f |
|---|---|---|---|---|---|---|
| 10 Vpp | ``sample_101x1_fsweep_10Vpp_20260518_173341`` | wide | 5 m/s | 2.88 MPa | 1.900 MHz | 0.157 MPa |
| 20 Vpp | ``sample_101x1_fsweep_20Vpp_20260518_185107`` | wide | 5 m/s | 6.47 MPa | 1.900 MHz | 0.697 MPa |
| 30 Vpp | ``sample_101x1_fsweep_30Vpp_20260518_191818`` | wide | 5 m/s | 8.42 MPa | 1.902 MHz | 1.534 MPa |
| 40 Vpp | ``sample_101x1_fsweep_40Vpp_20260518_195418`` | wide | 5 m/s | 11.95 MPa | 1.905 MHz | 2.601 MPa |
| 50 Vpp | ``sample_101x1_fsweep_50Vpp_20260518_202106`` | wide | 6 m/s | **10.37 MPa** | **1.913 MHz** | 2.44 MPa |
| 60 Vpp | ``sample_101x1_fsweep_narrow_60Vpp_20260518_204855`` | narrow | 6 m/s | 15.73 MPa | 1.9027 MHz | 3.84 MPa |
| 50 Vpp remeas | ``sample_101x1_fsweep_narrow_50Vpp_20260518_205751`` | narrow | 6 m/s | 14.46 MPa | 1.9050 MHz | 3.589 MPa |
| 40 Vpp remeas | ``sample_101x1_fsweep_narrow_40Vpp_20260518_210751`` | narrow | 6 m/s | 11.56 MPa | 1.9050 MHz | 2.584 MPa |
| 30 Vpp remeas | ``sample_101x1_fsweep_narrow_30Vpp_20260518_211632`` | narrow | 6 m/s | 8.93 MPa | 1.9050 MHz | 1.778 MPa |
| 20 Vpp remeas | ``sample_101x1_fsweep_narrow_20Vpp_20260518_212516`` | narrow | 6 m/s | 6.63 MPa | 1.9050 MHz | 0.884 MPa |
| 10 Vpp remeas | ``sample_101x1_fsweep_narrow_10Vpp_20260518_213357`` | narrow | 6 m/s | 3.58 MPa | 1.9050 MHz | 0.245 MPa |
| 10 Vpp **no-splitter** | ``sample_101x1_fsweep_narrow_10Vpp_20260518_214800`` | narrow | 6 m/s | 3.40 MPa | 1.9050 MHz | 0.230 MPa |

Bold = bad / anomalous (see below).

### Failure-mode references

- ``sample_101x1_fsweep_20Vpp_20260518_181210_failed_lowH2O`` —
  operator forgot to fill water in the channel before scanning.
  P_1f peak shifted from chip's water-filled ~1.905 MHz down to
  ~1.965 MHz (the PZT's electrical current-peak frequency, see
  below) and amplitude dropped by ~14x relative to water-filled at
  the same drive.  Folder kept as an air-filled reference.
- ``sample_101x1_fsweep_50Vpp_20260518_202106`` — measurement
  completed cleanly but reported P_1f = 10.37 MPa at 1.913 MHz,
  lower than both the 40 Vpp (11.95 MPa @ 1.905) and 60 Vpp
  (15.73 MPa @ 1.903) bracketing scans, with a +8 kHz peak-frequency
  shift.  The narrow-band remeasurement at 50 Vpp (205751) gives
  14.46 MPa @ 1.905 MHz, fitting between the bracket values.
  Cause of the original anomaly not pinpointed; this scan is
  excluded from the scaling analysis below.

## Electrical impedance summary (across all narrow-band sweeps)

From the V, I, V-I phase, and \|Z\|=V/I panels of
``freq_vs_current.png``:

- Drive freq of current peak (\|Z\| min): **~1.965 MHz**
- Drive freq of current min (\|Z\| max): **~1.998 MHz**
- Drive freq where V-I phase crosses 0: **~1.987 MHz**
- \|Z\| at current-peak freq: ~100 Ohm (reported; ~200 Ohm corrected
  for splitter, see calibration note)
- \|Z\| at current-min freq: ~500 Ohm (reported; ~1 kOhm corrected)
- V-I phase baseline (away from current peak): ~−70 deg (capacitive)

## Voltage scaling (narrow remeasurement series)

| AFG Vpp label | true PZT Vpp (gain x170) | P_1f (MPa) | P_2f (MPa) | P_1f/Vpp_label | P_2f/P_1f | P_2f/P_1f^2 (1/MPa) |
|---|---|---|---|---|---|---|
| 10 | 21.25 | 3.58 | 0.245 | 358 | 6.84 % | 0.0191 |
| 20 | 42.50 | 6.63 | 0.884 | 332 | 13.3 % | 0.0201 |
| 30 | 63.75 | 8.93 | 1.778 | 298 | 19.9 % | 0.0223 |
| 40 | 85.00 | 11.56 | 2.584 | 289 | 22.4 % | 0.0193 |
| 50 | 106.25 | 14.46 | 3.589 | 289 | 24.8 % | 0.0172 |
| 60 | 127.50 | 15.73 | 3.840 | 262 | 24.4 % | 0.0155 |

Observations:

- P_1f scales approximately linearly with drive across the 6 points.
  P_1f / Vpp_label drops from 358 → 262 kPa/Vpp (a 27% reduction)
  going from the lowest to the highest drive.
- P_2f scales approximately quadratically with drive (P_2f/P_1f^2
  range: 0.0155 to 0.0223, scatter ±18 %, mean 0.019 / MPa).
- P_2f / P_1f rises from 6.8 % at the lowest drive to ~25 % at the
  highest, with the increase tapering off between 50 and 60 Vpp
  labels.
- P_1f peak frequency is stable at 1.9050 MHz from 10 to 50 Vpp
  labels.  At 60 Vpp it shifts down by 2.3 kHz to 1.9027 MHz.
- P_2f peak frequency stays at 1.9050 MHz across the entire range
  (60 Vpp included).

Offset between the P_1f and P_2f peak frequencies in the lower-drive
scans: 0 kHz (at 1 kHz resolution).  In the wide-scan data the offset
was 5 kHz (P_1f peak at 1.900 MHz, P_2f peak at 1.905 MHz); this is
not seen in the narrow-band rescans, which captured the chip in a
more stable state in a single ~1 hr session.

## Pipeline cross-validations

End-to-end check on the first 3x3 v2 smoke test:

- ``validate_hdf5_v2`` returns [] on real DAQ HDF5 output
- ``load_scan_hdf5`` returns a ``ScanData`` with all v2 metadata
- ``fft_cache.load_or_compute`` produces per-point quantities for
  harmonics 1f–5f plus voltage, current, RSSI, V-I phase
- Drive-voltage 1f amplitude scaled 2x between conditions V0p1 and
  V0p2 (0.945 V vs 1.890 V; ratio 2.000), confirming writer–reader
  consistency

Two LDV ranges cross-checked at the same chip and drive on the 11x1
fsweep series:

| LDV range | velocity_scale | Peak P_1f | Frequency |
|---|---|---|---|
| 1m_s_max | 1.0 m/s/V | 2.84 MPa | 1.905 MHz |
| 5m_s_max | 2.5 m/s/V | 2.88 MPa | 1.900 MHz |

Difference 1.4 %, within session scatter.

## Splitter calibration issue (discovered during the voltage sweep)

Pico CH A was connected via a BNC T-splitter to both the PicoScope
and a Tek TBS100C oscilloscope.  The 10x passive probe was therefore
loaded by 0.5 MOhm (two 1 MOhm scope inputs in parallel) instead of
the design 1 MOhm.  The probe's effective divider ratio shifts from
1:10 to ~1:19.

Effect on the analysis:
- V_1f reported by the analysis = true V_1f at amp output / ~2
- \|Z\| = V/I reported = true \|Z\| / ~2
- All other quantities (pressures, current, all phases) are
  unaffected — they don't go through CH A or its probe.

The hardware.yaml ``amplifier_gain_v_per_v: 80`` was bench-measured
on 2026-05-17 through the same splitter-loaded probe, so it inherits
the same 2x error.  The true amp gain inferred from the splitter-off
remeasurement: ~x170.

Discovery sequence (chronological):

1. Initial comparison of W16 vs W21 P_1f-per-Vpp showed W21 ~10x
   higher than W16.
2. Investigation found a hardcoded ``VELOCITY_SCALE = 0.5`` in the
   W16 plot script that should have been ``2.5`` for the
   ``_5m_s_max`` Polytec range.  Switching to filename auto-detection
   (``io_utils._detect_velocity_scale_from_name``) reduced the
   discrepancy from ~10x to ~2x.
3. Operator recalled the BNC splitter on CH A.
4. Test: re-acquired the 10 Vpp narrow scan without the splitter.
   V_1f at 1.905 MHz changed from 4.85 V (with splitter) to 10.82 V
   (without); P_1f and P_2f changed by <5 %.

After correction:

| Chip | Drive label | Drive at PZT | Peak P_1f |
|---|---|---|---|
| W16 (no splitter) | "20 Vpp" | 20 Vpp | 2.93 MPa |
| W21 (with splitter) | "10 Vpp" | ~22 Vpp | 2.88 MPa |

See ``experiments/2026W21_freq_sweep/CALIBRATION_NOTE.md``
for the math and the in-place fix procedure (edit the snapshotted
``hardware.yaml`` for each affected run; the analysis script reads
it automatically).

## Analysis pipeline updates this session

- ``src/ldv_analysis/config.py`` — added a no-LaTeX fallback for
  scienceplots (``shutil.which("latex")`` check) so plots render
  with the IEEE typography on machines without LaTeX installed.
- ``src/ldv_analysis/fft_cache.py`` — fixed a non-ASCII ``×`` in the
  ``Grid:`` print that was getting mangled by Windows cp1252
  console encoding.
- New ``experiments/2026W16_freq_sweep/freq_vs_current.py`` —
  aggregates v1 TDMS freq-sweep into an 8-row sweep summary plus
  per-frequency mode-shape fits.  Uses
  ``io_utils._detect_velocity_scale_from_name`` for the
  filename-encoded LDV range.
- New ``experiments/2026W21_freq_sweep/freq_vs_current.py`` — v2
  HDF5 counterpart.  Per-dataset OUT_DIR subfolder routing so
  multiple sweep analyses coexist without overwriting.  Auto-reads
  ``amplifier_gain_v_per_v`` from the snapshotted ``hardware.yaml``
  and reports both AFG-side and PZT-side Vpp in the plot title.
- New ``experiments/2026W21_freq_sweep/vpp_vs_pressure.py`` —
  cascade plot: P_1f and P_2f peak vs true PZT Vpp, with linear and
  V^2 fits and the P_2f/P_1f and P_2f/P_1f^2 ratios.

## Next steps

1. Re-measure the amp gain on-bench without the splitter to confirm
   the x170 figure derived from the 10 Vpp 214800 scan.
2. Update ``../../ldv-daq/configs/hardware_default.yaml`` (live
   config, not snapshots) with the corrected amp gain and a
   correspondingly lowered ``max_voltage_vpp`` cap.
3. Batch-update the snapshotted ``hardware.yaml`` files in the
   affected run directories (see CALIBRATION_NOTE.md for the
   PowerShell one-liner) so the analysis script reports
   corrected PZT-side Vpp on re-run.
4. 101x101 follow-up 2D scan at a drive level inside the regime
   where P_1f scales approximately linearly (e.g. ``40 Vpp`` label
   = ~85 Vpp at PZT) and at 1.905 MHz.
5. Direct 2f sweep at ~3.6–3.9 MHz to characterise the chip's 2f
   response directly (analogous to the W10
   ``stepA_2f_{3860,3880}.tdms`` files).
6. LDV decoder extension unit would unlock drive levels above 60 Vpp
   label (~130 Vpp at PZT, peak \|v(t)\| at the antinode then
   approaches the 6 m/s ceiling).

## Outputs

All analysis output in
``experiments/2026W21_freq_sweep/output/<run_dir_name>/``:

- ``freq_vs_current.png`` — 8-row sweep summary (P_1f, P_2f,
  phase_1f, phase_2f, V-I phase, I, V, \|Z\|)
- ``mode_shapes_overview.png`` — all per-frequency 1f fits in one grid
- ``mode_shapes/mode_<freq>kHz.png`` — per-frequency 2-panel
  (amplitude + phase) mode fit
- ``fft_cache/`` — per-file ``_fft_cache_<stem>.npz`` for fast reruns
- ``run.log`` — tabular dump

Top-level outputs in ``experiments/2026W21_freq_sweep/output/``:

- ``vpp_vs_pressure.png`` — drive sweep summary across all narrow
  remeasurement scans
- ``CALIBRATION_NOTE.md`` — splitter-issue documentation and fix

Equivalent W16 outputs in
``experiments/2026W16_freq_sweep/output/``.
