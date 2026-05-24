# Calibration note — W21 sample-chip freq-sweep series

For every scan in this folder taken on 2026-05-18 before ~21:15 UTC,
the V_1f reading from Pico CH A was approximately **2x lower than
the actual voltage at the amplifier output**.  The hardware.yaml
``amplifier_gain_v_per_v: 80`` value was bench-measured through the
same splitter-loaded probe and is also off by ~2x.  Direct
remeasurement without the splitter gives an amp gain of ~x170.

This means each ``X Vpp`` folder label (derived from AFG_Vpp x 80) is
approximately **2.16x smaller than the actual peak-to-peak voltage at
the PZT**.

## Setup difference between affected and unaffected scans

The oscilloscope probe on Pico CH A (drive_voltage channel) was fed
through a BNC T-splitter to both the PicoScope (1 MOhm input) and a
Tek TBS100C oscilloscope (1 MOhm input).  Two 1 MOhm inputs in
parallel = 500 kOhm, which changes the 10x probe's effective divider
ratio:

```
nominal load (1 MOhm):  9 MOhm / (9 MOhm + 1 MOhm)   = 1/10
splitter load (500 kOhm): 9 MOhm / (9 MOhm + 0.5 MOhm) = 1/19
```

So the Pico sees ~half the voltage it would with a single 1 MOhm
load.  The analysis script's hardcoded ``x10`` probe factor then
under-reports V_1f by ~2x.

## What is and isn't affected

| Channel / quantity | Affected? |
|---|---|
| Pico CH A raw drive_voltage | YES — halved |
| V_1f reported by analysis | YES — multiply reported value by ~2.16 for true |
| \|Z\| = V/I reported | YES — multiply reported value by ~2.16 for true |
| Folder labels ``X Vpp`` (derived from AFG_Vpp x 80) | YES — multiply by ~2.16 for true PZT Vpp |
| Pico CH B (LDV velocity) | No — splitter not on CH B |
| pressure_1f, pressure_2f | No — derived from CH B only |
| Pico CH D (current) | No — current channel not on a probe |
| V-I phase, phase_1f, phase_2f | No — phase is independent of amplitude |
| HDF5 ``drive_voltage_vpp`` attr (AFG-side) | No — this is the commanded AFG output |

## Direct measurement of the effect

10 Vpp narrow scan, splitter on (213357) vs splitter off (214800):

| Quantity | Splitter on | Splitter off | Ratio |
|---|---|---|---|
| V_1f at 1.905 MHz | 4.85 V | 10.82 V | 2.23x |
| P_1f peak | 3.58 MPa | 3.40 MPa | 0.95x |
| P_2f peak | 0.245 MPa | 0.230 MPa | 0.94x |

V_1f doubled; pressures essentially unchanged.

## Effect on W16 vs W21 chip comparison

| Chip | Drive label | Drive at PZT (corrected) | Measured peak P_1f |
|---|---|---|---|
| W16 (no splitter) | "20 Vpp" (gain x80 real) | 20 Vpp | 2.93 MPa |
| W21 (with splitter) | "10 Vpp" (gain x170 real) | ~22 Vpp | 2.88 MPa |

The ~2x discrepancy in P_1f-per-Vpp that prompted this investigation
goes away once the labels are corrected.

## How to correct historical data

The analysis script ``experiments/2026W21/freq_vs_current.py``
auto-reads ``amplifier_gain_v_per_v`` from each run's snapshotted
``hardware.yaml`` and reports both AFG-side and PZT-side Vpp in the
plot title.  To correct an old run in place, edit the snapshotted
``hardware.yaml`` value:

```yaml
transducer:
  amplifier_gain_v_per_v: 170.0   # was 80.0; corrected 2026-05-18
```

Or batch-update all old W21 snapshots:

```powershell
Get-ChildItem "C:\Users\tatsuki\OneDrive - Lund University\Data\output\sample_101x1_fsweep_*\hardware.yaml" |
  ForEach-Object {
    (Get-Content $_) -replace 'amplifier_gain_v_per_v: 80.0', 'amplifier_gain_v_per_v: 170.0' |
    Set-Content $_
  }
```

Re-run ``freq_vs_current.py`` afterwards; the title and tabular dump
will then report the corrected PZT-side Vpp.

## Going forward

- Remove the splitter from CH A.  If you need the Tek scope as a
  second display, use the PicoSDK plot or the Pico's signal-output
  instead of splitting probe outputs.
- Re-measure the amp gain on-bench without the splitter to confirm
  the x170 number from the 10 Vpp 214800 scan.
- Update ``../../ldv-daq/configs/hardware_default.yaml`` (the live
  DAQ config, not the snapshots) once the recalibration is firm.
- ``max_voltage_vpp`` in the live config was set to 1.25 to cap PZT
  at 100 Vpp under the assumed x80 gain.  Under the corrected x170
  gain it should be lowered to ~0.6 to preserve the same cap.

## Affected scans in this folder

All ``sample_101x1_fsweep_*`` and ``sample_101x21_*`` directories
acquired on 2026-05-18 before ~21:15 UTC.  The 10 Vpp narrow scan
``sample_101x1_fsweep_narrow_10Vpp_20260518_214800`` is the first
un-affected scan (splitter removed).
