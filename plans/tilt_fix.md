# Plan: Robust channel geometry detection

## Problem

`pressure_map_2d.py` detects the tilted channel centre by minimising
`sum(pressure_1f² outside strip)` via `scipy.optimize.brute`.
This fails at low drive voltages (e.g. test10 5 Vpp) because the pressure
SNR is too poor — the optimiser latches onto noise and returns wrong tilt.

Evidence (test10, same chip position):

| Vpp | Tilt (deg) | Centre left (mm) | Centre right (mm) |
|-----|-----------|-------------------|-------------------|
| 5   | -0.232    | 27.112            | 27.072            |
| 25  | -0.116    | 27.089            | 27.069            |

Should be identical.  The 25 Vpp result is trustworthy; the 5 Vpp result is
noise-corrupted.

## Root cause

Pressure amplitude scales with drive voltage, so at 5 Vpp the pressure field
near the channel walls is comparable to the noise floor.  The pressure²
objective becomes flat and the brute-force grid (Ns=100) cannot distinguish
the true boundary from random fluctuations.

## Proposed fix: multi-dataset geometry calibration

The channel is a physical structure — its position and tilt are the same
across all scans taken at the same chip geometry (same chip, same mounting,
same scan region).  Instead of detecting geometry per-file, **jointly
estimate from all available datasets** and save the result.

### Design

#### 1. Geometry file: `channel_geometry.json`

Stored in the **output/cache directory**, named by dataset to avoid
collisions between geometry groups:

```
experiments/2026W10_stepA/output/cache/
  channel_geometry_20260306experimentA.json
  channel_geometry_20260307experimentB.json
```

Contents:

```json
{
    "channel_width_mm": 0.375,
    "centre_left_mm": 27.089,
    "centre_right_mm": 27.069,
    "tilt_deg": -0.116,
    "calibrated_from": [
        "test6_1907.tdms",
        "test10_1907_25Vpp_5m_s_max.tdms"
    ],
    "method": "rssi",
    "created": "2026-03-11"
}
```

#### 2. Calibration script: `calibrate_geometry.py`

New standalone script.

**Input:** List of TDMS paths (or glob patterns) from the same geometry.

**Algorithm:**
1. For each file, load the **FFT cache** (not raw TDMS — caches already
   contain RSSI and positions).
2. Stack all RSSI + position data from all files into one combined point
   cloud.  This is simpler and more robust than per-file averaging, and
   works because all files share the same scan region.
3. Run single `brute(Ns=100)` + `fmin` refinement on the combined cloud,
   maximising mean RSSI inside the tilted strip.
4. Write `channel_geometry_{dataset}.json` to the cache directory.

If scan regions differ between files (e.g. test6 is 101×101, test8 is
51×51 at a different x range), the stacking still works — the combined
point cloud just has non-uniform density.  The RSSI-based objective
handles this naturally since it uses `mean(rssi[inside])` not `sum`.

#### 3. Dataset-to-geometry mapping

The calibration script infers the dataset name from the TDMS file path
(same logic as `get_data_dir()`).  All files from the same dataset
directory share one geometry file.

| Dataset | Geometry file | Files |
|---------|--------------|-------|
| 20260303experimentA | `channel_geometry_20260303experimentA.json` | stepA1967*.tdms |
| 20260306experimentA | `channel_geometry_20260306experimentA.json` | test6_*, test8_* |
| 20260307experimentB | `channel_geometry_20260307experimentB.json` | test10_* |

#### 4. Usage in `pressure_map_2d.py`

**Fallback hierarchy (highest priority first):**

1. `--channel-centre` CLI override (zero tilt, user-specified)
2. `--geometry-file` CLI arg (explicit path to JSON)
3. Auto-discover `channel_geometry_{dataset}.json` in cache directory
4. Per-file RSSI-based detection (if RSSI available)
5. Per-file pressure-based detection (current method, last resort)

```python
def load_or_detect_geometry(tdms_path, cache_dir, pos_x, pos_y,
                            rssi, pressure, args):
    # 1. CLI centre override
    if args.channel_centre is not None:
        return args.channel_centre, args.channel_centre, 0.0

    # 2. Explicit geometry file
    if args.geometry_file is not None:
        return _load_geometry_file(args.geometry_file)

    # 3. Auto-discover from dataset name
    dataset = tdms_path.parent.name  # e.g. "20260307experimentB"
    geom_path = cache_dir / f"channel_geometry_{dataset}.json"
    if geom_path.exists():
        print(f"  Using saved geometry: {geom_path.name}")
        return _load_geometry_file(geom_path)

    # 4. Per-file RSSI
    if rssi is not None:
        print("  Detecting geometry from RSSI (no saved geometry found)")
        return _detect_from_rssi(pos_x, pos_y, rssi, CHANNEL_WIDTH)

    # 5. Per-file pressure (last resort)
    print("  Detecting geometry from pressure (no RSSI available)")
    return _detect_from_pressure(pos_x, pos_y, pressure, CHANNEL_WIDTH)
```

### RSSI-based detection algorithm

Used in both the calibration script and the per-file fallback (levels 4-5):

```
objective(c_left, c_right):
    centre(y) = c_left + (c_right - c_left)/(y_max - y_min) * (y - y_min)
    inside = |pos_x - centre(y)| <= W/2
    return -mean(rssi[inside])
```

Optimise with `brute(Ns=100, finish=fmin)`.

## Files to create / modify

| File | Action |
|------|--------|
| `experiments/2026W10_stepA/calibrate_geometry.py` | **New** — calibration script |
| `experiments/2026W10_stepA/pressure_map_2d.py` | Modify — add geometry loading + RSSI fallback |

No new `src/` modules needed for this step.  If boundary detection is later
extracted (see `refactoring.md` item 3), the RSSI logic moves with it.

## Verification

1. Run `calibrate_geometry.py` on all 5 test10 files → produces
   `channel_geometry_20260307experimentB.json`.
2. Run `calibrate_geometry.py` on test6_1907 + test6_1974 → produces
   `channel_geometry_20260306experimentA.json`.
3. Rerun `pressure_map_2d.py` on test10 5 Vpp — uses saved geometry,
   tilt matches 25 Vpp result.
4. `map2d_1f_vs_2f_test10_*_5Vpp_*.png` shows horizontal node line.
5. All test10 files report same tilt (from saved geometry).
