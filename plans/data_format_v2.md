# Data format v2 — what the new DAQ must include

Status: **draft for review** before the rebuilt DAQ starts writing files.

This document specifies the minimum content the new acquisition format
must provide so the existing analysis pipeline (and v2 evolutions of
it) can run unchanged after format migration. The v1 pipeline reads
TDMS plus brittle filename-encoded metadata; v2 must replace the
filename heuristics with explicit fields.

## Why this exists

Today the pipeline derives several critical parameters from filenames:

| Parameter | Today's source | Failure mode if wrong |
|---|---|---|
| LDV velocity scale (m/s per V) | `_Nm_s_max` in stem | silent factor-of-2 pressure error |
| Drive voltage Vpp | `_NVpp` in stem | wrong figure axis |
| Drive frequency | `_1907` (kHz) in stem | usually overridden by FFT detection, but used as initial guess |
| Test-run identifier (test10, etc.) | filename prefix | scripts hardcoded per dataset |

v2 moves all of these into explicit metadata so renaming a file can
never change its physics.

---

## Conceptual model

Each acquisition produces **one ScanData** unit:

```
ScanData
├── coordinates      per-point arrays (x, y, optional z, RSSI)
├── waveforms        per-point, per-channel time series with role tags
├── acquisition      session-level metadata (sample rate, LDV range, drive)
├── scan             grid metadata (n_x, n_y, pattern, step)
├── chip             pointer or inline copy of per-chip calibration
└── provenance       timestamp, operator, software version, free-text notes
```

The python interface (see `src/ldv_analysis/io_utils.py:ScanData` once
implemented) provides:

```python
@dataclass
class ScanData:
    pos_x: np.ndarray          # (N_points,) metres
    pos_y: np.ndarray          # (N_points,) metres
    rssi: np.ndarray | None    # (N_points,)
    dt: float                  # seconds, one value (channels share timebase)
    n_points: int
    n_samples: int             # per point, per channel
    metadata: dict             # see below — semantic keys, not channel numbers
    def load_waveforms(self, role: str, points: slice | np.ndarray) -> np.ndarray:
        """Return (k, n_samples) waveforms for k requested point indices."""
```

The reader (`load_scan_v2(path) -> ScanData`) is the only piece that
changes between formats; everything downstream consumes `ScanData`.

---

## Per-point arrays  (length = N_points)

| Field | Units | Required? | Notes |
|---|---|---|---|
| `pos_x` | metres | yes | scan x — match figure convention (channel length) |
| `pos_y` | metres | yes | scan y (channel width) |
| `pos_z` | metres | no | only if a z-stage is involved |
| `rssi` (or analogous) | arbitrary | recommended | optical signal-strength proxy; used by `make_rssi_mask` |

If the DAQ also computes per-point burst ON/OFF timing in real time,
include them as arrays `pt_burst_on_us`, `pt_burst_off_us` (else the
pipeline will detect them post-hoc).

---

## Per-point waveforms

Stored as `(N_points, N_samples)` per **role**, not per channel number.
Roles are semantic labels the analysis code uses; the DAQ chooses which
physical channel maps to each role.

| Role (string key) | What it carries | Required? |
|---|---|---|
| `"drive_voltage"` | Vpp piezo drive (was Ch1) | yes |
| `"ldv_output"` | LDV decoder voltage proportional to apparent velocity (was Ch2) | yes |
| `"current"` | Piezo drive current via shunt or probe (was Ch4) | no, but used for P_in and V-I phase |

All channels in one ScanData must share the same `dt` and `N_samples`.
Streaming-capable storage (HDF5, TDMS, zarr) is fine; the reader's
`load_waveforms(role, points)` is the access point.

---

## Acquisition metadata

These map to `ScanData.metadata[...]`. Names are the **canonical keys**
the pipeline expects.

| Key | Type | Required? | Replaces |
|---|---|---|---|
| `sample_rate_hz` (or equivalently `dt_s`) | float | yes | `wf_increment` TDMS property |
| `ldv_velocity_scale_mps_per_v` | float | yes | filename `_Nm_s_max` heuristic |
| `ldv_decoder_label` | string | optional | description, e.g. `"VD-09 ±1 m/s"` |
| `drive_frequency_hz_nominal` | float | yes | filename freq; pipeline still verifies via FFT |
| `drive_voltage_vpp` | float | yes | filename `_NVpp` |
| `burst_on_us_nominal` | float | yes | inferred from Ch1 today; explicit nominal allows sanity check |
| `burst_off_us_nominal` | float | yes | same |
| `voltage_probe_attenuation` | float | optional | currently global `VOLTAGE_ATTENUATION` |
| `current_scale_a_per_v` | float | optional | currently global `CURRENT_SCALE` |
| `channel_roles` | dict | yes | maps physical channel index → role string (see above) |
| `chip_id` | string | yes | links to per-chip JSON; see below |
| `session_id` | string | yes | groups runs taken under the same calibration |

---

## Scan metadata

| Key | Type | Required? | Notes |
|---|---|---|---|
| `n_x`, `n_y` | int | yes | grid shape; for sparse/irregular scans, set to 0 and rely on `pos_x`/`pos_y` |
| `scan_pattern` | string | yes | one of `"line"`, `"area"`, `"sparse"` |
| `scan_step_x_um`, `scan_step_y_um` | float | optional | reproducibility |
| `scan_order` | string | optional | `"raster"`, `"snake"` — needed if waveform names are not position-aligned |

---

## Chip / session-level metadata (sidecar JSON)

One JSON per chip, referenced from `metadata["chip_id"]`. Pipeline reads
it via `load_channel_geometry(chip_id, cache_dir)`. Suggested structure:

```json
{
  "chip_id": "ldv_chip_2026_W17_A",
  "channel": {
    "width_m": 3.75e-4,
    "height_m": 1.5e-4,
    "centre_axis_in_scan": "y",
    "tilt_rad": 0.0
  },
  "pzt": {
    "length_m": 6.0e-3,
    "x_range_m": [5.6e-3, 11.6e-3],
    "side": "x_positive"
  },
  "fluid": {
    "name": "water",
    "rho_kg_per_m3": 1000.0,
    "speed_of_sound_m_per_s": 1500.0,
    "dn_dp_per_pa": 1.4e-10,
    "beta_nonlinearity": 3.5
  },
  "rssi_threshold": 1.0,
  "voltage_quality_factor": 0.5
}
```

The pipeline today reads channel geometry from
`experiments/2026W10_stepA/cache/channel_geometry_*.json`; v2 extends
the same convention with the additional sections above.

---

## Provenance

| Key | Type | Required? |
|---|---|---|
| `timestamp_utc` | ISO-8601 string | yes |
| `operator` | string | yes |
| `daq_software_version` | string | yes |
| `notes` | string | optional, free-text |

---

## Example: minimal valid file

```jsonc
{
  "version": "v2.0",
  "coordinates": {
    "pos_x_m":   [/* N_points floats */],
    "pos_y_m":   [/* N_points floats */],
    "rssi":      [/* N_points floats */]
  },
  "waveforms": {
    "dt_s": 8e-9,
    "n_samples": 65536,
    "channel_roles": {
      "Ch1": "drive_voltage",
      "Ch2": "ldv_output",
      "Ch4": "current"
    },
    "data": { /* HDF5 datasets or TDMS groups keyed by role */ }
  },
  "acquisition": {
    "ldv_velocity_scale_mps_per_v": 0.5,
    "ldv_decoder_label": "VD-09 ±1 m/s",
    "drive_frequency_hz_nominal": 1907000.0,
    "drive_voltage_vpp": 5.0,
    "burst_on_us_nominal": 5.0,
    "burst_off_us_nominal": 525.0,
    "voltage_probe_attenuation": 10.0,
    "current_scale_a_per_v": 0.2
  },
  "scan": {
    "n_x": 101,
    "n_y": 101,
    "scan_pattern": "area",
    "scan_step_x_um": 5.0,
    "scan_step_y_um": 50.0,
    "scan_order": "snake"
  },
  "chip_id": "ldv_chip_2026_W17_A",
  "session_id": "2026W17_session_03",
  "provenance": {
    "timestamp_utc": "2026-05-13T11:23:08Z",
    "operator": "tatsuki",
    "daq_software_version": "v2.0.0",
    "notes": "first run after refit"
  }
}
```

(In practice the waveform array goes into HDF5 / TDMS / similar, not
into JSON — only the small fields above need to be human-readable.)

---

## Migration semantics

- `LDV_DATA_ROOT` + extension dispatcher (`.tdms` → v1 reader,
  `.h5` → v2 reader, etc.) means both formats coexist.
- v1 files do not need to be re-acquired; v1 reader extracts the same
  `ScanData` from filename heuristics.
- `_FFT_CACHE_VERSION` will bump when the v2 reader lands, so old
  caches don't silently mismatch new outputs.

---

## Open questions

1. **Storage container** — TDMS again, HDF5, zarr, raw binary + JSON
   manifest? My recommendation is HDF5: native Python support, hierarchical
   groups for `coordinates/`, `waveforms/`, fast partial reads, embedded
   attributes for metadata.
2. **Channel role registry** — fixed set (`drive_voltage`, `ldv_output`,
   `current`) or extensible? Reservation today: keep it small;
   add a `role_aux_N` namespace if needed.
3. **Per-point validity flags** — should the DAQ emit a boolean
   "this point is trustworthy" array, or is post-hoc filtering enough?
   Recommend post-hoc (already in `filters.py`); flagging at acquisition
   time hides issues the analysis should catch.
4. **Lazy vs eager reading** — 30+ GB per file means lazy is mandatory;
   the v2 reader must support `load_waveforms(role, slice)` without
   pulling everything into RAM.

Resolve these with the DAQ author before implementation begins.
