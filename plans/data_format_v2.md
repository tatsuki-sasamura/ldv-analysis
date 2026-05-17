# Data format v2 — what the new DAQ must include

Status: **contract for the rebuilt DAQ.**

This document specifies the minimum content the new acquisition format
must provide so the existing analysis pipeline (and v2 evolutions of
it) can run unchanged after format migration. The v1 pipeline reads
TDMS plus brittle filename-encoded metadata; v2 replaces the filename
heuristics with explicit fields in an HDF5 file plus one JSON sidecar.

---

## DAQ author checklist

To write a v2-compliant acquisition program, do these five steps in
order:

1. **Create one chip sidecar JSON.** Copy the example block under
   *Chip + mounting sidecar* into `chip_{your_chip_id}.json` and fill
   in the channel/PZT geometry and the mounting calibration. One file
   per (chip, mounting) pair.
2. **In the acquisition loop, write one HDF5 file per scan.** Use the
   streaming pattern in *Example: minimal valid file*: pre-allocate
   the waveform datasets with `chunks=(1, n_samples)`, then
   `dset[i] = waveform_i` after each captured point. Required root
   attributes are listed under *Per-acquisition*.
3. **Match canonical names exactly.** Channel roles are
   `drive_voltage`, `ldv_output`, `current`. Attribute keys
   (`sample_rate_hz`, `chip_id`, etc.) are case-sensitive.
4. **Convert stage coordinates to meters.** Stages typically report
   mm; the schema is `pos_x_m`, `pos_y_m` in meters.
5. **Run the validator on your first output:**
   ```bash
   python -c "from ldv_analysis.io_utils import validate_hdf5_v2; \
              print(validate_hdf5_v2('your_file.h5') or 'OK')"
   ```
   Empty list (or `OK`) = pass. Any non-empty list lists exactly what
   to fix.

DAQ hardware constants (probe attenuation, current scale, LDV decoder
model) are not in the file — they stay as Python globals in
`config.py` and are applied at analysis time. The DAQ doesn't need to
emit them.

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
    pos_x: np.ndarray          # (N_points,) meters
    pos_y: np.ndarray          # (N_points,) meters
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
| `pos_x` | meters | yes | scan x — match figure convention (channel length) |
| `pos_y` | meters | yes | scan y (channel width) |
| `pos_z` | meters | no | only if a z-stage is involved |
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

## Layered configuration

Configuration splits across four layers by lifetime and what it
describes. Keeping them separated means changing one (e.g. remounting
the chip) doesn't force editing the others.

| Layer | Where it lives | Lifetime |
|---|---|---|
| **Per-acquisition** | HDF5 root attrs in the scan file | one file |
| **Chip + mounting** | `chip_{chip_id}.json` sidecar | one chip in one mounting |
| **Analysis policy** | `analysis.json` (or Python globals in `config.py`) | site-wide |
| **DAQ hardware** | Python globals in `config.py` for now; promote to `daq_{id}.json` if you change DAQ rigs | per DAQ session |

### Per-acquisition (HDF5 root attrs)

Only things that genuinely vary per file. Names are the **canonical
keys** the pipeline expects.

| Key | Type | Required? | Replaces / notes |
|---|---|---|---|
| `sample_rate_hz` | float | yes | `wf_increment` TDMS property |
| `n_samples` | int | yes | per point, per channel |
| `ldv_velocity_scale_mps_per_v` | float | yes | filename `_Nm_s_max` heuristic; can change between runs if the LDV decoder range is switched |
| `drive_frequency_hz_nominal` | float | yes | pipeline still verifies via FFT |
| `drive_voltage_vpp` | float | yes | filename `_NVpp` |
| `burst_on_us_nominal` | float | yes | **time within waveform when burst starts** (onset), not duration |
| `burst_off_us_nominal` | float | yes | time when burst ends |
| `scan_n_x`, `scan_n_y` | int | yes | grid shape; both 0 means sparse — rely on `/coordinates/pos_*` |
| `scan_step_x_um`, `scan_step_y_um` | float | optional | reproducibility |
| `scan_order` | string | optional | `"raster"` or `"snake"`; affects waveform ordering relative to position |
| `chip_id` | string | yes | links to chip sidecar |
| `session_id` | string | yes | groups runs taken under one chip mounting + calibration |
| `timestamp_utc` | ISO-8601 | yes | provenance |
| `operator` | string | yes | provenance |
| `daq_software_version` | string | yes | provenance |
| `notes` | string | optional | free-text |

Field-semantics clarifications worth being explicit about:
- **Coordinates are in meters** (`pos_x_m`, `pos_y_m`). Stages typically report mm — convert before writing.
- **`burst_on_us_nominal`** = the time *within the waveform window* at which the burst begins (e.g. 5 µs). Not the burst's duration.
- `scan_pattern` is dropped — it's derivable from `scan_n_x`/`scan_n_y`.

### Chip + mounting sidecar (`chip_{chip_id}.json`)

Physical chip properties plus how the chip sits in this mounting. One
JSON per (chip, mounting) pair; if you remount the same chip, bump the
suffix or start a new file.

```json
{
  "chip_id": "ldv_chip_2026_W17_A",
  "channel": {
    "width_m": 3.75e-4,
    "height_m": 1.5e-4
  },
  "pzt": {
    "length_m": 6.0e-3,
    "side": "x_positive"
  },
  "mounting": {
    "center_axis_in_scan": "y",
    "tilt_rad": 0.0,
    "pzt_x_range_m_in_scan": [5.6e-3, 11.6e-3]
  }
}
```

Why `mounting` is a subsection of the chip JSON (not its own file):
in practice you don't remount during a study, so splitting it into a
separate file is friction for no payoff. If you ever do change
mountings often, promote `mounting` to its own sidecar then.

### Analysis policy (`analysis.json` or `config.py` globals)

Independent of the data — defines what the analysis considers valid
or how it fits. Default to keeping these as `config.py` globals
(`RSSI_THRESHOLD`, `VOLTAGE_QUALITY_FACTOR`); promote to JSON only if
you start running multiple analysis policies against the same data.

```json
{
  "rssi_threshold": 1.0,
  "voltage_quality_factor": 0.5,
  "sigma_clip_default": 3.0
}
```

### Fluid properties

Currently global in `config.py` (`RHO`, `C_SOUND`, `DN_DP`, plus β in
analysis code). Promote to `fluid_{id}.json` if you start switching
fluids; today, water everywhere makes a sidecar overkill.

### DAQ hardware

`VOLTAGE_ATTENUATION`, `CURRENT_SCALE`, the LDV decoder model live in
`config.py` for now. Promote to `daq_{id}.json` only if you swap DAQ
rigs.

The pipeline today reads chip geometry from
`experiments/2026W10_stepA/cache/channel_geometry_*.json`; v2 extends
the same convention with the chip-and-mounting layout shown above.

---

## Example: minimal valid file (streaming acquisition pattern)

The DAQ does not need to buffer the whole scan in RAM. Pre-allocate
the waveform datasets once and assign one row per scan point as data
arrives:

```python
import h5py
import numpy as np

N = 101 * 101            # total scan points
n_samples = 65536
sample_rate_hz = 125e6

with h5py.File("acquisition.h5", "w") as f:
    # ---- Root attributes (set once, before the loop) ----------------
    f.attrs["version"] = "2.0"
    f.attrs["timestamp_utc"] = "2026-05-13T11:23:08Z"
    f.attrs["operator"] = "tatsuki"
    f.attrs["daq_software_version"] = "v2.0.0"
    f.attrs["sample_rate_hz"] = sample_rate_hz
    f.attrs["n_samples"] = n_samples
    f.attrs["ldv_velocity_scale_mps_per_v"] = 0.5
    f.attrs["drive_frequency_hz_nominal"] = 1907000.0
    f.attrs["drive_voltage_vpp"] = 5.0
    f.attrs["burst_on_us_nominal"] = 5.0     # onset within waveform window
    f.attrs["burst_off_us_nominal"] = 525.0
    f.attrs["scan_n_x"] = 101
    f.attrs["scan_n_y"] = 101
    f.attrs["scan_step_x_um"] = 5.0
    f.attrs["scan_step_y_um"] = 50.0
    f.attrs["scan_order"] = "snake"
    f.attrs["chip_id"] = "ldv_chip_2026_W17_A"
    f.attrs["session_id"] = "2026W17_session_03"

    # ---- Coordinates (write per-point or in one block at the end) ---
    coords = f.create_group("coordinates")
    pos_x = coords.create_dataset("pos_x_m", shape=(N,), dtype="float64")
    pos_y = coords.create_dataset("pos_y_m", shape=(N,), dtype="float64")
    rssi  = coords.create_dataset("rssi",    shape=(N,), dtype="float32")

    # ---- Waveforms: pre-allocate, chunked one-per-point -------------
    wf = f.create_group("waveforms")
    common = dict(shape=(N, n_samples), dtype="float32",
                  chunks=(1, n_samples))
    drive_v = wf.create_dataset("drive_voltage", **common)
    ldv_out = wf.create_dataset("ldv_output",    **common)
    current = wf.create_dataset("current",       **common)   # optional

    # ---- Streaming acquisition loop ---------------------------------
    for i in range(N):
        x_mm, y_mm = next_stage_position()    # stage reports mm
        pos_x[i] = x_mm * 1e-3                # convert to meters
        pos_y[i] = y_mm * 1e-3
        rssi[i]  = read_rssi()

        # Each capture blocks until the DAQ delivers one full waveform.
        # Assignment writes exactly one HDF5 chunk (cheap, ~ms).
        wfs = capture_one_burst()             # dict[role -> (n_samples,)]
        drive_v[i] = wfs["drive_voltage"]
        ldv_out[i] = wfs["ldv_output"]
        current[i] = wfs["current"]
```

Memory footprint is bounded by `n_samples` per channel (e.g. 65k×4 B
= 256 kB per role) regardless of `N`.

---

## Migration semantics

- `LDV_DATA_ROOT` + extension dispatcher (`.tdms` → v1 reader,
  `.h5` → v2 reader, etc.) means both formats coexist.
- v1 files do not need to be re-acquired; v1 reader extracts the same
  `ScanData` from filename heuristics.
- `_FFT_CACHE_VERSION` will bump when the v2 reader lands, so old
  caches don't silently mismatch new outputs.

---

## Resolved decisions

1. **Storage container: HDF5.** Native h5py support, hierarchical
   groups, fast partial reads, attributes carry metadata. DAQ is in
   Python anyway, so it can write the file directly with no
   intermediate format.
2. **Channel roles: fixed set** — exactly `drive_voltage`,
   `ldv_output`, `current`. No `role_aux_N` namespace until a real
   need arises.
3. **No DAQ-time validity flags.** All quality filtering stays
   post-hoc in `src/ldv_analysis/filters.py`. The DAQ writes
   everything; the analysis decides what to keep.
4. **Lazy reading is required.** Files are >30 GB; the v2 reader
   must serve `load_waveforms(role, slice)` without pulling all
   waveforms into RAM. Use HDF5 chunked datasets with point-axis
   chunking (one chunk per point, or per small block of points) so
   the access pattern matches.

---

## Concrete HDF5 layout (target for the DAQ)

```
acquisition.h5
├── /coordinates/
│   ├── pos_x_m              dataset (N,)  float64     # meters, channel length
│   ├── pos_y_m              dataset (N,)  float64     # meters, channel width
│   └── rssi                 dataset (N,)  float32     # optional
│
├── /waveforms/                              # chunked along point axis
│   ├── drive_voltage        dataset (N, n_samples) float32   chunks=(1, n_samples)
│   ├── ldv_output           dataset (N, n_samples) float32   chunks=(1, n_samples)
│   └── current              dataset (N, n_samples) float32   chunks=(1, n_samples)
│                                                              optional group
│
└── root attrs:                                          # per-acquisition only
    version                          = "2.0"
    timestamp_utc                    = "2026-..."
    operator                         = "..."
    daq_software_version             = "..."
    sample_rate_hz                   = 125000000.0
    n_samples                        = 65536
    ldv_velocity_scale_mps_per_v     = 0.5
    drive_frequency_hz_nominal       = 1907000.0
    drive_voltage_vpp                = 5.0
    burst_on_us_nominal              = 5.0                 # onset within waveform
    burst_off_us_nominal             = 525.0
    scan_n_x                         = 101
    scan_n_y                         = 101
    scan_step_x_um                   = 5.0                 # optional
    scan_step_y_um                   = 50.0                # optional
    scan_order                       = "snake"             # optional
    chip_id                          = "ldv_chip_2026_W17_A"   # -> chip_*.json sidecar
    session_id                       = "2026W17_session_03"
    notes                            = ""                  # optional
```

### Notes on the HDF5 choice

- **Float32 waveforms** are sufficient (the LDV ADC is 14-16 bit; float32 has
  24-bit mantissa). Halves disk footprint vs float64.
- **One chunk per point** keeps `load_waveforms(role, [i, j, k])` cheap: only
  the requested chunks are read. If a future analysis frequently reads
  10-100 consecutive points, switching to `chunks=(64, n_samples)` may
  improve throughput; revisit after profiling.
- **Attributes vs datasets for scalars**: scalars (sample_rate_hz,
  drive_voltage_vpp, etc.) live as root attributes — fast to read, no
  load required. Per-point arrays must be datasets, not attributes
  (attribute size is limited to 64 KB in HDF5).
- **Compression**: leave off by default — gzip can quadruple write
  time on a streaming acquisition and the disk savings are modest on
  float-32 burst-mode waveforms (mostly zero outside the burst, but
  the burst itself doesn't compress much). Revisit if disk pressure
  becomes real.
- **Chip-level sidecar JSON** (chip_id → channel geometry, PZT
  position, β, dn/dp) stays separate; the HDF5 only carries `chip_id`
  as a pointer.

### Reader implications

`src/ldv_analysis/io_utils.py:load_scan_v2(path)` should:

1. Open the file lazily (`h5py.File(path, "r", swmr=True)` or similar).
2. Read `/coordinates/pos_x_m`, `pos_y_m`, optional `rssi` eagerly into numpy arrays.
3. Read all root attributes into `ScanData.metadata` with the same keys.
4. Build the `_loader` closure to read `/waveforms/<role>[indices, :]` on demand.
5. Keep the file handle alive for the lifetime of the `ScanData` object — close it explicitly via a `close()` method or context manager (or use a per-call `h5py.File` open inside the loader, simpler and only ~ms slower for typical chunk reads).

---

## Open questions

None right now — the four design decisions above are settled. Reopen
if the rebuild surfaces new constraints (e.g., multi-LDV-head
acquisition, multi-fluid runs in one file, or external trigger
metadata).
