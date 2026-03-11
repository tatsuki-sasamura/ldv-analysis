# Plan: Extract duplicated code to shared library

## Overview

The 11 experiment scripts in `experiments/2026W10_stepA/` share significant
duplicated logic.  This plan extracts the most-repeated patterns into
`src/ldv_analysis/` modules.

New modules: **`mode_fit.py`** (items 1+3 merged).
Modified modules: **`fft_cache.py`** (item 2), **`config.py`** (item 5).
Item 4 (grid construction) stays inline — only 2 scripts use it.

---

## 1. Mode-shape fitting + centre search → `src/ldv_analysis/mode_fit.py` (HIGH)

### What is duplicated

Two distinct patterns appear across 8 scripts:

**(a) Line-scan fit with centre search** (5 scripts):
Brute-force scan for channel centre, then project onto sinusoidal mode:

```python
for xc in x_trial:
    y_c = (x_line - xc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    sin_prof = np.abs(np.sin(k * y_c[inside]))
    p0_cand = np.sum(p_line[inside] * sin_prof) / np.sum(sin_prof ** 2)
```

Used in: freq_sweep_coarse, freq_sweep_fine, freq_sweep_25vpp,
freq_axial_sweep, pressure_buildup.

**(b) Column-wise fit on 2D grid** (2 scripts):
Channel centre is already known; fit each axial column independently:

```python
for j in range(n_y):
    col = grid[:, j]
    valid = ~np.isnan(col) & quality_mask
    p0[j] = np.sum(col[valid] * mode[valid]) / np.sum(mode[valid] ** 2)
```

Used in: pressure_map_2d, voltage_sweep.

Plus R² computation (both patterns, 7 scripts).

### Where it appears

| Script | Line-scan | Column-wise | 1f | 2f | R² |
|--------|-----------|-------------|----|----|-----|
| freq_sweep_coarse.py | yes | — | yes | — | yes |
| freq_sweep_fine.py | yes | — | yes | — | yes |
| freq_sweep_25vpp.py | yes | — | yes | yes | yes |
| freq_sweep_2f.py | yes | — | — | yes | yes |
| freq_axial_sweep.py | yes | — | yes | — | yes |
| pressure_map_2d.py | — | yes | yes | yes | yes |
| voltage_sweep.py | — | yes | yes | yes | yes |
| pressure_buildup.py | yes | — | yes | — | — |

### Proposed API

```python
# src/ldv_analysis/mode_fit.py

from dataclasses import dataclass

@dataclass
class ModeFitResult:
    p0: float           # pressure amplitude (Pa)
    centre: float       # channel centre position (mm)
    r2: float           # goodness of fit
    inside: np.ndarray  # boolean mask of points inside channel

def fit_mode_1f(
    positions_mm: np.ndarray,   # scan positions in mm
    pressure: np.ndarray,       # pressure values (Pa)
    channel_width_mm: float,
    centre: float | None = None,  # None → brute-force search
    quality_mask: np.ndarray | None = None,
    n_trial: int = 200,
) -> ModeFitResult:
    """Fit p(y) = p0 * |sin(π y/W)| to line-scan data."""

def fit_mode_2f(
    positions_mm: np.ndarray,
    pressure: np.ndarray,
    channel_width_mm: float,
    centre: float,              # required (reuse 1f centre)
    quality_mask: np.ndarray | None = None,
) -> ModeFitResult:
    """Fit p(y) = p0 * |cos(2π y/W)| to line-scan data."""

def fit_columns(
    grid: np.ndarray,             # (n_width, n_length)
    width_positions_m: np.ndarray,  # centred positions (m)
    channel_width_m: float,
    harmonic: int = 1,            # 1 → sin, 2 → cos
    quality_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit mode shape at each axial position. Returns p0 array (n_length,)."""
```

The centre search is internal to `fit_mode_1f(centre=None)` — not a
separate function.  `fit_columns` is the grid-based variant for 2D data.

### Migration example

Before:
```python
for xc in x_trial:
    y_c = (x_line - xc) * 1e-3
    inside = np.abs(y_c) <= W / 2
    if inside.sum() < 3: continue
    sin_prof = np.abs(np.sin(k * y_c[inside]))
    p0_cand = np.sum(p_line[inside] * sin_prof) / np.sum(sin_prof ** 2)
    if p0_cand > best_p0:
        best_p0, best_xc = p0_cand, xc
# ... R² computation ...
```

After:
```python
from ldv_analysis.mode_fit import fit_mode_1f
result = fit_mode_1f(x_line, p_line, CHANNEL_WIDTH)
best_p0, best_xc, r2 = result.p0, result.centre, result.r2
```

---

## 2. 2f harmonic extraction → `src/ldv_analysis/fft_cache.py` (MEDIUM)

### What is duplicated

Per-point DFT at 2×f_drive, reading raw Ch2 waveforms from TDMS:

```python
tone_2f = np.exp(-2j * np.pi * (2 * f_drive) * np.arange(ss_n) * dt)
for i in range(n_points):
    wf = wf_group[ch2_names[i]][ss_start:ss_end]
    dft = np.dot(wf, tone_2f)
    vel = np.abs(dft) * 2 / ss_n * VELOCITY_SCALE
    pressure_2f[i] = vel / (2 * np.pi * 2 * f_drive * SENSITIVITY)
```

### Where it appears (4 scripts)

- `pressure_map_2d.py`
- `voltage_sweep.py`
- `freq_sweep_25vpp.py`
- `freq_sweep_2f.py` (could use it — drive is already at 2f frequency)

### Proposed change

Add `velocity_2f` and `pressure_2f` to the cache computation in
`_compute()`.  The 2f DFT is one extra dot product per point — negligible
cost since the TDMS waveforms are already loaded for the 1f computation.

```python
# In _compute(), inside the per-point loop (after 1f DFT):
dft_2f = np.dot(wf_ss, tone_2f)
velocity_2f[i] = np.abs(dft_2f) * 2 / ss_n * VELOCITY_SCALE
pressure_2f[i] = velocity_2f[i] / (2 * np.pi * 2 * f_drive * SENSITIVITY)
```

Cache fields added: `velocity_2f`, `pressure_2f`.

### Cache compatibility

Existing `.npz` caches lack `pressure_2f`.  **Delete old caches and
regenerate.**  This is simpler than a lazy `_ensure_2f()` helper which
would need to re-open the TDMS file (path not stored in cache).  Cache
regeneration re-reads waveforms anyway, so the 2f extraction adds no
significant extra time.

### LDV range correction

Scripts that use non-default LDV ranges (e.g. voltage_sweep.py with
`vel_correction`) apply the correction **after** loading the cache, same
as for `pressure_1f`.  No change to this pattern:

```python
pressure_2f = cache["pressure_2f"] * vel_correction
```

### Migration

Scripts replace inline 2f extraction loops with:
```python
pressure_2f = cache["pressure_2f"]  # * vel_correction if needed
```

Eliminates 4 copies of the slow TDMS-reading loop.  For
`freq_sweep_25vpp.py` and `voltage_sweep.py`, this also removes the
`from nptdms import TdmsFile` import from those scripts.

---

## 3. Grid construction — keep inline (LOW)

Only `pressure_map_2d.py` and `voltage_sweep.py` use the `to_grid()`
pattern.  Extracting to a separate module (`grid.py` + `ChannelGrid`
dataclass) adds a module for just 2 call sites.

**Decision:** Keep inline.  If a third script needs gridding in the
future, extract then.

---

## 4. Quality filtering → `src/ldv_analysis/config.py` (LOW)

### What is duplicated

Constants defined independently in 7 scripts:

```python
RSSI_THRESHOLD = 1.0
EDGE_MARGIN = 1
```

Plus the mask construction pattern (5-line block).

### Proposed change

Add constants to `config.py`:

```python
RSSI_THRESHOLD = 1.0   # V — exclude poor LDV signal
EDGE_MARGIN = 1         # grid points to exclude at each channel wall
```

Add a utility function to `mode_fit.py` (not config — it depends on numpy):

```python
def make_quality_mask(
    n_width: int,
    edge_margin: int = EDGE_MARGIN,
    rssi_grid: np.ndarray | None = None,
    rssi_threshold: float = RSSI_THRESHOLD,
) -> np.ndarray:
    """Build boolean mask excluding edge points and low-RSSI columns."""
```

Scripts import the constants from config and call `make_quality_mask()`
instead of building the mask inline.

---

## Execution order

| Step | Items | Scope | Verify against |
|------|-------|-------|----------------|
| 1 | Tilt fix (`tilt_fix.md`) | pressure_map_2d.py + new calibrate script | test10 maps |
| 2 | Item 2 (2f in fft_cache) | fft_cache.py + 4 scripts | delete caches, rerun all 11 |
| 3 | Items 1+4 (mode_fit + quality) | new mode_fit.py + config.py + 8 scripts | rerun all 11, diff outputs |

Steps 1 and 2 are independent — can be done in either order or in
parallel.  Step 3 depends on neither but is the largest change, so do
it last.

### What NOT to do

- Don't create `boundary.py` as a separate module.  The 1D centre search
  lives inside `fit_mode_1f()`.  The 2D tilt detection stays in
  `pressure_map_2d.py` (or `calibrate_geometry.py`) — it's only used there.
- Don't create `grid.py` for 2 call sites.
- Don't add `_ensure_2f()` lazy loading — just delete old caches.

## Verification

After each step, rerun all 11 scripts and confirm:
1. No script errors
2. Numerical output (p0 values, R², tilt) matches pre-refactoring
3. Output PNGs are visually identical
