# %%
"""Frequency sweep analysis: resonance identification from Step A sweep data.

Processes all stepA_sweep_*.tdms files in a directory.  For each file
extracts drive frequency, voltage, current, V-I phase, and fits the
sinusoidal mode shape p(y) = p0|sin(pi*y/W)| to get the pressure
amplitude p0.  Plots all quantities vs frequency.

Usage:
    python freq_sweep.py [data_directory]
    python freq_sweep.py                    # uses default
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DPI, figsize_for_layout
from fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_DIR = Path("G:/My Drive/20260303experimentA")
GLOB_PATTERN = "stepA_sweep_*.tdms"

CHANNEL_WIDTH = 0.375  # mm (known physical width)
RSSI_THRESHOLD = 1.0   # V — exclude poor LDV signal (gap in RSSI distribution)

OUT_DIR = Path(__file__).parent.parent.parent / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Discover and process files
# =============================================================================

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
tdms_files = sorted(data_dir.glob(GLOB_PATTERN))
# Exclude index files and files with stray characters in name
tdms_files = [f for f in tdms_files if not f.name.endswith("_index")
              and "'" not in f.name]
print(f"Found {len(tdms_files)} sweep files in {data_dir}")

if not tdms_files:
    print("No files found.")
    sys.exit(1)

# Results arrays
freqs_hz = []
voltage_med = []
current_med = []
phase_vi_med = []
p0_fit = []

# Per-file data for mode-shape plots
mode_shape_data = []

hw = CHANNEL_WIDTH / 2
W_m = CHANNEL_WIDTH * 1e-3  # mm -> m
k = np.pi / W_m

for tdms_path in tdms_files:
    print(f"\n--- {tdms_path.name} ---")
    cache = load_or_compute(tdms_path, OUT_DIR)

    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]
    f_drive = float(cache["f_drive"])
    pressure_1f = cache["pressure_1f"]
    V = cache["voltage_1f"]

    # --- Quality filters ---
    V_med = np.median(V)
    valid = V > V_med * 0.5                         # missed bursts
    n_burst_bad = np.sum(~valid)

    if "rssi" in cache:
        valid &= cache["rssi"] > RSSI_THRESHOLD     # poor LDV signal
    n_rssi_bad = np.sum(~valid) - n_burst_bad

    n_valid = np.sum(valid)
    excluded = []
    if n_burst_bad > 0:
        excluded.append(f"{n_burst_bad} missed bursts")
    if n_rssi_bad > 0:
        excluded.append(f"{n_rssi_bad} low RSSI")
    if excluded:
        print(f"  Excluded: {', '.join(excluded)} "
              f"({n_valid}/{len(V)} valid)")

    freqs_hz.append(f_drive)
    voltage_med.append(np.median(V[valid]))

    has_ch4 = "current_1f" in cache
    if has_ch4:
        current_med.append(np.median(cache["current_1f"][valid]))
        phase_vi_med.append(np.median(cache["phase_vi"][valid]))
    else:
        current_med.append(np.nan)
        phase_vi_med.append(np.nan)

    # ---------------------------------------------------------------
    # Sinusoidal mode-shape fit across channel width
    # ---------------------------------------------------------------
    # All points are at nominally the same y — fit all valid points together.
    x_line = pos_x[valid]
    p_line = pressure_1f[valid]

    # Find channel centre: x_c that maximises projection onto |sin|
    x_trial = np.linspace(x_line.min() + hw, x_line.max() - hw, 200)
    best_p0 = 0
    best_xc = x_trial[0]
    for xc in x_trial:
        y_c = (x_line - xc) * 1e-3  # mm -> m, centred
        inside = np.abs(y_c) <= W_m / 2
        if inside.sum() < 3:
            continue
        sin_prof = np.abs(np.sin(k * y_c[inside]))
        # Least-squares projection: p0 = sum(p * sin) / sum(sin^2)
        p0_cand = np.sum(p_line[inside] * sin_prof) / np.sum(sin_prof ** 2)
        if p0_cand > best_p0:
            best_p0 = p0_cand
            best_xc = xc

    p0_fit.append(best_p0)

    # Store centred data for mode-shape plot
    x_centred = pos_x[valid] - best_xc
    mode_shape_data.append(dict(
        f_drive=f_drive, p0=best_p0, xc=best_xc,
        x_c=x_centred, p=pressure_1f[valid],
    ))

    print(f"  f = {f_drive/1e6:.4f} MHz,  p0 = {best_p0/1e3:.1f} kPa,  "
          f"V = {voltage_med[-1]:.2f} V,  I = {current_med[-1]*1e3:.1f} mA,  "
          f"phase = {phase_vi_med[-1]:.1f} deg")

# Convert to arrays
freqs_hz = np.array(freqs_hz)
voltage_med = np.array(voltage_med)
current_med = np.array(current_med)
phase_vi_med = np.array(phase_vi_med)
p0_fit = np.array(p0_fit)

freqs_mhz = freqs_hz / 1e6
p0_kpa = p0_fit / 1e3

# Sort by frequency
order = np.argsort(freqs_hz)
freqs_mhz = freqs_mhz[order]
p0_kpa = p0_kpa[order]
voltage_med = voltage_med[order]
current_med = current_med[order]
phase_vi_med = phase_vi_med[order]

# %%
# =============================================================================
# Print summary
# =============================================================================

print(f"\n{'='*60}")
print(f"Frequency sweep: {len(freqs_mhz)} points")
print(f"  f range: {freqs_mhz.min():.4f} -- {freqs_mhz.max():.4f} MHz")
best_idx = np.nanargmax(p0_kpa)
print(f"  Peak p0: {p0_kpa[best_idx]:.1f} kPa at {freqs_mhz[best_idx]:.4f} MHz")
print(f"  V range: {voltage_med.min():.2f} -- {voltage_med.max():.2f} V")
if not np.all(np.isnan(current_med)):
    print(f"  I range: {np.nanmin(current_med)*1e3:.1f} -- "
          f"{np.nanmax(current_med)*1e3:.1f} mA")

# %%
# =============================================================================
# Plot: 4-panel frequency sweep
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=figsize_for_layout(4, 1, sharex=True),
                         sharex=True)

# p0 vs frequency
axes[0].plot(freqs_mhz, p0_kpa, "-o", markersize=3, linewidth=0.8)
axes[0].set_ylabel("$p_0$ (kPa)")
axes[0].set_title("Frequency sweep --- Step A")
axes[0].grid(True, alpha=0.3)

# V-I phase vs frequency
axes[1].plot(freqs_mhz, phase_vi_med, "-o", markersize=3, linewidth=0.8,
             color="C1")
axes[1].set_ylabel("V--I phase (deg)")
axes[1].grid(True, alpha=0.3)

# Current vs frequency
axes[2].plot(freqs_mhz, current_med * 1e3, "-o", markersize=3, linewidth=0.8,
             color="C2")
axes[2].set_ylabel("Current (mA)")
axes[2].grid(True, alpha=0.3)

# Voltage vs frequency
axes[3].plot(freqs_mhz, voltage_med, "-o", markersize=3, linewidth=0.8,
             color="C3")
axes[3].set_ylabel("Voltage (V)")
axes[3].set_xlabel("Frequency (MHz)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / "freq_sweep.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")

# %%
# =============================================================================
# Plot: mode-shape fits (data + sinusoidal fit for each frequency)
# =============================================================================

mode_dir = OUT_DIR / "mode_shapes"
mode_dir.mkdir(parents=True, exist_ok=True)

x_fine = np.linspace(-hw, hw, 200)
sin_fine = np.abs(np.sin(k * x_fine * 1e-3))

for md in mode_shape_data:
    fig, ax = plt.subplots(figsize=figsize_for_layout())
    p0_kpa = md["p0"] / 1e3
    x_c_m = md["x_c"] * 1e-3
    inside = np.abs(x_c_m) <= W_m / 2

    # Plot inside / outside separately
    ax.plot(md["x_c"][~inside], md["p"][~inside] / 1e3,
            "x", markersize=3, alpha=0.3, color="0.6")
    ax.plot(md["x_c"][inside], md["p"][inside] / 1e3,
            ".", markersize=3, alpha=0.6)

    # R² inside channel
    if inside.sum() > 2:
        p_in = md["p"][inside]
        p_pred = md["p0"] * np.abs(np.sin(k * x_c_m[inside]))
        ss_res = np.sum((p_in - p_pred) ** 2)
        ss_tot = np.sum((p_in - np.mean(p_in)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2 = 0

    ax.plot(x_fine, p0_kpa * sin_fine, "--", linewidth=1, color="C3",
            label=f"$p_0$ = {p0_kpa:.0f} kPa, $R^2$ = {r2:.2f}")
    ax.axvline(-hw, color="0.5", ls=":", lw=0.5)
    ax.axvline(hw, color="0.5", ls=":", lw=0.5)
    f_khz = md["f_drive"] / 1e3
    ax.set_title(f"Mode Shape at {f_khz:.1f} kHz")
    ax.set_xlabel("Channel width (mm)")
    ax.set_ylabel("Pressure (kPa)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-hw * 1.3, hw * 1.3)
    plt.tight_layout()
    fname = mode_dir / f"mode_shape_{f_khz:.0f}kHz.png"
    plt.savefig(fname, dpi=FIG_DPI)
    plt.close()

print(f"Saved {len(mode_shape_data)} mode-shape plots to {mode_dir}")

# %%
print(f"\n=== Done ===")
