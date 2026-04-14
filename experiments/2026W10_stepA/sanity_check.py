# %%
"""Unified electrical & data-quality sanity check for a single TDMS scan.

Checks:
  1. Thermal drift — Ch1 (voltage) and Ch4 (current) stability across scan
  2. Electrical harmonics — amplifier THD: Ch1 and Ch4 at 2f, 3f, 4f
  3. Missed bursts — V_1f < 0.5 × median
  4. Burst timing — pt_burst_on/off outliers
  5. RSSI — fraction below threshold

Usage:
    python sanity_check.py <tdms_path> [--ldv-range {1,2,5}]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np

from ldv_analysis.config import (
    FIG_DPI,
    RSSI_THRESHOLD,
    figsize_for_layout,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.filters import (
    make_burst_timing_mask,
    make_rssi_mask,
    make_voltage_mask,
)

# %%
# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tdms_path", type=Path)
parser.add_argument("--ldv-range", type=int, choices=[1, 2, 5], default=2,
                    help="LDV velocity range (m/s) — for reference only")
args = parser.parse_args()

tdms_path = args.tdms_path
stem = tdms_path.stem

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load cache
# =============================================================================

print(f"Loading: {tdms_path.name}")
cache = load_or_compute(tdms_path, CACHE_DIR)
n_points = len(cache["pos_x"])

V_1f = cache["voltage_1f"]
has_ch4 = "current_1f" in cache
I_1f = cache["current_1f"] if has_ch4 else None
Z_mag = cache["impedance_1f"] if has_ch4 else None
phase_VI = cache["phase_vi"] if "phase_vi" in cache else None
rssi = cache["rssi"] if "rssi" in cache else None

f_drive = float(cache["f_drive"])
dt = float(cache["dt"])
ss_start = int(cache["ss_start"])
ss_end = int(cache["ss_end"])
ss_n = ss_end - ss_start

# %%
# =============================================================================
# 1. Missed bursts
# =============================================================================

voltage_valid = make_voltage_mask(V_1f)
n_missed = int(np.sum(~voltage_valid))
pct_missed = n_missed / n_points * 100

print(f"\n{'='*60}")
print(f"1. MISSED BURSTS")
print(f"{'='*60}")
print(f"  Valid: {np.sum(voltage_valid)} / {n_points}  "
      f"({n_missed} missed, {pct_missed:.1f}%)")
if pct_missed > 5:
    print(f"  ** WARNING: {pct_missed:.1f}% missed bursts (>5%)")
elif pct_missed > 1:
    print(f"  * Note: {pct_missed:.1f}% missed bursts")
else:
    print(f"  OK")

# %%
# =============================================================================
# 2. Burst timing
# =============================================================================

print(f"\n{'='*60}")
print(f"2. BURST TIMING")
print(f"{'='*60}")

has_burst = "pt_burst_on_us" in cache
if has_burst:
    burst_on = cache["pt_burst_on_us"]
    burst_off = cache["pt_burst_off_us"]
    burst_valid = make_burst_timing_mask(burst_on, burst_off)
    n_timing_bad = int(np.sum(~burst_valid & voltage_valid))
    pct_timing = n_timing_bad / max(np.sum(voltage_valid), 1) * 100
    med_on = np.nanmedian(burst_on)
    med_off = np.nanmedian(burst_off)
    print(f"  Burst ON median:  {med_on:.1f} us")
    print(f"  Burst OFF median: {med_off:.1f} us")
    print(f"  Duration: {med_off - med_on:.1f} us")
    print(f"  Timing outliers: {n_timing_bad} ({pct_timing:.1f}% of valid)")
    if pct_timing > 5:
        print(f"  ** WARNING: {pct_timing:.1f}% burst timing outliers")
    else:
        print(f"  OK")
else:
    burst_valid = np.ones(n_points, dtype=bool)
    print(f"  No burst timing data in cache")

# %%
# =============================================================================
# 3. RSSI
# =============================================================================

print(f"\n{'='*60}")
print(f"3. RSSI")
print(f"{'='*60}")

if rssi is not None:
    rssi_valid = make_rssi_mask(rssi, RSSI_THRESHOLD)
    n_low_rssi = int(np.sum(~rssi_valid))
    pct_rssi = n_low_rssi / n_points * 100
    print(f"  Threshold: {RSSI_THRESHOLD}")
    print(f"  Below threshold: {n_low_rssi} ({pct_rssi:.1f}%)")
    print(f"  Median RSSI: {np.median(rssi):.1f}")
    if pct_rssi > 20:
        print(f"  ** WARNING: {pct_rssi:.1f}% low RSSI")
    else:
        print(f"  OK")
else:
    print(f"  No RSSI data")

# %%
# =============================================================================
# 4. Thermal drift — Ch1, Ch4 stability across scan
# =============================================================================

print(f"\n{'='*60}")
print(f"4. THERMAL DRIFT")
print(f"{'='*60}")

valid = voltage_valid
valid_idx = np.where(valid)[0]
n_check = min(50, len(valid_idx) // 4)
early_idx = valid_idx[:n_check]
late_idx = valid_idx[-n_check:]

drift_data = [("V_1f", V_1f, "V")]
if has_ch4:
    drift_data += [
        ("I_1f", I_1f * 1e3, "mA"),
        ("|Z|", Z_mag, "ohm"),
        ("phase_VI", phase_VI, "deg"),
    ]

for name, arr, unit in drift_data:
    med = np.median(arr[valid])
    std = arr[valid].std()
    early = arr[early_idx].mean()
    late = arr[late_idx].mean()
    shift = late - early
    rel = abs(shift / early) * 100 if early != 0 else 0

    if name == "phase_VI":
        print(f"  {name:>8}: median={med:.3f} deg, std={std:.4f} deg, "
              f"drift={shift:+.4f} deg")
        if abs(shift) > 1.0:
            print(f"           ** WARNING: phase drift > 1 deg")
    else:
        print(f"  {name:>8}: median={med:.4f} {unit}, std={std:.4f} {unit} "
              f"({std/med*100:.2f}%), drift={shift:+.4f} ({rel:.2f}%)")
        if rel > 2.0:
            print(f"           ** WARNING: drift > 2%")

# %%
# =============================================================================
# 5. Electrical harmonics — amplifier THD
# =============================================================================

print(f"\n{'='*60}")
print(f"5. ELECTRICAL HARMONICS (amplifier THD)")
print(f"{'='*60}")

# Sample harmonics at multiple points across the scan
n_sample = 5
sample_idx = np.linspace(0, n_points - 1, n_sample + 2, dtype=int)[1:-1]
# Pick from valid points near these positions
sample_pts = []
for si in sample_idx:
    nearby_valid = valid_idx[np.argmin(np.abs(valid_idx - si))]
    sample_pts.append(int(nearby_valid))

ch1_harmonics_all = []
ch4_harmonics_all = []

for pt in sample_pts:
    try:
        channels = (1, 4) if has_ch4 else (1,)
        wfs, _ = load_point_waveforms(tdms_path, pt, channels=channels)
        ch1_ss = wfs[1][ss_start:ss_end]
        t_ss = np.arange(ss_n)

        ch1_h = {}
        for h in range(1, 5):
            tone = np.exp(-2j * np.pi * h * f_drive * t_ss * dt)
            ch1_h[h] = np.abs(ch1_ss @ tone) * 2 / ss_n
        ch1_harmonics_all.append(ch1_h)

        if has_ch4:
            ch4_ss = wfs[4][ss_start:ss_end]
            ch4_h = {}
            for h in range(1, 5):
                tone = np.exp(-2j * np.pi * h * f_drive * t_ss * dt)
                ch4_h[h] = np.abs(ch4_ss @ tone) * 2 / ss_n
            ch4_harmonics_all.append(ch4_h)

    except Exception as e:
        print(f"  Warning: could not load waveform at point {pt}: {e}")

if ch1_harmonics_all:
    # Average across sampled points
    ch1_avg = {h: np.mean([d[h] for d in ch1_harmonics_all]) for h in range(1, 5)}
    print(f"  Ch1 (voltage): 1f = {ch1_avg[1]:.4f} V")
    for h in [2, 3, 4]:
        ratio = ch1_avg[h] / ch1_avg[1] * 100 if ch1_avg[1] > 0 else 0
        print(f"    {h}f/1f = {ratio:.3f}%")
        if h == 2 and ratio > 1.0:
            print(f"    ** WARNING: Ch1 2f/1f = {ratio:.2f}% (>1%)")

    if ch4_harmonics_all:
        ch4_avg = {h: np.mean([d[h] for d in ch4_harmonics_all]) for h in range(1, 5)}
        print(f"  Ch4 (current): 1f = {ch4_avg[1]*1e3:.3f} mA")
        for h in [2, 3, 4]:
            ratio = ch4_avg[h] / ch4_avg[1] * 100 if ch4_avg[1] > 0 else 0
            print(f"    {h}f/1f = {ratio:.3f}%")
            if h == 2 and ratio > 1.0:
                print(f"    ** WARNING: Ch4 2f/1f = {ratio:.2f}% (>1%)")

    # Check variation across sampled points
    ch1_2f_ratios = [d[2] / d[1] * 100 for d in ch1_harmonics_all if d[1] > 0]
    if len(ch1_2f_ratios) > 1:
        print(f"  Ch1 2f/1f variation across scan: "
              f"{min(ch1_2f_ratios):.3f}% -- {max(ch1_2f_ratios):.3f}%")
else:
    print(f"  Could not load waveforms for harmonic analysis")

# %%
# =============================================================================
# Plot
# =============================================================================

pts = np.arange(n_points)

# Running median
med_win = max(n_points // 100 | 1, 5)


def running_median(arr, win):
    pad = win // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - pad)
        hi = min(len(arr), i + pad + 1)
        chunk = arr[lo:hi]
        out[i] = np.median(chunk[valid[lo:hi]]) if valid[lo:hi].any() else np.nan
    return out


# Layout: drift panels (left column) + harmonics bar (right column)
n_drift = 4 if has_ch4 else 1
n_rows = max(n_drift, 2)
plt.style.use(["science", "ieee"])
fig, axes = plt.subplots(n_rows, 2, figsize=(10, 2.2 * n_rows),
                         gridspec_kw={"width_ratios": [3, 1]})

# --- Left: drift panels ---
drift_specs = [("Voltage [V]", V_1f, "C0")]
if has_ch4:
    drift_specs += [
        ("Current [mA]", I_1f * 1e3, "C1"),
        (r"$|Z|$ [$\Omega$]", Z_mag, "C2"),
        ("V--I phase [deg]", phase_VI, "C3"),
    ]

for row, (ylabel, arr, color) in enumerate(drift_specs):
    ax = axes[row, 0]
    ax.plot(pts[valid], arr[valid], ".", markersize=0.5, color=color, alpha=0.4)
    if n_missed > 0:
        ax.plot(pts[~valid], arr[~valid], "x", markersize=2, color="red",
                alpha=0.6, label=f"{n_missed} missed" if row == 0 else None)
    rm = running_median(arr, med_win)
    ax.plot(pts, rm, "-", linewidth=0.8, color="k", alpha=0.7,
            label=f"median (n={med_win})" if row == 0 else None)
    lo, hi = np.percentile(arr[valid], [0.5, 99.5])
    margin = (hi - lo) * 0.15
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_ylabel(ylabel)
    if row == 0:
        ax.legend(fontsize=5, loc="lower left", frameon=False)

for row in range(len(drift_specs), n_rows):
    axes[row, 0].set_visible(False)

axes[0, 0].set_title("Electrical stability -- " + stem.replace("_", r"\_"))
axes[n_rows - 1, 0].set_xlabel("Scan point index")

# --- Right top: Ch1 harmonics ---
ax_h1 = axes[0, 1]
if ch1_harmonics_all:
    harmonics = [2, 3, 4]
    ratios_ch1 = [ch1_avg[h] / ch1_avg[1] * 100 for h in harmonics]
    bars = ax_h1.bar([f"${h}f$" for h in harmonics], ratios_ch1, color="C0", alpha=0.7)
    ax_h1.set_ylabel(r"Ratio to $1f$ [\%]")
    ax_h1.set_title(r"Ch1 (voltage) THD")
    ax_h1.axhline(1.0, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    for bar, r in zip(bars, ratios_ch1):
        ax_h1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{r:.2f}\\%", ha="center", va="bottom", fontsize=6)
else:
    ax_h1.set_visible(False)

# --- Right bottom: Ch4 harmonics ---
ax_h4 = axes[1, 1]
if ch4_harmonics_all:
    ratios_ch4 = [ch4_avg[h] / ch4_avg[1] * 100 for h in harmonics]
    bars = ax_h4.bar([f"${h}f$" for h in harmonics], ratios_ch4, color="C1", alpha=0.7)
    ax_h4.set_ylabel(r"Ratio to $1f$ [\%]")
    ax_h4.set_title(r"Ch4 (current) THD")
    ax_h4.axhline(1.0, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    for bar, r in zip(bars, ratios_ch4):
        ax_h4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{r:.2f}\\%", ha="center", va="bottom", fontsize=6)
else:
    ax_h4.set_visible(False)

# --- Right row 3: RSSI histogram ---
if n_rows > 2:
    ax_rssi = axes[2, 1]
    if rssi is not None:
        ax_rssi.hist(rssi, bins=50, color="C4", alpha=0.7, edgecolor="none")
        ax_rssi.axvline(RSSI_THRESHOLD, color="red", linewidth=0.8, linestyle="--",
                        label=f"threshold = {RSSI_THRESHOLD}")
        ax_rssi.set_xlabel("RSSI")
        ax_rssi.set_ylabel("Count")
        ax_rssi.set_yscale("log")
        ax_rssi.set_title(f"RSSI distribution ({pct_rssi:.1f}\\% low)"
                          if rssi is not None else "RSSI")
        ax_rssi.legend(fontsize=5, frameon=False)
    else:
        ax_rssi.set_visible(False)

# --- Right row 4: missed burst location ---
if n_rows > 3:
    ax_miss = axes[3, 1]
    if n_missed > 0:
        # Show where missed bursts occur across the scan
        n_bins = 20
        bin_edges = np.linspace(0, n_points, n_bins + 1)
        miss_counts, _ = np.histogram(np.where(~voltage_valid)[0], bins=bin_edges)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_miss.bar(bin_centres, miss_counts, width=bin_edges[1] - bin_edges[0],
                    color="red", alpha=0.6, edgecolor="none")
        ax_miss.set_xlabel("Scan point")
        ax_miss.set_ylabel("Missed count")
        ax_miss.set_title(f"Missed bursts ({n_missed})")
    else:
        ax_miss.text(0.5, 0.5, f"No missed bursts\n({n_points} pts)",
                     ha="center", va="center", transform=ax_miss.transAxes,
                     fontsize=8)
        ax_miss.set_title("Missed bursts")
        ax_miss.set_xticks([])
        ax_miss.set_yticks([])

plt.tight_layout()
out_path = OUT_DIR / f"sanity_{stem}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  File: {tdms_path.name}")
print(f"  Points: {n_points}")
print(f"  Drive: {f_drive/1e6:.6f} MHz")
print(f"  LDV range: {args.ldv_range} m/s")
print(f"  Missed bursts: {n_missed} ({pct_missed:.1f}%)")
if has_burst:
    print(f"  Burst timing outliers: {n_timing_bad} ({pct_timing:.1f}%)")
if rssi is not None:
    print(f"  Low RSSI: {n_low_rssi} ({pct_rssi:.1f}%)")
if ch1_harmonics_all:
    ch1_2f = ch1_avg[2] / ch1_avg[1] * 100
    print(f"  Ch1 2f/1f: {ch1_2f:.3f}%", end="")
    print("  ** HIGH" if ch1_2f > 1.0 else "  OK")
if ch4_harmonics_all:
    ch4_2f = ch4_avg[2] / ch4_avg[1] * 100
    print(f"  Ch4 2f/1f: {ch4_2f:.3f}%", end="")
    print("  ** HIGH" if ch4_2f > 1.0 else "  OK")
print(f"  Plot: {out_path}")
