# %%
"""Quick check: Ch1/Ch4 stability across the scan to confirm no thermal drift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DPI, figsize_for_layout
from fft_cache import load_or_compute

# %%
DEFAULT_TDMS = Path("G:/My Drive/20260303experimentA/stepA1967.tdms")
OUT_DIR = Path(__file__).parent.parent.parent / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem
print(f"Loading: {tdms_path.name}")

cache = load_or_compute(tdms_path, OUT_DIR)
n_points = len(cache["pos_x"])

V_1f = cache["voltage_1f"]
has_ch4 = "current_1f" in cache
if has_ch4:
    I_1f = cache["current_1f"]
    Z_mag = cache["impedance_1f"]
    phase_VI = cache["phase_vi"]

# %%
# =============================================================================
# Flag invalid points (missed bursts: V_1f << median)
# =============================================================================

V_med = np.median(V_1f)
valid = V_1f > V_med * 0.5
n_bad = np.sum(~valid)
print(f"  Valid points: {np.sum(valid)} / {n_points} "
      f"({n_bad} missed bursts, {n_bad/n_points*100:.1f}%)")

# %%
# Stats (valid points only)
print(f"\nCh1 voltage:  median={np.median(V_1f[valid]):.3f} V,  "
      f"std={V_1f[valid].std():.4f} V  "
      f"({V_1f[valid].std()/np.median(V_1f[valid])*100:.2f}%)")
if has_ch4:
    print(f"Ch4 current:  median={np.median(I_1f[valid])*1e3:.2f} mA,  "
          f"std={I_1f[valid].std()*1e3:.3f} mA  "
          f"({I_1f[valid].std()/np.median(I_1f[valid])*100:.2f}%)")
    print(f"|Z|:          median={np.median(Z_mag[valid]):.1f} ohm,  "
          f"std={Z_mag[valid].std():.2f} ohm  "
          f"({Z_mag[valid].std()/np.median(Z_mag[valid])*100:.2f}%)")
    print(f"V-I phase:    median={np.median(phase_VI[valid]):.2f} deg,  "
          f"std={phase_VI[valid].std():.3f} deg")

# Drift: first/last 50 valid points
n_check = 50
valid_idx = np.where(valid)[0]
early_idx = valid_idx[:n_check]
late_idx = valid_idx[-n_check:]
print(f"\nDrift (first {n_check} vs last {n_check} valid points):")
drift_items = [("V", V_1f, "V")]
if has_ch4:
    drift_items += [("I", I_1f * 1e3, "mA"),
                    ("|Z|", Z_mag, "ohm"), ("phase", phase_VI, "deg")]
for name, arr, unit in drift_items:
    early = arr[early_idx].mean()
    late = arr[late_idx].mean()
    shift = late - early
    if name == "phase":
        print(f"  {name:>6}: {early:.4f} -> {late:.4f} {unit}  (shift = {shift:+.4f} deg)")
    else:
        print(f"  {name:>6}: {early:.4f} -> {late:.4f} {unit}  "
              f"(shift = {shift:+.4f}, {shift/early*100:+.2f}%)")

# %%
# Running median (window = 1% of points, at least 5)
med_win = max(n_points // 100 | 1, 5)  # ensure odd


def running_median(arr, win):
    pad = win // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - pad)
        hi = min(len(arr), i + pad + 1)
        chunk = arr[lo:hi]
        out[i] = np.median(chunk[valid[lo:hi]])  # skip invalid in window
    return out


# %%
# Plot
pts = np.arange(n_points)
n_rows = 4 if has_ch4 else 1
fig, axes = plt.subplots(
    n_rows, 1, figsize=figsize_for_layout(n_rows, 1, sharex=True),
    sharex=True, squeeze=False)
axes = axes[:, 0]  # flatten to 1-D

plot_specs = [("Voltage (V)", V_1f, 1.0, "C0")]
if has_ch4:
    plot_specs += [
        ("Current (mA)", I_1f * 1e3, 1.0, "C1"),
        (r"$|Z|$ ($\Omega$)", Z_mag, 1.0, "C2"),
        ("V--I phase (deg)", phase_VI, 1.0, "C3"),
    ]

for row, (ylabel, arr, _, color) in enumerate(plot_specs):
    ax = axes[row]
    # Valid points
    ax.plot(pts[valid], arr[valid], ".", markersize=0.5, color=color, alpha=0.4)
    # Invalid points
    if n_bad > 0:
        ax.plot(pts[~valid], arr[~valid], "x", markersize=2, color="red",
                alpha=0.6, label=f"{n_bad} missed" if row == 0 else None)
    # Running median
    rm = running_median(arr, med_win)
    ax.plot(pts, rm, "-", linewidth=0.8, color="k", alpha=0.7,
            label=f"median (n={med_win})" if row == 0 else None)
    # Y-limits from valid data
    lo, hi = np.percentile(arr[valid], [0.5, 99.5])
    margin = (hi - lo) * 0.15
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[0].set_title(f"Electrical stability --- {stem}")
axes[0].legend(fontsize=5, loc="lower left")
axes[-1].set_xlabel("Scan point index")

plt.tight_layout()
output_path = OUT_DIR / f"thermal_drift_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")
