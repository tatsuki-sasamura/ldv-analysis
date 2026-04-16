# %%
"""Diagnose LDV focus sensitivity from a z-sweep at fixed drive.

Loads a series of TDMS files acquired at different LDV head z positions
(same drive voltage, frequency, and xy scan area) and compares electrical
quantities and apparent pressure.

The electrical channels (V, I, Z, V-I phase) should be independent of
LDV position — drift in these indicates real temporal effects, not
optical ones.

The acoustic pressure (from refracto-vibrometry) is sensitive to z via
optical interference between reflections from glass surfaces and the
reflector.  Non-monotonic behaviour with a minimum at focus is a
signature of multi-surface interference, not simple beam-size averaging.

Usage:
    python ldv_z_sweep.py <tdms_1> <z_1> [<tdms_2> <z_2> ...]
    python ldv_z_sweep.py --data-dir <dir> --pattern "...z{z}.tdms" \\
        --z-values 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6

Example:
    python ldv_z_sweep.py \\
        --data-dir "G:/My Drive/260416ldv" \\
        --pattern "W16test5_30Vpp1900kHz_5m_s_maxz{z}.tdms" \\
        --z-values 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 \\
        --label test5
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

from ldv_analysis.config import FIG_DPI, figsize_for_layout, get_output_dir
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pairs", nargs="*", default=[],
                    help="Alternating: tdms_path z_value tdms_path z_value ...")
parser.add_argument("--data-dir", type=Path,
                    help="Directory containing TDMS files")
parser.add_argument("--pattern", type=str,
                    help="Filename pattern with {z} placeholder")
parser.add_argument("--z-values", type=str, nargs="+",
                    help="z values (as strings, e.g. 0.0 0.2 ...)")
parser.add_argument("--label", type=str, default="z_sweep",
                    help="Label for output filename")
args = parser.parse_args()

# Build file list
files = []  # list of (Path, z as float, z as str)
if args.data_dir:
    if not args.pattern or not args.z_values:
        parser.error("--data-dir requires --pattern and --z-values")
    for z_str in args.z_values:
        fname = args.pattern.replace("{z}", z_str)
        files.append((args.data_dir / fname, float(z_str), z_str))
else:
    pairs = args.pairs
    if len(pairs) % 2 != 0 or len(pairs) == 0:
        parser.error("Positional args must be pairs: tdms_path z tdms_path z ...")
    for i in range(0, len(pairs), 2):
        files.append((Path(pairs[i]), float(pairs[i + 1]), pairs[i + 1]))

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Load cached quantities at each z
# =============================================================================

z_arr = []
V_arr = []
I_arr = []
Z_arr = []
phi_arr = []
p1f_arr = []
p2f_arr = []
p3f_arr = []
rssi_arr = []
f_drive_arr = []

for tdms_path, z_val, z_str in files:
    if not tdms_path.exists():
        print(f"  SKIP (not found): {tdms_path.name}")
        continue

    c = load_or_compute(tdms_path, CACHE_DIR)
    z_arr.append(z_val)
    V_arr.append(float(np.median(c["voltage_1f"])))
    if "current_1f" in c:
        I_arr.append(float(np.median(c["current_1f"]) * 1e3))
        Z_arr.append(float(np.median(c["impedance_1f"])))
        phi_arr.append(float(np.median(c["phase_vi"])))
    else:
        I_arr.append(np.nan)
        Z_arr.append(np.nan)
        phi_arr.append(np.nan)
    p1f_arr.append(float(np.median(c["pressure_1f"])) / 1e3)
    p2f_arr.append(float(np.median(c["pressure_2f"])) / 1e3 if "pressure_2f" in c else np.nan)
    p3f_arr.append(float(np.median(c["pressure_3f"])) / 1e3 if "pressure_3f" in c else np.nan)
    rssi_arr.append(float(np.median(c["rssi"])) if "rssi" in c else np.nan)
    f_drive_arr.append(float(c["f_drive"]))

    print(f"  z={z_str:>5} mm: V={V_arr[-1]:.3f} V, I={I_arr[-1]:.2f} mA, "
          f"|Z|={Z_arr[-1]:.1f} ohm, phi={phi_arr[-1]:+.2f} deg, "
          f"p1f={p1f_arr[-1]:.0f} kPa, RSSI={rssi_arr[-1]:.3f} V")

z = np.array(z_arr)
V = np.array(V_arr)
I = np.array(I_arr)
Zmag = np.array(Z_arr)
phi = np.array(phi_arr)
p1f = np.array(p1f_arr)
p2f = np.array(p2f_arr)
p3f = np.array(p3f_arr)
rssi = np.array(rssi_arr)

# Statistics
def rel_spread(arr):
    return (np.nanmax(arr) - np.nanmin(arr)) / np.nanmedian(arr) * 100

print(f"\nDrive frequency: {np.mean(f_drive_arr)/1e6:.6f} MHz")
print(f"\nRelative spread across z:")
print(f"  V_1f:     {rel_spread(V):.2f}%")
print(f"  I_1f:     {rel_spread(I):.2f}%")
print(f"  |Z|:      {rel_spread(Zmag):.2f}%")
print(f"  phi_VI:   {np.nanmax(phi) - np.nanmin(phi):.2f} deg")
print(f"  p_1f:     {rel_spread(p1f):.2f}%")
print(f"  p_2f:     {rel_spread(p2f):.2f}%")
print(f"  RSSI:     {rel_spread(rssi):.2f}%")

# %%
# =============================================================================
# Plot: 3 rows of panels — electrical, pressure, RSSI
# =============================================================================

plt.style.use(["science", "ieee"])
fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.0), sharex=True)

# Row 1: V, I
ax = axes[0, 0]
ax.plot(z, V, "o-", color="tab:blue", markersize=4, label=r"$V_{1f}$ [V]")
ax.set_ylabel(r"$V_{1f}$ [V]")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(z, I, "s-", color="tab:red", markersize=4, label=r"$I_{1f}$ [mA]")
ax.set_ylabel(r"$I_{1f}$ [mA]")
ax.grid(True, alpha=0.3)

# Row 2: |Z|, phi_VI
ax = axes[1, 0]
ax.plot(z, Zmag, "^-", color="tab:green", markersize=4)
ax.set_ylabel(r"$|Z|$ [$\Omega$]")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(z, phi, "d-", color="tab:orange", markersize=4)
ax.set_ylabel(r"V--I phase [deg]")
ax.grid(True, alpha=0.3)

# Row 3: pressure, RSSI
ax = axes[2, 0]
ax.plot(z, p1f, "o-", color="tab:blue", markersize=4, label=r"$p_{1f}$")
ax.plot(z, p2f, "s-", color="tab:red", markersize=4, label=r"$p_{2f}$")
ax.plot(z, p3f, "^-", color="tab:green", markersize=4, label=r"$p_{3f}$")
ax.set_ylabel(r"Pressure [kPa]")
ax.set_xlabel(r"LDV head z offset [mm]")
ax.set_yscale("log")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3, which="both")

ax = axes[2, 1]
ax.plot(z, rssi, "v-", color="tab:purple", markersize=4)
ax.set_ylabel(r"RSSI [V]")
ax.set_xlabel(r"LDV head z offset [mm]")
ax.grid(True, alpha=0.3)

fig.suptitle(f"LDV z-sweep --- {args.label}", fontsize=9)
plt.tight_layout()
out_path = OUT_DIR / f"ldv_z_sweep_{args.label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Normalised plot: everything as fractional deviation from median
# =============================================================================

fig, ax = plt.subplots(figsize=figsize_for_layout())
med_V = np.nanmedian(V)
med_I = np.nanmedian(I)
med_Z = np.nanmedian(Zmag)
med_p1f = np.nanmedian(p1f)
med_rssi = np.nanmedian(rssi)

ax.plot(z, (V / med_V - 1) * 100, "o-", markersize=3, label=r"$V_{1f}$", alpha=0.7)
ax.plot(z, (I / med_I - 1) * 100, "s-", markersize=3, label=r"$I_{1f}$", alpha=0.7)
ax.plot(z, (Zmag / med_Z - 1) * 100, "^-", markersize=3, label=r"$|Z|$", alpha=0.7)
ax.plot(z, (p1f / med_p1f - 1) * 100, "D-", markersize=4, label=r"$p_{1f}$", linewidth=1.5)
ax.plot(z, (rssi / med_rssi - 1) * 100, "v-", markersize=3, label=r"RSSI", alpha=0.7)
ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
ax.set_xlabel(r"LDV head z offset [mm]")
ax.set_ylabel(r"Relative deviation from median [\%]")
ax.legend(fontsize=6, frameon=False, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = OUT_DIR / f"ldv_z_sweep_normalised_{args.label}.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path}")

print("\n=== Done ===")
