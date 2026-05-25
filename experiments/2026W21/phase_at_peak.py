# %%
"""Phase of P_1f (and v_LDV) at the per-scan P_1f-peak frequency.

For each scan directory passed on the CLI, walks the cached FFT files
(``_fft_cache_*.npz`` in the analysis output's ``fft_cache/``
subfolder), fits ``|sin(pi y/W)|`` per frequency, finds the discrete
argmax of |P_1f|, and reports:

    peak f, P_1f (MPa), <v_LDV - <V_drive (deg), <P_1f - <V_drive (deg)

The phase shown in ``freq_vs_current.py``'s panel labelled
``angle P_{1f}`` is actually ``angle(v_LDV) - angle(V_drive)`` because
``fft_cache.phase[h]`` stores the LDV-velocity phase relative to the
drive-voltage phase (the ``-1`` of refracto-vibrometry is only applied
to the magnitude via ``abs(velocity_to_pressure)``).  To get the true
pressure-vs-drive phase, add 180 deg.  Both columns are reported here
so the script's panel value and the physical pressure phase can be
cross-checked.

Peak detection is a simple discrete argmax of fitted |P_1f| across the
cached frequency points -- no curve fit, no interpolation.

Usage::

    python experiments/2026W21/phase_at_peak.py <scan_dir> [<scan_dir> ...]
    python experiments/2026W21/phase_at_peak.py            # default: narrow remeasurement series
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import CHANNEL_WIDTH  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import fit_mode_1f  # noqa: E402

OUTPUT_ROOT = ROOT / "experiments" / "2026W21" / "output"

# Narrow-band voltage-sweep remeasurement series (default)
DEFAULT_SCAN_DIRS = [
    "sample_101x1_fsweep_narrow_10Vpp_20260518_213357",
    "sample_101x1_fsweep_narrow_20Vpp_20260518_212516",
    "sample_101x1_fsweep_narrow_30Vpp_20260518_211632",
    "sample_101x1_fsweep_narrow_40Vpp_20260518_210751",
    "sample_101x1_fsweep_narrow_50Vpp_20260518_205751",
    "sample_101x1_fsweep_narrow_60Vpp_20260518_204855",
]


def analyse(scan_dir: Path) -> dict | None:
    cache_dir = scan_dir / "fft_cache"
    if not cache_dir.exists():
        print(f"[skip] no fft_cache at {scan_dir}")
        return None
    freqs: list[float] = []
    p_arr: list[float] = []
    ph_arr: list[float] = []
    for cache_path in sorted(cache_dir.glob("_fft_cache_*.npz")):
        c = np.load(cache_path)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if int(valid.sum()) < 3:
            continue
        pos = np.asarray(c["pos_x"])
        P1abs = np.asarray(c["pressure_1f"])
        P1ph = np.asarray(c["phase_1f"])
        P1c = P1abs * np.exp(1j * np.radians(P1ph))
        r1 = fit_mode_1f(pos[valid], P1c[valid], CHANNEL_WIDTH)
        freqs.append(float(c["f_drive"]))
        p_arr.append(abs(r1.p0))
        ph_arr.append(float(np.degrees(np.angle(r1.p0))))

    if not freqs:
        print(f"[skip] no cached frequencies in {scan_dir}")
        return None

    freqs_a = np.asarray(freqs)
    p_a = np.asarray(p_arr)
    ph_a = np.asarray(ph_arr)
    i = int(np.argmax(p_a))
    ph_v_ldv = float(ph_a[i])
    ph_p = (ph_v_ldv + 180.0) % 360.0
    if ph_p > 180.0:
        ph_p -= 360.0
    return dict(
        name=scan_dir.name,
        n_files=len(freqs_a),
        f_min=freqs_a.min(),
        f_max=freqs_a.max(),
        peak_f=freqs_a[i],
        peak_P1=p_a[i],
        ph_v_ldv=ph_v_ldv,
        ph_P1=ph_p,
    )


def resolve(arg: Path | str) -> Path:
    """Accept either an absolute scan dir or a name relative to OUTPUT_ROOT."""
    p = Path(arg)
    return p if p.is_absolute() else (OUTPUT_ROOT / p)


def main(scan_dirs: list[Path]) -> None:
    print(f"{'label / name':<60}  {'N':>3}  {'f range (MHz)':>16}  "
          f"{'peak f (MHz)':>13}  {'|P_1f| peak (MPa)':>17}  "
          f"{'<v_LDV (deg)':>13}  {'<P_1f (deg)':>12}")
    for sd in scan_dirs:
        sd = resolve(sd)
        if not sd.exists():
            print(f"[skip] {sd} not found")
            continue
        r = analyse(sd)
        if r is None:
            continue
        print(f"{r['name']:<60}  {r['n_files']:>3}  "
              f"{r['f_min']/1e6:7.4f}-{r['f_max']/1e6:7.4f}  "
              f"{r['peak_f']/1e6:>13.4f}  "
              f"{r['peak_P1']/1e6:>17.3f}  "
              f"{r['ph_v_ldv']:>13.1f}  "
              f"{r['ph_P1']:>12.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_dirs", nargs="*", default=DEFAULT_SCAN_DIRS,
        help=("One or more scan directory names (relative to "
              "experiments/2026W21/output/) or absolute paths.  "
              "Default: the narrow-band 10..60 Vpp remeasurement series."),
    )
    args = parser.parse_args()
    main([Path(s) for s in args.scan_dirs])
