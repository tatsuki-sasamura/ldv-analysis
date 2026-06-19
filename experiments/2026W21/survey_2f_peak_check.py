# %%
"""Probe the 2f-band peak in the W21 single-Y survey lines.

The 2026-06-13 PRL plan and 2026-06-18 resource inventory both cite
``f_{2f,cavity} ~ 3.845 MHz`` from "survey lines", which the W21 area
scan (``f2_eigenmode_scan.py``) revised to ``3.794 MHz``.  The 3.845
value originally came from W10 Step A
(``reports/2026-03-06_stepA_resonance_shift.md``), where the author
already flagged it as a 1D-sweep-position artefact: the W10 line scan
hit three peaks (3.785, 3.845, 3.885 MHz) corresponding to different
axial orders of the SAME n=2 transverse mode.  3.845 was the dominant
one along that particular y line.

Two competing possibilities for what the W21 surveys say:

  (a) the chip remount between W10 -> W21 shifted the mode (the W21
      area scan finds 3.794 as the lowest peak; the survey should
      track that), OR

  (b) the W21 surveys still see 3.845 too, just at a different y line
      (the 3.785/3.845/3.885 ladder reorganised slightly under the
      remount but the survey-line dominant peak is still near 3.845).

This script settles which by looping the 21x1 (or 101x1) survey .h5
files in 3.7-3.9 MHz, computing FFT caches on the fly (small grid,
fast), and reporting the peak frequency of |pressure_1f| -- both as
the max over the 21 width points (which is the relevant single-point
observable that the survey originally returned) and as the
cos(2pi y/W) projected amplitude (mode-selective).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, LDV_DATA_ROOT, figsize_for_layout, get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import fit_mode  # noqa: E402

DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "survey_2f_peak_check"

DEFAULT_SCAN = "sample_21x1_fsweep_survey_60Vpp_20260530_001834"

F_LO_MHZ = 3.70
F_HI_MHZ = 3.90


def in_band(p: Path, f_lo: float, f_hi: float) -> bool:
    """f<NNNNNNN>.h5  -> True if NNNNNNN is in [f_lo, f_hi] MHz."""
    try:
        f_hz = float(p.stem[1:])
    except ValueError:
        return False
    return f_lo * 1e6 <= f_hz <= f_hi * 1e6


def main(scan_dir_name: str, f_lo: float = F_LO_MHZ,
         f_hi: float = F_HI_MHZ) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = DATA_ROOT / scan_dir_name
    cache_dir = get_cache_dir(scan_dir_name, __file__)
    files = sorted(p for p in run_dir.glob("f*.h5") if in_band(p, f_lo, f_hi))
    if not files:
        raise FileNotFoundError(
            f"No band files in {run_dir} for {f_lo}-{f_hi} MHz")
    print(f"Scan: {scan_dir_name}")
    print(f"Band: {f_lo}-{f_hi} MHz, {len(files)} files  "
          f"({files[0].name} ... {files[-1].name})")

    f_list: list[float] = []
    pmax_list: list[float] = []     # max |P_1f| across width points (raw)
    pcos_list: list[float] = []     # cos(2 pi y/W) projection amplitude
    r2cos_list: list[float] = []

    for p in files:
        with h5py.File(p, "r") as h:
            f_nom = float(h.attrs["drive_frequency_hz_nominal"])
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        nv = int(np.sum(valid))
        if nv < 5:
            print(f"  {p.name}: only {nv} valid pts, skip")
            continue
        P1_abs = np.asarray(c["pressure_1f"])
        P1_ph = np.asarray(c["phase_1f"])
        P1c = P1_abs * np.exp(1j * np.radians(P1_ph))
        pos_x = np.asarray(c["pos_x"])  # width direction (m)

        pmax = float(np.max(P1_abs[valid]))
        # mode-selective n=2 projection only -- the n=1 (sin) projection at
        # 3.8 MHz is a fit-pipeline artefact (sigma-clip iteration + an
        # independent center search make it nonzero even on a pure cos2
        # field; see sin_vs_cos_proof2.py for the decomposition).  No real
        # n=1 transverse mode exists at 3.8 MHz; f_1(n=1) = c/(2W) ~ 1.9 MHz.
        r2 = fit_mode(pos_x[valid], P1c[valid], CHANNEL_WIDTH, harmonic=2)
        pcos = float(abs(r2.p0))
        f_list.append(f_nom)
        pmax_list.append(pmax)
        pcos_list.append(pcos)
        r2cos_list.append(float(r2.r2))
        print(f"  {f_nom/1e6:.4f} MHz  pmax={pmax/1e3:6.1f} kPa  "
              f"|cos2|={pcos/1e3:6.1f} (R^2={r2.r2:+.2f})  "
              f"valid={nv}")

    order = np.argsort(f_list)
    f = np.asarray(f_list)[order] / 1e6           # MHz
    pmax = np.asarray(pmax_list)[order]            # Pa
    pcos = np.asarray(pcos_list)[order]
    r2cos = np.asarray(r2cos_list)[order]

    # peak frequencies under each observable
    def peak(f_arr, y_arr):
        k = int(np.argmax(y_arr))
        return float(f_arr[k]), float(y_arr[k])

    f_pmax, v_pmax = peak(f, pmax)
    f_pcos, v_pcos = peak(f, pcos)

    print(f"\nPeak summary:")
    print(f"  max |P_1f| across width points : {f_pmax:.4f} MHz  "
          f"({v_pmax/1e3:.1f} kPa)")
    print(f"  |cos(2 pi y/W)| projection      : {f_pcos:.4f} MHz  "
          f"({v_pcos/1e3:.1f} kPa)")
    print(f"\nCompare against:")
    print(f"  W21 area scan (fine, 2 kHz)     : 3.7942 MHz  (Lorentzian fit)")
    print(f"  W10 Step A 1D sweep (legacy)    : 3.845 MHz   "
          f"(known 1D-position artefact)")

    # ---- CSV --------------------------------------------------------------
    csv_path = OUT_DIR / f"survey_2f_peak_{scan_dir_name}.csv"
    rows = ["f_MHz,pmax_kPa,p_cos2_kPa,r2_cos2"]
    for i in range(len(f)):
        rows.append(
            f"{f[i]:.4f},{pmax[i]/1e3:.3f},{pcos[i]/1e3:.3f},{r2cos[i]:.4f}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True
    )

    axes[0].plot(f, pmax / 1e3, "o-", markersize=4, linewidth=0.7,
                 color="0.3", label="max |P| across width pts (raw)")
    axes[0].plot(f, pcos / 1e3, "s-", markersize=4, linewidth=0.7,
                 color="C0", label=r"$|\cos(2\pi y/W)|$ projection ($n=2$)")
    axes[0].axvline(3.7942, color="C2", lw=0.5, ls="--",
                    label="area-scan f$_{2f}$ = 3.7942 MHz")
    axes[0].axvline(3.845, color="C3", lw=0.5, ls=":",
                    label="W10 legacy 3.845 MHz")
    axes[0].set_ylabel(r"$P_{1f}$ [kPa]")
    axes[0].set_title(
        f"W21 single-Y survey, 2f-band peak check  ({scan_dir_name})"
    )
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f, r2cos, "s-", markersize=4, linewidth=0.7, color="C0",
                 label=r"$R^2(\cos 2\pi y/W)$")
    axes[1].axhline(0.9, color="0.5", lw=0.4, ls="--")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_xlabel(r"drive frequency [MHz]")
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"survey_2f_peak_{scan_dir_name}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--scan-dir", default=DEFAULT_SCAN,
                        help=f"survey scan dir under DATA_ROOT/output/W21 "
                             f"(default: {DEFAULT_SCAN})")
    parser.add_argument("--f-lo", type=float, default=F_LO_MHZ,
                        help=f"low edge in MHz (default: {F_LO_MHZ})")
    parser.add_argument("--f-hi", type=float, default=F_HI_MHZ,
                        help=f"high edge in MHz (default: {F_HI_MHZ})")
    args = parser.parse_args()
    main(args.scan_dir, args.f_lo, args.f_hi)
