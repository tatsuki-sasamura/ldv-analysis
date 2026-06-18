# %%
"""B1 refinement: bound the drive-leakage P_2f via the measured 2f-band transfer.

The basic B1 (``drive_purity_b1.py``) shows that the amplifier leaks
V_2f onto the PZT at ~0.14-1.2 % of V_1f.  The previous leakage estimate
multiplied that ratio by the *1f* acoustic gain to bound the leakage
contribution to the observed P_2f, which assumes the chip+water transfer
at 2f is the same as at 1f.  It isn't.

This script measures that 2f-band transfer **directly** from the
``sample_101x77_fsweep_3p7to3p9_60Vpp_*`` scan, where the chip is driven
at frequencies 3.70-3.90 MHz at 60 Vpp.  At each f the cached
``pressure_1f`` IS the acoustic response at the drive freq; we project
it onto the ``cos(2 pi x / W)`` cavity mode (the natural transverse
shape in this band) via ``AxialFit.p1_n2_mag``.  Dividing by V_drive at
PZT gives:

    T(f) = |P_cavity_cos|(f) / V_drive_PZT(f)     [Pa / V]

Then for each cascade Vpp the predicted leakage acoustic P_2f is

    P_2f,leak(Vpp) = T(2 f_drive_peak(Vpp)) * V_2f_PZT(Vpp)

with V_2f_PZT from ``drive_purity_b1.csv``.  Since the leakage adds to
the cascade-generated P_2f as a complex phasor, the *magnitude* share is
a worst-case bound.

Output: ``output/drive_purity_b1/drive_purity_b1_transfer.{png,csv}``
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH,
    FIG_DPI,
    LDV_DATA_ROOT,
    figsize_for_layout,
    get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.sweep_fit import fit_axial  # noqa: E402

# -- inputs ------------------------------------------------------------------
TRANSFER_DIR = "sample_101x77_fsweep_3p7to3p9_60Vpp_20260530_013206"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"

OUT_DIR = Path(__file__).resolve().parent / "output" / "drive_purity_b1"
B1_CSV = OUT_DIR / "drive_purity_b1.csv"  # produced by drive_purity_b1.py

# Cascade peak-fundamental frequencies (from vpp_vs_pressure.py latest run).
# These set the 2f frequency at which we sample T(f).  Hardcoded here so
# the script doesn't need to re-run sweep_peaks on every cascade dir; the
# 11-pt freq sweep gives 2 kHz resolution and the peak drifts by ~6 kHz
# across the cascade -- much smaller than the 31 kHz Lorentzian-tail
# detuning, so a constant 1.906 MHz also works to ~3% accuracy.
CASCADE_PEAK_F1_MHZ = {
    10: 1.908, 20: 1.908, 30: 1.908, 40: 1.906, 50: 1.906, 60: 1.906,
    70: 1.906, 80: 1.906, 90: 1.904, 100: 1.904, 110: 1.902, 120: 1.902,
}


def build_transfer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T(f) at the 3p7to3p9 sweep frequencies.

    Returns
    -------
    f_drive_mhz : ndarray of float, sorted
    V_drive_pzt_V : ndarray of float, median over scan points (V)
    P_cos2_pa : ndarray of float, |cos(2pi x/W)| fit of pressure_1f (Pa)
    """
    run_dir = DATA_ROOT / TRANSFER_DIR
    cache_dir = get_cache_dir(TRANSFER_DIR, __file__)
    files = sorted(run_dir.glob("f*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files in {run_dir}")

    f_list: list[float] = []
    v_list: list[float] = []
    p_list: list[float] = []
    geom = None

    print(f"\n[transfer] {TRANSFER_DIR}  ({len(files)} files)")
    for p in files:
        with h5py.File(p, "r") as h:
            f_nom = float(h.attrs["drive_frequency_hz_nominal"])
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if int(np.sum(valid)) < 3:
            print(f"  {p.name}: only {int(np.sum(valid))} valid pts, skip")
            continue
        fit, geom = fit_axial(c, valid, CHANNEL_WIDTH, geom=geom)
        v_pzt = float(np.median(V[valid]))
        # The 2f-band cavity mode is cos(2pi x/W) -> use p1_n2_mag.
        p_cos2 = float(fit.p1_n2_mag)
        print(f"  f = {f_nom/1e6:.3f} MHz  V_PZT = {v_pzt:5.1f} V  "
              f"|P_cos2| = {p_cos2/1e3:7.1f} kPa  R^2(cos2) = {fit.r2_p1_n2:.2f}")
        f_list.append(f_nom)
        v_list.append(v_pzt)
        p_list.append(p_cos2)

    order = np.argsort(f_list)
    return (np.asarray(f_list)[order] / 1e6,
            np.asarray(v_list)[order],
            np.asarray(p_list)[order])


def load_b1_csv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read PZT_Vpp, V_1f_V, V_2f_V columns from drive_purity_b1.csv."""
    if not B1_CSV.exists():
        raise FileNotFoundError(
            f"{B1_CSV} not found -- run drive_purity_b1.py first.")
    vpp: list[float] = []
    v1: list[float] = []
    v2: list[float] = []
    with B1_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vpp.append(float(row["PZT_Vpp"]))
            v1.append(float(row["V_1f_V"]))
            v2.append(float(row["V_2f_V"]))
    return np.asarray(vpp), np.asarray(v1), np.asarray(v2)


def load_cascade_observed_p2f() -> dict[int, float]:
    """Pull observed P_2f,peak (kPa) per Vpp from the cascade scans.

    Uses sweep_peaks like vpp_vs_pressure.py so the numbers match.
    Hardcoded to the 12-point canonical list.
    """
    from ldv_analysis.sweep_fit import sweep_peaks
    scans = [
        (10, "sample_101x21_fsweep_peak_10Vpp_20260524_182813"),
        (20, "sample_101x21_fsweep_peak_20Vpp_20260524_203544"),
        (30, "sample_101x21_fsweep_peak_30Vpp_20260524_210338"),
        (40, "sample_101x21_fsweep_peak_40Vpp_20260524_213135"),
        (50, "sample_101x21_fsweep_peak_50Vpp_20260524_215928"),
        (60, "sample_101x21_fsweep_peak_60Vpp_20260524_223919"),
        (70, "sample_101x21_fsweep_peak_70Vpp_20260524_233721"),
        (80, "sample_101x21_fsweep_peak_80Vpp_20260525_000520"),
        (90, "sample_101x21_fsweep_peak_90Vpp_20260525_003315"),
        (100, "sample_101x21_fsweep_peak_100Vpp_20260525_010113"),
        (110, "sample_101x21_fsweep_peak_110Vpp_20260525_012912"),
        (120, "sample_101x21_fsweep_peak_120Vpp_20260525_020136"),
    ]
    out: dict[int, float] = {}
    print(f"\n[cascade observed P_2f, kPa]")
    for label, dirname in scans:
        cache_dir = get_cache_dir(dirname, __file__)
        sp = sweep_peaks(DATA_ROOT / dirname, CHANNEL_WIDTH, cache_dir)
        out[label] = sp.peak_p2_kpa
        print(f"  {label:>3} V: peak P_2f = {sp.peak_p2_kpa:6.1f} kPa "
              f"@ {sp.peak_p2_freq_mhz:.3f} MHz")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. T(f) at 2f-band drives
    f_t_mhz, v_t, p_t = build_transfer()
    # transfer in Pa / V_drive_PZT
    T_pa_per_v = p_t / v_t
    print(f"\nT(f) at 60 V_PZT drive in the 2f band:")
    for f, v, p, t in zip(f_t_mhz, v_t, p_t, T_pa_per_v):
        print(f"  {f:.3f} MHz  V = {v:5.1f} V  P = {p/1e3:7.1f} kPa  "
              f"T = {t/1e3:6.2f} kPa/V")

    # 2. B1 V_2f per Vpp
    vpp_b1, v1f_b1, v2f_b1 = load_b1_csv()

    # 3. observed P_2f per Vpp from cascade
    p2f_obs = load_cascade_observed_p2f()
    vpp = np.asarray(sorted(p2f_obs.keys()), dtype=float)
    p2f_obs_arr = np.asarray([p2f_obs[int(v)] for v in vpp])  # kPa

    # interpolate V_2f and v1f_b1 onto vpp grid (already 1:1, but be safe)
    v1f_arr = np.interp(vpp, vpp_b1, v1f_b1)
    v2f_arr = np.interp(vpp, vpp_b1, v2f_b1)

    # 2 f_drive per Vpp -- T sampled here
    f_2f_mhz = np.asarray([2.0 * CASCADE_PEAK_F1_MHZ[int(v)] for v in vpp])
    T_at_2f = np.interp(f_2f_mhz, f_t_mhz, T_pa_per_v)
    p2f_leak_pa = T_at_2f * v2f_arr  # Pa (worst-case magnitude)
    p2f_leak_kpa = p2f_leak_pa / 1e3
    leak_share = p2f_leak_kpa / p2f_obs_arr  # dimensionless

    # ---- console -----------------------------------------------------------
    print(
        f"\n{'PZT Vpp':>7}  {'V_1f (V)':>9}  {'V_2f (V)':>9}  "
        f"{'2f_drv MHz':>10}  {'T (kPa/V)':>10}  "
        f"{'P_2f,leak (kPa)':>15}  {'P_2f,obs (kPa)':>14}  "
        f"{'leak/obs':>9}"
    )
    for i, v in enumerate(vpp):
        print(
            f"{v:>7.0f}  {v1f_arr[i]:>9.3f}  {v2f_arr[i]:>9.4f}  "
            f"{f_2f_mhz[i]:>10.3f}  {T_at_2f[i]/1e3:>10.2f}  "
            f"{p2f_leak_kpa[i]:>15.2f}  {p2f_obs_arr[i]:>14.1f}  "
            f"{leak_share[i]*100:>8.2f}%"
        )

    # ---- CSV ---------------------------------------------------------------
    csv_path = OUT_DIR / "drive_purity_b1_transfer.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "PZT_Vpp,V_1f_V,V_2f_V,f_2f_drive_MHz,T_Pa_per_V,"
            "P_2f_leak_kPa,P_2f_obs_kPa,leak_share\n"
        )
        for i, v in enumerate(vpp):
            fh.write(
                f"{v:.0f},{v1f_arr[i]:.4f},{v2f_arr[i]:.6g},"
                f"{f_2f_mhz[i]:.4f},{T_at_2f[i]:.4g},"
                f"{p2f_leak_kpa[i]:.4g},{p2f_obs_arr[i]:.3f},"
                f"{leak_share[i]:.4g}\n"
            )
    print(f"\nSaved {csv_path}")

    # ---- plots -------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=figsize_for_layout(3, 1, sharex=False), sharex=False
    )

    # (a) transfer curve T(f) over 2f band
    axes[0].plot(f_t_mhz, T_pa_per_v / 1e3, "o-", markersize=4,
                 linewidth=0.8, color="C0", label="T(f) measured")
    for i, v in enumerate(vpp):
        axes[0].axvline(f_2f_mhz[i], color="0.7", lw=0.3, ls=":")
    axes[0].plot(f_2f_mhz, T_at_2f / 1e3, "x", color="C3", markersize=4,
                 label="cascade 2$f_\\mathrm{drive}$ sample pts")
    axes[0].set_xlabel(r"drive frequency [MHz]")
    axes[0].set_ylabel(r"$T(f) = |P_\mathrm{cos2}|/V_\mathrm{PZT}$  [kPa/V]")
    axes[0].set_title(
        f"Linear electroacoustic transfer in the 2f band  "
        f"({TRANSFER_DIR.split('_')[-2]} 60 Vpp drive)"
    )
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(True, alpha=0.3)

    # (b) leakage vs observed P_2f, per Vpp
    axes[1].plot(vpp, p2f_obs_arr, "o-", markersize=4, linewidth=0.8,
                 color="C0", label=r"$P_{2f}$ observed (cascade)")
    axes[1].plot(vpp, p2f_leak_kpa, "s-", markersize=4, linewidth=0.8,
                 color="C3", label=r"$P_{2f,\mathrm{leak}}=T(2f)\,V_{2f}$  "
                                    r"(worst case)")
    axes[1].set_xlabel(r"True PZT drive [Vpp]")
    axes[1].set_ylabel(r"$P_{2f}$ [kPa]")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    axes[1].grid(True, which="both", alpha=0.3)

    # (c) leakage share
    axes[2].plot(vpp, leak_share * 100, "o-", markersize=4, linewidth=0.8,
                 color="C5")
    axes[2].axhline(10, color="0.5", lw=0.5, ls="--",
                    label="10\\% (perturbation-breakdown deviation)")
    axes[2].set_xlabel(r"True PZT drive [Vpp]")
    axes[2].set_ylabel(r"$P_{2f,\mathrm{leak}}/P_{2f,\mathrm{obs}}$  [\%]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, frameon=False)

    plt.tight_layout()
    out_path = OUT_DIR / "drive_purity_b1_transfer.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
