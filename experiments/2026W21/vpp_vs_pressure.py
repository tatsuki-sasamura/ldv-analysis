# %%
"""Voltage cascade: true PZT Vpp vs peak P_1f and P_2f for the sample chip.

Peak pressures are auto-computed per scan directory via the shared
per-axial mode fit (:mod:`ldv_analysis.sweep_fit`) -- the same method as
``freq_vs_current.py`` and ``pressure_map_2d.py``.  Adding a voltage
point is just adding its scan directory to ``SCANS``; no pasted numbers.

Clean 2026-05-24 peak-band series: 101x21 area scans, 1.890-1.910 MHz at
2 kHz steps, water-filled, BNC splitter removed.  The X axis is the true
PZT drive voltage, AFG_Vpp (each file's ``drive_voltage_vpp`` attr) times
amp_gain (the run's snapshotted ``hardware.yaml``).

The linear (P_1f) and quadratic (P_2f) trend lines are fit on the
*low-voltage* points only (the perturbative regime, V <= PERTURB_MAX_VPP)
and extrapolated across the full drive range, so the high-drive
saturation -- P_1f falling below the perturbative line, P_2f rolling off
the V^2 parabola -- stands out as the "innate throughput limitation".

Plots:
  1. P_1f peak vs PZT Vpp + perturbative linear fit, extrapolated (Coppens-1)
  2. P_2f peak vs PZT Vpp + perturbative V^2 fit, extrapolated (Coppens-2)
  3. P_2f / P_1f vs PZT Vpp
  4. P_2f / P_1f^2 vs PZT Vpp (Coppens prefactor)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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
from ldv_analysis.sweep_fit import sweep_peaks  # noqa: E402

# ---------------------------------------------------------------------------
# Clean 2026-05-24 peak-band voltage series (per-axial fit, water-filled).
# Each entry is (label_vpp, run_dir_name).  Bubble-contaminated runs
# (renamed *_failed_air) are intentionally excluded.  Add 70/80V here as
# they are measured.
# ---------------------------------------------------------------------------
SCANS = [
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

DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = ROOT / "experiments" / "2026W21" / "output"

# True PZT Vpp ceiling for the perturbative-regime fit.  Only points at
# or below this drive feed the linear P_1f and quadratic P_2f fits, so
# the trend lines reflect the small-signal regime (P_2f/P_1f still a few
# percent here -- 4.2% at 10 V, 9.8% at 30 V).  Higher-drive points are
# plotted but excluded from the fit; the line is extrapolated over them.
# Override on the CLI with --perturb-max.
PERTURB_MAX_VPP = 30.0


def read_amp_gain(run_dir: Path) -> float | None:
    """Read ``amplifier_gain_v_per_v`` from the snapshotted hardware.yaml."""
    hw = run_dir / "hardware.yaml"
    if not hw.exists():
        return None
    m = re.search(r"^\s*amplifier_gain_v_per_v:\s*([\d.]+)", hw.read_text(encoding="utf-8"), re.M)
    return float(m.group(1)) if m else None


def read_drive_vpp(run_dir: Path) -> float | None:
    """Read the AFG-side drive Vpp from ``protocol.yaml``.

    Mirrors the ``drive_voltage_vpp`` attr the .h5 stores (``conditions.
    voltage_vpp``).  Reading the small sidecar instead of opening a .h5
    keeps the run cache-only -- a .h5 open would hydrate the whole file on
    online-only (OneDrive Files-On-Demand) stores.
    """
    proto = run_dir / "protocol.yaml"
    if not proto.exists():
        return None
    m = re.search(r"^\s*voltage_vpp:\s*([\d.]+)", proto.read_text(encoding="utf-8"), re.M)
    return float(m.group(1)) if m else None


def main(perturb_max_vpp: float = PERTURB_MAX_VPP) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pzt_vpp: list[float] = []
    p1f_kpa: list[float] = []
    p2f_kpa: list[float] = []
    p1f_freq_mhz: list[float] = []

    print(
        f"{'label':>5}  {'AFG (V)':>8}  {'gain':>6}  {'PZT (V)':>8}  "
        f"{'P1 pk (kPa)':>11}  {'@MHz':>6}  {'P2 pk (kPa)':>11}  {'@MHz':>6}"
    )
    for label, dirname in SCANS:
        run_dir = DATA_ROOT / dirname
        cache_dir = get_cache_dir(dirname, __file__)
        try:
            sp = sweep_peaks(run_dir, CHANNEL_WIDTH, cache_dir)
        except (ValueError, FileNotFoundError) as e:
            # Scan still in progress / not yet written -- skip for now.
            print(f"{label:>5}  -- skipped ({e})")
            continue

        afg = read_drive_vpp(run_dir)
        gain = read_amp_gain(run_dir)
        if afg is not None and gain is not None:
            true_vpp = afg * gain
        else:
            true_vpp = float(label)  # fall back to folder label
        afg_disp = afg if afg is not None else float("nan")
        gain_disp = gain if gain is not None else float("nan")

        pzt_vpp.append(true_vpp)
        p1f_kpa.append(sp.peak_p1_kpa)
        p2f_kpa.append(sp.peak_p2_kpa)
        p1f_freq_mhz.append(sp.peak_p1_freq_mhz)
        print(
            f"{label:>5}  {afg_disp:>8.4f}  {gain_disp:>6.0f}  "
            f"{true_vpp:>8.2f}  {sp.peak_p1_kpa:>11.1f}  "
            f"{sp.peak_p1_freq_mhz:>6.3f}  {sp.peak_p2_kpa:>11.1f}  "
            f"{sp.peak_p2_freq_mhz:>6.3f}"
        )

    v = np.asarray(pzt_vpp)
    p1 = np.asarray(p1f_kpa)
    p2 = np.asarray(p2f_kpa)
    f1 = np.asarray(p1f_freq_mhz)

    # Perturbative-regime fit: use only the low-drive points (V <=
    # perturb_max_vpp), where P_1f ~ a*V and P_2f ~ b*V^2 hold, then
    # extrapolate across the full range to expose the high-drive
    # saturation (P_1f below the line) and 2f roll-off.
    fit_mask = v <= perturb_max_vpp
    n_fit = int(np.count_nonzero(fit_mask))
    if n_fit < 2:
        raise ValueError(
            f"only {n_fit} point(s) at or below {perturb_max_vpp:g} Vpp; "
            "raise --perturb-max so at least 2 points feed the fit"
        )
    v_fit, p1_fit, p2_fit = v[fit_mask], p1[fit_mask], p2[fit_mask]

    # Fits through origin on the perturbative subset.
    a_lin = float(np.sum(v_fit * p1_fit) / np.sum(v_fit * v_fit))  # kPa/Vpp
    b_v2 = float(np.sum(v_fit * v_fit * p2_fit) / np.sum(v_fit**4))  # kPa/Vpp^2
    print(
        f"\n[perturbative fits through origin, V <= {perturb_max_vpp:g} "
        f"Vpp, {n_fit}/{len(v)} points]"
    )
    print(f"  P_1f = {a_lin:.2f} kPa/Vpp x V_PZT")
    print(f"  P_2f = {b_v2:.4f} kPa/Vpp^2 x V_PZT^2")

    # Quantify the saturation: how far the top-drive P_1f falls below the
    # perturbative line extrapolated to the same voltage.
    v_top = float(v.max())
    p1_top = float(p1[int(np.argmax(v))])
    p1_extrap_top = a_lin * v_top
    suppress_pct = (1.0 - p1_top / p1_extrap_top) * 100.0
    print(
        f"  P_1f at {v_top:.0f} Vpp is {suppress_pct:.1f}% below the "
        f"perturbative line ({p1_top:.0f} vs {p1_extrap_top:.0f} kPa)"
    )

    ratio_21 = p2 / p1
    coppens_pref = p2 / (p1 * p1) * 1e3  # /MPa  (cancel the kPa^-2)

    # ---- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(
        5,
        1,
        figsize=figsize_for_layout(5, 1, sharex=True),
        sharex=True,
    )
    v_dense = np.linspace(0, v.max() * 1.05, 200)
    in_fit = v_dense <= perturb_max_vpp  # solid (fit) vs dashed (extrapolation)

    # --- P_1f: perturbative linear fit, extrapolated ----------------------
    axes[0].plot(
        v,
        p1,
        "o-",
        markersize=4,
        linewidth=0.8,
        color="C0",
        markerfacecolor="none",
        label="data (per-axial peak)",
    )
    axes[0].plot(
        v_fit,
        p1_fit,
        "o",
        markersize=4,
        color="C0",
        label=rf"fit points ($\leq {perturb_max_vpp:g}$ Vpp)",
    )
    axes[0].plot(
        v_dense[in_fit],
        a_lin * v_dense[in_fit],
        "-",
        linewidth=0.8,
        color="0.4",
        label=rf"$P_{{1f}} = {a_lin:.1f}$ kPa/Vpp $\times V$",
    )
    axes[0].plot(
        v_dense[~in_fit],
        a_lin * v_dense[~in_fit],
        "--",
        linewidth=0.8,
        color="0.4",
        label="extrapolation",
    )
    axes[0].axvline(perturb_max_vpp, color="0.7", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel(r"$P_{1f}$ peak [kPa]")
    axes[0].set_title(
        rf"sample chip cascade --- {len(v)} points, "
        rf"clean 2026-05-24 peak-band series; "
        rf"perturbative fit $\leq {perturb_max_vpp:g}$ Vpp"
    )
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(True, alpha=0.3)

    # --- P_2f: perturbative V^2 fit, extrapolated -------------------------
    axes[1].plot(
        v, p2, "o-", markersize=4, linewidth=0.8, color="C5", markerfacecolor="none", label="data"
    )
    axes[1].plot(
        v_fit,
        p2_fit,
        "o",
        markersize=4,
        color="C5",
        label=rf"fit points ($\leq {perturb_max_vpp:g}$ Vpp)",
    )
    axes[1].plot(
        v_dense[in_fit],
        b_v2 * v_dense[in_fit] ** 2,
        "-",
        linewidth=0.8,
        color="0.4",
        label=rf"$P_{{2f}} = {b_v2:.3f}$ kPa/Vpp$^2$ $\times V^2$",
    )
    axes[1].plot(
        v_dense[~in_fit],
        b_v2 * v_dense[~in_fit] ** 2,
        "--",
        linewidth=0.8,
        color="0.4",
        label="extrapolation",
    )
    axes[1].axvline(perturb_max_vpp, color="0.7", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel(r"$P_{2f}$ peak [kPa]")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(v, ratio_21 * 100, "o-", markersize=4, linewidth=0.8, color="C3")
    axes[2].set_ylabel(r"$P_{2f} / P_{1f}$ [\%]")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="0.5", lw=0.5)

    axes[3].plot(v, coppens_pref * 1e3, "o-", markersize=4, linewidth=0.8, color="C2")
    mean_pref = float(np.mean(coppens_pref * 1e3))
    axes[3].axhline(mean_pref, color="0.5", lw=0.5, ls="--", label=f"mean = {mean_pref:.1f}")
    axes[3].set_ylabel(r"$P_{2f} / P_{1f}^2$  [1/GPa]")
    axes[3].legend(fontsize=7, frameon=False, loc="upper right")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(v, f1 * 1e3, "o-", markersize=4, linewidth=0.8, color="C4")
    axes[4].set_ylabel(r"$P_{1f}$ peak freq [kHz]")
    axes[4].set_xlabel(r"True PZT drive [Vpp]")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "vpp_vs_pressure.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Compact tabular dump for cross-reference
    print(
        f"\n{'PZT Vpp':>8}  {'P_1f (kPa)':>10}  {'P_2f (kPa)':>10}  "
        f"{'ratio (%)':>9}  {'pref [1/GPa]':>13}"
    )
    for vi, p1i, p2i, ri, pi in zip(v, p1, p2, ratio_21, coppens_pref * 1e3):
        print(f"{vi:>8.2f}  {p1i:>10.1f}  {p2i:>10.1f}  " f"{ri*100:>9.2f}  {pi:>13.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voltage cascade: PZT Vpp vs peak P_1f / P_2f, with a "
        "perturbative low-voltage fit extrapolated to high drive.",
    )
    parser.add_argument(
        "--perturb-max",
        type=float,
        default=PERTURB_MAX_VPP,
        help="max true PZT Vpp included in the perturbative fit " f"(default: {PERTURB_MAX_VPP:g})",
    )
    args = parser.parse_args()
    main(args.perturb_max)
