# %%
"""Overlay T_1(f) lineshapes across the 10-120 Vpp cascade.

``q1_linewidth.py`` reports Q1(V) ranging 95-158 across the cascade.
That swing could be (a) genuine drive-dependence of the 1f cavity
linewidth (Coppens assumes Q V-independent), (b) fit-fragility from
the cascade's coarse 2 kHz freq step, or (c) sweep-window edge
effects at low V.  This script disambiguates by overlaying the raw
and peak-centered+normalized lineshapes per Vpp.

The headline result for F3 input is the **linear Q1** -- the cavity's
passive (small-signal) damping, measured in the regime where
nonlinear cascade depletion of the 1f mode is negligible.  This
script extracts Q1_linear as the mean of Q1(V) over the perturbative
band V_label in PERTURB_VPP_RANGE.  The harmonic_model iterative
Kuznetsov solver takes Q1_linear as input and then PREDICTS a
V-dependent Q1_eff(V) from the cascade physics; the measured Q1(V)
curve is the cross-check that the predicted broadening matches.

Visualisation:
  (a) raw T_1(f) per Vpp, one curve per drive level, colored by V;
  (b) normalized T_1(f)/T_peak(f) curves overlaid -- if Q were V-
      independent the curves would overlap when centered on each
      peak; if Q varies, the FWHM differs;
  (c) Q1(V) scatter with the perturbative-regime average + the fine-
      scan reference + the AF2026/W10 transient reference.

For comparison, the fine 1p89to1p92 1 kHz scan from
``f1_eigenmode_scan.py`` is overlaid on (a)/(b).

Output: ``output/f1_lineshape_vs_V/{png,csv}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpp_vs_pressure import SCANS, read_amp_gain, read_drive_vpp  # noqa: E402

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, LDV_DATA_ROOT, figsize_for_layout, get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.sweep_fit import fit_axial  # noqa: E402

DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "f1_lineshape_vs_V"
FINE_SCAN = "sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"

# Perturbative-regime cascade voltages used to extract Q1_linear: the
# 10 V scan is excluded because its peak sits at the sweep-window edge
# (unreliable Lorentzian fit); 60-120 V are excluded because the 1f
# linewidth visibly broadens there due to cascade depletion (P_2f/P_1f
# > 17 % at 60 V), so Q_eff < Q_linear in this regime.  20-50 V is the
# clean window: P_2f/P_1f < 13 %, R^2 > 0.99, FWHM ~ 12-14 kHz.
PERTURB_VPP_RANGE = (20, 50)
# Reference Q1 from the W10 transient (AF2026 abstract): tau1=20.2 us
# at f1=1.907 MHz -> Q1 = pi f1 tau1 = 121.
TRANSIENT_Q1_W10 = 121.0


def lorentzian(f, A, f0, hwhm, c):
    """|T|^2 Lorentzian: A * hwhm^2 / ((f-f0)^2 + hwhm^2) + c.  FWHM = 2*hwhm."""
    return A * hwhm ** 2 / ((f - f0) ** 2 + hwhm ** 2) + c


def fit_q_T2(f_hz, T):
    """Lorentzian fit on |T|^2(f); returns (f0, fwhm, Q, residual)."""
    y = T ** 2
    i = int(np.argmax(y))
    span = float(f_hz[-1] - f_hz[0])
    p0 = [float(y[i] - y.min()), float(f_hz[i]), span / 4, max(float(y.min()), 0.0)]
    bounds = (
        [0.0, float(f_hz.min()), span / 100, 0.0],
        [10 * float(y.max()), float(f_hz.max()), 5 * span, float(y.max()) + 1.0],
    )
    try:
        popt, _ = curve_fit(lorentzian, f_hz, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan"), float("nan")
    _, f0, hwhm, _ = popt
    fwhm = 2.0 * hwhm
    pred = lorentzian(f_hz, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(f0), float(fwhm), float(f0 / fwhm), float(r2)


def scan_T1(scan_dir_name: str):
    """Return (f_Hz, T_kPa_per_V, V_PZT_mean) for one cascade dir."""
    run_dir = DATA_ROOT / scan_dir_name
    cache_dir = get_cache_dir(scan_dir_name, __file__)
    files = sorted(run_dir.glob("f*.h5"))
    if not files:
        return np.array([]), np.array([]), float("nan")

    f_hz, T_list, v_list = [], [], []
    geom = None
    for p in files:
        with h5py.File(p, "r") as h:
            f_nom = float(h.attrs["drive_frequency_hz_nominal"])
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if int(np.sum(valid)) < 3:
            continue
        fit, geom = fit_axial(c, valid, CHANNEL_WIDTH, geom=geom)
        f_hz.append(f_nom)
        T_list.append(float(fit.p1_mag) / float(np.median(V[valid])))
        v_list.append(float(np.median(V[valid])))

    order = np.argsort(f_hz)
    return (
        np.asarray(f_hz)[order],
        np.asarray(T_list)[order],
        float(np.mean(v_list)) if v_list else float("nan"),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- fine 1f scan for reference ---------------------------------------
    f_fine, T_fine, V_fine = scan_T1(FINE_SCAN)
    f0_fine, fwhm_fine, q_fine, r2_fine = fit_q_T2(f_fine, T_fine)
    print(f"\nfine 1p89to1p92 (60 Vpp, {len(f_fine)} freqs @ 1 kHz):")
    print(f"  f0 = {f0_fine/1e6:.4f} MHz, FWHM = {fwhm_fine/1e3:.2f} kHz, "
          f"Q1 = {q_fine:.1f}, R^2 = {r2_fine:.3f}")

    # --- cascade dirs -----------------------------------------------------
    print(f"\nCascade (11 freqs @ 2 kHz):")
    print(f"{'Vpp':>5} {'V_PZT':>7} {'f0(MHz)':>8} {'FWHM(kHz)':>10} "
          f"{'Q1':>7} {'R^2':>6}  npts")
    casc = []
    for label, dirname in SCANS:
        f_v, T_v, V_pzt = scan_T1(dirname)
        if not len(f_v):
            print(f"{label:>5}   -- skipped (no files)")
            continue
        afg = read_drive_vpp(DATA_ROOT / dirname)
        gain = read_amp_gain(DATA_ROOT / dirname)
        v_label = afg * gain if (afg and gain) else float(label)
        f0i, fwhmi, qi, r2i = fit_q_T2(f_v, T_v)
        casc.append({
            "label": label, "v_true": v_label, "V_PZT": V_pzt,
            "f_hz": f_v, "T_kpa": T_v / 1e3,
            "f0": f0i, "fwhm": fwhmi, "Q1": qi, "r2": r2i,
        })
        print(f"{label:>5} {V_pzt:>7.2f} {f0i/1e6:>8.4f} "
              f"{fwhmi/1e3:>10.2f} {qi:>7.1f} {r2i:>6.3f}  {len(f_v)}")

    # --- linear-Q1 from the perturbative subset ---------------------------
    v_lo, v_hi = PERTURB_VPP_RANGE
    pert = [r for r in casc if v_lo <= r["label"] <= v_hi]
    if pert:
        q_perturb_arr = np.array([r["Q1"] for r in pert])
        Q1_linear = float(np.mean(q_perturb_arr))
        Q1_linear_std = float(np.std(q_perturb_arr, ddof=1)) if len(pert) > 1 else float("nan")
        print(f"\n=== F3 input -- Q1_linear from perturbative subset "
              f"({v_lo}-{v_hi} Vpp) ===")
        for r in pert:
            print(f"  {r['label']:>3} Vpp  ->  Q1 = {r['Q1']:.1f}  "
                  f"(R^2 = {r['r2']:.3f})")
        print(f"  mean = {Q1_linear:.1f}  +/-  {Q1_linear_std:.1f}  "
              f"(N={len(pert)})")
        print(f"  cross-checks:  fine 60 V Q1 = {q_fine:.0f}  "
              f"(borderline, some cascade loss);  "
              f"W10 transient Q1 = {TRANSIENT_Q1_W10:.0f}.")
    else:
        Q1_linear = float("nan")
        Q1_linear_std = float("nan")
        print(f"\n[no cascade points in {v_lo}-{v_hi} Vpp; Q1_linear undefined]")

    # --- CSV --------------------------------------------------------------
    csv_path = OUT_DIR / "f1_lineshape_vs_V.csv"
    rows = ["v_label,V_PZT,f0_MHz,FWHM_kHz,Q1,R2"]
    for r in casc:
        rows.append(
            f"{r['label']},{r['V_PZT']:.3f},{r['f0']/1e6:.4f},"
            f"{r['fwhm']/1e3:.3f},{r['Q1']:.2f},{r['r2']:.4f}"
        )
    rows.append(f"fine,{V_fine:.3f},{f0_fine/1e6:.4f},"
                f"{fwhm_fine/1e3:.3f},{q_fine:.2f},{r2_fine:.4f}")
    rows.append(f"Q1_linear_perturb_{v_lo}_{v_hi},,"
                f",,{Q1_linear:.2f},")
    rows.append(f"Q1_linear_perturb_std,,"
                f",,{Q1_linear_std:.2f},")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # --- plot --------------------------------------------------------------
    v_min = min(r["v_true"] for r in casc)
    v_max = max(r["v_true"] for r in casc)
    norm = Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        3, 1, figsize=figsize_for_layout(3, 1, sharex=False), sharex=False
    )

    # (a) raw T_1(f) overlays
    ax = axes[0]
    for r in casc:
        ax.plot(r["f_hz"] / 1e6, r["T_kpa"], "o-", markersize=3, linewidth=0.7,
                color=cmap(norm(r["v_true"])), label=f"{r['label']} Vpp")
    ax.plot(f_fine / 1e6, T_fine / 1e3, "k-", linewidth=1.0, alpha=0.7,
            label=f"fine 60V (1 kHz, Q={q_fine:.0f})")
    ax.set_xlabel("drive frequency [MHz]")
    ax.set_ylabel(r"$T_1(f) = |P_\mathrm{sin1}|/V_\mathrm{PZT}$  [kPa/V]")
    ax.set_title("Raw lineshape per Vpp")
    ax.grid(True, alpha=0.3)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(r"PZT drive [Vpp]", fontsize=7)

    # (b) normalized: each shifted by its own f0, divided by its own peak
    ax = axes[1]
    for r in casc:
        f_shift = (r["f_hz"] - r["f0"]) / 1e3  # kHz
        T_norm = r["T_kpa"] / r["T_kpa"].max()
        ax.plot(f_shift, T_norm, "o-", markersize=3, linewidth=0.7,
                color=cmap(norm(r["v_true"])))
    # fine scan
    f_shift_fine = (f_fine - f0_fine) / 1e3
    T_norm_fine = T_fine / T_fine.max()
    ax.plot(f_shift_fine, T_norm_fine, "k-", linewidth=1.0, alpha=0.7,
            label=f"fine 60V")
    ax.axvline(0, color="0.5", lw=0.3, ls=":")
    ax.set_xlabel(r"$f - f_0$ [kHz]")
    ax.set_ylabel(r"$T_1(f) / T_1(f_0)$")
    ax.set_title("Lineshape centered on each Vpp peak, normalized")
    ax.set_xlim(-12, 12)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (c) Q1(V) extracted
    ax = axes[2]
    v_arr = np.array([r["v_true"] for r in casc])
    Q_arr = np.array([r["Q1"] for r in casc])
    is_pert = np.array([v_lo <= r["label"] <= v_hi for r in casc])
    # plot perturbative-regime points as filled, others as open
    ax.plot(v_arr[~is_pert], Q_arr[~is_pert], "o", markerfacecolor="white",
            markeredgecolor="C0", markersize=5, linewidth=0,
            label=r"cascade $Q_1(V)$ (V outside perturbative window)")
    ax.plot(v_arr[is_pert], Q_arr[is_pert], "o", color="C0", markersize=5,
            linewidth=0,
            label=fr"cascade $Q_1(V)$ ($\{{{v_lo},{v_hi}\}}$ Vpp -> $Q_1$ linear)")
    ax.plot(v_arr, Q_arr, "-", linewidth=0.6, color="C0", alpha=0.4)
    if np.isfinite(Q1_linear):
        ax.axhline(Q1_linear, color="C2", lw=1.0, ls="-",
                   label=fr"$Q_{{1,\mathrm{{linear}}}}$ = {Q1_linear:.0f} "
                         fr"$\pm$ {Q1_linear_std:.0f} (F3 input)")
        if np.isfinite(Q1_linear_std):
            ax.axhspan(Q1_linear - Q1_linear_std,
                       Q1_linear + Q1_linear_std,
                       color="C2", alpha=0.12)
    ax.axhline(q_fine, color="C3", lw=0.8, ls="--",
               label=f"fine 60 V: $Q_1$ = {q_fine:.0f}  (cascade-loss biased)")
    ax.axhline(TRANSIENT_Q1_W10, color="0.4", lw=0.8, ls=":",
               label=f"W10 transient: $Q_1$ = {TRANSIENT_Q1_W10:.0f}")
    ax.set_xlabel("PZT drive [Vpp]")
    ax.set_ylabel(r"$Q_1 = f_0/\mathrm{FWHM}$")
    ax.set_ylim(0, max(180, Q_arr.max() * 1.1))
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        r"$Q_1(V)$ (cascade 11 pts @ 2 kHz); F3 input = perturbative-mean "
        fr"$Q_{{1,\mathrm{{linear}}}}$"
    )

    plt.tight_layout()
    out = OUT_DIR / "f1_lineshape_vs_V.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
