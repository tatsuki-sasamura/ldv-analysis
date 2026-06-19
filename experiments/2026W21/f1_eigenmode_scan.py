# %%
"""Visualize the 1f cavity-mode line shape T_1(f) on the fine 1f-band scan.

The Q1 work in ``q1_linewidth.py`` fits a single Lorentzian to
|P_1f|^2 vs f and reports Q1 = f0 / FWHM = 139 (60 Vpp).  Before
trusting that single-Lorentzian framing we want a direct look at the
line shape: is the 1f resonance a clean single mode (one Lorentzian
fits, R^2 of the sin(pi y/W) projection is high across the band), or
does it have fine structure analogous to the two-mode 2f cavity at
3.794 + 3.818 MHz?

Method mirrors ``f2_eigenmode_scan.py`` for the 2f band:

    T_1(f) = |P_sin1(f)| / V_PZT(f)    [Pa / V]

projected onto the n=1 transverse mode (sin(pi y/W)) via the
``AxialFit.p1_mag`` field returned by ``fit_axial``, dividing by the
measured PZT voltage.  Also reports R^2 of the sin1 fit per frequency
so we can see immediately whether the mode is pure (high R^2 in the
peak region, like 2f's R^2 = 0.99 plateau) or contaminated.

Output: ``output/f1_eigenmode_scan/{png,csv}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

SCAN_DIR = "sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "f1_eigenmode_scan"


def lorentzian(f, A, f0, gamma, c):
    """|P|^2 Lorentzian: A * (gamma/2)^2 / ((f-f0)^2 + (gamma/2)^2) + c."""
    return A * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2) + c


def parabolic_peak(x, y) -> tuple[float, float]:
    """3-point log-parabolic interpolation around the max sample."""
    k = int(np.argmax(y))
    if k == 0 or k == len(x) - 1:
        return float(x[k]), float(y[k])
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    if y1 <= y0 or y1 <= y2:
        return float(x[k]), float(y[k])
    a = np.log(y0)
    b = np.log(y1)
    c = np.log(y2)
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
    dx = float(x[k + 1] - x[k])
    return float(x[k] + delta * dx), float(np.exp(b + 0.25 * (a - c) * delta))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = DATA_ROOT / SCAN_DIR
    cache_dir = get_cache_dir(SCAN_DIR, __file__)
    files = sorted(run_dir.glob("f*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 in {run_dir}")

    print(f"Fine 1f-band scan: {len(files)} files, "
          f"{files[0].name} ... {files[-1].name}")
    f_hz: list[float] = []
    v_pzt: list[float] = []
    p_sin1: list[float] = []
    r2_sin1: list[float] = []
    geom = None

    for p in files:
        with h5py.File(p, "r") as h:
            f_nom = float(h.attrs["drive_frequency_hz_nominal"])
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if int(np.sum(valid)) < 3:
            print(f"  {p.name}: only {int(np.sum(valid))} valid; skip")
            continue
        fit, geom = fit_axial(c, valid, CHANNEL_WIDTH, geom=geom)
        f_hz.append(f_nom)
        v_pzt.append(float(np.median(V[valid])))
        # 1f cavity mode is the n=1 transverse sin(pi y/W) -> p1_mag.
        p_sin1.append(float(fit.p1_mag))
        r2_sin1.append(float(fit.r2_1))

    f = np.asarray(f_hz)                  # Hz
    v = np.asarray(v_pzt)                 # V
    p1 = np.asarray(p_sin1)                # Pa, sin(pi y/W) amplitude
    r2 = np.asarray(r2_sin1)
    T = p1 / v                            # Pa / V
    order = np.argsort(f)
    f, v, p1, r2, T = f[order], v[order], p1[order], r2[order], T[order]

    # ---- pin the peak via parabolic interpolation on T -------------------
    f_peak_par, T_peak_par = parabolic_peak(f, T)
    print(f"\nParabolic interpolation on T(f):")
    print(f"  peak f_1f = {f_peak_par/1e6:.4f} MHz   "
          f"T_peak = {T_peak_par/1e3:.2f} kPa/V")

    # ---- Lorentzian fit on T^2 over a window around the peak -------------
    half_window_hz = 20e3   # 20 kHz half-window
    mask_fit = np.abs(f - f_peak_par) < half_window_hz
    f_fit = f[mask_fit]
    T2_fit = T[mask_fit] ** 2
    A0 = float(T2_fit.max() - T2_fit.min())
    p0 = [A0, f_peak_par, 15e3, float(T2_fit.min())]
    bounds = (
        [0.0, f_fit.min(), 1e3, 0.0],
        [10 * A0, f_fit.max(), 200e3, A0],
    )
    try:
        popt, _ = curve_fit(lorentzian, f_fit, T2_fit, p0=p0, bounds=bounds)
        A_fit, f0_fit, gamma_fit, c_fit = popt
        Q_field = f0_fit / gamma_fit
        print(f"\nLorentzian fit on T^2 (window +/-{half_window_hz/1e3:.0f} kHz):")
        print(f"  f_1f          = {f0_fit/1e6:.4f} MHz")
        print(f"  FWHM (gamma)  = {gamma_fit/1e3:.2f} kHz")
        print(f"  Q_field       = f0/FWHM = {Q_field:.1f}")
        print(f"  (q1_linewidth.py report:  Q1 = 139,  FWHM ~= 13.8 kHz)")
        print(f"  (AF2026/W10 transient:    Q1 = 121)")
    except RuntimeError as e:
        print(f"\nLorentzian fit failed: {e}")
        popt = None
        f0_fit = f_peak_par
        gamma_fit = float("nan")
        Q_field = float("nan")

    # ---- CSV --------------------------------------------------------------
    csv_path = OUT_DIR / "f1_eigenmode_scan.csv"
    rows = ["f_MHz,V_PZT,P_sin1_kPa,R2_sin1,T_kPa_per_V"]
    for i in range(len(f)):
        rows.append(
            f"{f[i]/1e6:.4f},{v[i]:.3f},{p1[i]/1e3:.3f},{r2[i]:.4f},"
            f"{T[i]/1e3:.4f}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True
    )

    # (a) T(f) + Lorentzian fit + peak marker
    axes[0].plot(f / 1e6, T / 1e3, "o-", markersize=3, linewidth=0.7,
                 color="C0", label="T(f) = |P_sin1|/V")
    if popt is not None:
        f_dense = np.linspace(f.min(), f.max(), 600)
        T_dense = np.sqrt(np.maximum(lorentzian(f_dense, *popt), 0))
        axes[0].plot(f_dense / 1e6, T_dense / 1e3, "--", linewidth=0.8,
                     color="C3",
                     label=f"Lorentzian: $f_{{1f}}={f0_fit/1e6:.4f}$ MHz, "
                           f"$Q_\\mathrm{{field}}={Q_field:.0f}$")
    axes[0].axvline(f_peak_par / 1e6, color="0.5", lw=0.5, ls=":",
                    label=f"parabolic peak: {f_peak_par/1e6:.4f} MHz")
    axes[0].set_ylabel(r"$T_1(f)$ [kPa/V]")
    axes[0].set_title(
        r"Fine 1f-band scan (1.890-1.920 MHz, 1 kHz steps, 60 Vpp; "
        r"$P_{1f}$ projected onto $\sin(\pi y/W)$)"
    )
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    axes[0].grid(True, alpha=0.3)

    # (b) R^2 of the sin(pi y/W) fit -- quality control
    axes[1].plot(f / 1e6, r2, "o-", markersize=3, linewidth=0.7, color="C5")
    axes[1].axhline(0.9, color="0.5", lw=0.5, ls="--",
                    label="$R^2 = 0.9$")
    axes[1].set_ylabel(r"$R^2(\sin \pi y/W)$")
    axes[1].set_xlabel(r"drive frequency [MHz]")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "f1_eigenmode_scan.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
