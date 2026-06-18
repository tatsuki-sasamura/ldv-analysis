# %%
"""Pin the 2f cavity-mode location from the fine 3p76-3p84 MHz scan.

The Coppens prefactor ``P_2f / P_1f^2 = beta * Q_2 * cos(theta) /
(4 rho c^2)`` depends linearly on ``cos(theta)``, which depends on the
detuning ``Df = f_{2f,cavity} - 2 f_{1f,drive}``.  B1's coarse 2f-band
scan (``sample_101x77_fsweep_3p7to3p9_60Vpp_*``, 20 kHz steps) located
a peak at 3.800 MHz with R^2(cos2) = 0.99, but at 20 kHz resolution
that pins ``Df`` only to within ~10 kHz, which propagates to a factor
of ~2.7 uncertainty in the Coppens prediction (see B2 report).

This script uses the fine ``sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_*``
scan (41 files at 2 kHz resolution, recovered 2026-06-18) to pin
``f_{2f}`` to within ~1 kHz via a parabolic-peak fit in log-magnitude on
``T(f) = |P_cos2|/V_PZT``.  A Lorentzian fit on the same data also
returns the inferred ``Q_{2f}^{(field)}`` for comparison with the
transient ``Q_2 = 100`` (Q_field = 2 Q_pressure, but it's a useful
cross-check).

Outputs under ``output/f2_eigenmode_scan/``:
    f2_eigenmode_scan.png     T(f) + fits
    f2_eigenmode_scan.csv     per-frequency T and R^2
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

SCAN_DIR = "sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_20260530_072344"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "f2_eigenmode_scan"


def lorentzian(f, A, f0, gamma, c):
    """|P|^2 Lorentzian: A * (gamma/2)^2 / ((f-f0)^2 + (gamma/2)^2) + c."""
    return A * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2) + c


def parabolic_peak(x, y) -> tuple[float, float]:
    """3-point log-parabolic interpolation around the maximum sample.

    Returns ``(x_peak, y_peak)``.  Falls back to the raw peak if the
    neighbouring samples are not strictly smaller.
    """
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

    print(f"Fine 2f-band scan: {len(files)} files, "
          f"{files[0].name} ... {files[-1].name}")
    f_hz: list[float] = []
    v_pzt: list[float] = []
    p_cos2: list[float] = []
    r2_cos2: list[float] = []
    p_sin1: list[float] = []   # for sanity: how strong is the n=1 mode here?
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
        p_cos2.append(float(fit.p1_n2_mag))
        r2_cos2.append(float(fit.r2_p1_n2))
        p_sin1.append(float(fit.p1_mag))

    f = np.asarray(f_hz)                  # Hz
    v = np.asarray(v_pzt)                 # V
    p2 = np.asarray(p_cos2)               # Pa, cos(2pi x/W) amplitude
    r2 = np.asarray(r2_cos2)
    p1 = np.asarray(p_sin1)               # Pa, sin(pi x/W) amplitude
    T = p2 / v                            # Pa / V
    order = np.argsort(f)
    f, v, p2, r2, p1, T = f[order], v[order], p2[order], r2[order], p1[order], T[order]

    # ---- pin the peak via parabolic interpolation on T -------------------
    f_peak_par, T_peak_par = parabolic_peak(f, T)
    print(f"\nParabolic interpolation on T(f):")
    print(f"  peak f_2f = {f_peak_par/1e6:.4f} MHz   T_peak = {T_peak_par/1e3:.2f} kPa/V")

    # ---- Lorentzian fit on T^2 (which is proportional to |P|^2) ----------
    # only fit a window around the parabolic peak so off-resonance second
    # modes don't drag the fit.
    half_window_hz = 25e3   # 25 kHz half-window
    mask_fit = np.abs(f - f_peak_par) < half_window_hz
    f_fit = f[mask_fit]
    T2_fit = T[mask_fit] ** 2

    A0 = float(T2_fit.max() - T2_fit.min())
    p0 = [A0, f_peak_par, 20e3, float(T2_fit.min())]
    bounds = (
        [0.0, f_fit.min(), 1e3, 0.0],
        [10 * A0, f_fit.max(), 200e3, A0],
    )
    try:
        popt, _ = curve_fit(lorentzian, f_fit, T2_fit, p0=p0, bounds=bounds)
        A_fit, f0_fit, gamma_fit, c_fit = popt
        Q_field = f0_fit / gamma_fit
        print(f"\nLorentzian fit on T^2 (window +/-{half_window_hz/1e3:.0f} kHz):")
        print(f"  f_2f          = {f0_fit/1e6:.4f} MHz")
        print(f"  FWHM (gamma)  = {gamma_fit/1e3:.2f} kHz")
        print(f"  Q_field       = f/FWHM = {Q_field:.1f}")
        print(f"  (transient Q_pressure = 100 => Q_field expected = 50;")
        print(f"   FWHM-from-data and FWHM-from-transient differ by "
              f"{Q_field / 50:.2f}x if the same mode is dominant)")
    except RuntimeError as e:
        print(f"\nLorentzian fit failed: {e}")
        popt = None
        f0_fit = f_peak_par
        gamma_fit = float("nan")
        Q_field = float("nan")

    # Implications for the cos(theta) used in Coppens:
    f1_cascade_mhz = 1.902
    df_kHz = (f0_fit - 2.0 * f1_cascade_mhz * 1e6) / 1e3
    cos_theta_old_q100 = 1.0 / np.sqrt(1.0 + (2 * 100 * df_kHz * 1e3 / f0_fit) ** 2)
    print(f"\nImplications for Coppens calibration (cascade plateau at "
          f"f_1f = {f1_cascade_mhz:.3f} MHz):")
    print(f"  Df = f_2f - 2 f_1f = {df_kHz:+.2f} kHz")
    print(f"  cos(theta) at Q_2 = 100: {cos_theta_old_q100:.3f}")
    print(f"  => Coppens P_2f/P_1f^2 = beta Q_2 cos(theta) / (4 rho c^2)")
    print(f"  =                       3.48 * 100 * {cos_theta_old_q100:.3f} "
          f"/ (4 * 2.23e9)")
    coppens = 3.48 * 100 * cos_theta_old_q100 / (4 * 2.23e9) * 1e9
    print(f"  =                       {coppens:.2f}  /GPa  "
          f"(measured plateau ~22 /GPa)")

    # ---- CSV --------------------------------------------------------------
    csv_path = OUT_DIR / "f2_eigenmode_scan.csv"
    rows = ["f_MHz,V_PZT,P_cos2_kPa,R2_cos2,P_sin1_kPa,T_kPa_per_V"]
    for i in range(len(f)):
        rows.append(
            f"{f[i]/1e6:.4f},{v[i]:.3f},{p2[i]/1e3:.3f},{r2[i]:.4f},"
            f"{p1[i]/1e3:.3f},{T[i]/1e3:.4f}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=figsize_for_layout(3, 1, sharex=True), sharex=True
    )

    # (a) T(f) + Lorentzian fit + peak marker
    axes[0].plot(f / 1e6, T / 1e3, "o-", markersize=3, linewidth=0.7,
                 color="C0", label="T(f) = |P_cos2|/V")
    if popt is not None:
        f_dense = np.linspace(f.min(), f.max(), 600)
        T_dense = np.sqrt(np.maximum(lorentzian(f_dense, *popt), 0))
        axes[0].plot(f_dense / 1e6, T_dense / 1e3, "--", linewidth=0.8,
                     color="C3",
                     label=f"Lorentzian: $f_{{2f}}={f0_fit/1e6:.4f}$ MHz, "
                           f"$Q_\\mathrm{{field}}={Q_field:.0f}$")
    axes[0].axvline(f_peak_par / 1e6, color="0.5", lw=0.5, ls=":",
                    label=f"parabolic peak: {f_peak_par/1e6:.4f} MHz")
    axes[0].axvline(2 * f1_cascade_mhz, color="C2", lw=0.7, ls="-.",
                    label=fr"2 $f_{{1f,\mathrm{{drive}}}}={2*f1_cascade_mhz:.3f}$ MHz")
    axes[0].set_ylabel(r"$T(f)$ [kPa/V]")
    axes[0].set_title(
        r"Fine 2f-band scan (3.760-3.840 MHz, 2 kHz steps, "
        r"$f_{1f}^{(2)} = \cos(2\pi x/W)$ projection)"
    )
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    axes[0].grid(True, alpha=0.3)

    # (b) R^2 of the cos(2pi x/W) fit -- quality control
    axes[1].plot(f / 1e6, r2, "o-", markersize=3, linewidth=0.7, color="C5")
    axes[1].axhline(0.9, color="0.5", lw=0.5, ls="--",
                    label="$R^2 = 0.9$")
    axes[1].set_ylabel(r"$R^2(\cos 2\pi x/W)$")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(True, alpha=0.3)

    # (c) sin vs cos amplitudes (mode purity)
    axes[2].plot(f / 1e6, p2 / 1e3, "o-", markersize=3, linewidth=0.7,
                 color="C0", label=r"$|\cos(2\pi x/W)|$ amplitude")
    axes[2].plot(f / 1e6, p1 / 1e3, "s-", markersize=3, linewidth=0.7,
                 color="C4", label=r"$|\sin(\pi x/W)|$ amplitude")
    axes[2].set_ylabel(r"mode amplitude [kPa]")
    axes[2].set_xlabel(r"drive frequency [MHz]")
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "f2_eigenmode_scan.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
