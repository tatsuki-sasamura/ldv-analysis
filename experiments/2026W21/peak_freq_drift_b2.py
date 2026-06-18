# %%
"""B2 peak-frequency drift across the cascade.

Two questions this script answers:

1. **Thermal bound** -- the fundamental peak frequency drifts down with
   drive because the water warms.  ``f_1 = c/(2 W_eff)`` and
   ``dc/dT ~ +2.5 m/s/K`` near 20 deg C, so
   ``DT = (Df_1 / f_1) * c / (dc/dT)``.  Sets a per-Vpp temperature
   rise that the cascade interpretation has to be robust against.

2. **Per-V cos theta -> per-V Coppens prediction.**  The Coppens
   prefactor ``P_2f / P_1f^2 = beta * Q_2 * cos(theta) / (4 rho c^2)``
   has a single Vpp dependence through cos(theta), which depends on
   ``Df = f_{2f,cavity} - 2 * f_1f_drive``.  If ``f_1f_drive`` drifts
   ~6 kHz across the cascade, ``2*f_1f_drive`` drifts ~12 kHz, which is
   comparable to ``|Df|`` itself (~14 kHz to the 3.80 MHz mode in the
   101x77 transfer, ~31 kHz to the 3.845 MHz mode in the survey
   lines).  So cos(theta) is **not** drive-independent.  Does the per-V
   Coppens prediction explain the 10 Vpp outlier
   (measured 26.7 /GPa vs static-cos(theta) Coppens 20.7 /GPa)?

We pull per-Vpp peak f from ``sweep_peaks`` (the same machinery used by
``vpp_vs_pressure.py``), recompute the Lorentzian-tail cos(theta) per
Vpp, and overlay both predictions against the measured prefactor.

Outputs under ``output/peak_freq_drift_b2/``:
    peak_freq_drift_b2.png   four-panel: f1, DT, cos(theta), prefactor
    peak_freq_drift_b2.csv   raw per-Vpp numbers
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, C_SOUND, FIG_DPI, LDV_DATA_ROOT, RHO,
    figsize_for_layout, get_cache_dir,
)
from ldv_analysis.sweep_fit import sweep_peaks  # noqa: E402

# Canonical 12-Vpp cascade (matches vpp_vs_pressure.py).
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
OUT_DIR = Path(__file__).resolve().parent / "output" / "peak_freq_drift_b2"

# ---- physics constants -----------------------------------------------------
BETA = 3.48                 # 1 + (B/A)/2 for water
Q2 = 100.0                  # 2f Q (transient measurement)
DC_DT = 2.5                 # m/s/K, water at 20 deg C (Bilaniuk-Wong)

# Two candidate 2f cavity-mode locations -- see B1 side finding.
# Keys are plain ASCII (LaTeX-safe): used both in legends and CSV column names.
F2_EIGENMODES_MHZ = {
    "3.845 MHz, survey 21x1/101x1": 3.845,
    "3.800 MHz, 101x77 transfer fit": 3.800,
}


def cos_theta_lorentzian(df_hz: np.ndarray, f2_hz: float, q2: float) -> np.ndarray:
    """Lorentzian-tail off-resonance factor: 1 / sqrt(1 + (2*Q2*Df/f2)^2)."""
    return 1.0 / np.sqrt(1.0 + (2.0 * q2 * df_hz / f2_hz) ** 2)


def coppens_prefactor(cos_th: np.ndarray) -> np.ndarray:
    """P_2f/P_1f^2 prediction in 1/Pa from Coppens.

    ``= beta * Q2 * cos(theta) / (4 * rho * c^2)``
    """
    return BETA * Q2 * cos_th / (4.0 * RHO * C_SOUND ** 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vpp: list[float] = []
    f1_mhz: list[float] = []
    p1_kpa: list[float] = []
    p2_kpa: list[float] = []

    print(f"{'Vpp':>4}  {'peak f_1f':>10}  {'P_1f':>8}  {'P_2f':>8}")
    for label, dirname in SCANS:
        cache_dir = get_cache_dir(dirname, __file__)
        sp = sweep_peaks(DATA_ROOT / dirname, CHANNEL_WIDTH, cache_dir)
        vpp.append(float(label))
        f1_mhz.append(sp.peak_p1_freq_mhz)
        p1_kpa.append(sp.peak_p1_kpa)
        p2_kpa.append(sp.peak_p2_kpa)
        print(f"{label:>4}  {sp.peak_p1_freq_mhz:>10.4f}  "
              f"{sp.peak_p1_kpa:>8.1f}  {sp.peak_p2_kpa:>8.1f}")

    v = np.asarray(vpp)
    f1 = np.asarray(f1_mhz) * 1e6        # Hz
    p1 = np.asarray(p1_kpa) * 1e3        # Pa
    p2 = np.asarray(p2_kpa) * 1e3        # Pa
    prefactor_meas = p2 / (p1 ** 2)       # 1/Pa
    prefactor_meas_per_gpa = prefactor_meas * 1e9

    # Reference (lowest-V) peak frequency for relative shifts.
    f1_ref = f1[0]                        # 10 Vpp anchor
    df1 = f1 - f1_ref                     # Hz
    df1_over_f1 = df1 / f1_ref
    # Thermal rise: Df/f = Dc/c -> DT = Df/f * c / (dc/dT)
    dT = df1_over_f1 * C_SOUND / DC_DT    # K

    # cos(theta) and Coppens prediction per Vpp, for each candidate eigenmode.
    results = {}
    for name, f2_mhz in F2_EIGENMODES_MHZ.items():
        f2 = f2_mhz * 1e6
        df = f2 - 2.0 * f1                 # Hz, signed
        cos_th = cos_theta_lorentzian(df, f2, Q2)
        cop_pa = coppens_prefactor(cos_th)
        results[name] = {
            "f2": f2_mhz,
            "df_khz": df / 1e3,
            "cos_th": cos_th,
            "coppens_per_gpa": cop_pa * 1e9,
        }

    # --- console -----------------------------------------------------------
    print(
        f"\n{'Vpp':>4}  {'f_1f (MHz)':>11}  {'Df_1f (kHz)':>13}  "
        f"{'DT (K)':>7}  || " +
        "  ".join(
            f"{'cosT(' + n.split()[0] + ')':>14}" for n in results
        )
        + "  || " +
        "  ".join(
            f"{'Coppens(' + n.split()[0] + ')':>16}" for n in results
        )
        + f"  || {'measured /GPa':>13}"
    )
    for i, vi in enumerate(v):
        row = (
            f"{vi:>4.0f}  {f1[i]/1e6:>11.4f}  "
            f"{df1[i]/1e3:>13.1f}  {dT[i]:>7.2f}  || "
            + "  ".join(
                f"{r['cos_th'][i]:>14.3f}" for r in results.values())
            + "  || "
            + "  ".join(
                f"{r['coppens_per_gpa'][i]:>16.2f}" for r in results.values())
            + f"  || {prefactor_meas_per_gpa[i]:>13.2f}"
        )
        print(row)

    # --- CSV ----------------------------------------------------------------
    csv_path = OUT_DIR / "peak_freq_drift_b2.csv"
    headers = ["PZT_Vpp", "f1_MHz", "df1_kHz_vs_10V", "dT_K",
               "P_1f_kPa", "P_2f_kPa", "prefactor_meas_perGPa"]
    for n in results:
        tag = n.split()[0]
        headers += [f"df_to_{tag}_kHz", f"cos_theta_{tag}",
                    f"coppens_{tag}_perGPa"]
    rows = [",".join(headers)]
    for i, vi in enumerate(v):
        parts = [f"{vi:.0f}",
                 f"{f1[i]/1e6:.6f}", f"{df1[i]/1e3:.3f}",
                 f"{dT[i]:.4f}",
                 f"{p1[i]/1e3:.2f}", f"{p2[i]/1e3:.2f}",
                 f"{prefactor_meas_per_gpa[i]:.3f}"]
        for r in results.values():
            parts += [f"{r['df_khz'][i]:.3f}",
                      f"{r['cos_th'][i]:.4f}",
                      f"{r['coppens_per_gpa'][i]:.3f}"]
        rows.append(",".join(parts))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # --- plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True
    )

    axes[0].plot(v, f1 / 1e6, "o-", markersize=4, linewidth=0.8, color="C0")
    axes[0].set_ylabel(r"$f_{1f}$ peak [MHz]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        r"B2 peak-frequency drift --- per-V $f_{1f}$, thermal rise, "
        r"per-V $\cos\theta$, per-V Coppens"
    )

    axes[1].plot(v, dT, "o-", markersize=4, linewidth=0.8, color="C3")
    axes[1].axhline(0, color="0.5", lw=0.5)
    axes[1].set_ylabel(r"$\Delta T$ vs 10 Vpp [K]")
    axes[1].grid(True, alpha=0.3)

    colors = ["C0", "C2"]
    for (name, r), c in zip(results.items(), colors):
        axes[2].plot(v, r["cos_th"], "o-", markersize=4, linewidth=0.8,
                     color=c, label=name)
    axes[2].set_ylabel(r"$\cos\theta$  (Lorentzian tail, $Q_2=100$)")
    axes[2].legend(fontsize=7, frameon=False, loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(v, prefactor_meas_per_gpa, "o-", markersize=4,
                 linewidth=0.8, color="C5", label="measured")
    for (name, r), c in zip(results.items(), colors):
        axes[3].plot(v, r["coppens_per_gpa"], "s--", markersize=3,
                     linewidth=0.8, color=c,
                     label=f"Coppens, $f_{{2f}}=${r['f2']:.3f} MHz")
    axes[3].set_ylabel(r"$P_{2f}/P_{1f}^2$  [1/GPa]")
    axes[3].set_xlabel(r"True PZT drive [Vpp]")
    axes[3].legend(fontsize=7, frameon=False, loc="best")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "peak_freq_drift_b2.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
