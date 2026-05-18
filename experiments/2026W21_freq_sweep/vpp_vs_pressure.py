# %%
"""Cascade-saturation plot: true PZT Vpp vs P_1f and P_2f for the sample chip.

Uses the W21 narrow-band freq sweeps (101x1 line, 1880-1920 kHz, fitted
``|sin(pi y/W)|`` mode shape, peak across the band).  X axis is the
TRUE drive voltage at the PZT, computed as AFG_Vpp x amp_gain, where
amp_gain is read from each run's snapshotted ``hardware.yaml``.

This corrects for the splitter-induced 2x under-reading of CH A that
was active during the 2026-05-18 voltage series; see
``output/CALIBRATION_NOTE.md`` for the full story.

Plots:
  1. P_1f vs PZT Vpp + linear fit through origin (Coppens-1)
  2. P_2f vs PZT Vpp + V^2 fit through origin (Coppens-2)
  3. P_2f / P_1f vs PZT Vpp (Coppens predicts linear: ratio prop. M prop. V)
  4. P_2f / P_1f^2 vs PZT Vpp (Coppens prefactor, should be constant)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import FIG_DPI, figsize_for_layout  # noqa: E402

# ---------------------------------------------------------------------------
# Data points: narrow-band 1.88-1.92 MHz sweep series, sample chip,
# water-filled, 2026-05-18 session.  Each tuple is:
#   (label_vpp, run_dir_name, P_1f_peak_kPa, P_2f_peak_kPa)
# Peak values are from the per-run freq_vs_current.py runs (final
# tabular dump).  AFG_Vpp = label_vpp / 80 (the old assumed amp gain
# the labels were derived from).
# ---------------------------------------------------------------------------
SCANS = [
    (10, "sample_101x1_fsweep_narrow_10Vpp_20260518_213357",  3580,  245),
    (20, "sample_101x1_fsweep_narrow_20Vpp_20260518_212516",  6630,  884),
    (30, "sample_101x1_fsweep_narrow_30Vpp_20260518_211632",  8930, 1778),
    (40, "sample_101x1_fsweep_narrow_40Vpp_20260518_210751", 11560, 2584),
    (50, "sample_101x1_fsweep_narrow_50Vpp_20260518_205751", 14460, 3589),
    (60, "sample_101x1_fsweep_narrow_60Vpp_20260518_204855", 15730, 3840),
]

# True amp gain, post-recalibration (2026-05-18, splitter removed).
# Fallback if individual run snapshots haven't been updated.
TRUE_AMP_GAIN = 170.0

DATA_ROOT = Path(
    r"C:\Users\tatsuki\OneDrive - Lund University\Data\output"
)
OUT_DIR = ROOT / "experiments" / "2026W21_freq_sweep" / "output"


def read_amp_gain(run_dir: Path) -> float | None:
    hw = run_dir / "hardware.yaml"
    if not hw.exists():
        return None
    m = re.search(r"^\s*amplifier_gain_v_per_v:\s*([\d.]+)",
                  hw.read_text(encoding="utf-8"), re.M)
    return float(m.group(1)) if m else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pzt_vpp: list[float] = []
    p1f_kpa: list[float] = []
    p2f_kpa: list[float] = []
    afg_vpp: list[float] = []
    label_vpp: list[float] = []

    print(f"{'label':>5}  {'AFG (V)':>8}  {'amp_gain':>9}  "
          f"{'PZT (V)':>8}  {'P_1f (kPa)':>10}  {'P_2f (kPa)':>10}")
    for label, dirname, p1, p2 in SCANS:
        # AFG-side Vpp was label_vpp / 80 (the assumed-but-wrong gain
        # that the operator used to derive the folder label).
        afg = label / 80.0
        gain_snap = read_amp_gain(DATA_ROOT / dirname)
        gain = gain_snap if (gain_snap is not None and gain_snap > 100.0) \
            else TRUE_AMP_GAIN
        true_vpp = afg * gain
        label_vpp.append(label)
        afg_vpp.append(afg)
        pzt_vpp.append(true_vpp)
        p1f_kpa.append(p1)
        p2f_kpa.append(p2)
        print(f"{label:>5}  {afg:>8.4f}  {gain:>9.1f}  "
              f"{true_vpp:>8.2f}  {p1:>10}  {p2:>10}")

    v = np.asarray(pzt_vpp)
    p1 = np.asarray(p1f_kpa)
    p2 = np.asarray(p2f_kpa)

    # Fits through origin.  P_1f ~ a*V (linear), P_2f ~ b*V^2 (V-squared).
    a_lin = float(np.sum(v * p1) / np.sum(v * v))           # slope kPa/Vpp
    b_v2  = float(np.sum(v * v * p2) / np.sum(v ** 4))      # slope kPa/Vpp^2
    print(f"\n[fits through origin]")
    print(f"  P_1f = {a_lin:.2f} kPa/Vpp x V_PZT")
    print(f"  P_2f = {b_v2:.4f} kPa/Vpp^2 x V_PZT^2")

    ratio_21 = p2 / p1
    coppens_pref = p2 / (p1 * p1) * 1e3   # /MPa  (cancel the kPa^-2)

    # ---- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(
        4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True,
    )

    v_dense = np.linspace(0, v.max() * 1.05, 100)

    axes[0].plot(v, p1, "o-", markersize=4, linewidth=0.8, color="C0",
                 label="data")
    axes[0].plot(v_dense, a_lin * v_dense, "--", linewidth=0.7, color="0.5",
                 label=fr"$P_{{1f}} = {a_lin:.1f}$ kPa/Vpp $\times V$")
    axes[0].set_ylabel(r"$P_{1f}$ peak [kPa]")
    axes[0].set_title(rf"sample chip: 1f \& 2f vs true PZT Vpp  "
                      rf"(amp gain $\approx$ {TRUE_AMP_GAIN:g}$\times$)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(v, p2, "o-", markersize=4, linewidth=0.8, color="C5",
                 label="data")
    axes[1].plot(v_dense, b_v2 * v_dense ** 2, "--", linewidth=0.7,
                 color="0.5",
                 label=fr"$P_{{2f}} = {b_v2:.3f}$ kPa/Vpp$^2$ $\times V^2$")
    axes[1].set_ylabel(r"$P_{2f}$ peak [kPa]")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(v, ratio_21 * 100, "o-", markersize=4, linewidth=0.8,
                 color="C3")
    axes[2].set_ylabel(r"$P_{2f} / P_{1f}$ [\%]")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="0.5", lw=0.5)

    axes[3].plot(v, coppens_pref * 1e3, "o-", markersize=4, linewidth=0.8,
                 color="C2")
    mean_pref = float(np.mean(coppens_pref * 1e3))
    axes[3].axhline(mean_pref, color="0.5", lw=0.5, ls="--",
                    label=f"mean = {mean_pref:.1f}")
    axes[3].set_ylabel(r"$P_{2f} / P_{1f}^2$  [1/GPa]")
    axes[3].set_xlabel(r"True PZT drive [Vpp]")
    axes[3].legend(fontsize=7, frameon=False, loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "vpp_vs_pressure.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Compact tabular dump for cross-reference
    print(f"\n{'PZT Vpp':>8}  {'P_1f (kPa)':>10}  {'P_2f (kPa)':>10}  "
          f"{'ratio (%)':>9}  {'pref [1/GPa]':>13}")
    for vi, p1i, p2i, ri, pi in zip(v, p1, p2, ratio_21, coppens_pref * 1e3):
        print(f"{vi:>8.2f}  {p1i:>10.1f}  {p2i:>10.1f}  "
              f"{ri*100:>9.2f}  {pi:>13.1f}")


if __name__ == "__main__":
    main()
