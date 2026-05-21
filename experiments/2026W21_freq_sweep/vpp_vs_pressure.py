# %%
"""Cascade-saturation plot: true PZT Vpp vs P_1f .. P_5f for the sample chip.

Uses the W21 narrow-band freq sweeps (101x1 line, 1880-1920 kHz),
fitting the per-harmonic mode shape (odd → |sin(n pi y/W)|,
even → |cos(n pi y/W)|) and taking the peak across the band for each
harmonic.  X axis is the TRUE drive voltage at the PZT, computed as
AFG_Vpp x amp_gain, where amp_gain is read from each run's snapshotted
``hardware.yaml`` (with a fallback if the snapshot still has the
pre-recalibration ``80.0`` value).

This corrects for the splitter-induced ~2x under-reading of CH A that
was active during the 2026-05-18 voltage series; see
``CALIBRATION_NOTE.md`` for the full story.

Plots (single figure, 3 rows x 2 cols):
  (0,0) P_1f vs Vpp + linear fit through origin       (Coppens-1)
  (0,1) P_2f vs Vpp + V^2 fit through origin           (Coppens-2)
  (1,0) P_3f vs Vpp + V^3 fit through origin
  (1,1) P_4f vs Vpp + V^4 fit through origin
  (2,0) P_5f vs Vpp + V^5 fit through origin
  (2,1) Ratios P_nf / P_1f for n=2..5

Per-scan analysis cache: ``output/vpp_vs_pressure_cache.npz``.  Delete
to force re-analysis.  The per-file FFT cache lives under
``output/resonance_survey/fft_cache/<dataset_id>/`` (shared with
``resonance_survey.py`` to avoid duplicate FFTs).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, figsize_for_layout,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import fit_mode  # noqa: E402

# ---------------------------------------------------------------------------
# Narrow-band scans (1.88-1.92 MHz around the chip 1f resonance).
# Each tuple: (folder_vpp_label, run_dir_name).
#
# Two layered corrections needed to go folder label → true PZT Vpp:
#   1. W21 ×2 mislabel: folder label is AFG Vp (peak), recorded as Vpp.
#      True AFG Vpp = label × W21_VPP_FACTOR.
#   2. Pre-recalibration assumed gain of 80× was used to derive the
#      folder label from AFG_Vpp; true gain is ~170× (snapshotted per run
#      where available, else TRUE_AMP_GAIN fallback).
#
# Net: True PZT Vpp = (label × W21_VPP_FACTOR / 80) × amp_gain.
# ---------------------------------------------------------------------------
NARROW_SCANS = [
    (10, "sample_101x1_fsweep_narrow_10Vpp_20260518_213357"),
    (20, "sample_101x1_fsweep_narrow_20Vpp_20260518_212516"),
    (30, "sample_101x1_fsweep_narrow_30Vpp_20260518_211632"),
    (40, "sample_101x1_fsweep_narrow_40Vpp_20260518_210751"),
    (50, "sample_101x1_fsweep_narrow_50Vpp_20260518_205751"),
    (60, "sample_101x1_fsweep_narrow_60Vpp_20260518_204855"),
]
HARMONICS = (1, 2, 3, 4, 5)
W21_VPP_FACTOR = 2        # AFG Vp → Vpp (recorded as Vpp by mistake)
TRUE_AMP_GAIN = 170.0     # fallback if snapshot still has the bad 80.0

DATA_ROOT = Path(r"D:/OneDrive - Lund University/Data/output")
OUT_DIR = ROOT / "experiments" / "2026W21_freq_sweep" / "output"
RES_CACHE = OUT_DIR / "resonance_survey" / "fft_cache"   # shared with resonance_survey.py
ANALYSIS_CACHE = OUT_DIR / "vpp_vs_pressure_cache.npz"


def _read_amp_gain(run_dir: Path) -> float | None:
    """Read ``amplifier_gain_v_per_v`` from the snapshotted hardware.yaml."""
    hw = run_dir / "hardware.yaml"
    if not hw.exists():
        return None
    m = re.search(r"^\s*amplifier_gain_v_per_v:\s*([\d.]+)",
                  hw.read_text(encoding="utf-8"), re.M)
    return float(m.group(1)) if m else None


def _dataset_id_for_narrow(label_vpp: int, dirname: str) -> str:
    """Mirror the resonance_survey.py id so we share the FFT cache dir."""
    suffix = dirname.split("_")[-1]
    return f"sample_narrow_{label_vpp}V_{suffix}"


def _per_harmonic_peak(run_dir: Path, cache_dir: Path) -> dict[int, float]:
    """Return {harmonic: peak P_nf in Pa} taken across the frequency band.

    For each file: load_or_compute, fit |sin/cos(n pi y/W)| for n=1..5,
    record p0 magnitude.  Then the peak across frequencies is the
    chip's peak P_nf response in this band.
    """
    files = sorted(p for p in run_dir.glob("*.h5")
                   if not p.name.endswith(".inprogress"))
    per_freq: dict[int, list[float]] = {n: [] for n in HARMONICS}

    for p in files:
        try:
            c = load_or_compute(p, cache_dir, velocity_scale=None)
        except Exception as e:  # noqa: BLE001
            print(f"  FFT error {p.name}: {e}")
            continue
        pos = np.asarray(c["pos_x"])
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if valid.sum() < 3:
            continue

        # 1f sets the channel center; reuse it for higher harmonics
        P1c = (np.asarray(c["pressure_1f"])
               * np.exp(1j * np.radians(np.asarray(c["phase_1f"]))))
        res1 = fit_mode(pos[valid], P1c[valid], CHANNEL_WIDTH, harmonic=1)
        center = res1.center
        per_freq[1].append(float(abs(res1.p0)))

        for h in (2, 3, 4, 5):
            kp = f"pressure_{h}f"
            kph = f"phase_{h}f"
            if kp not in c.files or kph not in c.files:
                per_freq[h].append(np.nan)
                continue
            Ph_c = (np.asarray(c[kp])
                    * np.exp(1j * np.radians(np.asarray(c[kph]))))
            try:
                resh = fit_mode(pos[valid], Ph_c[valid], CHANNEL_WIDTH,
                                harmonic=h, center=center)
                per_freq[h].append(float(abs(resh.p0)))
            except Exception:  # noqa: BLE001
                per_freq[h].append(np.nan)

    # Peak across frequencies for each harmonic (NaN-safe)
    peaks: dict[int, float] = {}
    for h, vals in per_freq.items():
        arr = np.asarray(vals, dtype=float)
        peaks[h] = float(np.nanmax(arr)) if arr.size and np.any(~np.isnan(arr)) else float("nan")
    return peaks


def _load_or_build_analysis() -> dict:
    """Return dict with arrays keyed 'label_vpp', 'afg_vpp', 'amp_gain',
    'pzt_vpp', 'P1', 'P2', 'P3', 'P4', 'P5' (all in Pa for P_*).
    Cached at ``ANALYSIS_CACHE``."""
    if ANALYSIS_CACHE.exists():
        d = dict(np.load(ANALYSIS_CACHE))
        if all(k in d for k in ("P1", "P2", "P3", "P4", "P5", "pzt_vpp")):
            print(f"Loaded analysis cache: {ANALYSIS_CACHE.name}")
            return {k: np.asarray(v) for k, v in d.items()}

    label_vpp: list[float] = []
    afg_vpp: list[float] = []
    amp_gain: list[float] = []
    pzt_vpp: list[float] = []
    P_by_n: dict[int, list[float]] = {n: [] for n in HARMONICS}

    for label, dirname in NARROW_SCANS:
        print(f"\n--- {label} Vpp label ({dirname}) ---")
        run_dir = DATA_ROOT / dirname
        if not run_dir.exists():
            print(f"  MISSING: {run_dir}")
            for n in HARMONICS:
                P_by_n[n].append(np.nan)
            label_vpp.append(label)
            afg_vpp.append((label * W21_VPP_FACTOR) / 80.0)
            amp_gain.append(TRUE_AMP_GAIN)
            pzt_vpp.append((label * W21_VPP_FACTOR / 80.0) * TRUE_AMP_GAIN)
            continue

        cache_dir = RES_CACHE / _dataset_id_for_narrow(label, dirname)
        cache_dir.mkdir(parents=True, exist_ok=True)
        peaks = _per_harmonic_peak(run_dir, cache_dir)

        afg = (label * W21_VPP_FACTOR) / 80.0
        gain_snap = _read_amp_gain(run_dir)
        gain = (gain_snap if (gain_snap is not None and gain_snap > 100.0)
                else TRUE_AMP_GAIN)
        true_vpp = afg * gain

        label_vpp.append(label)
        afg_vpp.append(afg)
        amp_gain.append(gain)
        pzt_vpp.append(true_vpp)
        for n in HARMONICS:
            P_by_n[n].append(peaks[n])

        print(f"  AFG {afg:.4f} V x gain {gain:.1f} -> PZT {true_vpp:.2f} Vpp")
        for n in HARMONICS:
            vstr = f"{peaks[n]/1e3:.1f} kPa" if np.isfinite(peaks[n]) else "n/a"
            print(f"    P_{n}f peak = {vstr}")

    out = dict(
        label_vpp=np.asarray(label_vpp, dtype=float),
        afg_vpp=np.asarray(afg_vpp, dtype=float),
        amp_gain=np.asarray(amp_gain, dtype=float),
        pzt_vpp=np.asarray(pzt_vpp, dtype=float),
        P1=np.asarray(P_by_n[1], dtype=float),
        P2=np.asarray(P_by_n[2], dtype=float),
        P3=np.asarray(P_by_n[3], dtype=float),
        P4=np.asarray(P_by_n[4], dtype=float),
        P5=np.asarray(P_by_n[5], dtype=float),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ANALYSIS_CACHE, **out)
    print(f"\nSaved analysis cache: {ANALYSIS_CACHE}")
    return out


def _power_fit_through_origin(v: np.ndarray, p: np.ndarray,
                              n: int) -> float:
    """Least-squares slope for p = a · v^n through origin (NaN-safe)."""
    ok = np.isfinite(v) & np.isfinite(p) & (v > 0)
    if ok.sum() < 2:
        return float("nan")
    vn = v[ok] ** n
    return float(np.sum(vn * p[ok]) / np.sum(vn * vn))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = _load_or_build_analysis()
    v = res["pzt_vpp"]
    P = {n: res[f"P{n}"] for n in HARMONICS}

    # Power-law fits a_n · V^n through origin, on Pa
    a = {n: _power_fit_through_origin(v, P[n], n) for n in HARMONICS}

    # Pretty print
    print(f"\n{'PZT Vpp':>8}  " + "  ".join(f"P_{n}f (kPa)".rjust(11) for n in HARMONICS))
    for i, vi in enumerate(v):
        row = f"{vi:>8.2f}"
        for n in HARMONICS:
            pi = P[n][i] / 1e3
            row += f"  {pi:>11.1f}" if np.isfinite(pi) else "         n/a"
        print(row)

    print(f"\n[power-law slopes, p_nf = a_n · V^n, V in PZT Vpp, p in Pa]")
    for n in HARMONICS:
        if np.isfinite(a[n]):
            print(f"  a_{n} = {a[n]:.4e}  Pa / Vpp^{n}")

    # ---- Plot ------------------------------------------------------------
    fw, fh = figsize_for_layout(3, 2)
    fig, axes = plt.subplots(3, 2, figsize=(fw, fh), sharex=False)
    flat = axes.ravel()
    v_dense = np.linspace(0, np.nanmax(v) * 1.05, 200)
    colors = {1: "C0", 2: "C5", 3: "C2", 4: "C4", 5: "C1"}

    # Panels 0..4: P_nf vs V with V^n fit
    for idx, n in enumerate(HARMONICS):
        ax = flat[idx]
        pn = P[n]
        ok = np.isfinite(pn)
        ax.plot(v[ok], pn[ok] / 1e3, "o-", markersize=4, linewidth=1.0,
                color=colors[n], label="data")
        if np.isfinite(a[n]):
            ax.plot(v_dense, a[n] * v_dense ** n / 1e3,
                    "--", linewidth=0.8, color="0.4",
                    label=rf"$a_{{{n}}} V^{{{n}}}$")
        ax.set_xlabel("PZT Vpp")
        ax.set_ylabel(rf"$P_{{{n}f}}$ peak [kPa]")
        ax.set_title(rf"${n}f$ harmonic")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Panel 5: ratios P_nf / P_1f
    ax = flat[5]
    p1 = P[1]
    for n in (2, 3, 4, 5):
        pn = P[n]
        ok = np.isfinite(pn) & np.isfinite(p1) & (p1 > 0)
        if ok.sum() == 0:
            continue
        ax.plot(v[ok], 100 * pn[ok] / p1[ok], "o-", markersize=4,
                linewidth=1.0, color=colors[n],
                label=rf"$P_{{{n}f}} / P_{{1f}}$")
    ax.set_xlabel("PZT Vpp")
    ax.set_ylabel(r"Ratio [\%]")
    ax.set_title("Harmonic ratios vs drive")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(rf"sample chip — harmonics 1f–5f vs PZT Vpp  "
                 rf"(amp gain $\approx$ {TRUE_AMP_GAIN:g}×)",
                 fontsize=10)
    plt.tight_layout()
    out_path = OUT_DIR / "vpp_vs_pressure.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
