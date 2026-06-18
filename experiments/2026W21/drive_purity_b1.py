# %%
"""B1 drive-purity check: Ch A (drive voltage) and Ch D (current) harmonic content.

Per-Vpp question
----------------
The PRA draft cites V_2f/V_1f < 0.07% at 25 V on the drive channel.  The
cascade table shows a +29% deviation of measured P_2f/P_1f^2 from the
Coppens prediction at 10 Vpp (vs +7% at 30-60 V, -2% at 120 V), which
could be drive-leakage (2f present on the electrical drive itself) rather
than physics.  This script settles which by reading the actual Ch A
spectrum across the full 10-120 Vpp cascade.

Method
------
For each of the 12 canonical cascade scan directories
(``sample_101x21_fsweep_peak_<V>Vpp_<ts>``):

  - for every f.h5 file in the directory (11 drive frequencies)
    * select a steady-state burst window from a reference Ch A waveform
      (re-uses ``fft_cache.detect_burst_window`` and
      ``fft_cache.find_drive_frequency``)
    * load Ch A and Ch D for ~5 mid-scan points (these channels are
      electrical and approximately position-independent, so a small
      sample is enough)
    * exact-frequency DFT at n*f_drive for n=1..MAX_HARMONIC, ms-window
      scaled by ``2/ss_n``, then multiplied by the probe-attenuation /
      transimpedance scale factors from ``config``
  - aggregate per Vpp: median across (point, freq) for each harmonic
    on both Ch A and Ch D.

Outputs
-------
- ``output/drive_purity_b1/drive_purity_b1.png``
  4-panel: V_nf vs V (top), V_nf/V_1f vs V (next), I_nf vs V, I_nf/I_1f vs V
- ``output/drive_purity_b1/drive_purity_b1.csv``
  raw numbers for the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CURRENT_SCALE,
    FIG_DPI,
    LDV_DATA_ROOT,
    VOLTAGE_ATTENUATION,
    figsize_for_layout,
)
from ldv_analysis.fft_cache import (  # noqa: E402
    MAX_HARMONIC,
    detect_burst_window,
    find_drive_frequency,
)
from ldv_analysis.io_utils import (  # noqa: E402
    ROLE_CURRENT,
    ROLE_DRIVE_VOLTAGE,
    load_scan,
)

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
OUT_DIR = Path(__file__).resolve().parent / "output" / "drive_purity_b1"

# Points to sample per file.  Ch A / Ch D are electrical so they barely
# move with stage position; 5 centred points is plenty.
N_PROBE_POINTS = 5


def analyse_file(h5_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """FFT Ch A and Ch D at 1f..MAX_HARMONIC for one .h5 file.

    Returns
    -------
    V_n : (MAX_HARMONIC,) array of Vpp-side amplitudes (V, after attenuation)
    I_n : (MAX_HARMONIC,) array of current amplitudes (A)
    f_drive : float (Hz)
    """
    scan = load_scan(h5_path)
    dt = scan.dt
    n_pts = scan.n_points
    n_samp = scan.n_samples

    # Reference point near the middle for burst detection / freq estimation.
    ref_idx = n_pts // 2
    ref_wf = scan.load_waveforms(ROLE_DRIVE_VOLTAGE, np.array([ref_idx]))[0]
    bw = detect_burst_window(ref_wf, n_samp, dt)
    f_drive = find_drive_frequency(ref_wf, dt)

    # Pull a small set of points around the centre.  Ch A / Ch D are
    # electrical so position is irrelevant; we average to beat single-point
    # noise.
    half = N_PROBE_POINTS // 2
    sel = np.arange(
        max(0, ref_idx - half),
        min(n_pts, ref_idx - half + N_PROBE_POINTS),
    )

    wfA = scan.load_waveforms(ROLE_DRIVE_VOLTAGE, sel)
    wfD = scan.load_waveforms(ROLE_CURRENT, sel)

    ss_seg_A = wfA[:, bw.ss_start:bw.ss_end]
    ss_seg_D = wfD[:, bw.ss_start:bw.ss_end]
    ss_n = bw.ss_end - bw.ss_start
    ss_time = np.arange(ss_n) * dt

    V_n = np.empty(MAX_HARMONIC)
    I_n = np.empty(MAX_HARMONIC)
    for h in range(1, MAX_HARMONIC + 1):
        tone = np.exp(-2j * np.pi * h * f_drive * ss_time)
        dftA = ss_seg_A @ tone
        dftD = ss_seg_D @ tone
        # 2/ss_n -> peak amplitude (one-sided), then apply scale factors.
        # Average magnitudes over the probe points (median is more robust).
        V_n[h - 1] = float(
            np.median(np.abs(dftA)) * 2 / ss_n * VOLTAGE_ATTENUATION
        )
        I_n[h - 1] = float(
            np.median(np.abs(dftD)) * 2 / ss_n * CURRENT_SCALE
        )

    return V_n, I_n, f_drive


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nh = MAX_HARMONIC
    vpp_labels: list[int] = []
    V_per_vpp: list[np.ndarray] = []   # (n_freq, nh)
    I_per_vpp: list[np.ndarray] = []
    fdrive_per_vpp: list[np.ndarray] = []

    for label, dirname in SCANS:
        run_dir = DATA_ROOT / dirname
        files = sorted(run_dir.glob("f*.h5"))
        if not files:
            print(f"[{label:>3} Vpp] no .h5 in {run_dir}; skip")
            continue
        print(f"\n[{label:>3} Vpp] {dirname}  ({len(files)} files)")
        V_freqs = np.empty((len(files), nh))
        I_freqs = np.empty((len(files), nh))
        fdr = np.empty(len(files))
        for i, h5 in enumerate(files):
            print(f"  - {h5.name}")
            V_n, I_n, f_drive = analyse_file(h5)
            V_freqs[i] = V_n
            I_freqs[i] = I_n
            fdr[i] = f_drive
        vpp_labels.append(label)
        V_per_vpp.append(V_freqs)
        I_per_vpp.append(I_freqs)
        fdrive_per_vpp.append(fdr)

    vpp = np.asarray(vpp_labels, dtype=float)
    V_med = np.array([np.median(v, axis=0) for v in V_per_vpp])  # (n_vpp, nh)
    V_std = np.array([np.std(v, axis=0) for v in V_per_vpp])
    I_med = np.array([np.median(c, axis=0) for c in I_per_vpp])
    I_std = np.array([np.std(c, axis=0) for c in I_per_vpp])

    # ---- console table -----------------------------------------------------
    print("\nMedian harmonic content across the 11-freq sweep per Vpp")
    print(
        f"{'PZT Vpp':>8}  "
        + "  ".join(f"V_{h}f (V)".rjust(10) for h in range(1, nh + 1))
        + "  || "
        + "  ".join(f"V_{h}f/V_1f".rjust(10) for h in range(2, nh + 1))
    )
    for i, v in enumerate(vpp):
        row = f"{v:>8.0f}  " + "  ".join(f"{V_med[i, h]:>10.4f}" for h in range(nh))
        row += "  || "
        row += "  ".join(
            f"{V_med[i, h] / V_med[i, 0] * 100:>9.3f}%" for h in range(1, nh)
        )
        print(row)
    print(
        f"\n{'PZT Vpp':>8}  "
        + "  ".join(f"I_{h}f (mA)".rjust(10) for h in range(1, nh + 1))
        + "  || "
        + "  ".join(f"I_{h}f/I_1f".rjust(10) for h in range(2, nh + 1))
    )
    for i, v in enumerate(vpp):
        row = f"{v:>8.0f}  " + "  ".join(
            f"{I_med[i, h] * 1e3:>10.3f}" for h in range(nh)
        )
        row += "  || "
        row += "  ".join(
            f"{I_med[i, h] / I_med[i, 0] * 100:>9.3f}%" for h in range(1, nh)
        )
        print(row)

    # ---- CSV ---------------------------------------------------------------
    csv_path = OUT_DIR / "drive_purity_b1.csv"
    header = (
        "PZT_Vpp,"
        + ",".join(f"V_{h}f_V" for h in range(1, nh + 1))
        + ","
        + ",".join(f"I_{h}f_A" for h in range(1, nh + 1))
        + ","
        + ",".join(f"V_{h}f_over_V_1f" for h in range(2, nh + 1))
        + ","
        + ",".join(f"I_{h}f_over_I_1f" for h in range(2, nh + 1))
    )
    rows: list[str] = [header]
    for i, v in enumerate(vpp):
        parts = [f"{v:.0f}"]
        parts += [f"{V_med[i, h]:.6g}" for h in range(nh)]
        parts += [f"{I_med[i, h]:.6g}" for h in range(nh)]
        parts += [f"{V_med[i, h] / V_med[i, 0]:.6g}" for h in range(1, nh)]
        parts += [f"{I_med[i, h] / I_med[i, 0]:.6g}" for h in range(1, nh)]
        rows.append(",".join(parts))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True
    )
    colors = ["C0", "C1", "C2", "C3", "C4"]

    # (a) V_nf vs Vpp
    for h in range(nh):
        axes[0].plot(
            vpp,
            V_med[:, h],
            "o-",
            markersize=4,
            linewidth=0.8,
            color=colors[h],
            label=f"$V_{{{h+1}f}}$",
        )
    axes[0].set_ylabel(r"$V_{nf}$ on drive [V]")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False, ncol=5, loc="lower right")
    axes[0].set_title(
        f"B1 drive purity --- Ch A / Ch D harmonic content over the "
        f"10-120 Vpp cascade  (median over 11 sweep freqs, "
        f"{N_PROBE_POINTS} centre points per file)"
    )

    # (b) V_nf/V_1f vs Vpp
    for h in range(1, nh):
        axes[1].plot(
            vpp,
            V_med[:, h] / V_med[:, 0] * 100,
            "o-",
            markersize=4,
            linewidth=0.8,
            color=colors[h],
            label=f"$V_{{{h+1}f}}/V_{{1f}}$",
        )
    axes[1].set_ylabel(r"$V_{nf}/V_{1f}$  [\%]")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, frameon=False, ncol=4, loc="upper right")
    # PRA bound annotation
    axes[1].axhline(
        0.07,
        color="0.5",
        linewidth=0.5,
        linestyle="--",
        label="PRA <0.07% @ 25 V",
    )

    # (c) I_nf vs Vpp
    for h in range(nh):
        axes[2].plot(
            vpp,
            I_med[:, h] * 1e3,
            "o-",
            markersize=4,
            linewidth=0.8,
            color=colors[h],
            label=f"$I_{{{h+1}f}}$",
        )
    axes[2].set_ylabel(r"$I_{nf}$ [mA]")
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=7, frameon=False, ncol=5, loc="lower right")

    # (d) I_nf/I_1f vs Vpp
    for h in range(1, nh):
        axes[3].plot(
            vpp,
            I_med[:, h] / I_med[:, 0] * 100,
            "o-",
            markersize=4,
            linewidth=0.8,
            color=colors[h],
            label=f"$I_{{{h+1}f}}/I_{{1f}}$",
        )
    axes[3].set_ylabel(r"$I_{nf}/I_{1f}$  [\%]")
    axes[3].set_yscale("log")
    axes[3].set_xlabel(r"True PZT drive [Vpp]")
    axes[3].grid(True, which="both", alpha=0.3)
    axes[3].legend(fontsize=7, frameon=False, ncol=4, loc="upper right")

    plt.tight_layout()
    out_path = OUT_DIR / "drive_purity_b1.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
