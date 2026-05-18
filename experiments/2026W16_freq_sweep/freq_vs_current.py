# %%
"""Drive electrical + peak acoustic pressure vs frequency for the W16 v1 sweep.

Aggregates one TDMS per drive frequency, runs the shared FFT cache,
fits a ``|sin(pi y / W)|`` mode shape per frequency to get a
calibrated peak P_1f (kPa, with R^2), and plots five rows vs drive
frequency: I_1f, V_1f, |Z|=V/I, V-I phase, and peak P_1f.

Outputs (in ``output/``):
  * ``freq_vs_current.png`` -- 5-row sweep summary
  * ``mode_shapes/mode_<freq>kHz.png`` -- per-frequency 2-panel
    (amplitude + phase) plot of the data points + ``|sin(pi y/W)|``
    fit + channel walls.  Use these to sanity-check every fit.
  * ``mode_shapes_overview.png`` -- all per-frequency fits in one grid

Drive frequency is parsed from each filename (``...NNNNkHz...``).

Default dataset: ``260413_ldv/W16test2_20Vpp*kHz_5m_s_max.tdms`` -- fine
1f sweep covering 1880--2020 kHz around the W16 chip's 1f resonance,
constant 20 Vpp drive.  LDV velocity scale auto-detected from the
filename suffix ``_Nm_s_max`` -> N/2 m/s per V at 1 MOhm HiZ (per
io_utils._detect_velocity_scale_from_name); ``_5m_s_max`` -> 2.5 m/s/V,
matching the W21 YAML convention.

Channel-width assumption
------------------------
Uses ``CHANNEL_WIDTH = 375 um`` from ``ldv_analysis.config`` for the
mode fit.  If R^2 ends up low across the sweep, check the chip width.

Run from the repo root::

    python experiments/2026W16_freq_sweep/freq_vs_current.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo root on sys.path so this script works when run directly from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, figsize_for_layout,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import fit_mode_1f, fit_mode_2f  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\tatsuki\OneDrive - Lund University\Data\260413_ldv"
)
GLOB_PATTERN = "W16test2_20Vpp*kHz_5m_s_max.tdms"
# velocity_scale=None lets fft_cache.load_or_compute use
# io_utils._detect_velocity_scale_from_name (parses ``_Nm_s_max`` ->
# N/2 m/s per V at 1 MOhm HiZ).  For W16test2's ``_5m_s_max`` files
# that yields 2.5 m/s/V, matching the W21 YAML convention.
VELOCITY_SCALE = None
OUT_DIR = ROOT / "experiments" / "2026W16_freq_sweep" / "output"
CACHE_DIR = OUT_DIR / "fft_cache"
MODE_DIR = OUT_DIR / "mode_shapes"

# Filename frequency parser: matches `...NNNNkHz...`
FREQ_PATTERN = re.compile(r"(\d+)kHz")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob(GLOB_PATTERN))
    print(f"Found {len(files)} files matching {GLOB_PATTERN} in {DATA_DIR}")
    if not files:
        return

    freq_khz: list[int] = []
    f_detected: list[float] = []
    i_med: list[float] = []
    v_med: list[float] = []
    phase_med: list[float] = []
    p0_1f_kpa: list[float] = []
    p0_1f_phase: list[float] = []
    r2_1f: list[float] = []
    p0_2f_kpa: list[float] = []
    p0_2f_phase: list[float] = []
    r2_2f: list[float] = []
    n_valid: list[int] = []
    mode_data: list[dict] = []
    errors: list[tuple[str, str, str]] = []

    for p in files:
        m = FREQ_PATTERN.search(p.name)
        if not m:
            print(f"  skip (no kHz in name): {p.name}")
            continue
        fkhz = int(m.group(1))
        try:
            c = load_or_compute(p, CACHE_DIR, velocity_scale=VELOCITY_SCALE)
        except Exception as e:  # noqa: BLE001
            errors.append((p.name, type(e).__name__, str(e)))
            print(f"  ERROR {fkhz} kHz: {type(e).__name__}: {e}")
            continue
        if "current_1f" not in c.files:
            errors.append((p.name, "no_current_channel", ""))
            print(f"  {fkhz} kHz: no current_1f -- skip")
            continue

        pos = np.asarray(c["pos_x"])
        I = np.asarray(c["current_1f"])
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        ph_vi = np.asarray(c["phase_vi"]) if "phase_vi" in c.files else None

        valid = make_valid_mask(V, rssi)
        n_ok = int(np.sum(valid))
        if n_ok < 3:
            errors.append((p.name, "too_few_valid_points", str(n_ok)))
            print(f"  {fkhz} kHz: only {n_ok} valid points -- skip")
            continue

        # 1f mode fit: |sin(pi y/W)|
        P1_abs = np.asarray(c["pressure_1f"])
        P1_phase_deg = np.asarray(c["phase_1f"])
        P1_complex = P1_abs * np.exp(1j * np.radians(P1_phase_deg))
        result1 = fit_mode_1f(pos[valid], P1_complex[valid], CHANNEL_WIDTH)
        p0_1 = abs(result1.p0)
        ph_1 = float(np.degrees(np.angle(result1.p0)))

        # 2f mode fit: |cos(2 pi y/W)|, reuses the 1f-fitted channel center
        if "pressure_2f" in c.files and "phase_2f" in c.files:
            P2_abs = np.asarray(c["pressure_2f"])
            P2_phase_deg = np.asarray(c["phase_2f"])
            P2_complex = P2_abs * np.exp(1j * np.radians(P2_phase_deg))
            result2 = fit_mode_2f(
                pos[valid], P2_complex[valid], CHANNEL_WIDTH, result1.center
            )
            p0_2 = abs(result2.p0)
            ph_2 = float(np.degrees(np.angle(result2.p0)))
            r2_2 = result2.r2
        else:
            p0_2 = float("nan")
            ph_2 = float("nan")
            r2_2 = float("nan")

        freq_khz.append(fkhz)
        f_detected.append(float(c["f_drive"]))
        i_med.append(float(np.median(I[valid])))
        v_med.append(float(np.median(V[valid])))
        phase_med.append(
            float(np.median(ph_vi[valid])) if ph_vi is not None else float("nan")
        )
        p0_1f_kpa.append(p0_1 / 1e3)
        p0_1f_phase.append(ph_1)
        r2_1f.append(result1.r2)
        p0_2f_kpa.append(p0_2 / 1e3)
        p0_2f_phase.append(ph_2)
        r2_2f.append(r2_2)
        n_valid.append(n_ok)

        mode_data.append(dict(
            fkhz=fkhz,
            f_mhz=float(c["f_drive"]) / 1e6,
            p0=p0_1,
            p0_complex=result1.p0,
            r2=result1.r2,
            center=result1.center,
            y_c=pos[valid] - result1.center,
            p=P1_abs[valid],
            phase_1f=P1_phase_deg[valid],
        ))

        print(f"  {fkhz:4d} kHz (detected {float(c['f_drive'])/1e3:.2f}): "
              f"I={float(np.median(I[valid]))*1e3:6.2f} mA, "
              f"P1={p0_1/1e3:7.1f} kPa (R2={result1.r2:5.2f}), "
              f"P2={p0_2/1e3:6.1f} kPa (R2={r2_2:5.2f}), "
              f"valid {n_ok}/{I.size}")

    if not freq_khz:
        print("\nNo files processed successfully.")
        if errors:
            print(f"{len(errors)} errors total.")
        return

    order = np.argsort(freq_khz)
    f_mhz = np.array(f_detected)[order] / 1e6
    i_arr = np.array(i_med)[order] * 1e3       # mA
    v_arr = np.array(v_med)[order]
    z_arr = np.where(np.abs(i_med) > 0,
                     np.array(v_med) / np.array(i_med),
                     np.nan)[order]
    ph_arr = np.array(phase_med)[order]
    p1_arr = np.array(p0_1f_kpa)[order]
    ph1_arr = np.array(p0_1f_phase)[order]
    r2_1_arr = np.array(r2_1f)[order]
    p2_arr = np.array(p0_2f_kpa)[order]
    ph2_arr = np.array(p0_2f_phase)[order]
    r2_2_arr = np.array(r2_2f)[order]
    n_arr = np.array(n_valid)[order]
    mode_sorted = [mode_data[i] for i in order]

    # -----------------------------------------------------------------------
    # Plot 1: 8-row sweep summary
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        8, 1, figsize=figsize_for_layout(8, 1, sharex=True), sharex=True,
    )

    axes[0].plot(f_mhz, p1_arr, ".-", markersize=3, linewidth=0.8, color="C0")
    axes[0].set_ylabel(r"$P_{1f}$ [kPa]")
    axes[0].set_title(
        rf"v1 freq sweep --- W16test2 ({len(f_mhz)} files, 20 Vpp)"
    )

    axes[1].plot(f_mhz, p2_arr, ".-", markersize=3, linewidth=0.8, color="C5")
    axes[1].set_ylabel(r"$P_{2f}$ [kPa]")

    axes[2].plot(f_mhz, ph1_arr, ".-", markersize=3, linewidth=0.8, color="C0")
    axes[2].axhline(0, color="0.5", lw=0.5)
    axes[2].set_ylabel(r"$\angle P_{1f}$ [deg]")

    axes[3].plot(f_mhz, ph2_arr, ".-", markersize=3, linewidth=0.8, color="C5")
    axes[3].axhline(0, color="0.5", lw=0.5)
    axes[3].set_ylabel(r"$\angle P_{2f}$ [deg]")

    axes[4].plot(f_mhz, ph_arr, ".-", markersize=3, linewidth=0.8, color="C3")
    axes[4].axhline(0, color="0.5", lw=0.5)
    axes[4].set_ylabel(r"V--I phase [deg]")

    axes[5].plot(f_mhz, i_arr, ".-", markersize=3, linewidth=0.8, color="C2")
    axes[5].set_ylabel("Current [mA]")

    axes[6].plot(f_mhz, v_arr, ".-", markersize=3, linewidth=0.8, color="C1")
    axes[6].set_ylabel("Voltage [V]")

    axes[7].plot(f_mhz, z_arr, ".-", markersize=3, linewidth=0.8, color="C4")
    axes[7].set_ylabel(r"$|Z|$ [$\Omega$]")
    axes[7].set_xlabel("Frequency [MHz]")

    plt.tight_layout()
    out_path = OUT_DIR / "freq_vs_current.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # -----------------------------------------------------------------------
    # Plot 2: per-frequency mode-shape plots
    # -----------------------------------------------------------------------
    W = CHANNEL_WIDTH
    hw_mm = W / 2 * 1e3
    k_mode = np.pi / W
    y_fine_mm = np.linspace(-1.5 * hw_mm, 1.5 * hw_mm, 400)
    sin_fine_signed = np.sin(k_mode * y_fine_mm * 1e-3)
    sin_fine = np.abs(sin_fine_signed)

    for md in mode_sorted:
        fig, (ax, axp) = plt.subplots(
            1, 2, figsize=figsize_for_layout(1, 2), sharex=True,
        )
        p0_kpa_val = md["p0"] / 1e3
        inside = np.abs(md["y_c"]) <= W / 2

        ax.plot(md["y_c"][~inside] * 1e3, md["p"][~inside] / 1e3,
                "x", markersize=3, alpha=0.3, color="0.5")
        ax.plot(md["y_c"][inside] * 1e3, md["p"][inside] / 1e3,
                ".", markersize=3, alpha=0.6)
        r2_str = f"{md['r2']:.2f}" if md["r2"] > -10 else "<-10"
        ax.plot(y_fine_mm, p0_kpa_val * sin_fine, "--", linewidth=1, color="C3",
                label=rf"$P_0$ = {p0_kpa_val:.0f} kPa, $R^2$ = {r2_str}")
        ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.6)
        ax.axvline(hw_mm, color="0.5", ls=":", lw=0.6)
        ax.set_xlabel("Width position [mm]")
        ax.set_ylabel("Pressure [kPa]")
        ax.set_title(f"{md['fkhz']} kHz")
        ax.legend(fontsize=6, frameon=False)
        ax.set_ylim(bottom=0)

        axp.plot(md["y_c"][~inside] * 1e3, md["phase_1f"][~inside],
                 "x", markersize=3, alpha=0.3, color="0.5")
        axp.plot(md["y_c"][inside] * 1e3, md["phase_1f"][inside],
                 ".", markersize=3, alpha=0.6)
        phase_model = np.degrees(np.angle(md["p0_complex"] * sin_fine_signed))
        axp.plot(y_fine_mm, phase_model, "--", linewidth=1, color="C3")
        axp.axvline(-hw_mm, color="0.5", ls=":", lw=0.6)
        axp.axvline(hw_mm, color="0.5", ls=":", lw=0.6)
        axp.set_xlabel("Width position [mm]")
        axp.set_ylabel(r"Phase [$^\circ$]")
        axp.set_title("Phase (rel. voltage)")
        axp.set_ylim(-200, 200)

        plt.tight_layout()
        fig.savefig(MODE_DIR / f"mode_{md['fkhz']}kHz.png", dpi=FIG_DPI)
        plt.close(fig)
    print(f"Saved {len(mode_sorted)} per-frequency mode plots to {MODE_DIR}")

    # -----------------------------------------------------------------------
    # Plot 3: overview grid
    # -----------------------------------------------------------------------
    n = len(mode_sorted)
    ncols = 8
    nrows = (n + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(1.4 * ncols, 1.0 * nrows),
        sharex=True, sharey=True,
    )
    axes_flat = axes_grid.flatten() if nrows > 1 else axes_grid
    for i, md in enumerate(mode_sorted):
        ax = axes_flat[i]
        p0_kpa_val = md["p0"] / 1e3
        inside = np.abs(md["y_c"]) <= W / 2
        ax.plot(md["y_c"][~inside] * 1e3, md["p"][~inside] / 1e3,
                "x", markersize=1.5, alpha=0.3, color="0.5")
        ax.plot(md["y_c"][inside] * 1e3, md["p"][inside] / 1e3,
                ".", markersize=1.5, alpha=0.6)
        ax.plot(y_fine_mm, p0_kpa_val * sin_fine, "--", linewidth=0.6, color="C3")
        ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.4)
        ax.axvline(hw_mm, color="0.5", ls=":", lw=0.4)
        ax.set_title(f"{md['fkhz']}\n{p0_kpa_val:.0f} kPa "
                     f"R={md['r2']:.2f}", fontsize=4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=4)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.supxlabel("Width position [mm]", fontsize=6)
    fig.supylabel("Pressure [kPa]", fontsize=6)
    plt.tight_layout()
    overview_path = OUT_DIR / "mode_shapes_overview.png"
    fig.savefig(overview_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {overview_path}")

    # Tabular dump
    print(f"\n{'f_nom (kHz)':>11}  {'P1 (kPa)':>9}  {'R2_1':>5}  "
          f"{'P2 (kPa)':>9}  {'R2_2':>5}  "
          f"{'<P1 deg':>8}  {'<P2 deg':>8}  "
          f"{'I (mA)':>7}  {'V (V)':>6}  {'Z (Ohm)':>8}  N")
    for fk, p1, r1, p2, r2, a1, a2, im, vm, z, n in zip(
            np.array(freq_khz)[order],
            p1_arr, r2_1_arr, p2_arr, r2_2_arr,
            ph1_arr, ph2_arr, i_arr, v_arr, z_arr, n_arr):
        print(f"{fk:11d}  {p1:9.1f}  {r1:5.2f}  {p2:9.1f}  {r2:5.2f}  "
              f"{a1:8.1f}  {a2:8.1f}  {im:7.2f}  {vm:6.2f}  {z:8.1f}  {n}")

    i_best = int(np.argmax(p1_arr))
    print(f"\nPeak P1f = {p1_arr[i_best]:.1f} kPa at f = {f_mhz[i_best]:.3f} MHz")
    j_best = int(np.nanargmax(p2_arr))
    print(f"Peak P2f = {p2_arr[j_best]:.1f} kPa at f = {f_mhz[j_best]:.3f} MHz")

    if errors:
        print(f"\n{len(errors)} errors:")
        for n, e, msg in errors:
            print(f"  {n}: {e} {msg}")


if __name__ == "__main__":
    main()
