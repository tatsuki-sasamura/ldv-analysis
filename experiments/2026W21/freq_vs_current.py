# %%
"""Drive electrical + peak acoustic pressure vs frequency for the W21 v2 freq sweep.

Aggregates one HDF5 per drive frequency in a v2 run directory, runs the
shared FFT cache, fits a ``|sin(pi y / W)|`` mode shape per frequency to
get a calibrated peak P_1f (kPa, with R^2), and plots five rows vs
drive frequency: I_1f, V_1f, |Z|=V/I, V-I phase, and peak P_1f.

For a 2D area scan the mode is fit at *each axial position* (like
``pressure_map_2d.py``) and the reported P_1f is the value at the axial
slice that maximizes it; P_2f is taken at that same slice.  For a 1D
line scan (a single axial position) the original single lumped fit over
all valid points is used unchanged.

Outputs (in ``output/``):
  * ``freq_vs_current.png`` -- 5-row sweep summary
  * ``mode_shapes/mode_<freq>kHz.png`` -- per-frequency 2-panel
    (amplitude + phase) plot of the data points + ``|sin(pi y/W)|``
    fit + channel walls.  Use these to sanity-check every fit.
  * ``mode_shapes_overview.png`` -- all per-frequency fits in one grid

Frequency and ``ldv_velocity_scale_mps_per_v`` are read from each file's
HDF5 root attrs (canonical source); no hardcoded calibration constants.

Channel-width assumption
------------------------
The mode fit needs the channel width.  The ``sample`` chip has no
``chip_sample.json`` sidecar yet, so we fall back to the global
``CHANNEL_WIDTH = 375 um`` from ``ldv_analysis.config``.  If R^2 ends
up low across the sweep, that's the first thing to revisit.

Default dataset: ``output/sample_11x1_fsweep_10Vpp_20260518_005643/`` --
v2 11x1 line-scan fine 1f sweep from 1.900 to 2.000 MHz at 1 kHz steps,
chip_id="sample", 10 Vpp post-amp drive (0.125 Vpp AFG x 80).

Run from the repo root::

    python experiments/2026W21/freq_vs_current.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _read_amp_gain(run_dir: Path) -> float | None:
    """Read ``amplifier_gain_v_per_v`` from the snapshotted hardware.yaml.

    The DAQ writes one ``hardware.yaml`` per run directory at scan start.
    Returns the bench-measured amp gain (Vpp at PZT / Vpp at AFG) so we
    can convert the AFG-side ``drive_voltage_vpp`` (stored in each HDF5)
    into a true-PZT-side Vpp.  Returns None if the file or key is missing.

    NOTE: for runs prior to 2026-05-18 the snapshot may say ``80.0``
    even though the true gain was ~170 (a 10x scope probe was loaded
    by a BNC splitter feeding both Pico CH A and a Tek scope, which
    halved the probe's reading and made the bench-measured gain
    artifactually low).  To correct a historical snapshot in place,
    just edit the snapshotted ``hardware.yaml`` to the true gain.
    """
    hw_path = run_dir / "hardware.yaml"
    if not hw_path.exists():
        return None
    try:
        text = hw_path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"^\s*amplifier_gain_v_per_v:\s*([\d.]+)", text, re.M)
    return float(m.group(1)) if m else None

# Repo root on sys.path so this script works when run directly from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, figsize_for_layout,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.sweep_fit import fit_axial  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default if no --run-dir arg is given.  Outputs always land in
# experiments/2026W21/output/<run_dir.name>/ so multiple datasets
# can be analyzed in parallel without overwriting each other.
DEFAULT_RUN_DIR = Path(
    r"C:\Users\tatsuki\OneDrive - Lund University\Data\output"
    r"\W21\sample_101x1_fsweep_coarse_10Vpp_20260524_130731"
)


def main(run_dir: Path, save_mode_shapes: bool = False) -> None:
    out_dir = ROOT / "experiments" / "2026W21" / "output" / run_dir.name
    cache_dir = out_dir / "fft_cache"
    mode_dir = out_dir / "mode_shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if save_mode_shapes:
        mode_dir.mkdir(parents=True, exist_ok=True)

    # Per-run amp gain: AFG Vpp x amp_gain = PZT-side Vpp.  Read from
    # the snapshotted hardware.yaml so the conversion follows the
    # calibration in effect when the run was acquired.
    amp_gain = _read_amp_gain(run_dir)
    if amp_gain is None:
        print(f"Warning: no amplifier_gain_v_per_v in "
              f"{run_dir / 'hardware.yaml'} -- reporting AFG-side Vpp only")
    else:
        print(f"Amplifier gain (from snapshot): {amp_gain}x "
              f"-> PZT_Vpp = AFG_Vpp x {amp_gain}")

    # Skip aborted writes (`*.h5.inprogress`)
    files = sorted(p for p in run_dir.glob("*.h5")
                   if not p.name.endswith(".inprogress"))
    print(f"Found {len(files)} HDF5 files in {run_dir}")
    if not files:
        return

    freq_hz: list[float] = []
    i_med: list[float] = []
    v_med: list[float] = []
    phase_med: list[float] = []
    p0_1f_kpa: list[float] = []
    p0_1f_phase: list[float] = []   # deg, from fitted complex p0 (rel. drive)
    r2_1f: list[float] = []
    p0_2f_kpa: list[float] = []
    p0_2f_phase: list[float] = []
    r2_2f: list[float] = []
    # 1f magnitude refit against the n=2 (full-wavelength) mode shape -- catches
    # cross-channel n=2 cavity resonances that show up at the drive frequency.
    p0_1f_n2_kpa: list[float] = []
    r2_1f_n2: list[float] = []
    n_valid: list[int] = []
    vpp_nom: list[float] = []
    x_best_mm_list: list[float] = []
    mode_data: list[dict] = []
    errors: list[tuple[str, str, str]] = []

    # Channel geometry is detected once from the first 2D file and reused
    # across the sweep (every frequency shares the same physical scan grid).
    geom_ab: tuple[float, float] | None = None

    for p in files:
        try:
            with h5py.File(p, "r") as f:
                f_nom = float(f.attrs["drive_frequency_hz_nominal"])
                vpp = float(f.attrs.get("drive_voltage_vpp", float("nan")))
        except Exception as e:  # noqa: BLE001
            errors.append((p.name, type(e).__name__, str(e)))
            print(f"  ERROR opening {p.name}: {type(e).__name__}: {e}")
            continue

        try:
            c = load_or_compute(p, cache_dir, velocity_scale=None)
        except Exception as e:  # noqa: BLE001
            errors.append((p.name, type(e).__name__, str(e)))
            print(f"  ERROR FFT {p.name}: {type(e).__name__}: {e}")
            continue

        if "current_1f" not in c.files:
            errors.append((p.name, "no_current_channel", ""))
            print(f"  {f_nom/1e3:.1f} kHz: no current_1f -- skip")
            continue

        I = np.asarray(c["current_1f"])
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        ph_vi = np.asarray(c["phase_vi"]) if "phase_vi" in c.files else None

        valid = make_valid_mask(V, rssi)
        n_ok = int(np.sum(valid))
        if n_ok < 3:
            errors.append((p.name, "too_few_valid_points", str(n_ok)))
            print(f"  {f_nom/1e3:.1f} kHz: only {n_ok} valid points -- skip")
            continue

        # Per-axial fit for 2D scans, lumped fit for 1D line scans.
        # Geometry is detected on the first 2D file and reused across the
        # sweep (every frequency shares the same physical scan grid).
        try:
            fit, geom_ab = fit_axial(c, valid, CHANNEL_WIDTH, geom=geom_ab)
        except ValueError as e:
            errors.append((p.name, "no_axial_fit", str(e)))
            print(f"  {f_nom/1e3:.1f} kHz: {e} -- skip")
            continue

        p0_1 = fit.p1_mag
        ph_1 = float(np.degrees(np.angle(fit.p1_complex)))
        r2_1 = fit.r2_1
        p0_2 = fit.p2_mag
        ph_2 = float(np.degrees(np.angle(fit.p2_complex)))
        r2_2 = fit.r2_2
        x_best_mm = fit.x_best_mm
        md_p0c = fit.p1_complex
        md_center = fit.center
        md_y_c = fit.y_c
        md_p = fit.p
        md_phase = fit.phase_deg

        freq_hz.append(f_nom)
        i_med.append(float(np.median(I[valid])))
        v_med.append(float(np.median(V[valid])))
        phase_med.append(
            float(np.median(ph_vi[valid])) if ph_vi is not None else float("nan")
        )
        p0_1f_kpa.append(p0_1 / 1e3)
        p0_1f_phase.append(ph_1)
        r2_1f.append(r2_1)
        p0_2f_kpa.append(p0_2 / 1e3)
        p0_2f_phase.append(ph_2)
        r2_2f.append(r2_2)
        p0_1f_n2_kpa.append(fit.p1_n2_mag / 1e3)
        r2_1f_n2.append(fit.r2_p1_n2)
        n_valid.append(n_ok)
        vpp_nom.append(vpp)
        x_best_mm_list.append(x_best_mm)

        # Retain the best axial slice for the per-frequency mode-shape plots
        mode_data.append(dict(
            f_hz=f_nom,
            f_mhz=f_nom / 1e6,
            p0=p0_1,
            p0_complex=md_p0c,
            r2=r2_1,
            center=md_center,
            y_c=md_y_c,
            p=md_p,
            phase_1f=md_phase,
        ))

        xb_str = "  lumped" if np.isnan(x_best_mm) else f"x*={x_best_mm:6.2f}mm"
        print(f"  {f_nom/1e3:7.1f} kHz @ {vpp} Vpp: "
              f"I={float(np.median(I[valid]))*1e3:6.2f} mA, "
              f"P1={p0_1/1e3:7.1f} kPa (R2={r2_1:5.2f}), "
              f"P2={p0_2/1e3:6.1f} kPa (R2={r2_2:5.2f}), "
              f"P1_n2={fit.p1_n2_mag/1e3:6.0f} kPa (R2={fit.r2_p1_n2:5.2f}), "
              f"{xb_str}, valid {n_ok}/{I.size}")

    if not freq_hz:
        print("\nNo files processed successfully.")
        if errors:
            print(f"{len(errors)} errors total.")
        return

    order = np.argsort(freq_hz)
    f_mhz = np.array(freq_hz)[order] / 1e6
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
    p1_n2_arr = np.array(p0_1f_n2_kpa)[order]
    r2_1_n2_arr = np.array(r2_1f_n2)[order]
    n_arr = np.array(n_valid)[order]
    xb_arr = np.array(x_best_mm_list)[order]
    mode_sorted = [mode_data[i] for i in order]

    # 1D scans report a single lumped fit; 2D scans report the per-axial
    # peak slice.  Reflect which method was used in the plot title.
    fit_method = ("lumped $|\\sin|$ fit"
                  if np.all(np.isnan(xb_arr))
                  else "peak-over-axial fit")

    # -----------------------------------------------------------------------
    # Plot 1: 8-row sweep summary
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        9, 1, figsize=figsize_for_layout(9, 1, sharex=True), sharex=True,
    )

    axes[0].plot(f_mhz, p1_arr, ".-", markersize=3, linewidth=0.8, color="C0")
    axes[0].set_ylabel(r"$P_{1f}$ [kPa]")
    if len(set(vpp_nom)) == 1:
        afg_vpp = vpp_nom[0]
        if amp_gain is not None:
            pzt_vpp = afg_vpp * amp_gain
            vpp_str = (f"${afg_vpp}$ Vpp AFG $\\times {amp_gain}$ "
                       f"= ${pzt_vpp:.2f}$ Vpp PZT")
        else:
            vpp_str = f"${afg_vpp}$ Vpp AFG (no amp_gain)"
    else:
        vpp_str = "(mixed Vpp)"
    title_name = run_dir.name.replace("_", r"\_")
    axes[0].set_title(
        rf"v2 freq sweep --- {title_name}  "
        rf"({len(f_mhz)} files, {vpp_str}, {fit_method})"
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

    # 1f magnitude refit against the n=2 (full-wavelength) shape -- peaks
    # here mark cross-channel n=2 cavity resonances driven at 1f, which
    # the standard P_1f (n=1 fit) panel above would miss / show as junk.
    axes[8].plot(f_mhz, p1_n2_arr, ".-", markersize=3, linewidth=0.8, color="C6")
    axes[8].set_ylabel(r"$P_{1f}$ (n=2 fit) [kPa]")
    axes[8].set_xlabel("Frequency [MHz]")

    plt.tight_layout()
    out_path = out_dir / "freq_vs_current.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # -----------------------------------------------------------------------
    # Plot 2: per-frequency mode-shape plots (2-panel amplitude + phase).
    # Optional -- can dominate wallclock on wide-band surveys (461 PNGs etc.),
    # gated behind --mode-shapes so it's only run when needed.
    # -----------------------------------------------------------------------
    if not save_mode_shapes:
        print(f"Skipping per-frequency mode plots "
              f"(pass --mode-shapes to enable)")
    else:
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
                    "x", markersize=4, alpha=0.4, color="0.5")
            ax.plot(md["y_c"][inside] * 1e3, md["p"][inside] / 1e3,
                    ".", markersize=5, alpha=0.7)
            r2_str = f"{md['r2']:.2f}" if md["r2"] > -10 else "<-10"
            ax.plot(y_fine_mm, p0_kpa_val * sin_fine, "--", linewidth=1,
                    color="C3",
                    label=rf"$P_0$ = {p0_kpa_val:.0f} kPa, $R^2$ = {r2_str}")
            ax.axvline(-hw_mm, color="0.5", ls=":", lw=0.6)
            ax.axvline(hw_mm, color="0.5", ls=":", lw=0.6)
            ax.set_xlabel("Width position [mm]")
            ax.set_ylabel("Pressure [kPa]")
            ax.set_title(f"{md['f_mhz']*1e3:.0f} kHz")
            ax.legend(fontsize=6, frameon=False)
            ax.set_ylim(bottom=0)

            axp.plot(md["y_c"][~inside] * 1e3, md["phase_1f"][~inside],
                     "x", markersize=4, alpha=0.4, color="0.5")
            axp.plot(md["y_c"][inside] * 1e3, md["phase_1f"][inside],
                     ".", markersize=5, alpha=0.7)
            phase_model = np.degrees(
                np.angle(md["p0_complex"] * sin_fine_signed))
            axp.plot(y_fine_mm, phase_model, "--", linewidth=1, color="C3")
            axp.axvline(-hw_mm, color="0.5", ls=":", lw=0.6)
            axp.axvline(hw_mm, color="0.5", ls=":", lw=0.6)
            axp.set_xlabel("Width position [mm]")
            axp.set_ylabel(r"Phase [$^\circ$]")
            axp.set_title("Phase (rel. voltage)")
            axp.set_ylim(-200, 200)

            plt.tight_layout()
            fig.savefig(mode_dir / f"mode_{md['f_mhz']*1e3:.0f}kHz.png",
                        dpi=FIG_DPI)
            plt.close(fig)
        print(f"Saved {len(mode_sorted)} per-frequency mode plots to "
              f"{mode_dir}")

    # Tabular dump
    print(f"\n{'f (MHz)':>9}  {'P1 (kPa)':>9}  {'R2_1':>5}  "
          f"{'P2 (kPa)':>9}  {'R2_2':>5}  "
          f"{'<P1 deg':>8}  {'<P2 deg':>8}  "
          f"{'I (mA)':>7}  {'V (V)':>6}  {'Z (Ohm)':>8}  {'x* (mm)':>8}  N")
    for f, p1, r1, p2, r2, a1, a2, im, vm, z, xb, n in zip(
            f_mhz, p1_arr, r2_1_arr, p2_arr, r2_2_arr,
            ph1_arr, ph2_arr, i_arr, v_arr, z_arr, xb_arr, n_arr):
        xb_s = "  lumped" if np.isnan(xb) else f"{xb:8.2f}"
        print(f"{f:9.3f}  {p1:9.1f}  {r1:5.2f}  {p2:9.1f}  {r2:5.2f}  "
              f"{a1:8.1f}  {a2:8.1f}  {im:7.2f}  {vm:6.2f}  {z:8.1f}  {xb_s}  {n}")

    i_best = int(np.argmax(p1_arr))
    print(f"\nPeak P1f = {p1_arr[i_best]:.1f} kPa at f = {f_mhz[i_best]:.3f} MHz")
    j_best = int(np.nanargmax(p2_arr))
    print(f"Peak P2f = {p2_arr[j_best]:.1f} kPa at f = {f_mhz[j_best]:.3f} MHz")
    # Report the n=2-fit peak only where the n=2 shape is genuinely the
    # better fit (R2_n2 > R2_n1 AND R2_n2 >= 0.5).  In the n=1-dominant
    # band, projecting 1f onto the n=2 shape still gives some R^2 from
    # spatial overlap, so a bare R^2>=0.5 filter snags the n=1 peak; the
    # n2_better_than_n1 condition picks only n=2-dominant resonances.
    n2_real = (np.isfinite(p1_n2_arr) & np.isfinite(r2_1_n2_arr)
               & (r2_1_n2_arr >= 0.5) & (r2_1_n2_arr > r2_1_arr))
    if np.any(n2_real):
        masked = np.where(n2_real, p1_n2_arr, -np.inf)
        k_best = int(np.argmax(masked))
        print(f"Peak P1f (n=2 dominant) = {p1_n2_arr[k_best]:.1f} kPa at f = "
              f"{f_mhz[k_best]:.3f} MHz "
              f"(R2_n2={r2_1_n2_arr[k_best]:.2f}, "
              f"R2_n1={r2_1_arr[k_best]:.2f})")
    else:
        print("Peak P1f (n=2 dominant) = none "
              "(no frequency where the n=2 shape beats the n=1 fit)")

    if errors:
        print(f"\n{len(errors)} errors:")
        for n, e, msg in errors:
            print(f"  {n}: {e} {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR,
        help=(f"Path to a v2 scan directory (contains one HDF5 per "
              f"drive frequency).  Defaults to: {DEFAULT_RUN_DIR}"),
    )
    parser.add_argument(
        "--mode-shapes", action="store_true",
        help=("Save one mode-shape PNG per frequency under mode_shapes/. "
              "Off by default -- can dominate wallclock on wide-band "
              "surveys (one matplotlib save per frequency)."),
    )
    args = parser.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {args.run_dir}")
    main(args.run_dir, save_mode_shapes=args.mode_shapes)
