"""Harmonic ladder P1f..P5f at one common operating point per drive level.

For each 10--120 Vpp cascade scan this:

1. selects the drive-frequency file that maximizes the fitted P1f mode;
2. selects the axial column that maximizes P1f in that same file;
3. extracts P1f...P5f from that same file and same axial column.

This avoids combining independently maximized harmonics from different
drive frequencies or axial positions, so the ratios describe one
physically simultaneous acoustic field (the 1f operating point).  This
is the canonical ladder; ``harmonic_scaling`` / ``harmonic_cascade`` /
``harmonic_energy_coverage`` / ``prl_draft`` read its npz.

Note vs the older max-over-frequency convention: because the 2f
generation peaks slightly below the 1f resonance (2 f_drive hitting the
~3.79 MHz 2f eigenmode), P2f read at the 1f-peak drive is a few %
lower than its own max -> P2f/P1f ~ 25 % (not ~30 %) at 120 Vpp.  That
is the operating-point ratio and differs from vpp_vs_pressure peak_p2.

Per-harmonic 1-sigma error is sqrt(fit-SE^2 + noise^2) on the modal
amplitude (same as the previous amp_std).

Cache-only.  Outputs harmonic_ladder.{npz,csv,png} in output/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpp_vs_pressure import DATA_ROOT, SCANS, read_amp_gain, read_drive_vpp  # noqa: E402

from ldv_analysis.config import (  # noqa: E402
    C_SOUND,
    CHANNEL_WIDTH,
    RHO,
    RSSI_THRESHOLD,
    get_cache_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.mode_fit import _mode_shape, _project, _r2  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry, fit_columns  # noqa: E402

OUT_DIR = ROOT / "experiments" / "2026W21" / "output"
HARMONICS = (1, 2, 3, 4, 5)
SIGMA_CLIP = 3.0


def _build_grid(cache, valid):
    pos_x = np.asarray(cache["pos_x"])
    pos_y = np.asarray(cache["pos_y"])
    rssi = np.asarray(cache["rssi"]) if "rssi" in cache.files else None
    half_width = CHANNEL_WIDTH / 2

    p1 = np.where(valid, np.asarray(cache["pressure_1f"]), np.nan)
    a_opt, b_opt = detect_channel_geometry(pos_x, pos_y, rssi, p1, half_width)
    pos_x_centered = pos_x - (a_opt * pos_y + b_opt)

    return make_channel_grid(
        pos_width_c=pos_x_centered,
        pos_length=pos_y,
        n_scan_width=int(cache["n_x_meta"]),
        n_scan_length=int(cache["n_y_meta"]),
        channel_width=CHANNEL_WIDTH,
        raw_width_span=float(pos_x.max() - pos_x.min()),
        inside=np.abs(pos_x_centered) <= half_width,
        rssi=rssi,
        rssi_threshold=RSSI_THRESHOLD,
    )


def _fit_column(col, width_grid, harmonic, noise_pa):
    """Modal amplitude p0 + sqrt(fit-SE^2 + noise^2) std + R^2 at one column."""
    finite = np.isfinite(col)
    if int(np.sum(finite)) < 3:
        return np.nan, np.nan, np.nan

    mode = _mode_shape(width_grid[finite], CHANNEL_WIDTH, harmonic, use_abs=True)
    p0, kept = _project(col[finite], mode, sigma_clip=SIGMA_CLIP)
    n_kept = int(np.sum(kept))
    sum_mode2 = float(np.sum(mode[kept] ** 2))
    if n_kept < 2 or sum_mode2 <= 0:
        return float(p0), np.nan, np.nan

    prediction = p0 * mode
    r2 = float(_r2(col[finite][kept], prediction[kept]))

    residual = col[finite][kept] - prediction[kept]
    residual_sd = float(np.sqrt(np.sum(residual**2) / max(n_kept - 1, 1)))
    fit_se = residual_sd / np.sqrt(sum_mode2)
    noise_se = float(noise_pa) / np.sqrt(sum_mode2)
    p0_std = float(np.sqrt(fit_se**2 + noise_se**2))
    return float(p0), p0_std, r2


def extract_scan(run_dir: Path, cache_dir: Path):
    files = sorted(p for p in run_dir.glob("*.h5") if not p.name.endswith(".inprogress"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {run_dir}")

    grid = None
    selected = None

    # First pass: select the P1f-resonance file and its P1f-best column.
    for path in files:
        cache = load_or_compute(path, cache_dir, velocity_scale=None)
        rssi = np.asarray(cache["rssi"]) if "rssi" in cache.files else None
        valid = make_valid_mask(np.asarray(cache["voltage_1f"]), rssi)
        if int(np.sum(valid)) < 3:
            continue

        if grid is None:
            grid = _build_grid(cache, valid)

        p1_grid = grid.to_grid(np.where(valid, np.asarray(cache["pressure_1f"]), np.nan))
        p1_by_axial = fit_columns(
            p1_grid, grid.width_grid, CHANNEL_WIDTH, harmonic=1, sigma_clip=SIGMA_CLIP
        )
        if np.all(np.isnan(p1_by_axial)):
            continue

        best_axial = int(np.nanargmax(p1_by_axial))
        peak_p1 = float(p1_by_axial[best_axial])
        if selected is None or peak_p1 > selected["peak_p1"]:
            selected = {
                "path": path,
                "peak_p1": peak_p1,
                "best_axial": best_axial,
                "f_drive": float(cache["f_drive"]),
            }

    if grid is None or selected is None:
        raise ValueError(f"No usable file found in {run_dir}")

    # Second pass: extract every harmonic from the same file and column.
    cache = load_or_compute(selected["path"], cache_dir, velocity_scale=None)
    rssi = np.asarray(cache["rssi"]) if "rssi" in cache.files else None
    valid = make_valid_mask(np.asarray(cache["voltage_1f"]), rssi)
    best_axial = int(selected["best_axial"])
    f_drive = float(selected["f_drive"])

    noise_velocity = np.asarray(cache["noise_rms_velocity"])
    noise_velocity_median = float(np.nanmedian(noise_velocity[valid]))

    p_pa = np.full(len(HARMONICS), np.nan)
    p_std_pa = np.full(len(HARMONICS), np.nan)
    noise_pa = np.full(len(HARMONICS), np.nan)
    r2 = np.full(len(HARMONICS), np.nan)

    for j, harmonic in enumerate(HARMONICS):
        key = f"pressure_{harmonic}f"
        if key not in cache.files:
            continue

        pressure_grid = grid.to_grid(np.where(valid, np.asarray(cache[key]), np.nan))
        column = pressure_grid[:, best_axial]
        harmonic_noise = noise_velocity_median * abs(velocity_to_pressure(harmonic * f_drive))
        p0, p0_std, fit_r2 = _fit_column(column, grid.width_grid, harmonic, harmonic_noise)
        p_pa[j] = p0
        p_std_pa[j] = p0_std
        noise_pa[j] = harmonic_noise
        r2[j] = fit_r2

    measured_vpp = 2.0 * float(np.nanmedian(np.asarray(cache["voltage_1f"])[valid]))
    axial_mm = float(grid.length_grid[best_axial] * 1e3)

    return {
        "p_pa": p_pa,
        "p_std_pa": p_std_pa,
        "noise_pa": noise_pa,
        "snr": p_pa / noise_pa,
        "r2": r2,
        "f_drive_hz": f_drive,
        "measured_vpp": measured_vpp,
        "axial_mm": axial_mm,
        "selected_file": selected["path"].name,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_scans = len(SCANS)
    n_harmonics = len(HARMONICS)

    nominal_vpp = np.full(n_scans, np.nan)
    measured_vpp = np.full(n_scans, np.nan)
    f_drive_mhz = np.full(n_scans, np.nan)
    axial_mm = np.full(n_scans, np.nan)
    p_kpa = np.full((n_scans, n_harmonics), np.nan)
    p_std_kpa = np.full_like(p_kpa, np.nan)
    noise_kpa = np.full_like(p_kpa, np.nan)
    snr = np.full_like(p_kpa, np.nan)
    r2 = np.full_like(p_kpa, np.nan)
    selected_files = []

    for i, (label, dirname) in enumerate(SCANS):
        run_dir = DATA_ROOT / dirname
        cache_dir = get_cache_dir(dirname, __file__)
        result = extract_scan(run_dir, cache_dir)

        afg_vpp = read_drive_vpp(run_dir)
        gain = read_amp_gain(run_dir)
        nominal_vpp[i] = (
            afg_vpp * gain if (afg_vpp is not None and gain is not None) else float(label)
        )
        measured_vpp[i] = result["measured_vpp"]
        f_drive_mhz[i] = result["f_drive_hz"] / 1e6
        axial_mm[i] = result["axial_mm"]
        p_kpa[i] = result["p_pa"] / 1e3
        p_std_kpa[i] = result["p_std_pa"] / 1e3
        noise_kpa[i] = result["noise_pa"] / 1e3
        snr[i] = result["snr"]
        r2[i] = result["r2"]
        selected_files.append(result["selected_file"])

        ratios = p_kpa[i] / p_kpa[i, 0]
        print(
            f"{label:>3} Vpp: f={f_drive_mhz[i]:.4f} MHz, x={axial_mm[i]:.2f} mm, "
            f"P2/P1={ratios[1]*100:.2f}%, P3/P1={ratios[2]*100:.2f}%"
        )

    ratio_to_p1 = p_kpa / p_kpa[:, [0]]
    local_energy = (p_kpa * 1e3) ** 2 / (4 * RHO * C_SOUND**2)

    np.savez(
        OUT_DIR / "harmonic_ladder.npz",
        harmonics=np.asarray(HARMONICS),
        pzt_vpp=nominal_vpp,  # nominal (AFG x gain) -- downstream x-axis
        pzt_vpp_meas=measured_vpp,  # measured (2 x median voltage_1f)
        f_drive_mhz=f_drive_mhz,
        axial_mm=axial_mm,
        p_kpa=p_kpa,
        p_std_kpa=p_std_kpa,
        noise_kpa=noise_kpa,
        snr=snr,
        r2=r2,
        ratio_to_p1=ratio_to_p1,
        eac_1f=local_energy[:, 0],  # J/m^3 (consumed by harmonic_cascade)
        local_energy_j_m3=local_energy,
        selected_files=np.asarray(selected_files),
        channel_width=np.asarray(CHANNEL_WIDTH),
        rho=np.asarray(RHO),
        c_sound=np.asarray(C_SOUND),
    )

    csv_path = OUT_DIR / "harmonic_ladder.csv"
    header = ["nominal_vpp", "measured_vpp", "f_drive_mhz", "axial_mm", "selected_file"]
    for n in HARMONICS:
        header.extend(
            [
                f"P{n}f_kpa",
                f"P{n}f_std_kpa",
                f"P{n}f_noise_kpa",
                f"P{n}f_snr",
                f"P{n}f_r2",
                f"P{n}f_over_P1f",
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(n_scans):
            row = [nominal_vpp[i], measured_vpp[i], f_drive_mhz[i], axial_mm[i], selected_files[i]]
            for j in range(n_harmonics):
                row.extend(
                    [
                        p_kpa[i, j],
                        p_std_kpa[i, j],
                        noise_kpa[i, j],
                        snr[i, j],
                        r2[i, j],
                        ratio_to_p1[i, j],
                    ]
                )
            writer.writerow(row)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)
    for j, n in enumerate(HARMONICS):
        axes[0].errorbar(
            measured_vpp,
            p_kpa[:, j],
            yerr=p_std_kpa[:, j],
            marker="o",
            markersize=3,
            linewidth=0.8,
            capsize=2,
            label=rf"$P_{{{n}f}}$",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$P_{nf}$ [kPa]")
    axes[0].set_title("Same drive frequency and axial column at each voltage (1f operating point)")
    axes[0].legend(ncol=5, frameon=False)
    axes[0].grid(True, which="both", alpha=0.3)

    for j, n in enumerate(HARMONICS[1:], start=1):
        axes[1].plot(
            measured_vpp,
            ratio_to_p1[:, j] * 100,
            marker="o",
            markersize=3,
            linewidth=0.8,
            label=rf"$P_{{{n}f}}/P_{{1f}}$",
        )
    axes[1].set_ylabel("ratio [%]")
    axes[1].legend(ncol=2, frameon=False)
    axes[1].grid(True, alpha=0.3)

    for j, n in enumerate(HARMONICS):
        axes[2].plot(
            measured_vpp, r2[:, j], marker="o", markersize=3, linewidth=0.8, label=rf"{n}f"
        )
    axes[2].axhline(0.9, linestyle="--", linewidth=0.7)
    axes[2].set_xlabel("measured PZT drive [Vpp]")
    axes[2].set_ylabel(r"mode-fit $R^2$ (inlier)")
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].legend(ncol=5, frameon=False)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "harmonic_ladder.png", dpi=180)
    plt.close(fig)
    print("\nSaved harmonic_ladder.{npz,csv,png}")


if __name__ == "__main__":
    main()
