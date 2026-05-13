"""Shared scan-data → figure-data extraction for PRA and AF figures.

Centralizes the heavy data-extraction logic (mode fits, voltage sweep,
2D maps) used by both ``manuscript_figures.py`` and
``af2026_figures.py``. Each top-level function returns a plain dict
ready to be saved to a `.npz` cache.

Format-agnostic: input paths are routed through ``load_or_compute``
which dispatches to TDMS or HDF5 v2 readers by file extension. The
module contains no plotting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    C_SOUND,
    RHO,
    RSSI_THRESHOLD,
    channel_centre_func,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.filters import make_voltage_mask
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.mode_fit import fit_columns, fit_mode_1f, fit_mode_2f


# ---------------------------------------------------------------------------
# Voltage-sweep inner loop
# ---------------------------------------------------------------------------

def compute_voltage_sweep_results(
    voltage_files: Sequence[tuple[Path, int]],
    cache_dir: Path,
    geom: dict,
) -> list[dict]:
    """Run the per-Vpp data-extraction loop (format-agnostic via load_or_compute) and return one dict per file.

    For each ``(scan_path, vpp)`` pair:
      - load FFT cache (computes from scan file if absent)
      - mode-fit 1f and 2f profiles in y, get p0(x), σ(x)
      - whole-channel ⟨E_ac⟩ and peak p0 at the axial antinode
      - drive electrical power P_in
      - Ch1 2f/1f drive-harmonics ratio
    """
    hw = CHANNEL_WIDTH / 2
    centre_fn = channel_centre_func(geom)

    results: list[dict] = []
    for scan_path, vpp in voltage_files:
        scan_path = Path(scan_path)
        if not scan_path.exists():
            print(f"  SKIP (not found): {scan_path.name}")
            continue

        cache = load_or_compute(scan_path, cache_dir)
        pos_x = cache["pos_x"]
        pos_y = cache["pos_y"]
        n_x_meta = int(cache["n_x_meta"])
        n_y_meta = int(cache["n_y_meta"])
        f_drive = float(cache["f_drive"])
        pressure_1f = cache["pressure_1f"]

        pos_x_c = pos_x - centre_fn(pos_y)
        inside = np.abs(pos_x_c) <= hw

        rssi = cache["rssi"] if "rssi" in cache else None
        cg = make_channel_grid(
            pos_x_c, pos_y, n_x_meta, n_y_meta,
            CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
            rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
        )

        grid_prs_1f = cg.to_grid(pressure_1f)
        p0_1f_y, sig_1f_y = fit_columns(
            grid_prs_1f, cg.width_grid, CHANNEL_WIDTH,
            harmonic=1, return_sigma=True,
        )
        Eac_1f = np.nanmean(p0_1f_y**2) / (4 * RHO * C_SOUND**2)

        pressure_2f = cache["pressure_2f"]
        grid_prs_2f = cg.to_grid(pressure_2f)
        p0_2f_y, sig_2f_y = fit_columns(
            grid_prs_2f, cg.width_grid, CHANNEL_WIDTH,
            harmonic=2, return_sigma=True,
        )
        Eac_2f = np.nanmean(p0_2f_y**2) / (4 * RHO * C_SOUND**2)

        V = cache["voltage_1f"]
        if "current_1f" not in cache or "phase_vi" not in cache:
            raise ValueError(
                f"{scan_path.name}: drive current ('current' role) is required "
                f"by figure_data — Fig 5/8 use it for P_in = ½VIcos(φ). "
                f"Re-acquire including the current channel, or skip this file."
            )
        I = cache["current_1f"]
        phase_vi = cache["phase_vi"]
        valid_e = make_voltage_mask(V)
        P_in = (
            0.5 * float(np.median(V[valid_e])) * float(np.median(I[valid_e]))
            * np.cos(np.radians(float(np.median(phase_vi[valid_e]))))
        )

        j_anti = int(np.nanargmax(p0_1f_y))
        p0_1f_peak = float(p0_1f_y[j_anti])
        p0_2f_peak = float(p0_2f_y[j_anti])
        p0_1f_std = float(sig_1f_y[j_anti])
        p0_2f_std = float(sig_2f_y[j_anti])
        Eac_1f_std = 2 * p0_1f_peak * p0_1f_std / (4 * RHO * C_SOUND**2)

        # Drive-voltage harmonics at the mid-point
        n_points = len(pos_x)
        mid_pt = n_points // 2
        wf_ch1, dt = load_point_waveforms(
            scan_path, mid_pt, roles=("drive_voltage",)
        )
        ch1 = wf_ch1["drive_voltage"]
        ss_start = int(cache["ss_start"])
        ss_end = int(cache["ss_end"])
        ss_n = ss_end - ss_start
        ch1_ss = ch1[ss_start:ss_end]
        tone_1f = np.exp(-2j * np.pi * f_drive * np.arange(ss_n) * dt)
        tone_2f = np.exp(-2j * np.pi * 2 * f_drive * np.arange(ss_n) * dt)
        ch1_1f_amp = np.abs(ch1_ss @ tone_1f) * 2 / ss_n
        ch1_2f_amp = np.abs(ch1_ss @ tone_2f) * 2 / ss_n
        ch1_ratio = ch1_2f_amp / ch1_1f_amp * 100

        results.append(dict(
            vpp=vpp, Eac_1f=Eac_1f, Eac_1f_std=Eac_1f_std,
            Eac_2f=Eac_2f, P_in=P_in,
            p0_1f_peak=p0_1f_peak, p0_2f_peak=p0_2f_peak,
            p0_1f_std=p0_1f_std, p0_2f_std=p0_2f_std,
            p0_1f_y=p0_1f_y, p0_2f_y=p0_2f_y,
            ch1_ratio=ch1_ratio,
            f_drive=f_drive, scan_path=scan_path,
            cache=cache, cg=cg,
        ))
        print(f"  {scan_path.name}: p0_1f = {p0_1f_peak/1e3:.0f} kPa, "
              f"p0_2f = {p0_2f_peak/1e3:.0f} kPa, "
              f"Ch1 2f/1f = {ch1_ratio:.2f}%, "
              f"P_in = {P_in*1e3:.1f} mW")
    return results


# ---------------------------------------------------------------------------
# Fig 8 cache (drive-resolved harmonics)
# ---------------------------------------------------------------------------

def extract_fig8_data(results: list[dict]) -> dict:
    """Pack voltage-sweep results into the Fig8.npz dict.

    Includes through-origin linear/quadratic fits (a_1f, b_2f, ratio_slope).
    """
    Vpp = np.array([r["vpp"] for r in results])
    Eac_1f_arr = np.array([r["Eac_1f"] for r in results])
    P_in_arr = np.array([r["P_in"] for r in results])
    p0_1f_peak_arr = np.array([r["p0_1f_peak"] for r in results])
    p0_2f_peak_arr = np.array([r["p0_2f_peak"] for r in results])
    p0_1f_std_arr = np.array([r["p0_1f_std"] for r in results])
    p0_2f_std_arr = np.array([r["p0_2f_std"] for r in results])
    ch1_ratio_arr = np.array([r["ch1_ratio"] for r in results])

    Vpp_0 = np.array([0] + list(Vpp))
    p0_1f_0 = np.array([0] + list(p0_1f_peak_arr))
    p0_2f_0 = np.array([0] + list(p0_2f_peak_arr))
    a_1f = float(np.sum(Vpp_0 * p0_1f_0) / np.sum(Vpp_0**2))
    b_2f = float(np.sum(Vpp_0**2 * p0_2f_0) / np.sum(Vpp_0**4))
    ratio_slope = b_2f / a_1f

    return dict(
        Vpp=Vpp,
        p0_1f=p0_1f_peak_arr, p0_2f=p0_2f_peak_arr,
        p0_1f_std=p0_1f_std_arr, p0_2f_std=p0_2f_std_arr,
        E_ac=Eac_1f_arr, P_in=P_in_arr,
        ratio=p0_2f_peak_arr / p0_1f_peak_arr,
        ch1_ratio_pct=ch1_ratio_arr,
        ch1_2f_1f=ch1_ratio_arr / 100,
        a_1f=a_1f, b_2f=b_2f, ratio_slope=ratio_slope,
    )


# ---------------------------------------------------------------------------
# Fig 7 cache (spatial mode profiles + 2D maps)
# ---------------------------------------------------------------------------

def extract_fig7_data(
    results: list[dict],
    fig7_vpps: Sequence[int] = (10, 25),
) -> dict:
    """Pack mode-shape profiles and 2D maps into the Fig7.npz dict.

    Fits 1f and 2f mode shapes (complex LSQ at antinode column) at the
    requested Vpp levels. The 2D maps come from the highest Vpp in
    ``results`` (typically 25).
    """
    hw = CHANNEL_WIDTH / 2

    # 25-Vpp is the last entry in the voltage sweep; antinode column there
    r_peak = results[-1]
    j_best = int(np.nanargmax(r_peak["p0_1f_y"]))
    cg_peak = r_peak["cg"]
    y_best = cg_peak.length_grid[j_best]

    y_th = np.linspace(-hw, hw, 200)
    mode_1f_th = np.abs(np.sin(np.pi * y_th / CHANNEL_WIDTH))
    mode_2f_th = np.abs(np.cos(2 * np.pi * y_th / CHANNEL_WIDTH))

    fig7_results = [r for r in results if r["vpp"] in fig7_vpps]

    cache_data: dict = {
        "y_best": y_best,
        "j_best": j_best,
        "y_th_um": y_th * 1e6,
        "n_vpp": len(fig7_results),
    }

    for i, r in enumerate(fig7_results):
        cg_i = r["cg"]
        w_grid = cg_i.width_grid

        grid_1f = cg_i.to_grid(r["cache"]["pressure_1f"])
        grid_2f = cg_i.to_grid(r["cache"]["pressure_2f"])
        grid_ph1 = cg_i.to_grid(r["cache"]["phase_1f"])
        grid_ph2 = cg_i.to_grid(r["cache"]["phase_2f"])

        p1f_row = grid_1f[:, j_best]
        p2f_row = grid_2f[:, j_best]
        ph1_row = grid_ph1[:, j_best]
        ph2_row = grid_ph2[:, j_best]
        p1f_complex = p1f_row * np.exp(1j * np.radians(ph1_row))
        p2f_complex = p2f_row * np.exp(1j * np.radians(ph2_row))

        valid = ~np.isnan(p1f_row)
        res_1f = fit_mode_1f(w_grid[valid], p1f_complex[valid], CHANNEL_WIDTH,
                              center=0.0)
        res_2f = fit_mode_2f(w_grid[valid], p2f_complex[valid], CHANNEL_WIDTH,
                              center=0.0)
        p0_1f = abs(res_1f.p0)
        p0_2f = abs(res_2f.p0)

        fit_1f_mpa = p0_1f / 1e6 * mode_1f_th
        fit_2f_mpa = p0_2f / 1e6 * mode_2f_th
        y_um = w_grid * 1e6

        cache_data[f"vpp_{i}"] = r["vpp"]
        cache_data[f"y_um_{i}"] = y_um
        cache_data[f"p1f_mpa_{i}"] = p1f_row / 1e6
        cache_data[f"p2f_mpa_{i}"] = p2f_row / 1e6
        cache_data[f"fit_1f_mpa_{i}"] = fit_1f_mpa
        cache_data[f"fit_2f_mpa_{i}"] = fit_2f_mpa
        cache_data[f"p0_1f_{i}"] = p0_1f
        cache_data[f"p0_2f_{i}"] = p0_2f
        cache_data[f"r2_1f_{i}"] = res_1f.r2
        cache_data[f"r2_2f_{i}"] = res_2f.r2

        print(f"  {r['vpp']:2d} Vpp: p0_1f = {p0_1f/1e3:.0f} kPa (R²={res_1f.r2:.3f}), "
              f"p0_2f = {p0_2f/1e3:.0f} kPa (R²={res_2f.r2:.3f})")

    # 2D maps from the peak (25 Vpp)
    grid_1f_25 = cg_peak.to_grid(r_peak["cache"]["pressure_1f"]) / 1e6
    grid_2f_25 = cg_peak.to_grid(r_peak["cache"]["pressure_2f"]) / 1e6
    cache_data["grid_1f_25_mpa"] = grid_1f_25
    cache_data["grid_2f_25_mpa"] = grid_2f_25
    cache_data["w_mm"] = cg_peak.width_grid * 1e3
    cache_data["l_mm"] = cg_peak.length_grid * 1e3

    return cache_data
