# %%
"""Cross-frequency consistency of the 1f beat-fit transient Q1.

Sweeps the same averaged-envelope + simple/beat fit as
``transient_ch2_acoustic.py`` across every .h5 in one cascade scan,
and produces:
  - transient_q1_vs_fdrive_<scan>.{png,csv}   -- 3-panel summary
  - transient_q1_fits_panel_<scan>.png        -- per-file fit overlays

Crucially, the envelope cache path matches ``transient_ch2_acoustic.py``:
    output/<scan_dir>/transient/cache/_transient_env_ch2_<stem>.npz
so any envelope already computed by the standalone script is reused
unmodified, and any envelope computed by this sweep is reusable by
the standalone script.

Single-point envelopes are unstable off-resonance because the
steady-state amplitude collapses at the chosen antinode point, so we
use the amplitude-weighted average across all valid points (same as
the standalone script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    FIG_DPI, LDV_DATA_ROOT, figsize_for_layout, get_cache_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_point_waveforms  # noqa: E402
from ldv_analysis.io_utils import ROLE_LDV_OUTPUT, load_scan  # noqa: E402
from ldv_analysis.transient import (  # noqa: E402
    DF_THRESHOLD, FIT_SKIP_US, RISE_FIT_WINDOW_US,
    compute_fit_windows, detect_burst, estimate_beat_freq,
    load_transient_data, rise_beat_residual, rise_simple,
    sliding_dft_envelope, smooth_envelope, tau_to_Q,
)

DEFAULT_SCAN_DIR = (
    LDV_DATA_ROOT / "output" / "W21"
    / "sample_101x21_fsweep_peak_30Vpp_20260524_210338"
)
OUT_DIR = Path(__file__).resolve().parent / "output" / "transient_q1_vs_fdrive"


def script_out_dir(scan_dir: Path) -> Path:
    """Match the OUT_DIR layout of transient_ch2_acoustic.py exactly."""
    return (ROOT / "experiments" / "2026W21" / "output"
            / scan_dir.name / "transient")


def compute_or_load_env(h5_path: Path):
    """Same env-compute and cache schema as transient_ch2_acoustic.py.

    Cache key: <script_OUT_DIR>/<scan_dir>/transient/cache/_transient_env_ch2_<stem>.npz
    with fields: env_1f_complex, env_2f_complex, env_3f_complex,
    env_1f_std, env_2f_std, env_3f_std, best_1f_kPa, best_2f_kPa,
    best_ch1, n_used, p_ss, p_ss_2f, p_ss_3f.

    Returns (env_1f_complex, ss_start, ss_end, f1, dt, n_used, best_ch1).
    """
    out_dir = script_out_dir(h5_path.parent)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env_cache = cache_dir / f"_transient_env_ch2_{h5_path.stem}.npz"

    # Use FFT cache from the shared OneDrive location.
    fft_cache_dir = get_cache_dir(h5_path.parent.name, __file__)
    td = load_transient_data(h5_path, fft_cache_dir)

    if env_cache.exists():
        _ec = np.load(env_cache)
        print(f"  Loading env cache: {env_cache.name}")
        return (_ec["env_1f_complex"],
                int(td.ss_start), int(td.ss_end),
                float(td.f1), float(td.dt),
                int(_ec["n_used"]),
                _ec["best_ch1"])

    print(f"  Computing averaged envelopes (per-point)...")
    f1 = td.f1
    dt = td.dt
    n_samples = td.n_samples
    to_kPa = velocity_to_pressure(f1) / 1e3
    to_kPa_2f = velocity_to_pressure(2 * f1) / 1e3
    to_kPa_3f = abs(velocity_to_pressure(3 * f1)) / 1e3

    scan = load_scan(h5_path)
    valid_idx = np.where(td.valid)[0]

    env1_wsum = np.zeros(n_samples, dtype=complex)
    env2_wsum = np.zeros(n_samples, dtype=complex)
    env3_wsum = np.zeros(n_samples, dtype=complex)
    env1_sq_wsum = np.zeros(n_samples)
    env2_sq_wsum = np.zeros(n_samples)
    env3_sq_wsum = np.zeros(n_samples)
    weight_sum = 0.0
    n_used = 0

    CHUNK = 100
    for c0 in range(0, len(valid_idx), CHUNK):
        chunk_idx = valid_idx[c0:c0 + CHUNK]
        wfs = scan.load_waveforms(ROLE_LDV_OUTPUT, chunk_idx)
        for k in range(len(chunk_idx)):
            wf = wfs[k]
            ec1 = sliding_dft_envelope(wf, dt, f1, return_complex=True) * to_kPa
            ec2 = sliding_dft_envelope(wf, dt, 2 * f1, return_complex=True) * to_kPa_2f
            ec3 = sliding_dft_envelope(wf, dt, 3 * f1, return_complex=True) * to_kPa_3f
            p1c = np.mean(ec1[td.ss_start:td.ss_end])
            p1m = abs(p1c)
            if p1m <= 0:
                continue
            w = p1m
            env1_wsum += w * (ec1 / p1c)
            env1_sq_wsum += w * (np.abs(ec1) / p1m) ** 2
            p2c = np.mean(ec2[td.ss_start:td.ss_end])
            p2m = abs(p2c)
            if p2m > 0:
                env2_wsum += w * (ec2 / p2c)
                env2_sq_wsum += w * (np.abs(ec2) / p2m) ** 2
            p3c = np.mean(ec3[td.ss_start:td.ss_end])
            p3m = abs(p3c)
            if p3m > 0:
                env3_wsum += w * (ec3 / p3c)
                env3_sq_wsum += w * (np.abs(ec3) / p3m) ** 2
            weight_sum += w
            n_used += 1
    env1 = env1_wsum / weight_sum
    env2 = env2_wsum / weight_sum
    env3 = env3_wsum / weight_sum
    env1_std = np.sqrt(np.maximum(env1_sq_wsum / weight_sum - np.abs(env1) ** 2, 0))
    env2_std = np.sqrt(np.maximum(env2_sq_wsum / weight_sum - np.abs(env2) ** 2, 0))
    env3_std = np.sqrt(np.maximum(env3_sq_wsum / weight_sum - np.abs(env3) ** 2, 0))

    wfs_best, _ = load_point_waveforms(
        h5_path, td.best_i, roles=("drive_voltage", "ldv_output")
    )
    best_1f_kPa = sliding_dft_envelope(wfs_best["ldv_output"], dt, f1) * to_kPa
    best_2f_kPa = sliding_dft_envelope(wfs_best["ldv_output"], dt, 2 * f1) * to_kPa_2f
    p_ss = float(np.mean(best_1f_kPa[td.ss_start:td.ss_end]))
    p_ss_2f = float(np.mean(best_2f_kPa[td.ss_start:td.ss_end]))
    p_ss_3f = float(np.mean(np.abs(env3)[td.ss_start:td.ss_end]))
    best_ch1 = smooth_envelope(wfs_best["drive_voltage"])
    del scan

    np.savez(env_cache,
             env_1f_complex=env1, env_2f_complex=env2, env_3f_complex=env3,
             env_1f_std=env1_std, env_2f_std=env2_std, env_3f_std=env3_std,
             best_1f_kPa=best_1f_kPa, best_2f_kPa=best_2f_kPa,
             best_ch1=best_ch1,
             n_used=n_used, p_ss=p_ss, p_ss_2f=p_ss_2f, p_ss_3f=p_ss_3f)
    print(f"  Saved env cache: {env_cache.name}  (n_used={n_used})")
    return env1, int(td.ss_start), int(td.ss_end), float(f1), float(dt), n_used, best_ch1


def fit_rise(env_norm, f1, dt, n_samples, best_ch1):
    """Match the rise-fit block of transient_ch2_acoustic.py exactly."""
    burst_on, burst_off = detect_burst(best_ch1, dt)
    fw = compute_fit_windows(burst_on, burst_off, dt, n_samples)
    rs, re = fw["rise_start"], fw["rise_end"]

    t_us = np.arange(re - rs) * dt * 1e6
    ec = env_norm[rs:re]

    po, pcov = curve_fit(
        rise_simple, t_us, np.abs(ec),
        p0=[10.0], bounds=([0.1], [500.0]),
    )
    tau_simple = float(po[0])

    df_guess = estimate_beat_freq(ec - 1.0, t_us[1] - t_us[0])
    res = least_squares(
        rise_beat_residual,
        x0=[tau_simple, df_guess, 0.0],
        args=(t_us, ec.real, ec.imag),
        bounds=([0.1, -1e6, -np.pi], [500.0, 1e6, np.pi]),
        method="trf",
    )
    tau_beat, df_beat, phi_beat = res.x
    return dict(
        f_drive=f1,
        tau_simple=tau_simple,
        Q_simple=float(tau_to_Q(f1, tau_simple)),
        tau_beat=float(tau_beat),
        Q_beat=float(tau_to_Q(f1, tau_beat)),
        df_beat=float(df_beat),
        f_cavity=f1 + float(df_beat),
        phi_beat=float(phi_beat),
        beat_significant=bool(abs(df_beat) > DF_THRESHOLD),
        t_us=t_us,
        ec=ec,
        burst_on_us=burst_on * dt * 1e6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--scan-dir", default=str(DEFAULT_SCAN_DIR),
        help="Scan directory under LDV_DATA_ROOT/output/W21/",
    )
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(scan_dir.glob("f*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 in {scan_dir}")

    print(f"\nSweeping over {len(files)} files in {scan_dir.name}\n")
    rows = []
    for p in files:
        print(f"--- {p.name} ---")
        env, ss_s, ss_e, f1, dt, n_used, ch1 = compute_or_load_env(p)
        r = fit_rise(env, f1, dt, len(env), ch1)
        rows.append(r)
        print(f"  Q_simple={r['Q_simple']:6.1f}  Q_beat={r['Q_beat']:6.1f}  "
              f"df={r['df_beat']/1e3:+7.2f} kHz  "
              f"f_cav={r['f_cavity']/1e6:.4f} MHz"
              + ("" if r['beat_significant'] else "  (df ~0)"))

    f_drive = np.array([r["f_drive"] for r in rows])
    Q_simple = np.array([r["Q_simple"] for r in rows])
    Q_beat = np.array([r["Q_beat"] for r in rows])
    df_beat = np.array([r["df_beat"] for r in rows])
    f_cavity = np.array([r["f_cavity"] for r in rows])
    sig = np.array([r["beat_significant"] for r in rows])

    Q_beat_mean = float(np.mean(Q_beat[sig])) if sig.any() else float("nan")
    Q_beat_std = float(np.std(Q_beat[sig], ddof=1)) if int(sig.sum()) > 1 else float("nan")
    f_cav_mean = float(np.mean(f_cavity[sig])) if sig.any() else float("nan")
    f_cav_std = float(np.std(f_cavity[sig], ddof=1)) if int(sig.sum()) > 1 else float("nan")
    print(f"\n=== summary across {int(sig.sum())} sig + "
          f"{len(rows) - int(sig.sum())} non-sig ===")
    print(f"  f_cavity = {f_cav_mean/1e6:.4f}   std = {f_cav_std/1e3:.2f} kHz")
    print(f"  Q_beat   = {Q_beat_mean:.1f}   std = {Q_beat_std:.1f}")

    # ---- CSV --------------------------------------------------------------
    csv_path = OUT_DIR / f"transient_q1_vs_fdrive_{scan_dir.name}.csv"
    out_rows = [
        "f_drive_MHz,Q_simple,tau_simple_us,Q_beat,tau_beat_us,"
        "df_beat_kHz,f_cavity_MHz,beat_significant"
    ]
    for r in rows:
        out_rows.append(
            f"{r['f_drive']/1e6:.4f},{r['Q_simple']:.2f},"
            f"{r['tau_simple']:.3f},{r['Q_beat']:.2f},{r['tau_beat']:.3f},"
            f"{r['df_beat']/1e3:.3f},{r['f_cavity']/1e6:.4f},"
            f"{int(r['beat_significant'])}"
        )
    csv_path.write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")

    # ---- 3-panel summary --------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=figsize_for_layout(3, 1, sharex=True), sharex=True
    )
    ax = axes[0]
    ax.plot(f_drive / 1e6, Q_simple, "o-", markersize=4, linewidth=0.7,
            color="C3", label="simple exp fit")
    ax.plot(f_drive / 1e6, Q_beat, "s-", markersize=4, linewidth=0.7,
            color="C0", label="beat fit")
    if not np.isnan(Q_beat_std):
        ax.axhspan(Q_beat_mean - Q_beat_std, Q_beat_mean + Q_beat_std,
                   color="C0", alpha=0.12)
        ax.axhline(Q_beat_mean, color="C0", lw=0.5, ls="--",
                   label=fr"beat mean: ${Q_beat_mean:.0f} \pm "
                         fr"{Q_beat_std:.0f}$")
    ax.set_ylabel(r"$Q_1$")
    ax.set_ylim(0, max(180, Q_simple.max() * 1.1, Q_beat.max() * 1.1))
    ax.set_title(scan_dir.name)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(f_drive / 1e6, df_beat / 1e3, "o-", markersize=4, linewidth=0.7,
            color="C2")
    if not np.isnan(f_cav_mean):
        ax.plot(f_drive / 1e6, (f_cav_mean - f_drive) / 1e3, "--",
                color="0.4", linewidth=0.6,
                label=fr"$\Delta f = f_\mathrm{{cav}} - f_\mathrm{{drive}}$, "
                      fr"$f_\mathrm{{cav}}$ = {f_cav_mean/1e6:.4f} MHz")
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_ylabel(r"$\Delta f$ [kHz]")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f_drive / 1e6, f_cavity / 1e6, "o-", markersize=4, linewidth=0.7,
            color="C4")
    if not np.isnan(f_cav_mean):
        ax.axhline(f_cav_mean / 1e6, color="0.4", lw=0.6, ls="--",
                   label=fr"mean: {f_cav_mean/1e6:.4f} MHz "
                         fr"(std = {f_cav_std/1e3:.2f} kHz)")
    ax.set_xlabel("drive frequency [MHz]")
    ax.set_ylabel(r"recovered $f_\mathrm{cavity}$ [MHz]")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_summary = OUT_DIR / f"transient_q1_vs_fdrive_{scan_dir.name}.png"
    fig.savefig(out_summary, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_summary}")

    # ---- per-file rise-window fit overlays --------------------------------
    n = len(rows)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.8 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, r in enumerate(rows):
        ax = axes[i]
        t = r["t_us"]
        ec = r["ec"]
        ax.plot(t, np.abs(ec), ".", markersize=2, color="0.3",
                alpha=0.6, label="data $|E|$")
        ax.plot(t, 1 - np.exp(-t / r["tau_simple"]),
                "-", color="C3", linewidth=0.8,
                label=f"simple: $Q$={r['Q_simple']:.0f}")
        ax.plot(t, np.abs(1 - np.exp(-t / r["tau_beat"]) *
                          np.exp(1j * (2 * np.pi * r["df_beat"] * t * 1e-6
                                        + r["phi_beat"]))),
                "-", color="C0", linewidth=0.8,
                label=(f"beat: $Q$={r['Q_beat']:.0f}, "
                       f"$\\Delta f$={r['df_beat']/1e3:+.1f} kHz"))
        ax.set_title(f"f={r['f_drive']/1e6:.4f} MHz", fontsize=8)
        ax.set_ylim(0, max(1.4, np.max(np.abs(ec)) * 1.1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    for ax in axes[(nrow - 1) * ncol:nrow * ncol]:
        ax.set_xlabel(r"$t$ since burst ON [\textmu s]")
    for ax in axes[::ncol]:
        ax.set_ylabel(r"normalised $|E_{1f}(t)|$")
    fig.suptitle(f"Per-file rise-window fit overlays  ({scan_dir.name})",
                 fontsize=10)
    plt.tight_layout()
    out_panel = OUT_DIR / f"transient_q1_fits_panel_{scan_dir.name}.png"
    fig.savefig(out_panel, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {out_panel}")


if __name__ == "__main__":
    main()
