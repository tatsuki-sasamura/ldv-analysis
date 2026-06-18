"""A1 + A2: harmonic ladder (P_nf, n=1..5) and per-harmonic SNR vs drive.

Measurement-only, cache-only.  For each of the 12 cascade scans
(101x21, 10-120 Vpp PZT) this extracts, per harmonic n = 1..5, the peak
modal pressure across the frequency sweep -- using the *same* transverse
mode-shape projection (``fit_columns``) that ``vpp_vs_pressure.py`` /
``sweep_fit`` use for n = 1, 2, extended to n = 3, 4, 5 -- together with
the post-burst noise floor referenced to each harmonic's frequency.

The point of this pass is to decide *how far up the ladder the signal is
real*: that sets which Q_n we actually need to identify later (if 3f is
the top real harmonic, Q_4 is irrelevant).

Outputs (to experiments/2026W21/output/):
  - harmonic_ladder.png : (1) P_nf vs PZT Vpp with noise floors,
    (2) SNR_n vs Vpp, (3) P_nf/P_1f ratio vs Vpp.
  - harmonic_ladder.npz : every array, for replotting / structure
    planning without recompute.

No .h5 is opened: ``load_or_compute`` serves the cached FFT (the .h5 are
online-only on this machine), and the drive voltage comes from the
sidecar YAMLs via ``vpp_vs_pressure.read_drive_vpp``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import

from vpp_vs_pressure import (  # noqa: E402
    DATA_ROOT,
    SCANS,
    read_amp_gain,
    read_drive_vpp,
)

from ldv_analysis.config import (  # noqa: E402
    C_SOUND,
    CHANNEL_WIDTH,
    RHO,
    RSSI_THRESHOLD,
    figsize_for_layout,
    get_cache_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry, fit_columns  # noqa: E402

OUT_DIR = ROOT / "experiments" / "2026W21" / "output"
HARMONICS = (1, 2, 3, 4, 5)


def extract_scan(run_dir: Path, cache_dir: Path) -> dict:
    """Co-located modal P_nf and noise floor per harmonic across one sweep.

    Returns dict keyed by harmonic n -> (peak_Pa, freq_hz, noise_Pa).
    All harmonics are read at the **1f antinode** (the axial slice that
    maximizes P_1f, per frequency), matching ``sweep_fit``/F2 so the
    ratios and energy budget are spatially consistent; the value is then
    maximized over the frequency sweep.
    """
    files = sorted(p for p in run_dir.glob("*.h5") if not p.name.endswith(".inprogress"))
    hw = CHANNEL_WIDTH / 2
    geom = None
    cg = None

    # per-harmonic running best across the frequency sweep
    best = {n: (np.nan, np.nan, np.nan) for n in HARMONICS}  # (peak, f, noise)
    meas_vpp = np.nan  # measured PZT drive Vpp at the 1f-resonance file

    for p in files:
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        f_drive = float(c["f_drive"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(np.asarray(c["voltage_1f"]), rssi)
        if int(np.sum(valid)) < 3:
            continue

        pos = np.asarray(c["pos_x"])
        pos_y = np.asarray(c["pos_y"])
        n_x, n_y = int(c["n_x_meta"]), int(c["n_y_meta"])

        # Geometry + grid detected once per scan (shared physical grid).
        if geom is None:
            p1g = np.where(valid, np.asarray(c["pressure_1f"]), np.nan)
            geom = detect_channel_geometry(pos, pos_y, rssi, p1g, hw)
            a_opt, b_opt = geom
            pos_x_c = pos - (a_opt * pos_y + b_opt)
            cg = make_channel_grid(
                pos_width_c=pos_x_c,
                pos_length=pos_y,
                n_scan_width=n_x,
                n_scan_length=n_y,
                channel_width=CHANNEL_WIDTH,
                raw_width_span=pos.max() - pos.min(),
                inside=np.abs(pos_x_c) <= hw,
                rssi=rssi,
                rssi_threshold=RSSI_THRESHOLD,
            )

        noise_v = np.asarray(c["noise_rms_velocity"])
        noise_v_med = float(np.nanmedian(noise_v[valid]))
        vpp_file = 2.0 * float(np.median(np.asarray(c["voltage_1f"])[valid]))

        # Co-location slice: the 1f antinode (axial slice maximizing P_1f),
        # so every harmonic is read at the same point (matches F2).
        g1 = cg.to_grid(np.where(valid, np.asarray(c["pressure_1f"]), np.nan))
        p0_y1 = fit_columns(g1, cg.width_grid, CHANNEL_WIDTH, harmonic=1, sigma_clip=3.0)
        if np.all(np.isnan(p0_y1)):
            continue
        bs = int(np.nanargmax(p0_y1))

        for n in HARMONICS:
            key = f"pressure_{n}f"
            if key not in c.files:
                continue
            grid = cg.to_grid(np.where(valid, np.asarray(c[key]), np.nan))
            p0_y = fit_columns(grid, cg.width_grid, CHANNEL_WIDTH, harmonic=n, sigma_clip=3.0)
            val = float(p0_y[bs])
            if np.isnan(val):
                continue
            if np.isnan(best[n][0]) or val > best[n][0]:
                noise_p = noise_v_med * abs(velocity_to_pressure(n * f_drive))
                best[n] = (val, f_drive, noise_p)
                if n == 1:
                    meas_vpp = vpp_file

    return best, meas_vpp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_h = len(HARMONICS)

    pzt_vpp = np.full(len(SCANS), np.nan)  # nominal (AFG x gain)
    pzt_meas = np.full(len(SCANS), np.nan)  # measured (2 x median voltage_1f)
    p_kpa = np.full((len(SCANS), n_h), np.nan)
    noise_kpa = np.full((len(SCANS), n_h), np.nan)
    res_mhz = np.full((len(SCANS), n_h), np.nan)

    hdr = "label  PZT(V)  " + "  ".join(f"P{n}f(kPa) SNR{n}" for n in HARMONICS)
    print(hdr)
    for i, (label, dirname) in enumerate(SCANS):
        run_dir = DATA_ROOT / dirname
        cache_dir = get_cache_dir(dirname, __file__)
        try:
            best, meas_vpp = extract_scan(run_dir, cache_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"{label:>5}  skipped ({e})")
            continue

        afg = read_drive_vpp(run_dir)
        gain = read_amp_gain(run_dir)
        true_vpp = afg * gain if (afg is not None and gain is not None) else float(label)
        pzt_vpp[i] = true_vpp
        pzt_meas[i] = meas_vpp

        cells = []
        for j, n in enumerate(HARMONICS):
            peak, f_hz, noise = best[n]
            p_kpa[i, j] = peak / 1e3
            noise_kpa[i, j] = noise / 1e3
            res_mhz[i, j] = f_hz / 1e6
            snr = peak / noise if (noise and noise > 0) else np.nan
            cells.append(f"{peak/1e3:8.1f} {snr:5.0f}")
        print(f"{label:>5}  {true_vpp:6.1f}  " + "  ".join(cells))

    snr = p_kpa / noise_kpa
    ratio = p_kpa / p_kpa[:, [0]]  # P_nf / P_1f
    p1_pa = p_kpa[:, 0] * 1e3
    eac_1f = p1_pa**2 / (4 * RHO * C_SOUND**2)

    np.savez(
        OUT_DIR / "harmonic_ladder.npz",
        harmonics=np.array(HARMONICS),
        pzt_vpp=pzt_vpp,  # nominal PZT Vpp (AFG x gain = folder label)
        pzt_vpp_meas=pzt_meas,  # measured PZT Vpp (2 x median voltage_1f);
        # ~0.92 x nominal, uniform across the cascade
        p_kpa=p_kpa,  # (n_scan, 5) peak modal pressure
        noise_kpa=noise_kpa,  # (n_scan, 5) noise floor at n*f
        snr=snr,  # (n_scan, 5)
        res_freq_mhz=res_mhz,  # (n_scan, 5) resonance freq per harmonic
        ratio_to_p1=ratio,  # (n_scan, 5) P_nf / P_1f
        eac_1f=eac_1f,  # (n_scan,) J/m^3
        scan_dirs=np.array([d for _, d in SCANS]),
        channel_width=np.array(CHANNEL_WIDTH),
        rho=np.array(RHO),
        c_sound=np.array(C_SOUND),
    )
    print(f"\nSaved {OUT_DIR / 'harmonic_ladder.npz'}")

    # ---- Figure: strength / SNR / ratio --------------------------------
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize_for_layout(3, 1, sharex=True),
        sharex=True,
    )

    for j, n in enumerate(HARMONICS):
        col = colors[j]
        axes[0].semilogy(pzt_vpp, p_kpa[:, j], "o-", ms=4, lw=0.9, color=col, label=f"$P_{{{n}f}}$")
        axes[0].semilogy(pzt_vpp, noise_kpa[:, j], ":", lw=0.8, color=col, alpha=0.6)
    axes[0].set_ylabel(r"peak $P_{nf}$ [kPa]")
    axes[0].set_title("Harmonic ladder --- 101x21 cascade (dotted = noise floor at $nf$)")
    axes[0].legend(fontsize=7, ncol=5, frameon=False)
    axes[0].grid(True, which="both", alpha=0.3)

    for j, n in enumerate(HARMONICS):
        axes[1].semilogy(
            pzt_vpp, snr[:, j], "o-", ms=4, lw=0.9, color=colors[j], label=f"$P_{{{n}f}}$"
        )
    axes[1].axhline(1, color="0.4", lw=0.8, ls="--")
    axes[1].axhline(10, color="0.7", lw=0.6, ls=":")
    axes[1].text(pzt_vpp[0], 1.1, "SNR = 1", fontsize=6, color="0.4")
    axes[1].set_ylabel("SNR  ($P_{nf}$ / noise)")
    axes[1].grid(True, which="both", alpha=0.3)

    for j, n in enumerate(HARMONICS):
        if n == 1:
            continue
        axes[2].semilogy(
            pzt_vpp,
            ratio[:, j] * 100,
            "o-",
            ms=4,
            lw=0.9,
            color=colors[j],
            label=f"$P_{{{n}f}}/P_{{1f}}$",
        )
    axes[2].set_ylabel(r"$P_{nf}/P_{1f}$ [\%]")
    axes[2].set_xlabel("PZT drive [Vpp]")
    axes[2].legend(fontsize=7, ncol=4, frameon=False)
    axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_png = OUT_DIR / "harmonic_ladder.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
