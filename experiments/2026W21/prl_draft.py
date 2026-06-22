"""PRL-draft Fig 1 (spatial modes) + Fig 2a (drive sweep) for W21, to 3f.

Follows the structure of ``experiments/2026W10_stepA/af2026_figures.py``
Fig 1 and Fig 2(a), but with W21 sample-chip data, extended to **3
harmonics** (1f, 2f, 3f):
  - Fig 1 : 120 Vpp scan at its P_1f-peak frequency (2x3: |P_nf(x,y)|
    maps on top, width-profile + |sin/cos(n pi y/W)| fit on the bottom).
  - Fig 2a: 10-120 Vpp cascade, twin **linear** y-axes (P_1f [MPa] left,
    P_2f & P_3f [kPa] right), with perturbative power-law guides
    P_nf ~ V^n fit through the lowest above-noise points.

Cache-only.  Fig 2a reads harmonic_ladder.npz (run harmonic_ladder.py
first).  Outputs prl_draft_fig1.{png,npz}, prl_draft_fig2a.{png,npz}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonic_mode_shapes import build_grid  # noqa: E402
from vpp_vs_pressure import DATA_ROOT  # noqa: E402

from ldv_analysis.config import (  # noqa: E402
    C_SOUND,
    CHANNEL_WIDTH,
    RHO,
    get_cache_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import _mode_shape, _r2  # noqa: E402
from ldv_analysis.sweep_fit import fit_columns  # noqa: E402

SCAN_120 = "sample_101x21_fsweep_peak_120Vpp_20260525_020136"
HARMS = (1, 2, 3)
SIGMA = 3.0
OUT_DIR = ROOT / "experiments" / "2026W21" / "output"
LADDER_NPZ = OUT_DIR / "harmonic_ladder.npz"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 0.75,
    }
)
COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:purple", 5: "tab:orange"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}


def _resonance(run_dir, cache_dir):
    files = sorted(p for p in run_dir.glob("*.h5") if not p.name.endswith(".inprogress"))
    cg = None
    best = (-1.0, None)
    for p in files:
        c = load_or_compute(p, cache_dir, velocity_scale=None)
        valid = make_valid_mask(
            np.asarray(c["voltage_1f"]),
            np.asarray(c["rssi"]) if "rssi" in c.files else None,
        )
        if int(np.sum(valid)) < 3:
            continue
        if cg is None:
            cg = build_grid(c, valid)
        g1 = cg.to_grid(np.where(valid, np.asarray(c["pressure_1f"]), np.nan))
        peak = float(
            np.nanmax(fit_columns(g1, cg.width_grid, CHANNEL_WIDTH, harmonic=1, sigma_clip=SIGMA))
        )
        if peak > best[0]:
            best = (peak, p)
    c = load_or_compute(best[1], cache_dir, velocity_scale=None)
    valid = make_valid_mask(
        np.asarray(c["voltage_1f"]),
        np.asarray(c["rssi"]) if "rssi" in c.files else None,
    )
    return cg, c, valid, float(c["f_drive"])


def fig1() -> None:
    cache_dir = get_cache_dir(SCAN_120, __file__)
    cg, c, valid, f_res = _resonance(DATA_ROOT / SCAN_120, cache_dir)
    print(f"Fig1: 120 Vpp resonance file f_drive = {f_res/1e6:.4f} MHz")

    length_mm = cg.length_grid * 1e3
    length_mm = length_mm - length_mm.mean()
    width_mm = cg.width_grid * 1e3

    # Single operating point: pick the 1f-best axial column once and read
    # every harmonic at that same column (matches harmonic_ladder.py).
    g1 = cg.to_grid(np.where(valid, np.asarray(c["pressure_1f"]), np.nan))
    bi = int(
        np.nanargmax(fit_columns(g1, cg.width_grid, CHANNEL_WIDTH, harmonic=1, sigma_clip=SIGMA))
    )
    noise_v_grid = cg.to_grid(np.where(valid, np.asarray(c["noise_rms_velocity"]), np.nan))

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0))
    saved = {}
    for j, n in enumerate(HARMS):
        grid_pa = cg.to_grid(np.where(valid, np.asarray(c[f"pressure_{n}f"]), np.nan))
        grid_mpa = grid_pa / 1e6

        ax = axes[0, j]
        lo, hi = np.nanpercentile(grid_mpa, [5, 95])
        im = ax.pcolormesh(
            length_mm, width_mm, grid_mpa, shading="nearest", cmap="viridis", vmin=lo, vmax=hi
        )
        ax.axvline(length_mm[bi], color="red", lw=0.8, ls="--")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.set_aspect("auto")
        ax.set_title(f"({chr(97+j)}) $|P_{{{n}f}}|$", fontsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(f"$|P_{{{n}f}}|$ [MPa]", fontsize=8)

        col = grid_pa[:, bi]
        cv = np.isfinite(col)
        w_mm = cg.width_grid[cv] * 1e3
        y = col[cv]
        mode = _mode_shape(cg.width_grid[cv], CHANNEL_WIDTH, n, use_abs=True)
        sig = noise_v_grid[:, bi][cv] * abs(velocity_to_pressure(n * f_res))  # per-pt noise [Pa]

        # All points are fit/plotted; a single-pass 5-sigma (robust, MAD)
        # guard excludes only lone instrument glitches from the fit/R^2
        # (still drawn, marked x).  Reduced chi^2 uses the per-point noise.
        p0_all = np.sum(y * mode) / np.sum(mode**2)
        resid0 = y - p0_all * mode
        sd_rob = 1.4826 * np.median(np.abs(resid0 - np.median(resid0)))
        if sd_rob <= 0:
            sd_rob = float(resid0.std())
        inlier = np.abs(resid0) <= 5.0 * sd_rob
        n_flag = int(np.sum(~inlier))
        p0 = float(np.sum(y[inlier] * mode[inlier]) / np.sum(mode[inlier] ** 2))
        r2 = _r2(y[inlier], p0 * mode[inlier])
        gs = inlier & np.isfinite(sig) & (sig > 0)
        dof = max(int(np.sum(gs)) - 1, 1)
        chi2nu = float(np.sum(((y[gs] - p0 * mode[gs]) / sig[gs]) ** 2) / dof)

        axp = axes[1, j]
        axp.plot(w_mm[inlier], y[inlier] / 1e6, "ko", ms=2)
        if n_flag:
            axp.plot(
                w_mm[~inlier],
                y[~inlier] / 1e6,
                "x",
                ms=5,
                color="C3",
                label=f"{n_flag} flagged ($>5\\sigma$)",
            )
        xf = np.linspace(cg.width_grid[0], cg.width_grid[-1], 300)
        axp.plot(
            xf * 1e3,
            p0 / 1e6 * _mode_shape(xf, CHANNEL_WIDTH, n, use_abs=True),
            "-",
            color="C3",
            lw=0.9,
            label=rf"$R^2$={r2:.2f}, $\chi^2_\nu$={chi2nu:.1f}",
        )
        axp.set_xlabel(r"$y$ [mm]")
        axp.set_ylabel(f"$|P_{{{n}f}}|$ [MPa]")
        axp.set_ylim(bottom=0)
        axp.set_title(f"({chr(100+j)}) $\\hat{{P}}_{{{n}f}}$ = {p0/1e3:.0f} kPa", fontsize=9)
        axp.legend(frameon=False, handlelength=1.2)
        saved[f"p0_{n}f_kpa"] = p0 / 1e3
        saved[f"r2_{n}f"] = r2
        saved[f"chi2nu_{n}f"] = chi2nu
        saved[f"nflag_{n}f"] = n_flag

    fig.suptitle(
        f"PRL draft Fig 1 (W21, to 3f) --- 120 Vpp, " f"$f_{{1f}}$ = {f_res/1e6:.3f} MHz",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "prl_draft_fig1.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    np.savez(OUT_DIR / "prl_draft_fig1.npz", f_res_mhz=f_res / 1e6, **saved)
    print(f"  Saved {out}")
    for n in HARMS:
        print(
            f"  {n}f: P = {saved[f'p0_{n}f_kpa']:.0f} kPa, R^2 = {saved[f'r2_{n}f']:.3f} "
            f"(chi2nu = {saved[f'chi2nu_{n}f']:.1f}, {saved[f'nflag_{n}f']} flagged >5sig)"
        )


PERTURB_RATIO_MAX = 0.10  # common perturbative window: P2f/P1f <= this


def _perturb_window(P):
    """Common perturbative window shared by all harmonics: the drive
    points with P2f/P1f < PERTURB_RATIO_MAX.  The SAME voltage set is
    used for every harmonic's V^n fit (plan sec.6); SNR is not used."""
    ratio = P[:, 1] / P[:, 0]
    return np.isfinite(ratio) & (ratio < PERTURB_RATIO_MAX)


def _pert_k(V, pn, n, window):
    """Coefficient k for P_nf = k V^n, least squares through the origin
    over the common perturbative ``window`` (the same P2f/P1f<0.1
    voltage set for every harmonic; SNR plays no role).  Returns None
    only if the window is empty."""
    use = window & np.isfinite(pn)
    if int(np.sum(use)) < 1:
        return None
    vf, pf = V[use], pn[use]
    return float(np.sum(vf**n * pf) / np.sum(vf ** (2 * n)))


def fig2a() -> None:
    if not LADDER_NPZ.exists():
        raise FileNotFoundError(f"{LADDER_NPZ} missing -- run harmonic_ladder.py first")
    d = np.load(LADDER_NPZ)
    V = d["pzt_vpp"]
    P = d["p_kpa"]  # kPa
    Pstd = d["p_std_kpa"] if "p_std_kpa" in d.files else np.zeros_like(P)  # kPa
    window = _perturb_window(P)
    print(
        f"  perturbative window: {int(np.sum(window))} pts, "
        f"V<={V[window].max():.0f} Vpp (P2f/P1f<{PERTURB_RATIO_MAX})"
    )
    v_fine = np.linspace(0, V.max() * 1.05, 200)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    # left axis: P_1f in MPa
    k1 = _pert_k(V, P[:, 0], 1, window)
    line1 = k1 * v_fine if k1 is not None else np.array([0.0])
    ax.errorbar(
        V,
        P[:, 0] / 1e3,
        yerr=Pstd[:, 0] / 1e3,
        fmt=MARKERS[1],
        ms=3.5,
        color=COLORS[1],
        capsize=2,
        elinewidth=0.6,
        label=r"$\hat{P}_{1f}$",
    )
    if k1 is not None:
        ax.plot(v_fine, line1 / 1e3, ":", lw=0.6, color=COLORS[1])
    ax.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax.set_ylabel(r"$\hat{P}_{1f}$ [MPa]", color=COLORS[1])
    ax.tick_params(axis="y", labelcolor=COLORS[1])
    ax.set_xlim(0, V.max() * 1.05)
    ax.set_ylim(0, max(line1.max(), float(P[:, 0].max())) * 1.1 / 1e3)

    # right axis: P_2f and P_3f in kPa (shared scale)
    axr = ax.twinx()
    right_top = 0.0
    for n in (2, 3):
        pn = P[:, n - 1]
        axr.errorbar(
            V,
            pn,
            yerr=Pstd[:, n - 1],
            fmt=MARKERS[n],
            ms=3.5,
            color=COLORS[n],
            capsize=2,
            elinewidth=0.6,
            label=rf"$\hat{{P}}_{{{n}f}}$",
        )
        right_top = max(right_top, float(np.nanmax(pn)))
        k = _pert_k(V, pn, n, window)
        if k is not None:
            line = k * v_fine**n
            right_top = max(right_top, float(line.max()))
            axr.plot(v_fine, line, ":", lw=0.6, color=COLORS[n])
    axr.set_ylabel(r"$\hat{P}_{2f},\ \hat{P}_{3f}$ [kPa]")
    axr.set_ylim(0, right_top * 1.1)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.text(-0.12, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title(
        r"PRL draft Fig 2a (W21, to 3f)"
        "\n"
        r"dotted $\propto V^n$ over $P_{2f}/P_{1f}\le0.1$ window",
        fontsize=9,
    )

    fig.tight_layout()
    out = OUT_DIR / "prl_draft_fig2a.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    np.savez(
        OUT_DIR / "prl_draft_fig2a.npz",
        pzt_vpp=V,
        p_kpa=P[:, : len(HARMS)],
        harmonics=np.array(HARMS),
    )
    print(f"  Saved {out}")


def fig2b() -> None:
    """Measured harmonic ratios + Coppens prefactor (no theory line).

    (b) P_nf/P_1f (n=2,3) and (c) P_2f/P_1f^2 vs Mach number, with error
    bars propagated from the per-harmonic P_nf std (p_std_kpa):
      sigma(P_n/P_1)   = (P_n/P_1)   sqrt((s_n/P_n)^2 + (s_1/P_1)^2)
      sigma(P_2/P_1^2) = (P_2/P_1^2) sqrt((s_2/P_2)^2 + (2 s_1/P_1)^2)
    """
    if not LADDER_NPZ.exists():
        raise FileNotFoundError(f"{LADDER_NPZ} missing -- run harmonic_ladder.py first")
    d = np.load(LADDER_NPZ)
    if "p_std_kpa" not in d.files:
        raise KeyError("p_std_kpa missing in npz -- re-run harmonic_ladder.py")
    P = d["p_kpa"] * 1e3  # Pa
    S = d["p_std_kpa"] * 1e3  # Pa
    P1, P2, P3 = P[:, 0], P[:, 1], P[:, 2]
    s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]
    M = P1 / (RHO * C_SOUND**2)

    R2 = P2 / P1
    sR2 = R2 * np.sqrt((s2 / P2) ** 2 + (s1 / P1) ** 2)
    R3 = P3 / P1
    sR3 = R3 * np.sqrt((s3 / P3) ** 2 + (s1 / P1) ** 2)
    K2 = P2 / P1**2 * 1e9  # 1/GPa
    sK2 = K2 * np.sqrt((s2 / P2) ** 2 + (2 * s1 / P1) ** 2)

    fig, (axb, axc) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axb.errorbar(
        M,
        R2,
        yerr=sR2,
        fmt=MARKERS[2],
        ms=3.5,
        color=COLORS[2],
        capsize=2,
        elinewidth=0.6,
        label=r"$\hat{P}_{2f}/\hat{P}_{1f}$",
    )
    axb.errorbar(
        M,
        R3,
        yerr=sR3,
        fmt=MARKERS[3],
        ms=3.5,
        color=COLORS[3],
        capsize=2,
        elinewidth=0.6,
        label=r"$\hat{P}_{3f}/\hat{P}_{1f}$",
    )
    axb.set_xlabel(r"$M=\hat{P}_{1f}/\rho c^2$")
    axb.set_ylabel("harmonic ratio")
    axb.set_xlim(0, M.max() * 1.05)
    axb.set_ylim(bottom=0)
    axb.legend(frameon=False, loc="upper left")
    axb.grid(True, alpha=0.3)
    axb.text(-0.18, 0.98, "(b)", transform=axb.transAxes, va="top", ha="left", fontweight="bold")

    axc.errorbar(
        M, K2, yerr=sK2, fmt=MARKERS[2], ms=3.5, color=COLORS[2], capsize=2, elinewidth=0.6
    )
    axc.set_xlabel(r"$M=\hat{P}_{1f}/\rho c^2$")
    axc.set_ylabel(r"$\hat{P}_{2f}/\hat{P}_{1f}^2$ [1/GPa]")
    axc.set_xlim(0, M.max() * 1.05)
    axc.set_ylim(bottom=0)
    axc.grid(True, alpha=0.3)
    axc.text(-0.22, 0.98, "(c)", transform=axc.transAxes, va="top", ha="left", fontweight="bold")

    fig.suptitle(
        "PRL draft Fig 2b/c (W21) --- measured ratios, no theory line "
        r"(error = propagated $P_{nf}$ std)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "prl_draft_fig2b.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    np.savez(
        OUT_DIR / "prl_draft_fig2b.npz",
        M=M,
        ratio_2f=R2,
        ratio_2f_std=sR2,
        ratio_3f=R3,
        ratio_3f_std=sR3,
        prefactor_2f_perGPa=K2,
        prefactor_2f_std=sK2,
    )
    print(f"  Saved {out}")


def fig2a_5f(yscale: str = "log") -> None:
    """Fig 2a variant: full harmonic ladder P_1f..P_5f on ONE y-axis.

    Single axis (``yscale`` = "log" or "linear") so all five harmonics
    appear together; error bars are the per-harmonic sqrt(fit-SE^2 +
    noise^2); filled = detected (SNR>=3), open = below noise; dotted
    lines are the perturbative ~V^n guides fit over the common
    P2f/P1f<=0.1 window (skipped for a harmonic with < 2 detected points
    in that window).
    """
    if not LADDER_NPZ.exists():
        raise FileNotFoundError(f"{LADDER_NPZ} missing -- run harmonic_ladder.py first")
    d = np.load(LADDER_NPZ)
    V = d["pzt_vpp"]
    P = d["p_kpa"]  # kPa, n_scans x 5
    Pstd = d["p_std_kpa"] if "p_std_kpa" in d.files else np.zeros_like(P)
    SNR = d["snr"]
    noise = d["noise_kpa"] if "noise_kpa" in d.files else np.zeros_like(P)
    window = _perturb_window(P)
    harms = (1, 2, 3, 4, 5)
    is_log = yscale == "log"
    v_fine = np.linspace(V.min() * 0.85, V.max() * 1.05, 200)

    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    guide_max = 0.0
    for n in harms:
        pn = P[:, n - 1]
        sn = Pstd[:, n - 1]
        snr_n = SNR[:, n - 1]
        det = snr_n >= 3.0
        if is_log:
            # Floor the lower whisker at the noise level so below-noise
            # points do not plunge a meaningless decade on the log axis.
            lower = np.maximum(pn - sn, noise[:, n - 1])
            yerr_lo = np.clip(pn - lower, 0, None)
        else:
            yerr_lo = sn  # symmetric on linear (clipped at ylim = 0)
        ax.errorbar(
            V[det],
            pn[det],
            yerr=[yerr_lo[det], sn[det]],
            fmt=MARKERS[n],
            ms=3.5,
            color=COLORS[n],
            capsize=2,
            elinewidth=0.6,
            label=rf"$\hat{{P}}_{{{n}f}}$",
        )
        if np.any(~det):  # below noise (SNR<3): open markers
            ax.errorbar(
                V[~det],
                pn[~det],
                yerr=[yerr_lo[~det], sn[~det]],
                fmt=MARKERS[n],
                ms=3.5,
                color=COLORS[n],
                mfc="none",
                capsize=2,
                elinewidth=0.6,
                alpha=0.6,
            )
        k = _pert_k(V, pn, n, window)
        if k is not None:
            guide = k * v_fine**n
            guide_max = max(guide_max, float(guide.max()))
            ax.plot(v_fine, guide, ":", lw=0.6, color=COLORS[n])

    ax.set_yscale(yscale)
    ax.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax.set_ylabel(r"$\hat{P}_{nf}$ [kPa]")
    ax.set_xlim(0, V.max() * 1.05)
    if is_log:
        ax.legend(frameon=False, ncol=2, loc="lower right")
        scale_note = r"single log axis"
    else:
        ax.set_ylim(0, 1.05 * max(float(P[:, 0].max()), guide_max))
        ax.legend(frameon=False, ncol=2, loc="upper left")
        scale_note = r"single linear axis"
    ax.grid(True, which="both", alpha=0.3)
    ax.text(-0.14, 0.99, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title(
        r"PRL draft Fig 2a variant (W21, to 5f)"
        "\n" + scale_note + r"; $\propto V^n$ over $P_{2f}/P_{1f}\le0.1$; open = SNR$<$3",
        fontsize=9,
    )
    fig.tight_layout()
    suffix = "" if is_log else "_linear"
    out = OUT_DIR / f"prl_draft_fig2a_5f{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    if is_log:
        np.savez(
            OUT_DIR / "prl_draft_fig2a_5f.npz",
            pzt_vpp=V,
            p_kpa=P,
            p_std_kpa=Pstd,
            snr=SNR,
            harmonics=np.array(harms),
        )
    print(f"  Saved {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2a()
    fig2a_5f("log")
    fig2a_5f("linear")
    fig2b()


if __name__ == "__main__":
    main()
