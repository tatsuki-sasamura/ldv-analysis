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
    CHANNEL_WIDTH,
    get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import _mode_shape, _project, _r2  # noqa: E402
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
COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
MARKERS = {1: "o", 2: "s", 3: "^"}


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

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.0))
    saved = {}
    for j, n in enumerate(HARMS):
        grid_pa = cg.to_grid(np.where(valid, np.asarray(c[f"pressure_{n}f"]), np.nan))
        grid_mpa = grid_pa / 1e6
        p0_y = fit_columns(grid_pa, cg.width_grid, CHANNEL_WIDTH, harmonic=n, sigma_clip=SIGMA)
        bi = int(np.nanargmax(p0_y))
        p0 = float(p0_y[bi])

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
        cv = ~np.isnan(col)
        w_mm = cg.width_grid[cv] * 1e3
        mode = _mode_shape(cg.width_grid[cv], CHANNEL_WIDTH, n, use_abs=True)
        _, clip = _project(col[cv], mode, sigma_clip=SIGMA)
        r2 = _r2(col[cv][clip], (p0 * mode)[clip])
        axp = axes[1, j]
        axp.plot(w_mm[clip], col[cv][clip] / 1e6, "ko", ms=2)
        if (~clip).any():
            axp.plot(w_mm[~clip], col[cv][~clip] / 1e6, "x", ms=3, color="0.6")
        xf = np.linspace(cg.width_grid[0], cg.width_grid[-1], 300)
        axp.plot(
            xf * 1e3,
            p0 / 1e6 * _mode_shape(xf, CHANNEL_WIDTH, n, use_abs=True),
            "-",
            color="C3",
            lw=0.9,
            label=f"fit ($R^2$={r2:.2f})",
        )
        axp.set_xlabel(r"$y$ [mm]")
        axp.set_ylabel(f"$|P_{{{n}f}}|$ [MPa]")
        axp.set_ylim(bottom=0)
        axp.set_title(f"({chr(100+j)}) $\\hat{{P}}_{{{n}f}}$ = {p0/1e3:.0f} kPa", fontsize=9)
        axp.legend(frameon=False, handlelength=1.2)
        saved[f"p0_{n}f_kpa"] = p0 / 1e3
        saved[f"r2_{n}f"] = r2

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
        print(f"  {n}f: P = {saved[f'p0_{n}f_kpa']:.0f} kPa, R^2 = {saved[f'r2_{n}f']:.3f}")


def _pert_k(v, pn, snr_n, n):
    """Power-law coefficient k for P_nf ~ k V^n through the lowest <=3
    above-noise (SNR>=3) points (perturbative regime for that harmonic)."""
    idx = np.where(snr_n >= 3.0)[0][:3]
    if len(idx) < 2:
        idx = np.arange(min(3, len(v)))
    vf, pf = v[idx], pn[idx]
    return float(np.sum(vf**n * pf) / np.sum(vf ** (2 * n)))


def fig2a() -> None:
    if not LADDER_NPZ.exists():
        raise FileNotFoundError(f"{LADDER_NPZ} missing -- run harmonic_ladder.py first")
    d = np.load(LADDER_NPZ)
    V = d["pzt_vpp"]
    P = d["p_kpa"]  # kPa
    SNR = d["snr"]
    v_fine = np.linspace(0, V.max() * 1.05, 200)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    # left axis: P_1f in MPa
    k1 = _pert_k(V, P[:, 0], SNR[:, 0], 1)
    line1 = k1 * v_fine
    ax.plot(V, P[:, 0] / 1e3, MARKERS[1], ms=3.5, color=COLORS[1], label=r"$\hat{P}_{1f}$")
    ax.plot(v_fine, line1 / 1e3, ":", lw=0.6, color=COLORS[1])
    ax.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax.set_ylabel(r"$\hat{P}_{1f}$ [MPa]", color=COLORS[1])
    ax.tick_params(axis="y", labelcolor=COLORS[1])
    ax.set_xlim(0, V.max() * 1.05)
    ax.set_ylim(0, line1.max() * 1.1 / 1e3)

    # right axis: P_2f and P_3f in kPa (shared scale)
    axr = ax.twinx()
    right_top = 0.0
    for n in (2, 3):
        pn = P[:, n - 1]
        k = _pert_k(V, pn, SNR[:, n - 1], n)
        line = k * v_fine**n
        right_top = max(right_top, line.max())
        axr.plot(V, pn, MARKERS[n], ms=3.5, color=COLORS[n], label=rf"$\hat{{P}}_{{{n}f}}$")
        axr.plot(v_fine, line, ":", lw=0.6, color=COLORS[n])
    axr.set_ylabel(r"$\hat{P}_{2f},\ \hat{P}_{3f}$ [kPa]")
    axr.set_ylim(0, right_top * 1.1)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.text(-0.12, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title(
        r"PRL draft Fig 2a (W21, to 3f)" "\n" r"dotted = perturbative $\propto V^n$", fontsize=9
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2a()


if __name__ == "__main__":
    main()
