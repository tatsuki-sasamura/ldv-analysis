# %%
"""Why does fit_mode return ~300 kPa on the wrong (sin) basis?

The cleaner answer: ``fit_mode`` independently brute-forces a center
per harmonic AND runs iterative sigma-clipping with min_keep=0.5.  On
a pure cos(2 pi y/W) field, the sin(pi y/W) fit's "amplitude" of
~300 kPa comes mostly from the clipping discarding the antinodes
(where residual is huge against a sin shape) and the center hunt
picking the offset that lets the surviving wings look sin-like.

This script demonstrates by re-running the same data with BOTH
sigma_clip=3 (default) and sigma_clip=None (no clipping):

  - shows which points the sigma=3 sin1 fit retained vs dropped;
  - reports R^2 over the full data vs over the retained subset;
  - both fits and the data on one axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, LDV_DATA_ROOT, RSSI_THRESHOLD,
    figsize_for_layout, get_cache_dir,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.grid_utils import make_channel_grid  # noqa: E402
from ldv_analysis.mode_fit import _mode_shape, fit_columns, fit_mode  # noqa: E402
from ldv_analysis.sweep_fit import detect_channel_geometry  # noqa: E402

SCAN_DIR = "sample_101x77_fsweep_3p76to3p84_2kHz_60Vpp_20260530_072344"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = Path(__file__).resolve().parent / "output" / "sin_vs_cos_proof"
F_HZ = 3_794_000


def r2_over(data_complex, p0, mode_signed):
    """R^2 of |data| against |p0 * mode_signed| over ALL points provided."""
    pred = np.abs(p0 * mode_signed)
    data = np.abs(data_complex)
    ss_res = np.sum((data - pred) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = get_cache_dir(SCAN_DIR, __file__)
    h5_path = DATA_ROOT / SCAN_DIR / f"f{F_HZ}.h5"
    c = load_or_compute(h5_path, cache_dir, velocity_scale=None)

    V = np.asarray(c["voltage_1f"])
    rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
    valid = make_valid_mask(V, rssi)
    P_abs = np.asarray(c["pressure_1f"])
    P_ph = np.asarray(c["phase_1f"])
    P_c = P_abs * np.exp(1j * np.radians(P_ph))
    pos_x = np.asarray(c["pos_x"])
    pos_y = np.asarray(c["pos_y"])
    n_x = int(c["n_x_meta"])
    n_y = int(c["n_y_meta"])

    hw = CHANNEL_WIDTH / 2
    Pg = P_abs.copy(); Pg[~valid] = np.nan
    a, b = detect_channel_geometry(pos_x, pos_y, rssi, Pg, hw)
    pos_x_c = pos_x - (a * pos_y + b)
    inside_c = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(
        pos_width_c=pos_x_c, pos_length=pos_y,
        n_scan_width=n_x, n_scan_length=n_y,
        channel_width=CHANNEL_WIDTH, raw_width_span=pos_x.max() - pos_x.min(),
        inside=inside_c, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    P_abs_grid = cg.to_grid(np.where(valid, P_abs, np.nan))
    P_re_grid = cg.to_grid(np.where(valid, P_c.real, np.nan))
    P_im_grid = cg.to_grid(np.where(valid, P_c.imag, np.nan))
    p0_y = fit_columns(P_abs_grid, cg.width_grid, CHANNEL_WIDTH,
                       harmonic=2, sigma_clip=3.0)
    ix = int(np.nanargmax(p0_y))
    x_best_mm = float(cg.length_grid[ix] * 1e3)
    print(f"axial slice: x = {x_best_mm:.2f} mm  (cos2 column peak)")

    y_grid = cg.width_grid
    col_c = (P_re_grid[:, ix] + 1j * P_im_grid[:, ix])
    finite = np.isfinite(col_c.real)
    y = y_grid[finite]
    col = col_c[finite]
    n_total = len(y)
    print(f"data points at this slice: {n_total}")

    # --- four fits to compare -------------------------------------------------
    r_cos_clip = fit_mode(y, col, CHANNEL_WIDTH, harmonic=2, sigma_clip=3.0)
    r_cos_none = fit_mode(y, col, CHANNEL_WIDTH, harmonic=2, sigma_clip=None)
    r_sin_clip = fit_mode(y, col, CHANNEL_WIDTH, harmonic=1, sigma_clip=3.0)
    r_sin_none = fit_mode(y, col, CHANNEL_WIDTH, harmonic=1, sigma_clip=None)

    def report(tag, r):
        # ModeFitResult has p0, center, r2 (over the kept subset), inside.
        n_kept = int(np.sum(r.inside))
        mode_signed_full = _mode_shape(
            y - r.center, CHANNEL_WIDTH, harmonic=2 if "cos" in tag else 1,
            use_abs=False)
        r2_full = r2_over(col, r.p0, mode_signed_full)
        print(f"  {tag:25s} |p0|={abs(r.p0)/1e3:6.1f} kPa  "
              f"y0={r.center*1e6:+6.1f} um  "
              f"R^2(kept,reported)={r.r2:+.3f}  "
              f"R^2(all data)={r2_full:+.3f}  "
              f"kept {n_kept}/{n_total}")
        return n_kept, r2_full

    print("\n--- fit results ---")
    nk_cc, r2_full_cc = report("cos2, sigma_clip=3", r_cos_clip)
    nk_cn, r2_full_cn = report("cos2, sigma_clip=None", r_cos_none)
    nk_sc, r2_full_sc = report("sin1, sigma_clip=3", r_sin_clip)
    nk_sn, r2_full_sn = report("sin1, sigma_clip=None", r_sin_none)

    # ---- plot ---------------------------------------------------------------
    y_dense = np.linspace(y_grid.min(), y_grid.max(), 600)

    fit_cos_curve = abs(r_cos_clip.p0) * np.abs(_mode_shape(
        y_dense - r_cos_clip.center, CHANNEL_WIDTH, harmonic=2, use_abs=False))
    fit_sin_clip_curve = abs(r_sin_clip.p0) * np.abs(_mode_shape(
        y_dense - r_sin_clip.center, CHANNEL_WIDTH, harmonic=1, use_abs=False))
    fit_sin_none_curve = abs(r_sin_none.p0) * np.abs(_mode_shape(
        y_dense - r_sin_none.center, CHANNEL_WIDTH, harmonic=1, use_abs=False))

    fig, axes = plt.subplots(
        2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True
    )

    # (a) data + cos2 fit + sin1 fits (clipped vs full)
    ax = axes[0]
    ax.plot(y * 1e3, np.abs(col) / 1e3, "o", markersize=4, color="0.2",
            label="data $|P_{1f}|$")
    ax.plot(y_dense * 1e3, fit_cos_curve / 1e3, "-", lw=1.2, color="C2",
            label=rf"$\cos(2\pi y/W)$ fit (any clip): "
                  rf"$|p_0|={abs(r_cos_clip.p0)/1e3:.0f}$ kPa")
    ax.plot(y_dense * 1e3, fit_sin_clip_curve / 1e3, "--", lw=1.2, color="C3",
            label=rf"$\sin(\pi y/W)$, $\sigma$-clip=3: "
                  rf"$|p_0|={abs(r_sin_clip.p0)/1e3:.0f}$ kPa")
    ax.plot(y_dense * 1e3, fit_sin_none_curve / 1e3, ":", lw=1.2, color="C5",
            label=rf"$\sin(\pi y/W)$, $\sigma$-clip=None: "
                  rf"$|p_0|={abs(r_sin_none.p0)/1e3:.0f}$ kPa")
    ax.set_ylabel(r"$|P_{1f}|$ [kPa]")
    ax.set_title(
        f"Transverse cross-section at x = {x_best_mm:.2f} mm, "
        f"{F_HZ/1e6:.3f} MHz drive"
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (b) which points the sigma=3 sin1 fit kept vs dropped
    ax = axes[1]
    kept_mask = r_sin_clip.inside
    ax.plot(y[kept_mask] * 1e3, np.abs(col)[kept_mask] / 1e3, "o",
            markersize=5, color="C0", label=f"kept by $\\sigma$=3 sin1 fit "
                                              f"({int(kept_mask.sum())} pts)")
    ax.plot(y[~kept_mask] * 1e3, np.abs(col)[~kept_mask] / 1e3, "x",
            markersize=8, color="C3", markeredgewidth=1.5,
            label=f"dropped ({int((~kept_mask).sum())} pts)")
    ax.plot(y_dense * 1e3, fit_sin_clip_curve / 1e3, "--", lw=1.0, color="C3",
            label=r"sin1 fit curve")
    ax.set_xlabel(r"$y$ [mm]")
    ax.set_ylabel(r"$|P_{1f}|$ [kPa]")
    ax.set_title(
        r"Sigma-clip excludes the antinodes -- "
        r"R$^2$ is reported only over kept points"
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "sin_vs_cos_proof2.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
