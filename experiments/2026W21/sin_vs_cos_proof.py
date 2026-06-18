# %%
"""Visual proof: at 3.794 MHz the field is pure cos(2 pi y/W), not sin(pi y/W).

Both projections (n=2 cos and n=1 sin) return non-trivial amplitudes
in the survey check because ``fit_mode`` does an independent
brute-force center search per harmonic.  But only the cos fit
actually matches the data shape -- the sin fit returns a sizeable p0
while leaving big residuals (R^2 << 1).

This script makes that visual: at the 3.794 MHz peak axial slice from
the W21 101x77 fine scan, overlay the transverse data with both fits
on the same axes.

Output: output/sin_vs_cos_proof/sin_vs_cos_proof.png
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
    pos_x = np.asarray(c["pos_x"])      # width direction (m)
    pos_y = np.asarray(c["pos_y"])      # length direction (m)
    n_x = int(c["n_x_meta"])
    n_y = int(c["n_y_meta"])

    # Channel geometry + grid
    hw = CHANNEL_WIDTH / 2
    Pg = P_abs.copy()
    Pg[~valid] = np.nan
    a, b = detect_channel_geometry(pos_x, pos_y, rssi, Pg, hw)
    pos_x_c = pos_x - (a * pos_y + b)
    inside_c = np.abs(pos_x_c) <= hw
    cg = make_channel_grid(
        pos_width_c=pos_x_c, pos_length=pos_y,
        n_scan_width=n_x, n_scan_length=n_y,
        channel_width=CHANNEL_WIDTH, raw_width_span=pos_x.max() - pos_x.min(),
        inside=inside_c, rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
    )

    # Best axial slice using cos(2 pi y/W) projection per column
    P_abs_grid = cg.to_grid(np.where(valid, P_abs, np.nan))   # (n_w, n_l)
    P_re_grid = cg.to_grid(np.where(valid, P_c.real, np.nan))
    P_im_grid = cg.to_grid(np.where(valid, P_c.imag, np.nan))
    p0_y = fit_columns(P_abs_grid, cg.width_grid, CHANNEL_WIDTH,
                       harmonic=2, sigma_clip=3.0)
    ix = int(np.nanargmax(p0_y))
    x_best_mm = float(cg.length_grid[ix] * 1e3)
    print(f"best axial slice: x = {x_best_mm:.2f} mm")

    # Cross-section data at the best slice
    y_grid = cg.width_grid                                    # m, length n_w
    col_re = P_re_grid[:, ix]
    col_im = P_im_grid[:, ix]
    col_c = col_re + 1j * col_im
    finite = np.isfinite(col_re)
    y_data = y_grid[finite]
    col_c = col_c[finite]
    col_abs = np.abs(col_c)

    # Fit both n=1 (sin) and n=2 (cos) independently via fit_mode
    r_cos = fit_mode(y_data, col_c, CHANNEL_WIDTH, harmonic=2, sigma_clip=3.0)
    r_sin = fit_mode(y_data, col_c, CHANNEL_WIDTH, harmonic=1, sigma_clip=3.0)
    print(f"cos(2 pi y/W) fit: |p0| = {abs(r_cos.p0)/1e3:.1f} kPa  "
          f"center = {r_cos.center*1e6:+.1f} um  R^2 = {r_cos.r2:+.3f}")
    print(f"sin(  pi y/W) fit: |p0| = {abs(r_sin.p0)/1e3:.1f} kPa  "
          f"center = {r_sin.center*1e6:+.1f} um  R^2 = {r_sin.r2:+.3f}")

    # Build the actual fit curves to overlay
    y_dense = np.linspace(y_grid.min(), y_grid.max(), 600)
    # cos fit reconstruction: |p_cos| * |cos(2 pi (y - center)/W)|
    fit_cos = abs(r_cos.p0) * np.abs(_mode_shape(
        y_dense - r_cos.center, CHANNEL_WIDTH, harmonic=2, use_abs=False))
    # sin fit reconstruction: |p_sin| * |sin(  pi (y - center)/W)|
    fit_sin = abs(r_sin.p0) * np.abs(_mode_shape(
        y_dense - r_sin.center, CHANNEL_WIDTH, harmonic=1, use_abs=False))

    # ----- plot ------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize_for_layout(1, 1))
    ax.plot(y_data * 1e3, col_abs / 1e3, "o", markersize=4, color="0.2",
            label="data $|P_{1f}|$")
    ax.plot(y_dense * 1e3, fit_cos / 1e3, "-", linewidth=1.2, color="C2",
            label=(rf"$\cos(2\pi y/W)$ fit: "
                   rf"$|p_0|={abs(r_cos.p0)/1e3:.0f}$ kPa, "
                   rf"$R^2={r_cos.r2:.2f}$"))
    ax.plot(y_dense * 1e3, fit_sin / 1e3, "--", linewidth=1.2, color="C3",
            label=(rf"$\sin(\pi y/W)$ fit: "
                   rf"$|p_0|={abs(r_sin.p0)/1e3:.0f}$ kPa, "
                   rf"$R^2={r_sin.r2:.2f}$"))
    ax.axvline(r_cos.center * 1e3, color="C2", lw=0.4, ls=":",
               label=fr"cos center: {r_cos.center*1e6:+.0f} $\mu$m")
    ax.axvline(r_sin.center * 1e3, color="C3", lw=0.4, ls=":",
               label=fr"sin center: {r_sin.center*1e6:+.0f} $\mu$m")
    ax.set_xlabel(r"channel width $y$ [mm]")
    ax.set_ylabel(r"$|P_{1f}|$ [kPa]")
    ax.set_title(
        f"Transverse cross-section at x={x_best_mm:.2f} mm, "
        f"drive {F_HZ/1e6:.3f} MHz\n"
        f"cos and sin fits compared on the same data"
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    out_path = OUT_DIR / "sin_vs_cos_proof.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
