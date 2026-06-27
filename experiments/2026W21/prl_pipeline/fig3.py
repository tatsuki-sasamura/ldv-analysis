"""Fig 3 — force reconstruction and central-stiffness reduction.

Decisional figure (per contract Outcome Matrix).

Coordinate convention: transverse position is the CENTERED coordinate
``y/W ∈ [-1/2, +1/2]`` with the channel center at y=0 and the walls at
``y = ±W/2``. This is the convention in which the contract's F̃ formula
is derived; the `(-1)^n` factor encodes the alternating parity of the
n-th transverse-mode shape about y=0.

Four panels (2x2):

- (a) ``E_n / E_1`` vs drive for n=2,3,4,5.
- (b) Dimensionless force-field shape
  ``F̃(y) = Σ_n (-1)^n · n · (E_n/E_1) · sin(2nπy/W)``  (centered y)
  overlaid at low / mid / top drive.
- (c) Harmonic-reshaping factor (one of two contributions to R_κ)
  ``E_foc / E_1
        = 1 − 4·E_2/E_1 + 9·E_3/E_1 − 16·E_4/E_1 + 25·E_5/E_1 − …``
  vs V_drive. NOT R_κ itself.
- (d) Two-factor decomposition vs P_in (contract paragraph_draft §P3
  Fig 3(d) — `[HYP-F2c, HYP-F3]`):
  ``R_κ ≡ κ_foc/κ_lin = E_foc/E_lin = (E_1/E_lin) · (E_foc/E_1)``
  Plots fundamental-response factor (read from fig2.npz), harmonic-
  reshaping factor (computed here), and their product.

Inviscid Gor'kov assumption (Φ_n/Φ_1 = 1) — no pressure-calibration
audit dependence; magnitudes only.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from conventions import (
    CHANNEL_WIDTH,
    HARMONIC_E_RATIO_SIGMA,
    HARMONIC_SNR_MIN,
    PIPE_OUT,
)
from _data import load_ladder  # noqa: E402  (sibling module)

HARMS = (1, 2, 3, 4, 5)
COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:purple", 5: "tab:orange"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
Y_GRID_POINTS = 401  # F̃(y) sampling, centered y/W in [-0.5, +0.5]
FIG2_NPZ = PIPE_OUT / "fig2.npz"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 0.8,
    }
)


def _en_over_e1_with_sigma(L: dict):
    """Return (ratio[N,H], sigma[N,H]) for E_n/E_1, including n=1."""
    e = L["e_n_j_m3"]
    s = L["e_n_sigma_j_m3"]
    e1 = e[:, 0:1]
    s1 = s[:, 0:1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e / e1
        sigma = ratio * np.sqrt((s / np.where(e > 0, e, np.nan)) ** 2
                                + (s1 / np.where(e1 > 0, e1, np.nan)) ** 2)
    # n=1: ratio is 1, sigma is 0 by construction
    ratio[:, 0] = 1.0
    sigma[:, 0] = 0.0
    return ratio, sigma


def _adoption_mask(L: dict, ratio: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-drive per-harmonic adoption per contract §3.

    n=1 is always adopted (reference). For n >= 2, adopted iff
    SNR_n > HARMONIC_SNR_MIN AND E_n/E_1 > HARMONIC_E_RATIO_SIGMA * sigma.
    """
    snr = L["snr"]
    N, H = ratio.shape  # noqa: N806
    adopted = np.zeros((N, H), dtype=bool)
    adopted[:, 0] = True
    for n in range(2, H + 1):
        j = n - 1
        snr_ok = np.isfinite(snr[:, j]) & (snr[:, j] > HARMONIC_SNR_MIN)
        finite = np.isfinite(ratio[:, j]) & np.isfinite(sigma[:, j]) & (sigma[:, j] > 0)
        signif = finite & (ratio[:, j] > HARMONIC_E_RATIO_SIGMA * sigma[:, j])
        adopted[:, j] = snr_ok & signif
    return adopted


def _global_depth(adopted: np.ndarray) -> int:
    """Single series depth applied to every drive.

    A harmonic enters the series iff it is adopted at *some* drive
    (contract R3 truncation: drop the harmonic from the series only if
    it is never adopted anywhere). This avoids the per-drive truncation
    artifact in R_kappa(V), where the series formula would change
    discretely as the drive crosses each adoption threshold.

    Walk n=2,3,4,5 over the "ever-adopted" flag; stop at the first n
    that is never adopted. Always include n=1.
    """
    H = adopted.shape[1]
    ever = adopted.any(axis=0)
    depth = 1
    for n in range(2, H + 1):
        if not ever[n - 1]:
            break
        depth = n
    return depth


def _e_foc_over_e1_and_sigma(ratio: np.ndarray, sigma: np.ndarray,
                              depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-reshaping factor ``E_foc/E_1`` per drive.

    ``E_foc/E_1 = sum_{n=1..depth} (-1)^(n+1) n^2 (E_n/E_1)``

    sigma propagated as sqrt(sum (n^2)^2 sigma_{En/E1}^2) over n >= 2.

    This is the per-drive ratio comparing the measured centered stiffness
    to a hypothetical pure-fundamental field with the same E_1. It is
    NOT R_kappa, which compares to E_lin (the linear extrapolation from
    low input power) — see ``_r_kappa_from_decomposition``.
    """
    N, _H = ratio.shape
    e_foc_over_e1 = np.full(N, np.nan)
    sig = np.full(N, np.nan)
    d = int(depth)
    for i in range(N):
        terms = np.array([(-1) ** (n + 1) * n**2 * ratio[i, n - 1]
                          for n in range(1, d + 1)])
        e_foc_over_e1[i] = float(np.sum(terms))
        if d >= 2:
            var = sum((n**2 * sigma[i, n - 1]) ** 2 for n in range(2, d + 1))
            sig[i] = float(np.sqrt(var))
        else:
            sig[i] = 0.0
    return e_foc_over_e1, sig


def _f_tilde_y(ratio_row: np.ndarray, depth: int,
               y_over_w_centered: np.ndarray) -> np.ndarray:
    """Contract F̃(y) on the CENTERED transverse coordinate.

    ``F̃(y) = Σ_{n=1..depth} (-1)^n · n · (E_n/E_1) · sin(2nπy/W)``

    ``y_over_w_centered`` ∈ [-1/2, +1/2]; y=0 is the channel center.
    The `(-1)^n` factor encodes the alternating parity of the n-th
    transverse-mode shape about the centered y=0 axis (analysis_contract.md
    line 223; paragraph_draft.md line 70).
    """
    s = np.zeros_like(y_over_w_centered)
    for n in range(1, depth + 1):
        s = s + ((-1) ** n) * n * float(ratio_row[n - 1]) * \
            np.sin(2 * n * np.pi * y_over_w_centered)
    return s


def _load_fig2_for_decomposition() -> tuple[np.ndarray, float, float] | None:
    """Read P_in and E_1-linear slope from fig2.npz (for panel d).

    Returns (p_in_w[N], slope_jm3_per_w, slope_se) or None if fig2.npz
    is missing (panel d is then skipped).
    """
    if not FIG2_NPZ.exists():
        return None
    f2 = np.load(FIG2_NPZ)
    p_in = np.asarray(f2["p_in_w"])
    s = float(f2["e_1_lin_slope_jm3_per_w"])
    s_se = float(f2["e_1_lin_slope_se"])
    return p_in, s, s_se


def compute() -> dict:
    L = load_ladder()
    V = L["pzt_vpp"]
    E1 = L["e_n_j_m3"][:, 0]
    E1_sig = L["e_n_sigma_j_m3"][:, 0]

    ratio, sigma = _en_over_e1_with_sigma(L)
    adopted = _adoption_mask(L, ratio, sigma)
    depth = _global_depth(adopted)              # single scalar, applied at every V
    e_foc_over_e1, e_foc_sig = _e_foc_over_e1_and_sigma(ratio, sigma, depth)

    # F~(y) at every drive on the CENTERED transverse coordinate.
    N = len(V)
    y_over_w_centered = np.linspace(-0.5, 0.5, Y_GRID_POINTS)
    f_shape_tilde_F = np.zeros((N, Y_GRID_POINTS))
    for i in range(N):
        f_shape_tilde_F[i] = _f_tilde_y(ratio[i], depth, y_over_w_centered)

    # Three picks for panel (b) overlay: low / mid / top drive.
    pick = np.asarray([0, N // 2, N - 1])

    # --- panel (d): R_kappa decomposition vs P_in (from fig2.npz) -----------
    fig2 = _load_fig2_for_decomposition()
    if fig2 is not None:
        p_in_w, lin_slope, lin_slope_se = fig2
        with np.errstate(divide="ignore", invalid="ignore"):
            E1_lin = lin_slope * p_in_w
            e1_over_e1_lin = E1 / E1_lin
            e1_factor_sigma = np.sqrt(
                (E1_sig / E1_lin) ** 2 +
                (E1 / E1_lin * (lin_slope_se / lin_slope)) ** 2
            )
        r_kappa = e_foc_over_e1 * e1_over_e1_lin
        # sigma_R: combined relative uncertainty
        with np.errstate(divide="ignore", invalid="ignore"):
            r_kappa_sigma = np.abs(r_kappa) * np.sqrt(
                (e_foc_sig / np.where(e_foc_over_e1 != 0, e_foc_over_e1, np.nan)) ** 2 +
                (e1_factor_sigma / np.where(e1_over_e1_lin != 0, e1_over_e1_lin, np.nan)) ** 2
            )
    else:
        p_in_w = np.full(N, np.nan)
        e1_over_e1_lin = np.full(N, np.nan)
        e1_factor_sigma = np.full(N, np.nan)
        r_kappa = np.full(N, np.nan)
        r_kappa_sigma = np.full(N, np.nan)

    # Decision (per contract Fig 3 criterion line 230): at the highest
    # drive, 1 − E_foc/E_1 exceeds 3sigma above zero.
    top = int(np.argmax(V))
    e_foc_top = float(e_foc_over_e1[top])
    e_foc_top_sig = float(e_foc_sig[top])
    reshape_loss = 1.0 - e_foc_top
    reshape_sig = abs(reshape_loss) / e_foc_top_sig if e_foc_top_sig > 0 else float("inf")
    decision_pass = bool(e_foc_top < 1.0 and reshape_sig > 3.0)

    # R3 sanity flag: did 4f or 5f ever get adopted?
    ever_adopted = adopted.any(axis=0)
    n4_ever = bool(ever_adopted[3])
    n5_ever = bool(ever_adopted[4])

    decision_msg = (
        f"Fig 3 at {int(V[top])} Vpp: E_foc/E_1 = {e_foc_top:.3f} +/- {e_foc_top_sig:.3f} "
        f"(reshaping loss {reshape_loss*100:+.1f}% at {reshape_sig:.1f} sigma) -> "
        f"{'PASS' if decision_pass else 'FAIL'}. "
        f"Full R_kappa at top drive = {float(r_kappa[top]) if np.isfinite(r_kappa[top]) else float('nan'):.3f}. "
        f"Global series depth: n=1..{depth}. "
        f"4f ever adopted: {n4_ever}; 5f ever adopted: {n5_ever}."
    )

    return dict(
        pzt_vpp=V,
        e_n_over_e1=ratio,
        e_n_over_e1_sigma=sigma,
        adopted=adopted,
        truncation_depth=np.asarray(depth),
        # harmonic-reshaping factor (the one quantity that fig3 alone owns)
        e_foc_over_e1=e_foc_over_e1,
        e_foc_over_e1_sigma=e_foc_sig,
        # full R_kappa decomposition (cross-references fig2)
        p_in_w=p_in_w,
        e1_over_e1_lin=e1_over_e1_lin,
        e1_over_e1_lin_sigma=e1_factor_sigma,
        r_kappa=r_kappa,
        r_kappa_sigma=r_kappa_sigma,
        # F~(y) on centered y in [-0.5, +0.5]
        y_over_w_centered=y_over_w_centered,
        f_shape_tilde_F=f_shape_tilde_F,
        f_tilde_v_idx=pick,
        ever_adopted_n=ever_adopted,
        decision_pass=np.asarray(decision_pass),
        decision_msg=np.asarray(decision_msg),
    )


def plot(d: dict):
    V = d["pzt_vpp"]
    ratio = d["e_n_over_e1"]
    sigma = d["e_n_over_e1_sigma"]
    e_foc = d["e_foc_over_e1"]
    e_foc_sig = d["e_foc_over_e1_sigma"]
    y_centered = d["y_over_w_centered"]
    f_tilde = d["f_shape_tilde_F"]
    pick = d["f_tilde_v_idx"]
    p_in = d["p_in_w"]
    e1_factor = d["e1_over_e1_lin"]
    e1_factor_sig = d["e1_over_e1_lin_sigma"]
    r_kappa = d["r_kappa"]
    r_kappa_sig = d["r_kappa_sigma"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))

    # --- (a) E_n/E_1 vs V_drive ------------------------------------------
    ax_a = axes[0, 0]
    for n in (2, 3, 4, 5):
        j = n - 1
        ax_a.errorbar(V, ratio[:, j], yerr=sigma[:, j],
                      fmt=MARKERS[n], ms=4, color=COLORS[n], capsize=2,
                      elinewidth=0.6, label=rf"$E_{{{n}f}}/E_1$")
    ax_a.set_xlabel(r"$V_\mathrm{drive}$ [V$_\mathrm{pp}$]")
    ax_a.set_ylabel(r"$E_n/E_1$")
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(frameon=False, ncol=2, loc="lower right", fontsize=7)
    ax_a.text(-0.15, 0.99, "(a)", transform=ax_a.transAxes, va="top",
              ha="left", fontweight="bold")
    ax_a.set_title("(a) per-harmonic energy ratios", fontsize=9)

    # --- (b) F̃(y) overlay on CENTERED y ---------------------------------
    # Centered coordinate y/W in [-1/2, +1/2]; channel center at y=0.
    # 10 Vpp curve doubles as the empirical single-mode reference
    # (E_{n>=2}/E_1 ~ 10^-3 there).
    ax_b = axes[0, 1]
    ax_b.axhline(0, color="0.6", lw=0.5, ls="--")
    ax_b.axvline(0, color="0.6", lw=0.5, ls="--")
    styles = ["-", "--", ":"]
    for k, i in enumerate(pick):
        ax_b.plot(y_centered, f_tilde[int(i)], styles[k % 3], lw=1.0,
                  color=f"C{k}", label=f"{int(V[int(i)])} Vpp")
    ax_b.set_xlabel(r"$y/W$ (centered, walls at $\pm 1/2$)")
    ax_b.set_ylabel(r"$\tilde{F}(y)$")
    ax_b.legend(frameon=False, loc="best", fontsize=7)
    ax_b.grid(True, alpha=0.3)
    ax_b.text(-0.15, 0.99, "(b)", transform=ax_b.transAxes, va="top",
              ha="left", fontweight="bold")
    ax_b.set_title(rf"(b) force shape $\tilde{{F}}(y)$  (W = {CHANNEL_WIDTH*1e6:.0f} $\mu$m)",
                   fontsize=9)

    # --- (c) harmonic-reshaping factor E_foc/E_1 vs V_drive -------------
    ax_c = axes[1, 0]
    ax_c.axhline(1, color="0.6", lw=0.5, ls="--",
                 label=r"$E_\mathrm{foc}/E_1=1$ (no reshaping)")
    ax_c.errorbar(V, e_foc, yerr=e_foc_sig, fmt="o", ms=4,
                  color="C0", capsize=2, elinewidth=0.6,
                  label=r"$E_\mathrm{foc}/E_1$ (harmonic-reshaping)")
    ax_c.set_xlabel(r"$V_\mathrm{drive}$ [V$_\mathrm{pp}$]")
    ax_c.set_ylabel(r"$E_\mathrm{foc}/E_1$")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(frameon=False, loc="lower left", fontsize=8)
    ax_c.text(-0.15, 0.99, "(c)", transform=ax_c.transAxes, va="top",
              ha="left", fontweight="bold")
    decision = bool(d["decision_pass"])
    ax_c.set_title(f"(c) harmonic-reshaping factor -- decision: "
                   f"{'PASS' if decision else 'FAIL'}", fontsize=9)

    # --- (d) two-factor decomposition vs P_in ---------------------------
    ax_d = axes[1, 1]
    finite = np.isfinite(p_in) & (p_in > 0)
    if np.any(finite):
        ax_d.axhline(1, color="0.6", lw=0.5, ls="--",
                     label=r"$R_\kappa=1$ (no penalty)")
        ax_d.errorbar(p_in[finite] * 1e3, e1_factor[finite],
                      yerr=e1_factor_sig[finite],
                      fmt="^", ms=4, color="C1", capsize=2, elinewidth=0.6,
                      label=r"$E_1/E_\mathrm{lin}$ (fundamental)")
        ax_d.errorbar(p_in[finite] * 1e3, e_foc[finite], yerr=e_foc_sig[finite],
                      fmt="s", ms=4, color="C2", capsize=2, elinewidth=0.6,
                      label=r"$E_\mathrm{foc}/E_1$ (reshaping)")
        ax_d.errorbar(p_in[finite] * 1e3, r_kappa[finite], yerr=r_kappa_sig[finite],
                      fmt="o", ms=4, color="C0", capsize=2, elinewidth=0.6,
                      label=r"$R_\kappa$ (product)")
        ax_d.set_xlabel(r"$P_\mathrm{in}=\langle vi\rangle$ [mW]")
        ax_d.set_xscale("log")
    else:
        ax_d.text(0.5, 0.5, "fig2.npz missing -- panel (d) unavailable",
                  transform=ax_d.transAxes, ha="center", va="center")
    ax_d.set_ylabel("ratio")
    ax_d.grid(True, which="both", alpha=0.3)
    ax_d.legend(frameon=False, loc="best", fontsize=7)
    ax_d.text(-0.15, 0.99, "(d)", transform=ax_d.transAxes, va="top",
              ha="left", fontweight="bold")
    top_idx = int(np.argmax(V))
    rk_top = float(r_kappa[top_idx]) if np.isfinite(r_kappa[top_idx]) else float("nan")
    ax_d.set_title(rf"(d) $R_\kappa = (E_1/E_\mathrm{{lin}})\cdot(E_\mathrm{{foc}}/E_1)$  "
                   rf"-- top drive $R_\kappa$ = {rk_top:.3f}", fontsize=9)

    fig.suptitle("PRL Fig 3 (W21) -- self-generated harmonics reshape the radiation force",
                 fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_ftilde_breakdown(d: dict):
    """Debug-only: F~(y) per-harmonic breakdown at the top drive on
    the CENTERED transverse coordinate y/W in [-1/2, +1/2].

    Each n=1..depth contribution is plotted separately, plus the
    cumulative sum. With centered y, odd n have NEGATIVE slope at y=0
    (focusing) and even n have POSITIVE slope (defocusing) -- the same
    parity that gives the alternating signs in R_kappa.

    Saved to ``fig3_debug_ftilde_breakdown.png``; NOT part of the
    publication figure set.
    """
    ratio = d["e_n_over_e1"]
    V = d["pzt_vpp"]
    depth = int(d["truncation_depth"])
    y = d["y_over_w_centered"]
    i_top = int(np.argmax(V))
    v_top = int(V[i_top])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.axvline(0, color="0.6", lw=0.5, ls="--")
    cumulative = np.zeros_like(y)
    for n in range(1, depth + 1):
        contrib = ((-1) ** n) * n * float(ratio[i_top, n - 1]) * np.sin(2 * n * np.pi * y)
        cumulative = cumulative + contrib
        ax.plot(y, contrib, "-", lw=0.9, color=COLORS[n],
                label=(rf"$n={n}$: $(-1)^{{{n}}}\cdot{n}\cdot(E_{{{n}f}}/E_1)$"
                       rf"$\sin(2\cdot{n}\pi y/W)$"
                       rf"   [$E_{{{n}f}}/E_1={float(ratio[i_top, n-1]):.2e}$]"))
    ax.plot(y, cumulative, "k--", lw=1.2, label=r"sum $= \tilde{F}(y)$")
    ax.set_xlabel(r"$y/W$ (centered, walls at $\pm 1/2$)")
    ax.set_ylabel(r"$\tilde{F}(y)$ contribution")
    ax.legend(frameon=False, loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title(rf"Fig 3 debug: $\tilde{{F}}(y)$ per-harmonic breakdown at {v_top} Vpp",
                 fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    PIPE_OUT.mkdir(parents=True, exist_ok=True)
    d = compute()
    fig = plot(d)
    fig.savefig(PIPE_OUT / "fig3.png", dpi=200)
    fig.savefig(PIPE_OUT / "fig3.pdf")
    plt.close(fig)
    # Debug-only breakdown — separate file, not part of the publication set
    fig_dbg = plot_ftilde_breakdown(d)
    fig_dbg.savefig(PIPE_OUT / "fig3_debug_ftilde_breakdown.png", dpi=200)
    plt.close(fig_dbg)
    np.savez(PIPE_OUT / "fig3.npz", **d)
    top = int(np.argmax(d["pzt_vpp"]))
    (PIPE_OUT / "fig3.json").write_text(json.dumps({
        "decision_pass": bool(d["decision_pass"]),
        "decision_msg": str(d["decision_msg"]),
        "global_series_depth": int(d["truncation_depth"]),
        "ever_adopted_n": [bool(x) for x in d["ever_adopted_n"]],
        "e_foc_over_e1_top_drive": float(d["e_foc_over_e1"][top]),
        "r_kappa_top_drive": (float(d["r_kappa"][top])
                              if np.isfinite(d["r_kappa"][top]) else None),
    }, indent=2), encoding="utf-8")
    print(f"  Saved {PIPE_OUT / 'fig3.png'}")
    print(f"  global series depth: n=1..{int(d['truncation_depth'])}")
    print(f"  ever-adopted n=1..5: {[bool(x) for x in d['ever_adopted_n']]}")
    print(f"  {d['decision_msg']}")


if __name__ == "__main__":
    main()
