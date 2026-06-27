"""Fig 2 — cascade scaling, Coppens slope, and input-power response.

Combined 4-panel figure (2x2 layout):

- (a) P_nf vs V_drive for n=1,2,3 (perturbative scaling) — descriptive,
  ports ``prl_draft.fig2a`` and adds bootstrap CI on the V^n exponent.
- (b) Harmonic ratios P_2f/P_1f and P_3f/P_1f vs Mach with K_exp
  origin-through fit (Coppens slope) — descriptive.
- (c) E_1 = |P_1f|^2/(4 rho c^2) vs time-domain P_in with linear
  baseline — **decisional** (Fig 2c per contract Outcome Matrix).
- (d) Deviation E_1/E_1^lin - 1 with combined 1-sigma band.

Output: ``PIPE_OUT/fig2.{png,pdf,npz}``. The npz contains all arrays for
panels a/b/c plus the panel-c decision result.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from conventions import (
    BETA,
    CASCADE_SCANS,
    C_0,
    LOW_DRIVE_P2_OVER_P1,
    MIN_LOW_DRIVE_POINTS,
    PIPE_OUT,
    RHO_0,
    W21_DATA_ROOT,
    get_cache_dir,
)
from _data import compute_pin_spectral, compute_pin_time, load_ladder

HARMS = (1, 2, 3, 4, 5)         # full set used in compute()
HARMS_3F = (1, 2, 3)            # shown in the main fig2 panel (a)
COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green",
          4: "tab:purple", 5: "tab:orange"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
ADOPTION_SNR = 3.0              # below this on the 5f variant: open markers

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


# ----- (a) cascade scaling -------------------------------------------------

def _pert_k(V, pn, n, window):
    """Through-origin coefficient k for P_nf = k V^n over the low-drive
    window. Returns None if no points."""
    use = window & np.isfinite(pn) & (pn > 0)
    if int(np.sum(use)) < 1:
        return None
    vf, pf = V[use], pn[use]
    return float(np.sum(vf**n * pf) / np.sum(vf ** (2 * n)))


def _exponent_sig_distance(central: float, lo: float, hi: float, ideal: float) -> float:
    """Asymmetric-bootstrap sigma distance from ``ideal`` exponent.

    Uses the relevant half-width (toward the ideal value): if central is
    above ideal, the lower 1-sigma half-width ``central - lo``; otherwise
    the upper half-width ``hi - central``. Returns inf if the relevant
    half-width is non-positive.
    """
    if not (np.isfinite(central) and np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    half = (central - lo) if central > ideal else (hi - central)
    if half <= 0:
        return float("inf")
    return abs(central - ideal) / half


def _exponent_label(sig_dist: float) -> str:
    if not np.isfinite(sig_dist):
        return "unresolved"
    if sig_dist < 1.0:
        return "consistent"
    if sig_dist < 3.0:
        return "weakly resolved deficit"
    return "significantly below"


def _exponent_with_se(V, pn, sigma_pn, mask):
    """Log-log weighted least-squares slope + analytic SE.

    Fit ``log(P_n) = log C + p * log(V)`` on the masked points with
    per-point weights ``w = 1 / (sigma_pn/pn)^2`` (variance of log P).
    Returns (p_central, p_minus_se, p_plus_se).

    No bootstrap. The bootstrap with N=3 fit points is over-tight
    because the resample picks duplicates with high probability,
    whereas the analytic SE propagates per-point uncertainties
    consistently. See response to user 2026-06-27 for the reasoning.
    """
    use = mask & np.isfinite(pn) & (pn > 0) & np.isfinite(sigma_pn) & (sigma_pn > 0)
    nk = int(np.sum(use))
    if nk < 2:
        return float("nan"), float("nan"), float("nan")
    lv = np.log(V[use])
    lp = np.log(pn[use])
    w = (pn[use] / sigma_pn[use]) ** 2          # 1 / var(log P)
    # Weighted LS: minimize sum w (lp - p*lv - c)^2
    lv_bar = float(np.sum(w * lv) / np.sum(w))
    lp_bar = float(np.sum(w * lp) / np.sum(w))
    Sxx = float(np.sum(w * (lv - lv_bar) ** 2))
    Sxy = float(np.sum(w * (lv - lv_bar) * (lp - lp_bar)))
    p_cen = Sxy / Sxx if Sxx > 0 else float("nan")
    if nk < 3 or not np.isfinite(p_cen):
        return p_cen, float("nan"), float("nan")
    # Residual MS scaled by reduced chi^2 (conservative when chi^2_nu > 1)
    resid = lp - lv_bar * 0 - lp_bar - p_cen * (lv - lv_bar)
    chi2_nu = float(np.sum(w * resid ** 2) / (nk - 2))
    se = float(np.sqrt(max(chi2_nu, 1.0) / Sxx))
    return p_cen, p_cen - se, p_cen + se


# ----- (b) K_exp Coppens slope --------------------------------------------

def _kexp_fit(M, ratio, sigma_ratio, mask):
    """Origin-through weighted fit ``P_2f/P_1f = (beta/4) * M * K_exp``.

    Returns (K_exp, K_se). Analytic standard error from the weighted-
    LS formula. No bootstrap (see ``_exponent_with_se``).
    """
    use = mask & np.isfinite(ratio) & np.isfinite(sigma_ratio) & (sigma_ratio > 0)
    nk = int(np.sum(use))
    if nk < 1:
        return float("nan"), float("nan")
    x = M[use]
    y = ratio[use]
    w = 1.0 / sigma_ratio[use] ** 2
    slope = float(np.sum(w * x * y) / np.sum(w * x * x))
    slope_se = float(np.sqrt(1.0 / np.sum(w * x * x)))
    K_exp = slope * 4.0 / BETA
    K_se = slope_se * 4.0 / BETA
    return K_exp, K_se


# ----- (c) E_1 vs P_in -----------------------------------------------------

def _gather_pin(L: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every cascade scan, time-domain <v i> at the operating-point file.

    Returns (p_in_w, p_in_se_w, p_in_spec_w, p_in_spec_se_w). Both forms
    are computed so Fig S1 can scatter them against each other.
    """
    scans = list(CASCADE_SCANS)
    selected = [str(x) for x in L["selected_files"]]
    N = len(scans)
    assert len(selected) == N, f"ladder vs SCANS length mismatch {N} vs {len(selected)}"
    pin = np.full(N, np.nan)
    pin_se = np.full(N, np.nan)
    pin_sp = np.full(N, np.nan)
    pin_sp_se = np.full(N, np.nan)
    for i, (label, dirname) in enumerate(scans):
        run_dir = W21_DATA_ROOT / dirname
        scan_path = run_dir / selected[i]
        cache_dir = get_cache_dir(dirname, str(Path(__file__).parent / "fig2.py"))
        if not scan_path.exists():
            print(f"  {label:>3} Vpp: missing {scan_path.name} -- skip")
            continue
        p, se = compute_pin_time(scan_path, cache_dir, n_points=5)
        sp, sp_se = compute_pin_spectral(scan_path, cache_dir)
        pin[i] = p
        pin_se[i] = se
        pin_sp[i] = sp
        pin_sp_se[i] = sp_se
        print(
            f"  {label:>3} Vpp ({selected[i]}): "
            f"P_in(time)={p*1e3:.2f} +/- {se*1e3:.2f} mW, "
            f"P_in(spec)={sp*1e3:.2f} mW"
        )
    return pin, pin_se, pin_sp, pin_sp_se


def compute() -> dict:
    L = load_ladder()

    V = L["pzt_vpp"]
    P = L["p_kpa"]
    Pstd = L["p_std_kpa"]
    mask = L["low_drive"]
    n_low = int(np.sum(mask))

    # ----- (a) cascade scaling: V^n coefficients + analytic SE on exponent --
    k_n = np.array([_pert_k(V, P[:, n - 1], n, mask) or float("nan") for n in HARMS])
    p_exp = np.full(len(HARMS), np.nan)
    p_exp_lo = np.full(len(HARMS), np.nan)
    p_exp_hi = np.full(len(HARMS), np.nan)
    for j, n in enumerate(HARMS):
        p_exp[j], p_exp_lo[j], p_exp_hi[j] = _exponent_with_se(
            V, P[:, n - 1], Pstd[:, n - 1], mask
        )

    # ----- (b) K_exp slope -------------------------------------------------
    P_pa = P * 1e3
    S_pa = Pstd * 1e3
    M = P_pa[:, 0] / (RHO_0 * C_0**2)
    ratio_21 = P_pa[:, 1] / P_pa[:, 0]
    sigma_r21 = ratio_21 * np.sqrt((S_pa[:, 1] / P_pa[:, 1]) ** 2 +
                                    (S_pa[:, 0] / P_pa[:, 0]) ** 2)
    K_exp, K_se = _kexp_fit(M, ratio_21, sigma_r21, mask)

    # P_3/P_1 (no fit, just plotted)
    ratio_31 = P_pa[:, 2] / P_pa[:, 0]
    sigma_r31 = ratio_31 * np.sqrt((S_pa[:, 2] / P_pa[:, 2]) ** 2 +
                                    (S_pa[:, 0] / P_pa[:, 0]) ** 2)

    # ----- (c) E_1 vs P_in -------------------------------------------------
    print("Fig2c: computing time-domain P_in for 12 cascade operating points...")
    p_in, p_in_se, p_in_spec, p_in_spec_se = _gather_pin(L)
    E1 = L["e_n_j_m3"][:, 0]
    E1_sigma = L["e_n_sigma_j_m3"][:, 0]

    # weighted origin-through fit on low-drive: E1 = s * P_in
    use_c = mask & np.isfinite(p_in) & (p_in > 0) & np.isfinite(E1) & (E1_sigma > 0)
    nuse = int(np.sum(use_c))
    if nuse >= 1:
        w = 1.0 / E1_sigma[use_c] ** 2
        s_slope = float(np.sum(w * p_in[use_c] * E1[use_c]) /
                        np.sum(w * p_in[use_c] ** 2))
        s_se = float(np.sqrt(1.0 / np.sum(w * p_in[use_c] ** 2)))
    else:
        s_slope = float("nan")
        s_se = float("nan")

    # deviation + combined sigma
    E1_lin = s_slope * p_in
    deviation = E1 / E1_lin - 1.0
    # delta f = f(E1, P_in, s) = E1/(s P_in) - 1 ; partial derivatives
    with np.errstate(divide="ignore", invalid="ignore"):
        d_dE1 = 1.0 / (s_slope * p_in)
        d_dPin = -E1 / (s_slope * p_in**2)
        d_ds = -E1 / (s_slope**2 * p_in)
    dev_sigma = np.sqrt(
        (d_dE1 * E1_sigma) ** 2
        + (d_dPin * p_in_se) ** 2
        + (d_ds * s_se) ** 2
    )

    # decision: top-drive deviation > 3 sigma?
    top = int(np.argmax(V))
    dev_top = float(deviation[top])
    sig_top = float(dev_sigma[top])
    decision_pass = bool(np.isfinite(dev_top) and abs(dev_top) > 3 * sig_top)
    decision_msg = (
        f"Fig 2c at {int(V[top])} Vpp: E1/E1_lin - 1 = {dev_top:+.3f} "
        f"+/- {sig_top:.3f}  -> {'PASS (>3 sigma)' if decision_pass else 'FAIL (<3 sigma)'}"
    )

    # ----- pack ------------------------------------------------------------
    return dict(
        # axes / cascade
        pzt_vpp=V,
        p_kpa=P[:, : len(HARMS)],
        p_std_kpa=Pstd[:, : len(HARMS)],
        snr=L["snr"][:, : len(HARMS)],
        noise_kpa=L["noise_kpa"][:, : len(HARMS)],
        low_drive_mask=mask,
        low_drive_count=np.asarray(n_low),
        # (a) scaling
        scaling_v_coeff=k_n,
        scaling_exponent=p_exp,
        scaling_exponent_lo=p_exp_lo,
        scaling_exponent_hi=p_exp_hi,
        scaling_sigma_from_ideal=np.array(
            [_exponent_sig_distance(p_exp[j], p_exp_lo[j], p_exp_hi[j], float(n))
             for j, n in enumerate(HARMS)]),
        scaling_label=np.array(
            [_exponent_label(_exponent_sig_distance(p_exp[j], p_exp_lo[j], p_exp_hi[j], float(n)))
             for j, n in enumerate(HARMS)]),
        # (b) Coppens
        m_mach=M,
        ratio_2f=ratio_21,
        ratio_2f_sigma=sigma_r21,
        ratio_3f=ratio_31,
        ratio_3f_sigma=sigma_r31,
        coppens_K_exp=np.asarray(K_exp),
        coppens_K_exp_se=np.asarray(K_se),
        # (c) input power
        p_in_w=p_in,
        p_in_se_w=p_in_se,
        p_in_spec_w=p_in_spec,
        p_in_spec_se_w=p_in_spec_se,
        e_1_j_m3=E1,
        e_1_sigma_j_m3=E1_sigma,
        e_1_lin_slope_jm3_per_w=np.asarray(s_slope),
        e_1_lin_slope_se=np.asarray(s_se),
        deviation=deviation,
        deviation_sigma=dev_sigma,
        # decision
        decision_pass=np.asarray(decision_pass),
        decision_msg=np.asarray(decision_msg),
    )


def plot(d: dict):
    V = d["pzt_vpp"]
    P = d["p_kpa"]
    Pstd = d["p_std_kpa"]
    mask = d["low_drive_mask"]
    M = d["m_mach"]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))

    # --- (a) P_nf vs V_drive ----------------------------------------------
    ax_a = axes[0, 0]
    v_fine = np.linspace(0, V.max() * 1.05, 200)
    ax_a.errorbar(V, P[:, 0] / 1e3, yerr=Pstd[:, 0] / 1e3,
                  fmt=MARKERS[1], ms=3.5, color=COLORS[1], capsize=2,
                  elinewidth=0.6, label=r"$\hat{P}_{1f}$")
    k1 = d["scaling_v_coeff"][0]
    if np.isfinite(k1):
        ax_a.plot(v_fine, k1 * v_fine / 1e3, ":", lw=0.6, color=COLORS[1])
    ax_a.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax_a.set_ylabel(r"$\hat{P}_{1f}$ [MPa]", color=COLORS[1])
    ax_a.tick_params(axis="y", labelcolor=COLORS[1])
    ax_a.set_xlim(0, V.max() * 1.05)
    ax_ar = ax_a.twinx()
    right_top = 0.0
    p_central = d["scaling_exponent"]
    for n in (2, 3, 4, 5):
        pn_mpa = P[:, n - 1] / 1e3            # kPa -> MPa for unit consistency
        sn_mpa = Pstd[:, n - 1] / 1e3
        ax_ar.errorbar(V, pn_mpa, yerr=sn_mpa,
                       fmt=MARKERS[n], ms=3.5, color=COLORS[n], capsize=2,
                       elinewidth=0.6, label=rf"$\hat{{P}}_{{{n}f}}$")
        right_top = max(right_top, float(np.nanmax(pn_mpa)))
        # V^n guide only if the fitted exponent is sanely near n
        k = d["scaling_v_coeff"][n - 1]
        p_n = float(p_central[n - 1])
        if np.isfinite(k) and np.isfinite(p_n) and abs(p_n - n) < 1.0:
            line = k * v_fine**n / 1e3
            right_top = max(right_top, float(line.max()))
            ax_ar.plot(v_fine, line, ":", lw=0.6, color=COLORS[n])
    ax_ar.set_ylabel(r"$\hat{P}_{2f}$..$\hat{P}_{5f}$ [MPa]")
    ax_ar.set_ylim(0, right_top * 1.1)
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_ar.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax_a.text(-0.15, 0.98, "(a)", transform=ax_a.transAxes, va="top",
              ha="left", fontweight="bold")
    p_c = d["scaling_exponent"]
    p_lo = d["scaling_exponent_lo"]
    p_hi = d["scaling_exponent_hi"]
    # asymmetric half-widths (toward each ideal n)
    def _hw(j, ideal):
        c, lo, hi = float(p_c[j]), float(p_lo[j]), float(p_hi[j])
        return (c - lo) if c > ideal else (hi - c)
    bits = []
    for n in HARMS_3F:                           # only the harmonics shown in (a)
        j = n - 1
        c, hw = float(p_c[j]), _hw(j, float(n))
        bits.append(rf"$p_{{{n}}}$={c:.2f}$\pm${hw:.2f}")
    ax_a.set_title(
        rf"(a) cascade exponents (1$\sigma$, low-drive window): " + ", ".join(bits),
        fontsize=9,
    )

    # --- (b) Coppens slope (2f only) -------------------------------------
    ax_b = axes[0, 1]
    R2 = d["ratio_2f"]; sR2 = d["ratio_2f_sigma"]
    ax_b.errorbar(M, R2, yerr=sR2, fmt=MARKERS[2], ms=3.5, color=COLORS[2],
                  capsize=2, elinewidth=0.6, label=r"$\hat{P}_{2f}/\hat{P}_{1f}$")
    K_exp = float(d["coppens_K_exp"])
    K_se = float(d["coppens_K_exp_se"])
    if np.isfinite(K_exp):
        m_fine = np.linspace(0, M.max() * 1.05, 100)
        ax_b.plot(m_fine, (BETA / 4.0) * m_fine * K_exp, "--",
                  color=COLORS[2], lw=0.8, label="low-drive fit")
    ax_b.set_xlabel(r"$M=\hat{P}_{1f}/\rho c^2$")
    ax_b.set_ylabel(r"$\hat{P}_{2f}/\hat{P}_{1f}$")
    ax_b.set_xlim(0, M.max() * 1.05)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(frameon=False, loc="upper left")
    ax_b.grid(True, alpha=0.3)
    ax_b.text(-0.15, 0.98, "(b)", transform=ax_b.transAxes, va="top",
              ha="left", fontweight="bold")
    ax_b.set_title(rf"(b) Coppens slope $K_\mathrm{{exp}}={K_exp:.1f}\pm{K_se:.1f}$",
                   fontsize=9)

    # --- (c) E_1 vs P_in --------------------------------------------------
    ax_c = axes[1, 0]
    p_in = d["p_in_w"]
    E1 = d["e_1_j_m3"]
    E1_sig = d["e_1_sigma_j_m3"]
    p_in_se = d["p_in_se_w"]
    s = float(d["e_1_lin_slope_jm3_per_w"])
    s_se = float(d["e_1_lin_slope_se"])
    use_plot = np.isfinite(p_in) & np.isfinite(E1)
    ax_c.errorbar(p_in[use_plot] * 1e3, E1[use_plot] / 1e3,
                  xerr=p_in_se[use_plot] * 1e3, yerr=E1_sig[use_plot] / 1e3,
                  fmt="o", ms=3.5, color="C0", capsize=2, elinewidth=0.6,
                  label="data")
    # mark which were used for the linear fit
    fit_mask = mask & use_plot
    ax_c.plot(p_in[fit_mask] * 1e3, E1[fit_mask] / 1e3, "o", ms=3.5,
              color="C3", mfc="white", mew=1.0, label="low-drive (fit)")
    if np.isfinite(s) and p_in[use_plot].max() > 0:
        p_fine = np.linspace(0, float(p_in[use_plot].max()) * 1.05, 100)
        ax_c.plot(p_fine * 1e3, s * p_fine / 1e3, "--", color="0.4",
                  lw=0.8, label=rf"$E_1^\mathrm{{lin}}={s:.2g}\cdot P_\mathrm{{in}}$")
    ax_c.set_xlabel(r"$P_\mathrm{in}=\langle v(t)i(t)\rangle$ [mW]")
    ax_c.set_ylabel(r"$E_1$ [kJ/m$^3$]")
    ax_c.set_xlim(left=0)
    ax_c.set_ylim(bottom=0)
    ax_c.legend(frameon=False, loc="upper left")
    ax_c.grid(True, alpha=0.3)
    ax_c.text(-0.15, 0.98, "(c)", transform=ax_c.transAxes, va="top",
              ha="left", fontweight="bold")
    ax_c.set_title(f"(c) input power -- slope {s:.2g}+/-{s_se:.1g}", fontsize=9)

    # --- (d) deviation panel ----------------------------------------------
    ax_d = axes[1, 1]
    dev = d["deviation"]
    dev_sig = d["deviation_sigma"]
    ax_d.axhline(0, color="0.6", lw=0.5, ls="--")
    ax_d.errorbar(V[use_plot], dev[use_plot] * 100,
                  yerr=dev_sig[use_plot] * 100,
                  fmt="o", ms=3.5, color="C0", capsize=2, elinewidth=0.6)
    ax_d.fill_between(V[use_plot], -dev_sig[use_plot] * 100,
                      dev_sig[use_plot] * 100, color="C0", alpha=0.12,
                      label=r"$\pm 1\sigma$ band")
    ax_d.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax_d.set_ylabel(r"$E_1/E_1^\mathrm{lin}-1$ [\%]")
    ax_d.legend(frameon=False, loc="upper left")
    ax_d.grid(True, alpha=0.3)
    ax_d.text(-0.15, 0.98, "(d)", transform=ax_d.transAxes, va="top",
              ha="left", fontweight="bold")
    decision = bool(d["decision_pass"])
    ax_d.set_title(f"(d) deviation -- decision: {'PASS' if decision else 'FAIL'}",
                   fontsize=9)

    fig.suptitle(
        rf"PRL Fig 2 (W21) -- low-drive window $|P_{{2f}}/P_{{1f}}|<{LOW_DRIVE_P2_OVER_P1}$ "
        f"({int(d['low_drive_count'])} pts)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_fig2a_5f(d: dict):
    """Standalone single-log-axis 5-harmonic variant of Fig 2a.

    All five harmonics on a single log y-axis in kPa, with V^n
    perturbative guides (n=1..5). Open markers for SNR<3 ("below
    noise"); filled markers otherwise. Same low-drive perturbative
    window as the main fig2 panel (a).
    """
    V = d["pzt_vpp"]
    P = d["p_kpa"]
    Pstd = d["p_std_kpa"]
    SNR = d["snr"]
    NOISE = d["noise_kpa"]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    v_fine = np.linspace(V.min() * 0.85, V.max() * 1.05, 200)
    guide_top = 0.0
    for n in HARMS:
        pn = P[:, n - 1]
        sn = Pstd[:, n - 1]
        snr_n = SNR[:, n - 1]
        # Floor lower whisker at the noise floor on log axis so below-
        # noise error bars don't plunge a meaningless decade.
        lower = np.maximum(pn - sn, NOISE[:, n - 1])
        yerr_lo = np.clip(pn - lower, 0, None)
        det = snr_n >= ADOPTION_SNR
        # filled: detected
        ax.errorbar(V[det], pn[det],
                    yerr=[yerr_lo[det], sn[det]],
                    fmt=MARKERS[n], ms=4, color=COLORS[n], capsize=2,
                    elinewidth=0.6, label=rf"$\hat{{P}}_{{{n}f}}$")
        # open: below SNR threshold
        if np.any(~det):
            ax.errorbar(V[~det], pn[~det],
                        yerr=[yerr_lo[~det], sn[~det]],
                        fmt=MARKERS[n], ms=4, color=COLORS[n], mfc="none",
                        capsize=2, elinewidth=0.6, alpha=0.65)
        # Only draw V^n perturbative guide if the fitted exponent is
        # within 1 of the ideal n (sanity check: the fit hasn't
        # degenerated into noise, which happens for 4f/5f at low drive).
        k = d["scaling_v_coeff"][n - 1]
        p_central = float(d["scaling_exponent"][n - 1])
        if np.isfinite(k) and np.isfinite(p_central) and abs(p_central - n) < 1.0:
            guide = k * v_fine**n
            guide_top = max(guide_top, float(guide.max()))
            ax.plot(v_fine, guide, ":", lw=0.6, color=COLORS[n])

    ax.set_yscale("log")
    ax.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
    ax.set_ylabel(r"$\hat{P}_{nf}$ [kPa]")
    ax.set_xlim(0, V.max() * 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    ax.text(-0.13, 0.99, "(a)", transform=ax.transAxes, va="top",
            ha="left", fontweight="bold")
    ax.set_title(
        rf"PRL Fig 2a (5-harmonic variant, W21) -- "
        rf"$\propto V^n$ over $P_{{2f}}/P_{{1f}}<{LOW_DRIVE_P2_OVER_P1}$; "
        rf"open = SNR$<${int(ADOPTION_SNR)}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def _strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    PIPE_OUT.mkdir(parents=True, exist_ok=True)
    d = compute()
    fig = plot(d)
    fig.savefig(PIPE_OUT / "fig2.png", dpi=200)
    fig.savefig(PIPE_OUT / "fig2.pdf")
    plt.close(fig)
    # 5-harmonic variant of panel (a) on a single log axis
    fig5 = plot_fig2a_5f(d)
    fig5.savefig(PIPE_OUT / "fig2a_5f.png", dpi=200)
    fig5.savefig(PIPE_OUT / "fig2a_5f.pdf")
    plt.close(fig5)
    np.savez(PIPE_OUT / "fig2.npz", **_strip_private(d))
    # sidecar: human-readable decision summary
    # Fig 2a decision: |p_n - n| < 0.1 with 1-sigma (analysis_contract.md:165)
    fig2a_per_harmonic = []
    for j, n in enumerate(HARMS_3F):                # 1f, 2f, 3f are shown in 2a
        p = float(d["scaling_exponent"][n - 1])
        lo = float(d["scaling_exponent_lo"][n - 1])
        hi = float(d["scaling_exponent_hi"][n - 1])
        passes = bool(abs(p - n) < 0.1)
        fig2a_per_harmonic.append({
            "n": int(n),
            "p_central": p,
            "p_lo_16th": lo,
            "p_hi_84th": hi,
            "sigma_from_ideal": float(d["scaling_sigma_from_ideal"][n - 1]),
            "interpretation": str(d["scaling_label"][n - 1]),
            "passes_contract": passes,
        })
    fig2a_decision_pass = all(h["passes_contract"] for h in fig2a_per_harmonic)
    if fig2a_decision_pass:
        fig2a_msg = "All p_n within 0.1 of ideal: (1,2,3) supported."
    else:
        failing = [h["n"] for h in fig2a_per_harmonic if not h["passes_contract"]]
        fig2a_msg = (f"|p_n - n| >= 0.1 for n in {failing}. "
                     f"Fallback: restrict explicit exponent claim to passing harmonics.")

    (PIPE_OUT / "fig2.json").write_text(json.dumps({
        "fig2a": {
            "decision_pass": fig2a_decision_pass,
            "decision_msg": fig2a_msg,
            "scaling_exponents": fig2a_per_harmonic,
        },
        "fig2b": {
            "coppens_K_exp_dimensionless": float(d["coppens_K_exp"]),
            "coppens_K_exp_se": float(d["coppens_K_exp_se"]),
            "coppens_prefactor_per_GPa": float(d["coppens_K_exp"]) *
                (BETA / 4.0) / (RHO_0 * C_0**2) * 1e9,
            "low_drive_count": int(d["low_drive_count"]),
        },
        "fig2c": {
            "decision_pass": bool(d["decision_pass"]),
            "decision_msg": str(d["decision_msg"]),
        },
    }, indent=2), encoding="utf-8")
    n_low = int(d["low_drive_count"])
    if n_low < MIN_LOW_DRIVE_POINTS:
        print(f"  WARNING: only {n_low} low-drive points "
              f"({MIN_LOW_DRIVE_POINTS} required); drop V^n claim from caption.")
    print(f"  Saved {PIPE_OUT / 'fig2.png'}")
    print(f"  Saved {PIPE_OUT / 'fig2a_5f.png'}")
    print("  exponents vs ideal (bootstrap 1sigma toward ideal):")
    for j, n in enumerate(HARMS):
        c = float(d["scaling_exponent"][j])
        lo = float(d["scaling_exponent_lo"][j])
        hi = float(d["scaling_exponent_hi"][j])
        sig = float(d["scaling_sigma_from_ideal"][j])
        label = str(d["scaling_label"][j])
        print(f"    p_{n} = {c:.2f}  [{lo:.2f}, {hi:.2f}]  ({sig:.1f}sigma from {n}: {label})")
    K = float(d['coppens_K_exp'])
    K_se = float(d['coppens_K_exp_se'])
    K_pref_perGPa = K * (BETA / 4.0) / (RHO_0 * C_0**2) * 1e9
    print(f"  K_exp (dimensionless) = {K:.2f} +/- {K_se:.2f}  (weighted-LS analytic SE)")
    print(f"  -> Coppens prefactor P_2f/P_1f^2 = {K_pref_perGPa:.2f} /GPa")
    print(f"  {d['decision_msg']}")


if __name__ == "__main__":
    main()
