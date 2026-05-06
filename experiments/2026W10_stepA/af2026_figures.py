# %%
"""Generate AF2026 abstract figures (Fig1, Fig2) from existing PRA caches.

Loads Fig7.npz and Fig8.npz produced by manuscript_figures.py (no
recomputation), and the external simulation cache fig3.npz (which
contains both self-consistent and perturbative theory curves).

Output: $MANUSCRIPT_DIR/acoustofluidics/figures/Fig{1,2}.{eps,png}

Notation differs from PRA per plans/af2026_figures.md:
  - hat on scalar peak amplitudes ($\\hat{P}_{nf}$)
  - no angle brackets around $E_\\mathrm{ac}$
  - perturbative-theory comparison instead of self-consistent simulation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import C_SOUND, RHO, MANUSCRIPT_DIR

if MANUSCRIPT_DIR is None:
    raise RuntimeError("MANUSCRIPT_DIR not set — add it to .env or environment")

PRA_FIG_DIR = MANUSCRIPT_DIR / "pra" / "figures"
AF_FIG_DIR = MANUSCRIPT_DIR / "acoustofluidics" / "figures"
AF_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "lines.linewidth": 0.75,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})


def save_fig(fig, name):
    # Save with full figure width preserved but tight vertical extent
    # (trim white space top/bottom only).
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_w, _ = fig.get_size_inches()
    bbox = Bbox.from_extents(0.0, tight.y0, fig_w, tight.y1)
    fig.savefig(AF_FIG_DIR / f"{name}.eps", dpi=1200, bbox_inches=bbox)
    fig.savefig(AF_FIG_DIR / f"{name}.png", dpi=1200, bbox_inches=bbox)
    print(f"Saved: {AF_FIG_DIR / name}.eps/.png")


# %%
# =============================================================================
# Fig 1 — Spatial mode shapes at 25 Vpp (from Fig7.npz)
# =============================================================================
d7 = np.load(PRA_FIG_DIR / "Fig7.npz")

# vpp_1 is 25 Vpp slot in the cache
vpp_25 = int(d7["vpp_1"])
y_um = d7["y_um_1"]
y_th = d7["y_th_um"]
p1f = d7["p1f_mpa_1"]
p2f = d7["p2f_mpa_1"]
fit_1f = d7["fit_1f_mpa_1"]
fit_2f = d7["fit_2f_mpa_1"]
p0_1f = float(d7["p0_1f_1"])
p0_2f = float(d7["p0_2f_1"])
grid_1f = d7["grid_1f_25_mpa"]
grid_2f = d7["grid_2f_25_mpa"]
w_mm = d7["w_mm"]
PZT_CENTER_MM = 8.6  # PZT spans 5.6-11.6 mm; centre at 8.6 mm
l_mm = d7["l_mm"] - PZT_CENTER_MM
y_best = float(d7["y_best"])
y_best_centred_mm = y_best * 1e3 - PZT_CENTER_MM

_y_label = (r"$y$ [\textmu m]"
            if plt.rcParams.get("text.usetex", False) else "$y$ [μm]")

fig, axes = plt.subplots(2, 2, figsize=(6.7, 3.0),
                          gridspec_kw={"height_ratios": [1, 1]})
ax_a, ax_b = axes[0]  # top row: 2D pressure maps
ax_c, ax_d = axes[1]  # bottom row: line profiles + fits
_lbl_kw = dict(va="top", ha="left", fontweight="bold")

# Top row: 2D pressure maps
for ax_map, grid, lbl, cbar_lbl in [
    (ax_a, grid_1f, "(a)", r"$|P_{1f}(x,y)|$ [MPa]"),
    (ax_b, grid_2f, "(b)", r"$|P_{2f}(x,y)|$ [MPa]"),
]:
    lo, hi = np.nanpercentile(grid, [5, 95])
    im = ax_map.pcolormesh(l_mm, w_mm, grid, shading="nearest",
                            cmap="viridis", vmin=lo, vmax=hi)
    ax_map.axvline(y_best_centred_mm, color="red", linewidth=0.8, ls="--")
    ax_map.set_xlabel("$x$ [mm]")
    ax_map.set_ylabel("$y$ [mm]")
    ax_map.set_xlim(-3.4, 3.4)
    ax_map.set_aspect("auto")
    ax_map.text(-0.15, 0.98, lbl, transform=ax_map.transAxes, **_lbl_kw)
    cb = fig.colorbar(im, ax=ax_map, pad=0.02)
    cb.set_label(cbar_lbl)

# Bottom row: line profiles + fits
ax_c.plot(y_um, p1f, "ko", markersize=2)
ax_c.plot(y_th, fit_1f, "-", color="C3", linewidth=0.8,
          label="Fit")
ax_c.set_xlabel(_y_label)
ax_c.set_ylabel(r"$|P_{1f}(x,y)|$ [MPa]")
ax_c.set_ylim(bottom=0)
ax_c.text(-0.15, 1.0, "(c)", transform=ax_c.transAxes,
          va="top", ha="left", fontweight="bold")
ax_c.legend(loc="best", handlelength=1.2)

ax_d.plot(y_um, p2f, "ko", markersize=2)
ax_d.plot(y_th, fit_2f, "-", color="C3", linewidth=0.8,
          label="Fit")
ax_d.set_xlabel(_y_label)
ax_d.set_ylabel(r"$|P_{2f}(x,y)|$ [MPa]")
ax_d.set_ylim(0, 1.2 * p0_2f / 1e6)
ax_d.text(-0.15, 1.0, "(d)", transform=ax_d.transAxes,
          va="top", ha="left", fontweight="bold")
ax_d.legend(loc="best", handlelength=1.2)

plt.tight_layout(h_pad=0.3)
np.savez(AF_FIG_DIR / "Fig1.npz",
         vpp=vpp_25, y_um=y_um, y_th=y_th,
         p1f_mpa=p1f, p2f_mpa=p2f,
         fit_1f_mpa=fit_1f, fit_2f_mpa=fit_2f,
         p0_1f=p0_1f, p0_2f=p0_2f,
         grid_1f_mpa=grid_1f, grid_2f_mpa=grid_2f,
         w_mm=w_mm, l_mm=l_mm, y_best=y_best)
save_fig(fig, "Fig1")
plt.close()

print(f"\n--- AF Fig 1: spatial modes at {vpp_25} Vpp ---")
print(f"  hat{{P}}_1f = {p0_1f/1e3:.0f} kPa, hat{{P}}_2f = {p0_2f/1e3:.0f} kPa")


# %%
# =============================================================================
# Fig 2 — Drive-sweep harmonic ratio with perturbative theory (Fig8 + fig3)
# =============================================================================
d8 = np.load(PRA_FIG_DIR / "Fig8.npz")
sim = np.load(PRA_FIG_DIR / "fig3.npz")

# Experiment data
Vpp = d8["Vpp"]
exp_E = d8["E_ac"]
exp_ratio = d8["ratio"]
exp_p1f = d8["p0_1f"]
exp_p2f = d8["p0_2f"]
exp_s1f = d8["p0_1f_std"]
exp_s2f = d8["p0_2f_std"]
exp_ratio_std = exp_ratio * np.sqrt((exp_s2f / exp_p2f)**2
                                     + (exp_s1f / exp_p1f)**2)
a_1f = float(d8["a_1f"])
b_2f = float(d8["b_2f"])

# Perturbative theory: x-axis is P_1f (peak amplitude) directly
sim_p1f_pert = sim["p1f_pert"]  # Pa
sim_E_pert = sim_p1f_pert**2 / (4 * RHO * C_SOUND**2)  # for npz cache only
sim_ratio_pert = sim["ratio_pert"]

# Mach number of fundamental: M = P_1f / (rho * c²)
exp_M = exp_p1f / (RHO * C_SOUND**2)
sim_M_pert = sim_p1f_pert / (RHO * C_SOUND**2)

# Linear fit through origin on M-axis: ratio = K_exp * M
_K_exp = float(np.sum(exp_M * exp_ratio) / np.sum(exp_M**2))
_pred = _K_exp * exp_M
_r2_ratio = float(1 - np.sum((exp_ratio - _pred)**2) / np.sum(exp_ratio**2))

# Theoretical slope on M-axis: K_th = beta * Q_2 / 4 (dimensionless)
_BETA, _Q2 = 3.5, 100
_K_th = _BETA * _Q2 / 4

# Lorentzian detuning factor cos(theta) for off-resonance 2f drive
_F2_HZ = 3.845e6   # cavity 2f eigenmode
_TWOF1_HZ = 3.814e6  # 2f drive (= 2 * f_1)
_tan_theta = 2 * _Q2 * (_TWOF1_HZ - _F2_HZ) / _F2_HZ
_cos_theta = float(1 / np.sqrt(1 + _tan_theta**2))
_K_th_detuned = _K_th * _cos_theta
_K_ratio = _K_exp / _K_th_detuned

# R² for through-origin fits (SS_tot = Σy², not centred)
_r2_1f = 1 - np.sum((exp_p1f - a_1f * Vpp)**2) / np.sum(exp_p1f**2)
_r2_2f = 1 - np.sum((exp_p2f - b_2f * Vpp**2)**2) / np.sum(exp_p2f**2)

XLIM_MAX_M = 4.5e6 / (RHO * C_SOUND**2)  # M corresponding to 4.5 MPa
V_fine = np.linspace(0, Vpp.max() * 1.05, 100)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.7, 2.0))
_lbl_kw = dict(va="top", ha="left", fontweight="bold")

# (a) P_1f (left axis, MPa) and P_2f (right axis, kPa) vs V_drive,
# linear with twin y-axes
ax_a.errorbar(Vpp, exp_p1f / 1e6, yerr=exp_s1f / 1e6,
              fmt="o", markersize=3, color="tab:blue",
              capsize=3, capthick=0.5, elinewidth=0.5)
ax_a.plot(V_fine, a_1f * V_fine / 1e6, ":", linewidth=0.5, color="tab:blue",
          label=r"$\hat{P}_{1f}\propto V_\mathrm{drive}$")
ax_a.set_xlabel(r"$V_\mathrm{drive}$ [$\mathrm{V_{pp}}$]")
ax_a.set_ylabel(r"$\hat{P}_{1f}$ [MPa]", color="tab:blue")
ax_a.tick_params(axis="y", labelcolor="tab:blue")
ax_a.set_xlim(0, Vpp.max() * 1.05)
ax_a.set_ylim(bottom=0)

ax_a_right = ax_a.twinx()
ax_a_right.errorbar(Vpp, exp_p2f / 1e3, yerr=exp_s2f / 1e3,
                    fmt="s", markersize=3, color="tab:red",
                    capsize=3, capthick=0.5, elinewidth=0.5)
ax_a_right.plot(V_fine, b_2f * V_fine**2 / 1e3, ":", linewidth=0.5,
                color="tab:red",
                label=r"$\hat{P}_{2f}\propto V_\mathrm{drive}^2$")
ax_a_right.set_ylabel(r"$\hat{P}_{2f}$ [kPa]", color="tab:red")
ax_a_right.tick_params(axis="y", labelcolor="tab:red")
ax_a_right.set_ylim(bottom=0)

# Combined legend (handles from both y-axes)
_h1, _l1 = ax_a.get_legend_handles_labels()
_h2, _l2 = ax_a_right.get_legend_handles_labels()
ax_a.legend(_h1 + _h2, _l1 + _l2, frameon=False,
            loc="upper left")
ax_a.text(-0.10, 0.98, "(a)", transform=ax_a.transAxes, **_lbl_kw)

# (b) ratio vs Mach number with detuned Coppens prediction and linear fit
_M_fine = np.linspace(0, XLIM_MAX_M, 100)
ax_b.plot(_M_fine, _K_th_detuned * _M_fine,
          "-", color="C0", linewidth=0.8,
          label="Coppens")
ax_b.plot(_M_fine, _K_exp * _M_fine, "--", color="k", linewidth=0.6,
          label="Linear fit")
ax_b.errorbar(exp_M, exp_ratio, yerr=exp_ratio_std,
              fmt="ko", markersize=3, capsize=3, capthick=0.5,
              elinewidth=0.5, zorder=3)
ax_b.set_xlabel(r"$M$")
ax_b.set_ylabel(r"$\hat{P}_{2f}/\hat{P}_{1f}$")
ax_b.text(0.95, 0.05,
          f"$K_\\mathrm{{exp}}/K_\\mathrm{{the}}$ = {_K_ratio:.2f}",
          transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8)
ax_b.legend(frameon=False, loc="upper left")
ax_b.set_xlim(0, XLIM_MAX_M)
ax_b.set_ylim(bottom=0)
ax_b.text(-0.17, 0.98, "(b)", transform=ax_b.transAxes, **_lbl_kw)

plt.tight_layout()
np.savez(AF_FIG_DIR / "Fig2.npz",
         Vpp=Vpp,
         exp_E_ac=exp_E, exp_ratio=exp_ratio, exp_ratio_std=exp_ratio_std,
         exp_p0_1f=exp_p1f, exp_p0_2f=exp_p2f,
         exp_p0_1f_std=exp_s1f, exp_p0_2f_std=exp_s2f,
         a_1f=a_1f, b_2f=b_2f, r2_1f=_r2_1f, r2_2f=_r2_2f,
         pert_E_ac=sim_E_pert, pert_ratio=sim_ratio_pert,
         K_exp=_K_exp, K_th=_K_th, K_ratio=_K_ratio,
         cos_theta=_cos_theta, K_th_detuned=_K_th_detuned,
         r2_ratio=_r2_ratio)
save_fig(fig, "Fig2")
plt.close()

print(f"\n--- AF Fig 2: drive-sweep ratio + perturbative theory ---")
print(f"  Experiment: P_1f = {exp_p1f[0]/1e6:.2f}–{exp_p1f[-1]/1e6:.2f} MPa, "
      f"ratio = {exp_ratio[0]:.3f}–{exp_ratio[-1]:.3f}")
print(f"  Linear fit through origin: ratio = {_K_exp:.3e} * P_1f, R² = {_r2_ratio:.3f}")
print(f"  Theory K_th = beta*Q2/(4*rho*c²) = {_K_th:.3e}  (beta={_BETA}, Q2={_Q2})")
print(f"  K_exp / K_th = {_K_ratio:.2f}")
print("\n=== Done ===")
