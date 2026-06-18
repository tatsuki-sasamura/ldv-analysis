"""A4 + A5 + A6: cascade analysis in P_1f / energy space (measurement-only).

Reads harmonic_ladder.npz (co-located modal P_nf, n=1..5, 12-point
cascade) and produces the harmonic-generation view that is independent
of the drive->P_1f transduction:

  A5  P_nf ~ P_1f^{m_n}   (perturbative m_n = n) -- exponents + running.
  A4  K_n = P_nf / P_1f^n (generalized Coppens prefactor), normalized to
      its low-drive value: flat = perturbative, declining = breakdown.
  A6  energy budget E_nf = P_nf^2 / (4 rho c^2): per-harmonic fractions,
      total diverted Sum_{n>=2} E_nf / E_1f, and the closure check
      (the 2f deficit below its perturbative prefactor vs the energy
      that appears in 3f+).

Points with SNR < 3 are treated as noise-dominated (excluded from fits,
drawn faint).  Outputs harmonic_cascade.{png,npz} in output/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "experiments" / "2026W21" / "output"
SNR_MIN = 3.0
PERT_V = (30.0, 60.0)  # perturbative window for the Coppens prefactor


def main() -> None:
    d = np.load(OUT / "harmonic_ladder.npz", allow_pickle=True)
    V = d["pzt_vpp"]
    P_kpa = d["p_kpa"]
    SNR = d["snr"]
    H = [int(h) for h in d["harmonics"]]
    rho = float(d["rho"])
    c = float(d["c_sound"])

    P = P_kpa * 1e3  # Pa
    P1 = P[:, 0]
    lnP1 = np.log(P1)
    efac = 1.0 / (4 * rho * c**2)
    E = P**2 * efac  # J/m^3 per harmonic

    # ---- A5: P_nf vs P_1f exponents ------------------------------------
    print(
        f"{'harm':>4} {'pred':>4} {'m(vs P1f)':>10} {'range':>14} "
        f"{'m(vs Eac)':>10} {'hi-end local':>12}"
    )
    m_vs_p1 = np.full(len(H), np.nan)
    for j, n in enumerate(H):
        if n == 1:
            continue
        good = (SNR[:, j] >= SNR_MIN) & (SNR[:, 0] >= SNR_MIN)
        if good.sum() >= 2:
            m_vs_p1[j] = np.polyfit(lnP1[good], np.log(P[good, j]), 1)[0]
            gi = np.where(good)[0]
            rng = f"{V[gi[0]]:.0f}-{V[gi[-1]]:.0f}V"
            mhi = np.log(P[gi[-1], j] / P[gi[-2], j]) / np.log(P1[gi[-1]] / P1[gi[-2]])
        else:
            rng, mhi = "<2 pt", np.nan
        # E_ac ~ P1^2, so exponent vs E_ac = m/2
        print(f"{n:>4} {n:>4} {m_vs_p1[j]:>10.2f} {rng:>14} " f"{m_vs_p1[j]/2:>10.2f} {mhi:>12.2f}")

    # ---- A4: generalized prefactor K_n = P_nf / P_1f^n ------------------
    # normalized to the lowest above-noise drive (so it starts at 1.0)
    Knorm = np.full_like(P, np.nan)
    for j, n in enumerate(H):
        if n == 1:
            continue
        K = P[:, j] / P1**n
        good = (SNR[:, j] >= SNR_MIN) & (SNR[:, 0] >= SNR_MIN)
        if good.any():
            Knorm[:, j] = K / K[np.where(good)[0][0]]

    # ---- A6: energy closure -------------------------------------------
    k2_arr = P[:, 1] / P1**2
    win = (V >= PERT_V[0]) & (V <= PERT_V[1])
    k2_pert = float(np.median(k2_arr[win]))
    P2_pert = k2_pert * P1**2
    E2_pert = P2_pert**2 * efac
    E2_deficit = E2_pert - E[:, 1]  # 2f shortfall vs perturbative
    E_ge3 = E[:, 2:].sum(axis=1)  # energy in 3f+
    E_total = E.sum(axis=1)
    frac = E / E_total[:, None]
    diverted = E[:, 1:].sum(axis=1) / E[:, 0]

    print(
        f"\nperturbative 2f prefactor k2 = P2/P1^2 (median {PERT_V[0]:.0f}-"
        f"{PERT_V[1]:.0f}V) = {k2_pert*1e9:.1f} /GPa"
    )
    print(
        f"{'V':>5} {'divert%':>8} {'f_2f%':>7} {'f_3f+%':>7} "
        f"{'2fdef[J/m3]':>11} {'E_3f+[J/m3]':>11}"
    )
    for i in range(len(V)):
        print(
            f"{V[i]:>5.0f} {diverted[i]*100:>8.2f} {frac[i,1]*100:>7.2f} "
            f"{frac[i,2:].sum()*100:>7.2f} {E2_deficit[i]:>11.2f} "
            f"{E_ge3[i]:>11.2f}"
        )

    np.savez(
        OUT / "harmonic_cascade.npz",
        harmonics=np.array(H),
        pzt_vpp=V,
        p_pa=P,
        m_vs_p1f=m_vs_p1,
        eac_1f=d["eac_1f"],
        e_nf=E,
        energy_frac=frac,
        diverted=diverted,
        knorm=Knorm,
        k2_pert=np.array(k2_pert),
        e2_deficit=E2_deficit,
        e_ge3=E_ge3,
    )
    print(f"\nSaved {OUT / 'harmonic_cascade.npz'}")

    # ---- figure: 2x2 contact sheet -------------------------------------
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # (a) P_nf vs P_1f, log-log, with perturbative slope refs
    a0 = ax[0, 0]
    for j, n in enumerate(H):
        good = (SNR[:, j] >= SNR_MIN) & (SNR[:, 0] >= SNR_MIN)
        a0.loglog(P1 / 1e3, P[:, j] / 1e3, "o", ms=3, color=colors[j], alpha=0.3)
        a0.loglog(
            P1[good] / 1e3,
            P[good, j] / 1e3,
            "o",
            ms=4,
            color=colors[j],
            label=f"$P_{{{n}f}}$" + ("" if n == 1 else f" (m={m_vs_p1[j]:.2f})"),
        )
        if n > 1 and good.sum() >= 2:
            xr = np.array([P1[good].min(), P1[good].max()])
            k = P[good, j][0] / P1[good][0] ** n
            a0.loglog(xr / 1e3, (k * xr**n) / 1e3, ":", lw=0.8, color=colors[j])
    a0.set_xlabel(r"$P_{1f}$ [kPa]")
    a0.set_ylabel(r"$P_{nf}$ [kPa]")
    a0.set_title(r"A5: $P_{nf}\propto P_{1f}^{m}$ (dotted = perturbative $m=n$)")
    a0.legend(fontsize=7, frameon=False)
    a0.grid(True, which="both", alpha=0.3)

    # (b) normalized prefactor K_n / K_n(low) vs P_1f
    a1 = ax[0, 1]
    for j, n in enumerate(H):
        if n == 1:
            continue
        good = (SNR[:, j] >= SNR_MIN) & (SNR[:, 0] >= SNR_MIN)
        a1.plot(P1 / 1e3, Knorm[:, j], "o-", ms=3, lw=0.7, color=colors[j], alpha=0.3)
        a1.plot(
            P1[good] / 1e3,
            Knorm[good, j],
            "o-",
            ms=4,
            lw=1.2,
            color=colors[j],
            label=f"$P_{{{n}f}}/P_{{1f}}^{n}$",
        )
    a1.axhline(1.0, color="0.5", lw=0.7, ls="--")
    a1.set_xlabel(r"$P_{1f}$ [kPa]")
    a1.set_ylabel(r"$K_n/K_n^{\rm low}$")
    a1.set_title(r"A4: generalized Coppens prefactor (flat = perturbative)")
    a1.legend(fontsize=7, frameon=False)
    a1.grid(True, alpha=0.3)

    # (c) energy fractions vs V
    a2 = ax[1, 0]
    for j, n in enumerate(H):
        a2.semilogy(V, frac[:, j] * 100, "o-", ms=4, lw=0.9, color=colors[j], label=f"$E_{{{n}f}}$")
    a2.set_xlabel("PZT drive [Vpp]")
    a2.set_ylabel(r"energy fraction $E_{nf}/\sum E$ [\%]")
    a2.set_title("A6: harmonic energy partition")
    a2.legend(fontsize=7, ncol=5, frameon=False)
    a2.grid(True, which="both", alpha=0.3)

    # (d) closure: 2f deficit vs energy in 3f+
    a3 = ax[1, 1]
    a3.plot(
        V,
        E2_deficit,
        "s-",
        ms=4,
        lw=1.1,
        color="C1",
        label=r"$E_{2f}^{\rm pert}-E_{2f}^{\rm obs}$ (2f deficit)",
    )
    a3.plot(V, E_ge3, "o-", ms=4, lw=1.1, color="C2", label=r"$\sum_{n\geq3} E_{nf}$ (in 3f+)")
    a3.axhline(0, color="0.6", lw=0.6)
    a3.set_xlabel("PZT drive [Vpp]")
    a3.set_ylabel(r"energy density [J/m$^3$]")
    a3.set_title(r"A6 closure: does the 2f shortfall reappear in 3f+?")
    a3.legend(fontsize=7, frameon=False)
    a3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "harmonic_cascade.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
