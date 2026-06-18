"""How many harmonics carry 99.9 % of the acoustic energy?

Reads harmonic_ladder.npz (co-located modal P_nf, n=1..5, 10-120 Vpp)
and computes the per-harmonic acoustic energy density
``E_nf = P_nf^2 / (4 rho c^2)`` [J/m^3], the cumulative coverage
``Sum_{1..k} E_nf / Sum_{1..5} E_nf``, and the smallest k reaching
99.9 % at each drive level.

Caveat: "total" is the measured Sum over n=1..5 (the FFT cache stops at
MAX_HARMONIC=5).  Points with SNR < 3 are noise-limited (marked); at the
cascade tail E_4f/E_5f are near the floor, so the measured total slightly
over-counts the true high-n tail -> the coverage is a conservative
lower bound on how few harmonics suffice.

Outputs harmonic_energy_coverage.{png,npz} in output/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "experiments" / "2026W21" / "output"
SNR_MIN = 3.0
TARGET = 0.999


def main() -> None:
    d = np.load(OUT / "harmonic_ladder.npz", allow_pickle=True)
    V = d["pzt_vpp"]
    P = d["p_kpa"] * 1e3  # Pa
    SNR = d["snr"]
    H = [int(h) for h in d["harmonics"]]
    rho = float(d["rho"])
    c = float(d["c_sound"])

    E = P**2 / (4 * rho * c**2)  # J/m^3, (n_scan, 5)
    cum = np.cumsum(E, axis=1)
    total = cum[:, -1]
    cum_frac = cum / total[:, None]  # coverage including up to harmonic k
    missing = 1.0 - cum_frac  # residual after k harmonics

    n_999 = np.array([int(np.argmax(cum_frac[i] >= TARGET)) + 1 for i in range(len(V))])

    print(
        f"{'Vpp':>5} "
        + " ".join(f"E{n}f[J/m3]".rjust(11) for n in H)
        + f" {'Etot':>10} | "
        + " ".join(f"<= {n}f".rjust(8) for n in H)
        + f" | {'n@99.9%':>7}"
    )
    for i in range(len(V)):
        e_cells = " ".join(f"{E[i, j]:>11.2f}" for j in range(len(H)))
        cum_cells = " ".join(f"{cum_frac[i, j]*100:>8.3f}" for j in range(len(H)))
        print(f"{V[i]:>5.0f} {e_cells} {total[i]:>10.1f} | {cum_cells} | " f"{n_999[i]:>7d}")

    vmax_i = int(np.argmax(V))
    print(
        f"\nAt the highest drive ({V[vmax_i]:.0f} Vpp): "
        f"<=2f={cum_frac[vmax_i,1]*100:.3f}%, "
        f"<=3f={cum_frac[vmax_i,2]*100:.3f}%, "
        f"<=4f={cum_frac[vmax_i,3]*100:.3f}%"
    )
    print(
        f"Harmonics needed for {TARGET*100:.1f}% coverage: "
        f"{n_999.min()} (low drive) -> {n_999.max()} (max drive)"
    )

    np.savez(
        OUT / "harmonic_energy_coverage.npz",
        harmonics=np.array(H),
        pzt_vpp=V,
        e_nf=E,
        e_total=total,
        cum_frac=cum_frac,
        n_for_999=n_999,
    )
    print(f"\nSaved {OUT / 'harmonic_energy_coverage.npz'}")

    # ---- figure --------------------------------------------------------
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) absolute E_nf per harmonic
    for j, n in enumerate(H):
        good = SNR[:, j] >= SNR_MIN
        a0.semilogy(V, E[:, j], "-", lw=0.7, color=colors[j], alpha=0.4)
        a0.semilogy(V[good], E[good, j], "o", ms=4, color=colors[j], label=f"$E_{{{n}f}}$")
        a0.semilogy(V[~good], E[~good, j], "x", ms=4, color=colors[j], alpha=0.6)
    a0.set_xlabel("PZT drive [Vpp]")
    a0.set_ylabel(r"$E_{nf}=P_{nf}^2/(4\rho c^2)$ [J/m$^3$]")
    a0.set_title(r"Acoustic energy per harmonic (x = SNR\,$<$\,3, noise)")
    a0.legend(fontsize=8, ncol=5, frameon=False)
    a0.grid(True, which="both", alpha=0.3)

    # (b) residual missing fraction after k harmonics
    for k in range(1, len(H)):  # k = 1..4 (after including up to kf)
        a1.semilogy(
            V,
            np.clip(missing[:, k - 1], 1e-6, 1),
            "o-",
            ms=4,
            lw=1.0,
            color=colors[k - 1],
            label=rf"after $\leq{H[k-1]}f$",
        )
    a1.axhline(1 - TARGET, color="0.4", lw=0.9, ls="--", label=f"{(1-TARGET)*100:.1f}% missing")
    a1.set_xlabel("PZT drive [Vpp]")
    a1.set_ylabel(r"missing energy fraction $1-\sum_{1}^{k}E/E_{\rm tot}$")
    a1.set_title("Energy not yet captured vs harmonics included")
    a1.legend(fontsize=8, frameon=False)
    a1.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "harmonic_energy_coverage.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
