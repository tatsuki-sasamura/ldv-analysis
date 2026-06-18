"""Drive-voltage scaling of each harmonic: P_nf ~ V^m.

Perturbation theory predicts m = n (P_1f ~ V, P_2f ~ V^2, ...).  This
reads the cache-only ladder (harmonic_ladder.npz) and reports, per
harmonic: the low-drive (perturbative) exponent, the full-range
exponent, and the local running exponent d ln P_nf / d ln V -- the last
shows where each harmonic departs from V^n.  Points with SNR < 3 are
treated as noise-dominated and excluded from fits / drawn faint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "experiments" / "2026W21" / "output"
SNR_MIN = 3.0
PERT_VMAX = 40.0  # low-drive cap for the "perturbative" exponent fit


def main() -> None:
    d = np.load(OUT / "harmonic_ladder.npz", allow_pickle=True)
    V = d["pzt_vpp"]
    P = d["p_kpa"]
    SNR = d["snr"]
    H = d["harmonics"]
    lnV = np.log(V)

    print(
        f"{'harm':>4} {'pred m':>6} {'pert m':>7} {'pert range':>14} "
        f"{'full m':>7} {'hi-V local m':>12}"
    )
    pert_m = np.full(len(H), np.nan)
    full_m = np.full(len(H), np.nan)
    for j, n in enumerate(H):
        p = P[:, j]
        good = SNR[:, j] >= SNR_MIN
        if good.sum() >= 2:
            full_m[j] = np.polyfit(lnV[good], np.log(p[good]), 1)[0]
        idx = np.where(good & (V <= PERT_VMAX))[0][:3]
        if len(idx) >= 2:
            pert_m[j] = np.polyfit(lnV[idx], np.log(p[idx]), 1)[0]
            rng = f"{V[idx[0]]:.0f}-{V[idx[-1]]:.0f}V/{len(idx)}pt"
        else:
            rng = "<2 pt > noise"
        mhi = np.log(p[-1] / p[-2]) / np.log(V[-1] / V[-2])
        print(
            f"{int(n):>4} {int(n):>6} {pert_m[j]:>7.2f} {rng:>14} "
            f"{full_m[j]:>7.2f} {mhi:>12.2f}"
        )

    # ---- running local exponent vs drive --------------------------------
    v_mid = np.sqrt(V[:-1] * V[1:])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for j, n in enumerate(H):
        p = P[:, j]
        m_loc = np.diff(np.log(p)) / np.diff(lnV)
        rel = (SNR[:-1, j] >= SNR_MIN) & (SNR[1:, j] >= SNR_MIN)
        ax.plot(v_mid, m_loc, "o-", color=f"C{j}", lw=0.7, ms=3, alpha=0.35)
        ax.plot(
            v_mid[rel], m_loc[rel], "o-", color=f"C{j}", lw=1.4, ms=4, label=f"$P_{{{int(n)}f}}$"
        )
        ax.axhline(n, color=f"C{j}", ls=":", lw=0.7, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("PZT drive [Vpp] (geom-mean of pair)")
    ax.set_ylabel(r"local exponent  $d\ln P_{nf}/d\ln V$")
    ax.set_title(r"Drive scaling $P_{nf}\sim V^{m}$  (dotted = perturbative $m=n$)")
    ax.legend(fontsize=8, ncol=5, frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "harmonic_scaling.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
