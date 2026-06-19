"""Q1 from resonance linewidth (cache-only).

Q1 = f0 / FWHM of the 1f resonance, fit as a Lorentzian on |P_1f|^2 vs
drive frequency.  For a single damped mode this equals the transient
Q = pi f1 tau1, so it cross-checks the report's transient Q1 = 121
(tau1 = 20.2 us).  Linewidth Q is the *total* loss Q (radiation +
material + viscous + structural) -- the quantity the F3 Kuznetsov
damping model wants.

Sources (all cache-only, via sweep_peaks which returns the per-frequency
P_1f curve):
  - headline: 101x77 1p89to1p92 60 Vpp scan (31 freq @ 1 kHz; FWHM ~16
    kHz sampled by ~16 points);
  - Q1(V): the 12-point 101x21 cascade (11 freq @ 2 kHz each) -> drive
    dependence + the low-drive (most linear) Q1.

Outputs q1_linewidth.{png,npz} in output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpp_vs_pressure import SCANS, read_amp_gain, read_drive_vpp  # noqa: E402

from ldv_analysis.config import CHANNEL_WIDTH, LDV_DATA_ROOT, get_cache_dir  # noqa: E402
from ldv_analysis.sweep_fit import sweep_peaks  # noqa: E402

FINE_1F = "sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"
DATA_ROOT = LDV_DATA_ROOT / "output" / "W21"
OUT_DIR = ROOT / "experiments" / "2026W21" / "output"
TRANSIENT_Q1 = 121.0  # report value (tau1 = 20.2 us @ 1.907 MHz)


def lorentzian(f, A, f0, hwhm, c):
    """|P|^2 Lorentzian; FWHM = 2*hwhm, Q = f0 / FWHM."""
    return A * hwhm**2 / ((f - f0) ** 2 + hwhm**2) + c


def fit_q(freq_hz, p_pa):
    """Lorentzian fit of |P|^2(f); returns (f0, fwhm, Q). NaN on failure."""
    y = p_pa**2
    i = int(np.argmax(y))
    span = freq_hz[-1] - freq_hz[0]
    p0 = [y[i] - y.min(), freq_hz[i], span / 8, max(y.min(), 0.0)]
    bounds = (
        [0.0, freq_hz.min(), span / 100, 0.0],
        [10 * y.max(), freq_hz.max(), span, y.max() + 1.0],
    )
    try:
        popt, _ = curve_fit(lorentzian, freq_hz, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    _, f0, hwhm, _ = popt
    fwhm = 2.0 * hwhm
    return float(f0), float(fwhm), float(f0 / fwhm)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- headline: fine 1f scan ---------------------------------------
    sp = sweep_peaks(DATA_ROOT / FINE_1F, CHANNEL_WIDTH, get_cache_dir(FINE_1F, __file__))
    f_fine = sp.freq_mhz * 1e6
    p_fine = sp.p1_kpa * 1e3
    f0, fwhm, q_fine = fit_q(f_fine, p_fine)
    print(f"=== headline Q1 (fine 1p89to1p92, 60 Vpp, {len(f_fine)} freqs) ===")
    print(f"  f0   = {f0/1e6:.4f} MHz")
    print(f"  FWHM = {fwhm/1e3:.2f} kHz")
    print(
        f"  Q1   = {q_fine:.1f}   (transient Q1 = {TRANSIENT_Q1:.0f}, "
        f"ratio {q_fine/TRANSIENT_Q1:.2f})"
    )

    # ---- Q1(V) from the cascade ---------------------------------------
    print(f"\n=== Q1(V) from cascade (11 freq @ 2 kHz) ===")
    print(f"{'Vpp':>5} {'f0(MHz)':>8} {'FWHM(kHz)':>10} {'Q1':>7}")
    vpp, q_v, f0_v = [], [], []
    for label, dirname in SCANS:
        try:
            spv = sweep_peaks(DATA_ROOT / dirname, CHANNEL_WIDTH, get_cache_dir(dirname, __file__))
        except (ValueError, FileNotFoundError):
            continue
        afg = read_drive_vpp(DATA_ROOT / dirname)
        gain = read_amp_gain(DATA_ROOT / dirname)
        v = afg * gain if (afg and gain) else float(label)
        f0i, fwhmi, qi = fit_q(spv.freq_mhz * 1e6, spv.p1_kpa * 1e3)
        vpp.append(v)
        q_v.append(qi)
        f0_v.append(f0i)
        print(f"{v:>5.0f} {f0i/1e6:>8.4f} {fwhmi/1e3:>10.2f} {qi:>7.1f}")
    vpp, q_v, f0_v = np.array(vpp), np.array(q_v), np.array(f0_v)

    np.savez(
        OUT_DIR / "q1_linewidth.npz",
        fine_f0=f0,
        fine_fwhm=fwhm,
        fine_Q1=q_fine,
        transient_Q1=TRANSIENT_Q1,
        cascade_vpp=vpp,
        cascade_Q1=q_v,
        cascade_f0=f0_v,
        fine_freq_hz=f_fine,
        fine_p1_pa=p_fine,
    )
    print(f"\nSaved {OUT_DIR / 'q1_linewidth.npz'}")

    # ---- figure --------------------------------------------------------
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    a0.plot(f_fine / 1e6, p_fine**2 / 1e12, "ko", ms=3, label="data")
    fd = np.linspace(f_fine.min(), f_fine.max(), 500)
    # reconstruct fit curve from f0/fwhm by refitting params for the plot
    popt, _ = curve_fit(
        lorentzian, f_fine, p_fine**2, p0=[(p_fine**2).max(), f0, fwhm / 2, 0.0], maxfev=20000
    )
    a0.plot(
        fd / 1e6,
        lorentzian(fd, *popt) / 1e12,
        "-",
        color="C3",
        lw=1.0,
        label=f"Lorentzian: $Q_1$={q_fine:.0f}",
    )
    a0.axvline(f0 / 1e6, color="0.6", lw=0.6, ls=":")
    a0.set_xlabel("drive frequency [MHz]")
    a0.set_ylabel(r"$|P_{1f}|^2$ [MPa$^2$]")
    a0.set_title(
        f"1f resonance (60 Vpp): $f_0$={f0/1e6:.4f} MHz, " f"FWHM={fwhm/1e3:.1f} kHz", fontsize=9
    )
    a0.legend(frameon=False)
    a0.grid(True, alpha=0.3)

    a1.plot(vpp, q_v, "o-", ms=4, color="C0", label="linewidth $Q_1(V)$ (cascade)")
    a1.axhline(
        TRANSIENT_Q1, color="0.4", lw=0.9, ls="--", label=f"transient $Q_1$={TRANSIENT_Q1:.0f}"
    )
    a1.plot([60], [q_fine], "s", ms=7, color="C3", label=f"fine scan ({q_fine:.0f})")
    a1.set_xlabel(r"PZT drive [Vpp]")
    a1.set_ylabel(r"linewidth $Q_1 = f_0/\mathrm{FWHM}$")
    a1.set_title("Q1 vs drive (apparent / total-loss Q)", fontsize=9)
    a1.set_ylim(bottom=0)
    a1.legend(frameon=False, fontsize=8)
    a1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "q1_linewidth.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
