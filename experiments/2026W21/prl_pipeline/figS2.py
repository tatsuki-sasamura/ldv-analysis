"""Fig S2 — complex resonance response and effective Q_n.

One-pole fit on the 1f frequency sweep; two-pole fit on the 2f sweep
(the 2f cavity has two transverse-mode-aliased peaks at 3.794 + 3.817 MHz
per ``reports/2026-06-18_f2_eigenmode_pin.md`` and the cascade drives at
~3.804 MHz between them — a one-pole 2f fit is physically wrong).

Outputs
-------
``PIPE_OUT/figS2.{png,pdf,npz}``. The npz contains complex sweep data,
fitted parameters, parameter covariance, and effective ``Q_n`` at the
cascade operating point.

Phase calibration
-----------------
All quantities (magnitude AND complex residue) are derived from Ch2
(velocity decoder) only; Ch3 (displacement) is not used. The complex
residue phases inherit the Polytec velocity-decoder phase response,
which the vendor spec quotes as < 1 deg distortion in the linear band.
The PLAN.md §11 R2 cross-channel audit does not apply to this
Ch2-only pipeline — see PLAN.md §11 R2 documentation note.

Self-test
---------
``python figS2.py --selftest`` synthesizes a one-pole and a two-pole
response with known parameters and verifies the fit recovers them
within tolerance before touching real data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from conventions import (
    CHANNEL_WIDTH,
    F2_MODE_A_HZ,
    F2_MODE_B_HZ,
    FSWEEP_1F,
    FSWEEP_2F_FINE,
    PIPE_OUT,
    W21_DATA_ROOT,
    get_cache_dir,
)

_W21 = Path(__file__).resolve().parents[1]
if str(_W21) not in sys.path:
    sys.path.insert(0, str(_W21))

from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.sweep_fit import fit_axial  # noqa: E402

# Cascade operating point for the effective-Q report
CASCADE_F_OP_HZ = 1.9020e6

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 0.9,
    }
)


# ---------------------------------------------------------------------------
# Sweep ingestion: f_i, complex H_i = P_modal / V_drive_1f
# ---------------------------------------------------------------------------

def _collect_sweep(scan_dir_name: str, projection: str) -> tuple[np.ndarray, np.ndarray]:
    """Iterate ``f*.h5`` in ``scan_dir_name`` and return (f_hz, H_complex).

    ``projection``:
      * ``"p1"``  — fit_axial's ``p1_complex`` (n=1 transverse). Use for
        the 1f sweep (drive ~1.9 MHz).
      * ``"p1n2"`` — fit_axial's ``p1_n2_complex`` (n=2 transverse). Use
        for the 2f-band sweep (drive ~3.8 MHz, drives the n=2 cavity).

    H is normalized by the per-file 1f drive voltage so amplitudes are
    Pa/V at the PZT terminal — that gives ``H_n^ext(f)`` directly.
    """
    run_dir = W21_DATA_ROOT / scan_dir_name
    cache_dir = get_cache_dir(scan_dir_name, str(_W21 / "figS2_proxy.py"))
    files = sorted(p for p in run_dir.glob("f*.h5") if not p.name.endswith(".inprogress"))
    if not files:
        raise FileNotFoundError(f"No .h5 in {run_dir}")

    f_list = []
    H_list = []
    geom = None
    for p in files:
        with h5py.File(p, "r") as h:
            f_nom = float(h.attrs["drive_frequency_hz_nominal"])
        cache = load_or_compute(p, cache_dir, velocity_scale=None)
        rssi = np.asarray(cache["rssi"]) if "rssi" in cache.files else None
        valid = make_valid_mask(np.asarray(cache["voltage_1f"]), rssi)
        if int(np.sum(valid)) < 5:
            print(f"  {p.name}: only {int(np.sum(valid))} valid pts, skip")
            continue
        fit, geom = fit_axial(cache, valid, CHANNEL_WIDTH, geom=geom)
        v_pzt = 2.0 * float(np.nanmedian(np.asarray(cache["voltage_1f"])[valid]))  # Vpp
        if projection == "p1":
            modal = complex(fit.p1_complex)
        elif projection == "p1n2":
            modal = complex(fit.p1_n2_complex)
        else:
            raise ValueError(f"unknown projection {projection!r}")
        if v_pzt <= 0 or not np.isfinite(v_pzt):
            continue
        f_list.append(f_nom)
        H_list.append(modal / v_pzt)  # Pa/V

    f_arr = np.asarray(f_list)
    H_arr = np.asarray(H_list, dtype=complex)
    order = np.argsort(f_arr)
    return f_arr[order], H_arr[order]


# ---------------------------------------------------------------------------
# Models: H(f) parameterized for least_squares
# ---------------------------------------------------------------------------

def _denom(f, f0, Q):
    return 1.0 + 1j * Q * (f / f0 - f0 / f)


def H_one_pole(f, b0, b1, r, f0, Q, f_c):
    return b0 + b1 * (f - f_c) + r / _denom(f, f0, Q)


def H_two_pole(f, b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b, f_c):
    return (b0 + b1 * (f - f_c)
            + r_a / _denom(f, f_a, Q_a)
            + r_b / _denom(f, f_b, Q_b))


def _pack_one(b0, b1, r, f0, Q):
    return np.array([b0.real, b0.imag, b1.real, b1.imag, r.real, r.imag, f0, Q])


def _unpack_one(p):
    return (p[0] + 1j * p[1], p[2] + 1j * p[3], p[4] + 1j * p[5], p[6], p[7])


def _resid_one(p, f, H_meas, f_c):
    b0, b1, r, f0, Q = _unpack_one(p)
    H = H_one_pole(f, b0, b1, r, f0, Q, f_c)
    d = H - H_meas
    return np.concatenate([d.real, d.imag])


def _pack_two(b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b):
    return np.array([b0.real, b0.imag, b1.real, b1.imag,
                     r_a.real, r_a.imag, f_a, Q_a,
                     r_b.real, r_b.imag, f_b, Q_b])


def _unpack_two(p):
    b0 = p[0] + 1j * p[1]
    b1 = p[2] + 1j * p[3]
    r_a = p[4] + 1j * p[5]
    f_a, Q_a = p[6], p[7]
    r_b = p[8] + 1j * p[9]
    f_b, Q_b = p[10], p[11]
    return b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b


def _resid_two(p, f, H_meas, f_c):
    b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b = _unpack_two(p)
    H = H_two_pole(f, b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b, f_c)
    d = H - H_meas
    return np.concatenate([d.real, d.imag])


def _initial_one(f, H):
    """Rough initial guess from |H|."""
    mag = np.abs(H)
    k = int(np.argmax(mag))
    f0 = float(f[k])
    half = mag[k] / np.sqrt(2)
    above = mag > half
    if above.sum() >= 2:
        idx = np.where(above)[0]
        fwhm = float(f[idx[-1]] - f[idx[0]])
        Q = max(f0 / max(fwhm, 1.0), 5.0)
    else:
        Q = 50.0
    b0 = complex(np.mean(H[[0, -1]]))
    r = complex(H[k]) - b0
    b1 = 0.0 + 0.0j
    return _pack_one(b0, b1, r, f0, Q)


def _initial_two(f, H, f_a_seed=F2_MODE_A_HZ, f_b_seed=F2_MODE_B_HZ):
    b0 = complex(np.mean(H[[0, -1]]))
    # Pick the H values nearest each seed for residue scale.
    ia = int(np.argmin(np.abs(f - f_a_seed)))
    ib = int(np.argmin(np.abs(f - f_b_seed)))
    r_a = complex(H[ia]) - b0
    r_b = complex(H[ib]) - b0
    Q_a = Q_b = 250.0
    b1 = 0.0 + 0.0j
    return _pack_two(b0, b1, r_a, f_a_seed, Q_a, r_b, f_b_seed, Q_b)


def _fit_one_pole(f, H):
    f_c = float(np.mean(f))
    p0 = _initial_one(f, H)
    res = least_squares(_resid_one, p0, args=(f, H, f_c), method="lm", max_nfev=5000)
    cov = _approx_cov(res)
    return res, cov, f_c


def _fit_two_pole(f, H, f_a_seed=F2_MODE_A_HZ, f_b_seed=F2_MODE_B_HZ):
    f_c = float(np.mean(f))
    p0 = _initial_two(f, H, f_a_seed, f_b_seed)
    res = least_squares(_resid_two, p0, args=(f, H, f_c), method="lm", max_nfev=10000)
    cov = _approx_cov(res)
    # Enforce pole ordering f_a < f_b
    b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b = _unpack_two(res.x)
    if f_a > f_b:
        res.x = _pack_two(b0, b1, r_b, f_b, Q_b, r_a, f_a, Q_a)
    return res, cov, f_c


def _approx_cov(res) -> np.ndarray:
    """Crude parameter covariance from the residual Jacobian."""
    J = res.jac
    m, n = J.shape
    if m <= n:
        return np.full((n, n), np.nan)
    rss = float(np.sum(res.fun ** 2))
    dof = m - n
    sigma2 = rss / dof
    try:
        cov = sigma2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((n, n), np.nan)
    return cov


def selftest() -> None:
    rng = np.random.default_rng(20260626)
    # one-pole synthetic
    f = np.linspace(1.88e6, 1.93e6, 60)
    f0_t, Q_t = 1.905e6, 800.0
    r_t = 1.5e4 + 0.0j
    b0_t = 50.0 + 10.0j
    b1_t = 0.0 + 0.0j
    H = H_one_pole(f, b0_t, b1_t, r_t, f0_t, Q_t, np.mean(f))
    H += (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * 50.0
    res, cov, f_c = _fit_one_pole(f, H)
    b0, b1, r, f0, Q = _unpack_one(res.x)
    fwhm = f0 / Q
    err_f0 = abs(f0 - f0_t) / fwhm
    err_Q = abs(Q / Q_t - 1.0)
    err_r = abs(r) / abs(r_t) - 1.0
    print(f"  ONE-POLE selftest: f0 err = {err_f0:.3f} fwhm, Q rel err = {err_Q:.3f}, "
          f"|r| rel err = {err_r:+.3f}")
    assert err_f0 < 0.05, "one-pole f0 recovery failed"
    assert err_Q < 0.05, "one-pole Q recovery failed"
    assert abs(err_r) < 0.10, "one-pole |r| recovery failed"

    # two-pole synthetic
    f = np.linspace(3.76e6, 3.84e6, 80)
    f_a_t, Q_a_t = 3.794e6, 400.0
    f_b_t, Q_b_t = 3.817e6, 350.0
    r_a_t = 8.0e3 + 0.0j
    r_b_t = 5.0e3 + 2.0e3j
    b0_t = 20.0 + 5.0j
    b1_t = 0.0 + 0.0j
    H = H_two_pole(f, b0_t, b1_t, r_a_t, f_a_t, Q_a_t,
                   r_b_t, f_b_t, Q_b_t, np.mean(f))
    H += (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * 30.0
    res, cov, f_c = _fit_two_pole(f, H)
    b0, b1, r_a, f_a, Q_a, r_b, f_b, Q_b = _unpack_two(res.x)
    fwhm_a = f_a / Q_a
    err_a = abs(f_a - f_a_t) / fwhm_a
    err_b = abs(f_b - f_b_t) / (f_b / Q_b)
    err_Qa = abs(Q_a / Q_a_t - 1)
    err_Qb = abs(Q_b / Q_b_t - 1)
    print(f"  TWO-POLE selftest: f_a err = {err_a:.3f} fwhm, f_b err = {err_b:.3f} fwhm, "
          f"Q_a rel err = {err_Qa:.3f}, Q_b rel err = {err_Qb:.3f}")
    assert err_a < 0.05 and err_b < 0.05, "two-pole pole-frequency recovery failed"
    assert err_Qa < 0.10 and err_Qb < 0.10, "two-pole Q recovery failed"
    print("  -- selftest PASS --")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _eff_Q_at_op(f, H, f_op):
    """Effective Q ~ f_op / FWHM(|H|^2) at the cascade operating point.

    Diagnostic only — not the fit Q. NaN if magnitude never reaches half-max
    on both sides of the peak.
    """
    m2 = np.abs(H) ** 2
    k = int(np.argmax(m2))
    half = m2[k] / 2.0
    lo = next((i for i in range(k, -1, -1) if m2[i] <= half), None)
    hi = next((i for i in range(k, len(f)) if m2[i] <= half), None)
    if lo is None or hi is None:
        return float("nan")
    return float(f[k] / (f[hi] - f[lo]))


def compute() -> dict:
    print("FigS2: 1f sweep ...")
    f1, H1 = _collect_sweep(FSWEEP_1F, projection="p1")
    print(f"  {len(f1)} frequencies, {f1.min()/1e6:.3f}-{f1.max()/1e6:.3f} MHz")
    res1, cov1, fc1 = _fit_one_pole(f1, H1)
    b0_1, b1_1, r_1, f0_1, Q_1 = _unpack_one(res1.x)
    H1_fit = H_one_pole(f1, b0_1, b1_1, r_1, f0_1, Q_1, fc1)
    Q_eff_1 = _eff_Q_at_op(f1, H1, CASCADE_F_OP_HZ)

    print("FigS2: 2f sweep ...")
    f2, H2 = _collect_sweep(FSWEEP_2F_FINE, projection="p1n2")
    print(f"  {len(f2)} frequencies, {f2.min()/1e6:.3f}-{f2.max()/1e6:.3f} MHz")
    res2, cov2, fc2 = _fit_two_pole(f2, H2)
    b0_2, b1_2, r_a, f_a, Q_a, r_b, f_b, Q_b = _unpack_two(res2.x)
    H2_fit = H_two_pole(f2, b0_2, b1_2, r_a, f_a, Q_a, r_b, f_b, Q_b, fc2)
    Q_eff_2 = _eff_Q_at_op(f2, H2, 2 * CASCADE_F_OP_HZ)

    print(f"  1f one-pole: f0 = {f0_1/1e6:.4f} MHz, Q = {Q_1:.1f}, |r| = {abs(r_1):.2e} Pa/V")
    print(f"  2f two-pole: f_a = {f_a/1e6:.4f} MHz, Q_a = {Q_a:.1f}, |r_a| = {abs(r_a):.2e}")
    print(f"               f_b = {f_b/1e6:.4f} MHz, Q_b = {Q_b:.1f}, |r_b| = {abs(r_b):.2e}")
    print(f"  effective Q at cascade operating point: Q_eff_1 = {Q_eff_1:.0f}, "
          f"Q_eff_2 = {Q_eff_2:.0f}")

    return dict(
        # 1f sweep + fit
        f_sweep_1f=f1,
        h_1f_complex_pa_per_v=H1,
        h_1f_fit_complex=H1_fit,
        one_pole_b0=np.asarray(b0_1),
        one_pole_b1=np.asarray(b1_1),
        one_pole_residue=np.asarray(r_1),
        one_pole_f0=np.asarray(f0_1),
        one_pole_Q=np.asarray(Q_1),
        one_pole_cov=cov1,
        # 2f sweep + fit
        f_sweep_2f=f2,
        h_2f_complex_pa_per_v=H2,
        h_2f_fit_complex=H2_fit,
        two_pole_b0=np.asarray(b0_2),
        two_pole_b1=np.asarray(b1_2),
        two_pole_residue_a=np.asarray(r_a),
        two_pole_f_a=np.asarray(f_a),
        two_pole_Q_a=np.asarray(Q_a),
        two_pole_residue_b=np.asarray(r_b),
        two_pole_f_b=np.asarray(f_b),
        two_pole_Q_b=np.asarray(Q_b),
        two_pole_cov=cov2,
        # diagnostics
        q_eff_1=np.asarray(Q_eff_1),
        q_eff_2=np.asarray(Q_eff_2),
        cascade_f_op_hz=np.asarray(CASCADE_F_OP_HZ),
        # provenance
        phase_calibration_note=np.asarray(
            "Ch2-only pipeline; phase inherits Polytec velocity-decoder "
            "spec (<1 deg distortion in band). PLAN.md R2 cross-channel "
            "audit does not apply (see Ch3-only debug in "
            "figS1_decoder_check)."
        ),
    )


def plot(d: dict):
    f1 = d["f_sweep_1f"]; H1 = d["h_1f_complex_pa_per_v"]; H1f = d["h_1f_fit_complex"]
    f2 = d["f_sweep_2f"]; H2 = d["h_2f_complex_pa_per_v"]; H2f = d["h_2f_fit_complex"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5))

    # (a) |H_1| Bode magnitude
    ax = axes[0, 0]
    ax.plot(f1 / 1e6, np.abs(H1), "o", ms=3, color="C0", label="data")
    ff = np.linspace(f1.min(), f1.max(), 400)
    H_fit_fine = H_one_pole(ff, complex(d["one_pole_b0"]), complex(d["one_pole_b1"]),
                             complex(d["one_pole_residue"]), float(d["one_pole_f0"]),
                             float(d["one_pole_Q"]), float(np.mean(f1)))
    ax.plot(ff / 1e6, np.abs(H_fit_fine), "-", color="C3", lw=0.8,
            label=rf"1-pole: $f_0$={float(d['one_pole_f0'])/1e6:.4f} MHz, $Q$={float(d['one_pole_Q']):.0f}")
    ax.set_ylabel(r"$|H_{1f}|$ [Pa/V]")
    ax.set_xlabel(r"$f$ [MHz]")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.text(-0.15, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title("(a) 1f sweep: one-pole magnitude", fontsize=9)

    # (b) 1f phase (R2-gated)
    ax = axes[0, 1]
    ax.plot(f1 / 1e6, np.degrees(np.angle(H1)), "o", ms=3, color="C0", label="data")
    ax.plot(ff / 1e6, np.degrees(np.angle(H_fit_fine)), "-", color="C3", lw=0.8, label="1-pole fit")
    ax.set_ylabel(r"$\arg H_{1f}$ [deg]")
    ax.set_xlabel(r"$f$ [MHz]")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.text(-0.15, 0.98, "(b)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title("(b) 1f phase", fontsize=9)

    # (c) |H_2| with two-pole fit
    ax = axes[1, 0]
    ax.plot(f2 / 1e6, np.abs(H2), "o", ms=3, color="C0", label="data")
    ff2 = np.linspace(f2.min(), f2.max(), 600)
    H2_fit_fine = H_two_pole(ff2, complex(d["two_pole_b0"]), complex(d["two_pole_b1"]),
                              complex(d["two_pole_residue_a"]), float(d["two_pole_f_a"]),
                              float(d["two_pole_Q_a"]),
                              complex(d["two_pole_residue_b"]), float(d["two_pole_f_b"]),
                              float(d["two_pole_Q_b"]), float(np.mean(f2)))
    ax.plot(ff2 / 1e6, np.abs(H2_fit_fine), "-", color="C3", lw=0.8, label="2-pole fit")
    ax.axvline(float(d["two_pole_f_a"]) / 1e6, color="0.4", ls=":", lw=0.5,
               label=rf"$f_a$={float(d['two_pole_f_a'])/1e6:.4f} MHz, $Q_a$={float(d['two_pole_Q_a']):.0f}")
    ax.axvline(float(d["two_pole_f_b"]) / 1e6, color="0.4", ls=":", lw=0.5)
    ax.axvline(2 * float(d["cascade_f_op_hz"]) / 1e6, color="C2", ls="--", lw=0.7,
               label=rf"$2 f_\mathrm{{op}}$={2*float(d['cascade_f_op_hz'])/1e6:.3f} MHz")
    ax.set_ylabel(r"$|H_{2f}|$ projected on $n=2$ [Pa/V]")
    ax.set_xlabel(r"$f$ [MHz]")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.text(-0.15, 0.98, "(c)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title(rf"(c) 2f sweep: two-pole magnitude ($f_b$={float(d['two_pole_f_b'])/1e6:.4f} MHz, $Q_b$={float(d['two_pole_Q_b']):.0f})",
                 fontsize=9)

    # (d) 2f Nyquist (Re vs Im) — visualises the two-pole structure
    ax = axes[1, 1]
    ax.plot(H2.real, H2.imag, "o", ms=3, color="C0", label="data")
    ax.plot(H2_fit_fine.real, H2_fit_fine.imag, "-", color="C3", lw=0.8, label="2-pole fit")
    ax.axhline(0, color="0.6", lw=0.4, ls="--")
    ax.axvline(0, color="0.6", lw=0.4, ls="--")
    ax.set_xlabel(r"Re $H_{2f}$ [Pa/V]")
    ax.set_ylabel(r"Im $H_{2f}$ [Pa/V]")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.text(-0.18, 0.98, "(d)", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    ax.set_title("(d) 2f Nyquist (complex two-pole)", fontsize=9)

    fig.suptitle("PRL Fig S2 (W21) -- complex resonance response and effective $Q_n$", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def main(do_selftest: bool = False) -> None:
    PIPE_OUT.mkdir(parents=True, exist_ok=True)
    if do_selftest:
        print("Running --selftest synthetic recovery ...")
        selftest()
        return
    d = compute()
    fig = plot(d)
    fig.savefig(PIPE_OUT / "figS2.png", dpi=200)
    fig.savefig(PIPE_OUT / "figS2.pdf")
    plt.close(fig)
    np.savez(PIPE_OUT / "figS2.npz", **d)
    (PIPE_OUT / "figS2.json").write_text(json.dumps({
        "one_pole": {
            "f0_MHz": float(d["one_pole_f0"]) / 1e6,
            "Q": float(d["one_pole_Q"]),
            "residue_abs_Pa_per_V": float(abs(complex(d["one_pole_residue"]))),
        },
        "two_pole": {
            "f_a_MHz": float(d["two_pole_f_a"]) / 1e6,
            "Q_a": float(d["two_pole_Q_a"]),
            "residue_a_abs": float(abs(complex(d["two_pole_residue_a"]))),
            "f_b_MHz": float(d["two_pole_f_b"]) / 1e6,
            "Q_b": float(d["two_pole_Q_b"]),
            "residue_b_abs": float(abs(complex(d["two_pole_residue_b"]))),
        },
        "q_eff_at_operating_point": {
            "Q_eff_1": float(d["q_eff_1"]),
            "Q_eff_2": float(d["q_eff_2"]),
            "f_op_MHz": float(d["cascade_f_op_hz"]) / 1e6,
        },
        "phase_calibration_note": str(d["phase_calibration_note"]),
    }, indent=2), encoding="utf-8")
    print(f"  Saved {PIPE_OUT / 'figS2.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="Run synthetic one/two-pole recovery and exit.")
    args = ap.parse_args()
    main(do_selftest=args.selftest)
