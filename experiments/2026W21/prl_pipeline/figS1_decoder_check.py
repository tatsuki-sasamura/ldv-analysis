"""Decoder cross-check (Fig S1 rule-out) — velocity vs displacement.

Physical identity for a single-tone steady-state oscillator:
    D(omega) = V(omega) / (i*omega)
so the complex ratio of the displacement-decoder DFT (Ch3) to the
velocity-decoder DFT (Ch2) at the drive frequency should equal -i/omega.

If the identity holds within tolerance, the LDV decoder's apparent-
quantity output is internally consistent. The complex pressure
calibration C_p<-d(f) inherits the validation: pressure is recovered
from EITHER channel via p = -d / (H * dn/dp) = -v / (2*pi*f * H * dn/dp),
and the two should give the same complex pressure.

This is a diagnostic Ch3-consistency check. It does NOT gate any
published P_nf value: the publishing pipeline uses Ch2 (velocity)
exclusively; Ch3 is not consumed by fft_cache or any figure script.
A failure here means the displacement decoder's scale/phase recorded
in config.py does not match what was used during acquisition — useful
to know if anyone later wants to derive pressure from Ch3, irrelevant
to everything we publish today. See PLAN.md sec.11 R2 documentation
note for the rewritten R2 status.

Output: ``PIPE_OUT/figS1_decoder_check.{png,pdf,json}``.

Usage::
    python experiments/2026W21/prl_pipeline/figS1_decoder_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from conventions import (
    CASCADE_SCANS,
    PIPE_OUT,
    W21_DATA_ROOT,
    get_cache_dir,
)
from ldv_analysis.config import VELOCITY_SCALE, DISPLACEMENT_SCALE
from ldv_analysis.fft_cache import load_or_compute

# Use the 120 Vpp / 1.902 MHz operating-point file (highest SNR)
SCAN_DIR = CASCADE_SCANS[-1][1]
H5_FILE = "f1902000.h5"
N_POINTS = 21  # cross-channel of the 101x21 grid; enough for spread

TOLERANCE_MAG = 0.05   # ±5% on |D/V| vs 1/omega
TOLERANCE_PHASE_DEG = 5.0  # ±5° on arg(D/V) vs -90°


def main() -> None:
    PIPE_OUT.mkdir(parents=True, exist_ok=True)

    run_dir = W21_DATA_ROOT / SCAN_DIR
    h5_path = run_dir / H5_FILE
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not present")

    # Use the cache's burst window + drive frequency to ensure consistency.
    cache_dir = get_cache_dir(SCAN_DIR, str(Path(__file__).parent / "figS1_decoder_check.py"))
    cache = load_or_compute(h5_path, cache_dir, velocity_scale=None)
    f_drive = float(cache["f_drive"])
    omega = 2 * np.pi * f_drive
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    dt = float(cache["dt"])
    n_total = int(cache["voltage_1f"].size)
    pts = np.linspace(0, n_total - 1, N_POINTS, dtype=int)

    # Exact-frequency DFT via dot product with the 1f phasor.
    ss_n = ss_end - ss_start
    t = np.arange(ss_n) * dt
    tone = np.exp(-2j * np.pi * f_drive * t)

    v_complex_mps = np.empty(N_POINTS, dtype=complex)
    d_complex_m = np.empty(N_POINTS, dtype=complex)
    with h5py.File(h5_path, "r") as f:
        ldv_v = f["waveforms/ldv_output"]
        ldv_d = f["waveforms/ldv_displacement"]
        for k, idx in enumerate(pts):
            v_seg = ldv_v[int(idx), ss_start:ss_end]
            d_seg = ldv_d[int(idx), ss_start:ss_end]
            v_complex_mps[k] = (v_seg @ tone) * (2 / ss_n) * VELOCITY_SCALE
            d_complex_m[k] = (d_seg @ tone) * (2 / ss_n) * DISPLACEMENT_SCALE

    # Cross-check: D/V should be 1/(i*omega) = -i/omega
    ratio = d_complex_m / v_complex_mps
    theory = -1j / omega                              # complex scalar

    mag_meas = np.abs(ratio)
    mag_theory = 1.0 / omega
    rel_mag_err = mag_meas / mag_theory - 1.0         # dimensionless

    phase_meas_deg = np.degrees(np.angle(ratio))
    phase_theory_deg = np.degrees(np.angle(theory))   # -90 deg
    phase_err_deg = ((phase_meas_deg - phase_theory_deg + 180) % 360) - 180

    med_mag_err = float(np.median(rel_mag_err))
    spread_mag_err = float(np.median(np.abs(rel_mag_err - med_mag_err)))
    med_phase_err = float(np.median(phase_err_deg))
    spread_phase_err = float(np.median(np.abs(phase_err_deg - med_phase_err)))

    mag_passes = bool(abs(med_mag_err) < TOLERANCE_MAG)
    phase_passes = bool(abs(med_phase_err) < TOLERANCE_PHASE_DEG)
    overall_pass = bool(mag_passes and phase_passes)

    print(f"Decoder cross-check on {SCAN_DIR}/{H5_FILE}:")
    print(f"  f_drive = {f_drive/1e6:.4f} MHz, omega = {omega:.3e} rad/s")
    print(f"  N points sampled: {N_POINTS}")
    print(f"  |D|/|V| median = {med_mag_err*100:+.2f}% from 1/omega "
          f"(spread MAD = {spread_mag_err*100:.2f}%)  -> {'PASS' if mag_passes else 'FAIL'}")
    print(f"  arg(D/V) median = {med_phase_err:+.2f} deg from -90 deg "
          f"(spread MAD = {spread_phase_err:.2f} deg)  -> {'PASS' if phase_passes else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'} "
          f"(R2 audit -- complex P_nf release {'enabled' if overall_pass else 'gated'})")

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Panel a: magnitude error
    ax = axes[0]
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.axhline(TOLERANCE_MAG * 100, color="C3", lw=0.4, ls=":",
               label=rf"$\pm$ {TOLERANCE_MAG*100:.0f}% tolerance")
    ax.axhline(-TOLERANCE_MAG * 100, color="C3", lw=0.4, ls=":")
    ax.plot(np.arange(N_POINTS), rel_mag_err * 100, "o", ms=4, color="C0")
    ax.axhline(med_mag_err * 100, color="C0", lw=0.5, ls="-",
               label=f"median {med_mag_err*100:+.2f}%")
    ax.set_xlabel("scan-point index")
    ax.set_ylabel(r"$|D|/|V| - 1/\omega$, relative [\%]")
    ax.set_title("(a) magnitude consistency", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="best")

    # Panel b: phase error
    ax = axes[1]
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.axhline(TOLERANCE_PHASE_DEG, color="C3", lw=0.4, ls=":",
               label=rf"$\pm$ {TOLERANCE_PHASE_DEG:.0f}$^\circ$ tolerance")
    ax.axhline(-TOLERANCE_PHASE_DEG, color="C3", lw=0.4, ls=":")
    ax.plot(np.arange(N_POINTS), phase_err_deg, "o", ms=4, color="C0")
    ax.axhline(med_phase_err, color="C0", lw=0.5, ls="-",
               label=f"median {med_phase_err:+.2f}$^\\circ$")
    ax.set_xlabel("scan-point index")
    ax.set_ylabel(r"$\arg(D/V) - (-90^\circ)$ [deg]")
    ax.set_title("(b) phase consistency", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="best")

    fig.suptitle(f"Fig S1 R2 audit: decoder cross-check  ({SCAN_DIR}/{H5_FILE})  -- "
                 f"{'PASS' if overall_pass else 'FAIL'}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PIPE_OUT / "figS1_decoder_check.png", dpi=200)
    fig.savefig(PIPE_OUT / "figS1_decoder_check.pdf")
    plt.close(fig)

    (PIPE_OUT / "figS1_decoder_check.json").write_text(json.dumps({
        "scan_dir": SCAN_DIR,
        "h5_file": H5_FILE,
        "f_drive_hz": f_drive,
        "n_points": int(N_POINTS),
        "magnitude_err_median_rel": med_mag_err,
        "magnitude_err_spread_mad_rel": spread_mag_err,
        "magnitude_tolerance_rel": TOLERANCE_MAG,
        "magnitude_passes": mag_passes,
        "phase_err_median_deg": med_phase_err,
        "phase_err_spread_mad_deg": spread_phase_err,
        "phase_tolerance_deg": TOLERANCE_PHASE_DEG,
        "phase_passes": phase_passes,
        "overall_pass": overall_pass,
        "r2_audit_status": "PASS" if overall_pass else "FAIL",
    }, indent=2), encoding="utf-8")
    print(f"  Saved {PIPE_OUT / 'figS1_decoder_check.png'}")


if __name__ == "__main__":
    main()
