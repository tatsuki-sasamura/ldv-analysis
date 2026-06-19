# %%
"""Per-axial-bin transient ring-up/down on a single 1f-band file.

Single-mode mass-spring-damper test: bin scan points by axial x and
compute the weighted-average complex 1f envelope per bin, then fit
the simple + beat models per bin.  If the cavity behaves as one ideal
oscillator, the rise tau (Q) should be the SAME in every axial bin.
A measurable spread tells us the spectral-vs-transient Q gap is
spatially structured -- e.g. the field has to fill axially, the
transducer is at one end, or there's mode-component beating.

Default file:
``sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_*/f1909000.h5``
(on cavity peak at 60 Vpp, 101x77 grid, 19 mm axial extent).

Output:
``output/<scan_dir>/transient_q1_axial_bins/transient_q1_axial_bins_<stem>.png``
plus a CSV of (x_bin, n_points, p_ss_kPa, Q_simple, Q_beat, df).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    FIG_DPI, LDV_DATA_ROOT, figsize_for_layout, get_cache_dir,
    velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_point_waveforms  # noqa: E402
from ldv_analysis.io_utils import ROLE_LDV_OUTPUT, load_scan  # noqa: E402
from ldv_analysis.transient import (  # noqa: E402
    DF_THRESHOLD, FIT_SKIP_US, RISE_FIT_WINDOW_US,
    compute_fit_windows, detect_burst, estimate_beat_freq,
    load_transient_data, rise_beat_residual, rise_simple,
    sliding_dft_envelope, smooth_envelope, tau_to_Q,
)

DEFAULT_INPUT = (
    LDV_DATA_ROOT / "output" / "W21"
    / "sample_101x77_fsweep_1p89to1p92_1kHz_60Vpp_20260530_031237"
    / "f1909000.h5"
)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
parser.add_argument("input", nargs="?", default=None,
                    help="Path to v2 HDF5 file")
parser.add_argument("--n-bins", type=int, default=6,
                    help="Number of axial bins (default: 6)")
args = parser.parse_args()

h5_path = Path(args.input) if args.input else DEFAULT_INPUT
print(f"Loading: {h5_path.name}")

OUT_DIR = (ROOT / "experiments" / "2026W21" / "output"
           / h5_path.parent.name / "transient_q1_axial_bins")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Per-point sliding DFT, accumulated into axial bins
# =============================================================================

fft_cache_dir = get_cache_dir(h5_path.parent.name, __file__)
td = load_transient_data(h5_path, fft_cache_dir)
f1 = td.f1
dt = td.dt
n_samples = td.n_samples
to_kPa = velocity_to_pressure(f1) / 1e3

cache = td.cache
pos_x = np.asarray(cache["pos_x"])     # width (m)
pos_y = np.asarray(cache["pos_y"])     # length / axial (m)
valid = td.valid

# Define axial bins on pos_y (which is the channel-length axis here).
y_min, y_max = float(pos_y.min()), float(pos_y.max())
edges = np.linspace(y_min, y_max, args.n_bins + 1)
centers_mm = 0.5 * (edges[:-1] + edges[1:]) * 1e3
print(f"\nAxial binning: {args.n_bins} bins between "
      f"{y_min*1e3:.2f} and {y_max*1e3:.2f} mm")

# Each bin accumulates a complex env wsum + weight sum (matching
# transient_ch2_acoustic.py's amplitude-weighted average).
env_wsum = [np.zeros(n_samples, dtype=complex) for _ in range(args.n_bins)]
weight_sum = np.zeros(args.n_bins)
n_used = np.zeros(args.n_bins, dtype=int)

valid_idx = np.where(valid)[0]
scan = load_scan(h5_path)
CHUNK = 200
print(f"Computing per-point envelopes ({len(valid_idx)} valid pts, "
      f"chunk={CHUNK})...")
for c0 in range(0, len(valid_idx), CHUNK):
    chunk_idx = valid_idx[c0:c0 + CHUNK]
    wfs = scan.load_waveforms(ROLE_LDV_OUTPUT, chunk_idx)
    for k, point_i in enumerate(chunk_idx):
        wf = wfs[k]
        env_c = sliding_dft_envelope(
            wf, dt, f1, return_complex=True
        ) * to_kPa
        p_ss_c = np.mean(env_c[td.ss_start:td.ss_end])
        p_ss_mag = abs(p_ss_c)
        if p_ss_mag <= 0:
            continue
        bin_i = int(np.clip(
            np.digitize(pos_y[point_i], edges) - 1, 0, args.n_bins - 1
        ))
        w = p_ss_mag
        env_wsum[bin_i] += w * (env_c / p_ss_c)
        weight_sum[bin_i] += w
        n_used[bin_i] += 1
    if (c0 // CHUNK) % 5 == 0:
        print(f"  chunk {c0}/{len(valid_idx)}")
del scan

# Best-point Ch1 envelope for burst-window detection.
wfs_best, _ = load_point_waveforms(
    h5_path, td.best_i, roles=("drive_voltage",)
)
best_ch1 = smooth_envelope(wfs_best["drive_voltage"])
burst_on, burst_off = detect_burst(best_ch1, dt)
fw = compute_fit_windows(burst_on, burst_off, dt, n_samples)
rs, re = fw["rise_start"], fw["rise_end"]
t_us = np.arange(re - rs) * dt * 1e6
print(f"Burst ON: {burst_on*dt*1e6:.0f}-{burst_off*dt*1e6:.0f} us; "
      f"rise window: {rs*dt*1e6:.0f}-{re*dt*1e6:.0f} us")

# =============================================================================
# Per-bin fits + plot
# =============================================================================

print(f"\n{'bin':>4}  {'x_mid (mm)':>10}  {'n_pts':>6}  {'p_ss(kPa)':>10}  "
      f"{'Q_simple':>9}  {'Q_beat':>7}  {'df(kHz)':>9}")
results = []
for bi in range(args.n_bins):
    if weight_sum[bi] <= 0:
        results.append(None)
        print(f"{bi:>4}  {centers_mm[bi]:>10.2f}  --skip--")
        continue
    env_norm = env_wsum[bi] / weight_sum[bi]
    ec = env_norm[rs:re]
    # simple fit
    po, _ = curve_fit(rise_simple, t_us, np.abs(ec),
                      p0=[10.0], bounds=([0.1], [500.0]))
    tau_simple = float(po[0])
    # beat fit
    df_guess = estimate_beat_freq(ec - 1.0, t_us[1] - t_us[0])
    res = least_squares(
        rise_beat_residual, x0=[tau_simple, df_guess, 0.0],
        args=(t_us, ec.real, ec.imag),
        bounds=([0.1, -1e6, -np.pi], [500.0, 1e6, np.pi]),
        method="trf",
    )
    tau_beat, df_beat, phi_beat = res.x
    Q_simple = float(tau_to_Q(f1, tau_simple))
    Q_beat = float(tau_to_Q(f1, tau_beat))
    results.append(dict(
        bin_i=bi, x_mid_mm=centers_mm[bi], n_pts=int(n_used[bi]),
        p_ss_kPa=float(weight_sum[bi] / n_used[bi]),
        env=env_norm, ec=ec,
        tau_simple=tau_simple, Q_simple=Q_simple,
        tau_beat=float(tau_beat), Q_beat=Q_beat,
        df_beat=float(df_beat), phi_beat=float(phi_beat),
        beat_significant=bool(abs(df_beat) > DF_THRESHOLD),
    ))
    print(f"{bi:>4}  {centers_mm[bi]:>10.2f}  {int(n_used[bi]):>6}  "
          f"{weight_sum[bi]/n_used[bi]:>10.1f}  "
          f"{Q_simple:>9.1f}  {Q_beat:>7.1f}  {df_beat/1e3:>+9.2f}")

# ---- CSV ----------------------------------------------------------------
csv_path = OUT_DIR / f"transient_q1_axial_bins_{h5_path.stem}.csv"
rows = ["bin,x_mid_mm,n_pts,p_ss_kPa,Q_simple,tau_simple_us,Q_beat,"
        "tau_beat_us,df_beat_kHz"]
for r in results:
    if r is None:
        continue
    rows.append(f"{r['bin_i']},{r['x_mid_mm']:.3f},{r['n_pts']},"
                f"{r['p_ss_kPa']:.1f},{r['Q_simple']:.2f},"
                f"{r['tau_simple']:.3f},{r['Q_beat']:.2f},"
                f"{r['tau_beat']:.3f},{r['df_beat']/1e3:.3f}")
csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"\nSaved {csv_path}")

# ---- plot --------------------------------------------------------------
have = [r for r in results if r is not None]
nb = len(have)
ncol = 3
nrow = (nb + ncol - 1) // ncol
fig = plt.figure(figsize=(3.0 * ncol, 1.9 * nrow + 1.4))
gs = fig.add_gridspec(nrow + 1, ncol, hspace=0.5, wspace=0.3,
                      height_ratios=[1] * nrow + [1.1])

cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=min(r["x_mid_mm"] for r in have),
                     vmax=max(r["x_mid_mm"] for r in have))

# per-bin subplots
for i, r in enumerate(have):
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    color = cmap(norm(r["x_mid_mm"]))
    t = t_us
    ax.plot(t, np.abs(r["ec"]), ".", markersize=2, color="0.3", alpha=0.6,
            label="data $|E|$")
    ax.plot(t, 1 - np.exp(-t / r["tau_simple"]),
            "-", color="C3", linewidth=0.8,
            label=f"simple: $Q$={r['Q_simple']:.0f}")
    ax.plot(t, np.abs(1 - np.exp(-t / r["tau_beat"])
                      * np.exp(1j * (2 * np.pi * r["df_beat"] * t * 1e-6
                                      + r["phi_beat"]))),
            "-", color="C0", linewidth=0.8,
            label=(f"beat: $Q$={r['Q_beat']:.0f}, "
                   f"$\\Delta f$={r['df_beat']/1e3:+.1f} kHz"))
    ax.set_title(f"bin {r['bin_i']}: $x \\approx {r['x_mid_mm']:.1f}$ mm  "
                 f"({r['n_pts']} pts)", fontsize=8)
    ax.set_ylim(0, max(1.4, np.max(np.abs(r["ec"])) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    if i // ncol == nrow - 1:
        ax.set_xlabel(r"$t$ since burst ON [\textmu s]")
    if i % ncol == 0:
        ax.set_ylabel(r"normalised $|E_{1f}(t)|$")

# bottom summary row: Q vs x and overlay of all envelopes
ax = fig.add_subplot(gs[nrow, 0:2])
for r in have:
    ax.plot(t_us, np.abs(r["ec"]), "-", linewidth=0.8,
            color=cmap(norm(r["x_mid_mm"])),
            label=f"x = {r['x_mid_mm']:.1f} mm")
ax.set_xlabel(r"$t$ since burst ON [\textmu s]")
ax.set_ylabel(r"normalised $|E_{1f}(t)|$")
ax.set_title("All bins overlaid (color = axial position)")
ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[nrow, 2])
x_vals = np.array([r["x_mid_mm"] for r in have])
Q_simple_vals = np.array([r["Q_simple"] for r in have])
Q_beat_vals = np.array([r["Q_beat"] for r in have])
ax.plot(x_vals, Q_simple_vals, "o-", markersize=4, linewidth=0.8,
        color="C3", label="simple")
ax.plot(x_vals, Q_beat_vals, "s-", markersize=4, linewidth=0.8,
        color="C0", label="beat")
ax.set_xlabel(r"$x_\mathrm{mid}$ [mm]")
ax.set_ylabel(r"$Q_1$")
ax.set_title("$Q_1$ per axial bin")
ax.legend(fontsize=7, frameon=False)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Per-axial-bin 1f ring-up on  {h5_path.stem}  "
    f"({h5_path.parent.name})",
    fontsize=10,
)
plt.tight_layout()
out = OUT_DIR / f"transient_q1_axial_bins_{h5_path.stem}.png"
fig.savefig(out, dpi=FIG_DPI)
plt.close(fig)
print(f"Saved {out}")
