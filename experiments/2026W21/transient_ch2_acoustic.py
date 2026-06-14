# %%
"""Ch2 acoustic transient analysis: ring-up/ring-down Q estimation.

1f simple exponential + damped-beat fit, 2f driven resonator fit.
Promoted from ``experiments/2026W10_stepA/`` for use on W21 v2 HDF5
sweeps; the only change is per-dataset output routing so that runs
on different scan directories (or different freq files within them)
do not overwrite each other.

Output layout::

    experiments/2026W21/output/<scan_dir_name>/transient/
        transient_ch2_<file_stem>.png
        transient_3f_<file_stem>.png
        cache/_transient_env_ch2_<file_stem>.npz

Usage::

    python transient_ch2_acoustic.py <path_to_h5_or_tdms>

By default points at the W21 coarse 10 Vpp scan's ``f1910000.h5``
(the P_1f peak file).  Pass any other v1 TDMS or v2 HDF5 file to
override.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

from ldv_analysis.config import (
    LDV_DATA_ROOT, figsize_for_layout, get_data_dir, velocity_to_pressure,
)
from ldv_analysis.fft_cache import load_point_waveforms
from ldv_analysis.io_utils import ROLE_LDV_OUTPUT, load_scan
from ldv_analysis.plotting import save_fig
from ldv_analysis.transient import (
    DF_THRESHOLD, FIT_SKIP_US, RISE_FIT_WINDOW_US,
    beat_model_complex, compute_fit_windows, detect_burst,
    estimate_beat_freq, fall_beat_residual, fall_simple,
    load_transient_data, rise_2f, rise_beat_residual, rise_simple,
    sliding_dft_envelope, smooth_envelope, tau_to_Q,
    add_burst_markers, add_burst_spans, plot_no_data, plot_phase_column,
)

# Default input: W21 coarse-sweep P_1f peak file
DEFAULT_INPUT = (LDV_DATA_ROOT / "output" / "W21"
                 / "sample_101x1_fsweep_coarse_10Vpp_20260524_130731"
                 / "f1910000.h5")

# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", nargs="?", default=None,
                    help="Path to v2 HDF5 (.h5) or v1 TDMS (.tdms) file")
parser.add_argument("--fresh", action="store_true",
                    help="Recompute envelopes (ignore cached .npz)")
_args = parser.parse_args()

tdms_path = Path(_args.input) if _args.input else DEFAULT_INPUT

# Per-dataset OUT_DIR keyed by the input file's parent directory
# name (the scan dir), so multiple scans coexist without overwriting.
OUT_DIR = (ROOT / "experiments" / "2026W21" / "output"
           / tdms_path.parent.name / "transient")
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUT_DIR  = {OUT_DIR}")
print(f"INPUT    = {tdms_path}")

td = load_transient_data(tdms_path, CACHE_DIR)

f1 = td.f1
dt = td.dt
n_samples = td.n_samples
t_us = td.t_us
ss_start = td.ss_start
ss_end = td.ss_end
valid = td.valid
best_i = td.best_i

to_kPa = velocity_to_pressure(f1) / 1e3       # signed: V → kPa
to_kPa_2f = velocity_to_pressure(2 * f1) / 1e3

# %%
# =============================================================================
# Envelope computation
# =============================================================================

env_cache_path = CACHE_DIR / f"_transient_env_ch2_{td.stem}.npz"
if env_cache_path.exists() and not _args.fresh:
    print(f"  Loading envelope cache: {env_cache_path.name}")
    _ec = np.load(env_cache_path)
    env_ch2_norm_complex = _ec["env_1f_complex"]
    env_ch2_2f_norm_complex = _ec["env_2f_complex"]
    env_ch2_3f_norm_complex = _ec["env_3f_complex"] if "env_3f_complex" in _ec else None
    env_1f_std = _ec["env_1f_std"] if "env_1f_std" in _ec else None
    env_2f_std = _ec["env_2f_std"] if "env_2f_std" in _ec else None
    env_3f_std = _ec["env_3f_std"] if "env_3f_std" in _ec else None
    env_ch2_kPa = _ec["best_1f_kPa"]
    env_ch2_2f_best_kPa = _ec["best_2f_kPa"]
    env_ch1 = _ec["best_ch1"]
    n_used = int(_ec["n_used"])
    p_ss = float(_ec["p_ss"])
    p_ss_2f = float(_ec["p_ss_2f"])
    p_ss_3f = float(_ec["p_ss_3f"]) if "p_ss_3f" in _ec else 0.0
    # Force recompute if 3f not in cache
    if env_ch2_3f_norm_complex is None:
        print("  3f envelope not in cache, recomputing...")
    del _ec
else:
    print(f"  Computing averaged Ch2 envelopes (per-point)...")
    scan = load_scan(tdms_path)

    to_kPa_3f = abs(velocity_to_pressure(3 * f1)) / 1e3

    env_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_mag_sq_wsum = np.zeros(n_samples)
    env_2f_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_2f_mag_sq_wsum = np.zeros(n_samples)
    env_3f_complex_wsum = np.zeros(n_samples, dtype=complex)
    env_3f_mag_sq_wsum = np.zeros(n_samples)
    weight_sum = 0.0
    n_used = 0

    valid_idx = np.where(valid)[0]
    CHUNK = 100  # read 100 waveforms at a time (~80 MB for 100k samples)
    for c0 in range(0, len(valid_idx), CHUNK):
        chunk_idx = valid_idx[c0:c0 + CHUNK]
        chunk_wfs = scan.load_waveforms(ROLE_LDV_OUTPUT, chunk_idx)
        for k_in_chunk in range(len(chunk_idx)):
            wf = chunk_wfs[k_in_chunk]
            env_c = sliding_dft_envelope(wf, dt, f1, return_complex=True) * to_kPa
            env_2f_c = sliding_dft_envelope(wf, dt, 2 * f1, return_complex=True) * to_kPa_2f
            env_3f_c = sliding_dft_envelope(wf, dt, 3 * f1, return_complex=True) * to_kPa_3f
            p_ss_c = np.mean(env_c[ss_start:ss_end])
            p_ss_mag = np.abs(p_ss_c)
            if p_ss_mag > 0:
                w = p_ss_mag
                norm_mag = np.abs(env_c) / p_ss_mag
                env_complex_wsum += w * (env_c / p_ss_c)
                env_mag_sq_wsum += w * norm_mag**2
                p_ss_2f_c = np.mean(env_2f_c[ss_start:ss_end])
                p_ss_2f_mag = np.abs(p_ss_2f_c)
                if p_ss_2f_mag > 0:
                    norm_2f_mag = np.abs(env_2f_c) / p_ss_2f_mag
                    env_2f_complex_wsum += w * (env_2f_c / p_ss_2f_c)
                    env_2f_mag_sq_wsum += w * norm_2f_mag**2
                p_ss_3f_c = np.mean(env_3f_c[ss_start:ss_end])
                p_ss_3f_mag = np.abs(p_ss_3f_c)
                if p_ss_3f_mag > 0:
                    norm_3f_mag = np.abs(env_3f_c) / p_ss_3f_mag
                    env_3f_complex_wsum += w * (env_3f_c / p_ss_3f_c)
                    env_3f_mag_sq_wsum += w * norm_3f_mag**2
                weight_sum += w
                n_used += 1
    env_ch2_norm_complex = env_complex_wsum / weight_sum
    env_ch2_2f_norm_complex = env_2f_complex_wsum / weight_sum
    env_ch2_3f_norm_complex = env_3f_complex_wsum / weight_sum
    env_1f_mean = np.abs(env_ch2_norm_complex)
    env_1f_std = np.sqrt(np.maximum(env_mag_sq_wsum / weight_sum - env_1f_mean**2, 0))
    env_2f_mean = np.abs(env_ch2_2f_norm_complex)
    env_2f_std = np.sqrt(np.maximum(env_2f_mag_sq_wsum / weight_sum - env_2f_mean**2, 0))
    env_3f_mean = np.abs(env_ch2_3f_norm_complex)
    env_3f_std = np.sqrt(np.maximum(env_3f_mag_sq_wsum / weight_sum - env_3f_mean**2, 0))
    print(f"  Averaged {n_used} normalized Ch2 envelopes (p_ss-weighted)")

    wfs_best, _ = load_point_waveforms(
        tdms_path, best_i, roles=("drive_voltage", "ldv_output")
    )
    env_ch2_kPa = sliding_dft_envelope(wfs_best["ldv_output"], dt, f1) * to_kPa
    p_ss = float(np.mean(env_ch2_kPa[ss_start:ss_end]))
    env_ch2_2f_best_kPa = sliding_dft_envelope(
        wfs_best["ldv_output"], dt, 2 * f1
    ) * to_kPa_2f
    p_ss_2f = float(np.mean(env_ch2_2f_best_kPa[ss_start:ss_end]))
    p_ss_3f = float(np.mean(np.abs(env_ch2_3f_norm_complex)[ss_start:ss_end]))
    env_ch1 = smooth_envelope(wfs_best["drive_voltage"])
    del scan

    np.savez(env_cache_path,
             env_1f_complex=env_ch2_norm_complex,
             env_2f_complex=env_ch2_2f_norm_complex,
             env_3f_complex=env_ch2_3f_norm_complex,
             env_1f_std=env_1f_std, env_2f_std=env_2f_std,
             env_3f_std=env_3f_std,
             best_1f_kPa=env_ch2_kPa, best_2f_kPa=env_ch2_2f_best_kPa,
             best_ch1=env_ch1,
             n_used=n_used, p_ss=p_ss, p_ss_2f=p_ss_2f, p_ss_3f=p_ss_3f)
    print(f"  Saved: {env_cache_path.name}")

burst_on, burst_off = detect_burst(env_ch1, dt)
fw = compute_fit_windows(burst_on, burst_off, dt, n_samples)
rise_start, rise_end = fw["rise_start"], fw["rise_end"]
fall_start, end_f, has_fall = fw["fall_start"], fw["end_f"], fw["has_fall"]
print(f"  Burst ON: {burst_on * dt * 1e6:.1f}--{burst_off * dt * 1e6:.1f} us")

# %%
# =============================================================================
# Fitting
# =============================================================================

# --- 1f: simple exponential + beat ---
ch2_rise = {
    "t": np.arange(rise_end - rise_start) * dt * 1e6,
    "e": np.abs(env_ch2_norm_complex)[rise_start:rise_end],
    "ec": env_ch2_norm_complex[rise_start:rise_end],
}
ch2_rise["po"], ch2_rise["pcov"] = curve_fit(
    rise_simple, ch2_rise["t"], ch2_rise["e"],
    p0=[10], bounds=([0.1], [500]))

_rise_res = ch2_rise["ec"] - 1.0
_df_guess_r = estimate_beat_freq(_rise_res, ch2_rise["t"][1] - ch2_rise["t"][0])
ch2_rise["beat"] = least_squares(
    rise_beat_residual,
    x0=[ch2_rise["po"][0], _df_guess_r, 0.0],
    args=(ch2_rise["t"], ch2_rise["ec"].real, ch2_rise["ec"].imag),
    bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
    method="trf",
)

if has_fall:
    ch2_fall = {
        "t": np.arange(end_f - fall_start) * dt * 1e6,
        "e": np.abs(env_ch2_norm_complex)[fall_start:end_f],
        "ec": env_ch2_norm_complex[fall_start:end_f],
    }
    ch2_fall["po"], ch2_fall["pcov"] = curve_fit(
        fall_simple, ch2_fall["t"], ch2_fall["e"],
        p0=[10], bounds=([0.1], [500]))
    _df_guess_f = estimate_beat_freq(ch2_fall["ec"], ch2_fall["t"][1] - ch2_fall["t"][0])
    ch2_fall["beat"] = least_squares(
        fall_beat_residual,
        x0=[ch2_fall["po"][0], _df_guess_f, 0.0],
        args=(ch2_fall["t"], ch2_fall["ec"].real, ch2_fall["ec"].imag),
        bounds=([0.1, -1e6, -np.pi], [500, 1e6, np.pi]),
        method="trf",
    )
else:
    ch2_fall = None
    print("  WARNING: no fall data (burst extends to end of recording)")

# --- 2f: driven resonator (source ~ p1^2) ---
tau1_fixed = ch2_rise["po"][0]
ch2_2f_rise = {
    "t": ch2_rise["t"],
    "e": np.abs(env_ch2_2f_norm_complex)[rise_start:rise_end],
}
_pre_burst = np.abs(env_ch2_2f_norm_complex)[max(0, burst_on - 100):burst_on]
noise_floor_2f = float(np.mean(_pre_burst)) if len(_pre_burst) > 0 else 0.0

ch2_2f_rise["po"], ch2_2f_rise["pcov"] = curve_fit(
    lambda t, tau2, A, b: b + A * rise_2f(t, tau2, tau1_fixed),
    ch2_2f_rise["t"], ch2_2f_rise["e"],
    p0=[tau1_fixed * 0.3, 1.0 - noise_floor_2f, noise_floor_2f],
    bounds=([0.1, 0, 0], [500, 5, 1]))
tau_2f, A_2f, baseline_2f = ch2_2f_rise["po"]
tau_2f_err = np.sqrt(ch2_2f_rise["pcov"][0, 0])

# --- Extract parameters ---
tau_ch2 = ch2_rise["po"][0]
tau_ch2_err = np.sqrt(ch2_rise["pcov"][0, 0])
tau_beat_r, df_beat_r, phi_beat_r = ch2_rise["beat"].x
beat_sig_r = abs(df_beat_r) > DF_THRESHOLD
if has_fall:
    tau_beat_f, df_beat_f, phi_beat_f = ch2_fall["beat"].x
    beat_sig_f = abs(df_beat_f) > DF_THRESHOLD
    tau_ch2_f = ch2_fall["po"][0]

# %%
# =============================================================================
# Results
# =============================================================================

print(f"\n--- Ch2 (acoustic) ---")
print(f"  p_ss_1f = {p_ss:.0f} kPa, p_ss_2f = {p_ss_2f:.0f} kPa")
print(f"  Rise (simple):  tau = {tau_ch2:.2f} +/- {tau_ch2_err:.2f} us"
      f"  ->  Q = {tau_to_Q(f1, tau_ch2):.0f}")
print(f"  Rise (beat):    tau = {tau_beat_r:.2f} us"
      f"  ->  Q = {tau_to_Q(f1, tau_beat_r):.0f}"
      f"  df = {df_beat_r:.0f} Hz" + ("" if beat_sig_r else "  (not significant)"))
if has_fall:
    print(f"  Fall (simple):  tau = {tau_ch2_f:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_ch2_f):.0f}")
    print(f"  Fall (beat):    tau = {tau_beat_f:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_beat_f):.0f}"
          f"  df = {df_beat_f:.0f} Hz" + ("" if beat_sig_f else "  (not significant)"))
    if beat_sig_r or beat_sig_f:
        df_avg = np.mean([abs(df_beat_r), abs(df_beat_f)])
        print(f"  Mode splitting: {df_avg/1e3:.1f} kHz")
else:
    print(f"  Fall: no data")
print(f"\n--- Ch2 2f (driven resonator, source ~ p1^2) ---")
print(f"  tau_1 (fixed from 1f) = {tau1_fixed:.2f} us")
print(f"  tau_2 = {tau_2f:.2f} +/- {tau_2f_err:.2f} us"
      f"  ->  Q_2f = {tau_to_Q(f1, tau_2f, harmonic=2):.0f}")
print(f"  baseline = {baseline_2f:.3f}, A = {A_2f:.3f}")

# %%
# =============================================================================
# Visualization: 3x4 (1f amp, 1f phase, 2f amp, 2f phase)
# =============================================================================

t_fine_rise = np.linspace(0, ch2_rise["t"][-1], 500)
curve_rise_simple = rise_simple(t_fine_rise, *ch2_rise["po"])
curve_rise_beat = 1.0 - beat_model_complex(
    t_fine_rise, tau_beat_r, df_beat_r, phi_beat_r)
if has_fall:
    t_fine_fall = np.linspace(0, ch2_fall["t"][-1], 500)
    curve_fall_simple = fall_simple(t_fine_fall, *ch2_fall["po"])
    curve_fall_beat = beat_model_complex(
        t_fine_fall, tau_beat_f, df_beat_f, phi_beat_f)
t_fine_2f = np.linspace(0, ch2_rise["t"][-1], 500)
curve_2f_model = baseline_2f + A_2f * rise_2f(t_fine_2f, tau_2f, tau1_fixed)
curve_2f_source = A_2f * rise_simple(t_fine_2f, tau1_fixed)**2

ss_start_us = ss_start * dt * 1e6
ss_end_us = ss_end * dt * 1e6
burst_on_us = burst_on * dt * 1e6
burst_off_us = burst_off * dt * 1e6

fig, axes = plt.subplots(3, 4, figsize=figsize_for_layout(3, 4))

# Col 0: 1f amplitude
ax = axes[0, 0]
ax.plot(t_us, env_ch2_kPa, linewidth=0.4, color="C0", alpha=0.5, label="best point")
ax.plot(t_us, np.abs(env_ch2_norm_complex) * p_ss, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 1f envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 0]
ax.plot(ch2_rise["t"], ch2_rise["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
ax.plot(t_fine_rise, curve_rise_simple, "--", color="0.6", linewidth=0.8, alpha=0.7,
        label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
        % (tau_ch2, tau_to_Q(f1, tau_ch2)))
ax.plot(t_fine_rise, np.real(curve_rise_beat), "--", color="C3", linewidth=1.2,
        label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
        % (tau_beat_r, tau_to_Q(f1, tau_beat_r), df_beat_r / 1e3))
ax.set_ylabel(r"Normalized $P / P_{ss}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title(f"Ch2 1f ring-up (avg {n_used} pts)")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 0]
if has_fall:
    ax.plot(ch2_fall["t"], ch2_fall["e"], "-", linewidth=0.5, color="C0", alpha=0.7)
    ax.plot(t_fine_fall, curve_fall_simple, "--", color="0.6", linewidth=0.8, alpha=0.7,
            label=r"simple: $\tau$ = %.1f \textmu s (Q = %d)"
            % (tau_ch2_f, tau_to_Q(f1, tau_ch2_f)))
    ax.plot(t_fine_fall, np.real(curve_fall_beat), "--", color="C3", linewidth=1.2,
            label=r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
            % (tau_beat_f, tau_to_Q(f1, tau_beat_f), df_beat_f / 1e3))
    ax.legend(fontsize=5, frameon=False)
else:
    plot_no_data(ax)
ax.set_ylabel(r"Normalized $P / P_{ss}$")
ax.set_title("Ch2 1f ring-down")

# Col 1: 1f phase
plot_phase_column(
    axes[:, 1], t_us, burst_on_us, burst_off_us,
    env_ch2_norm_complex,
    ch2_rise["t"], ch2_rise["ec"],
    ch2_fall["t"] if has_fall else None,
    ch2_fall["ec"] if has_fall else None,
    "Ch2 1f",
    beat_model_rise=curve_rise_beat,
    beat_model_fall=curve_fall_beat if has_fall else None)

# Col 2: 2f amplitude
ax = axes[0, 2]
ax.plot(t_us, env_ch2_2f_best_kPa, linewidth=0.4, color="C0", alpha=0.5, label="best point")
ax.plot(t_us, np.abs(env_ch2_2f_norm_complex) * p_ss_2f, linewidth=0.6, color="k", alpha=0.8,
        label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Pressure [kPa]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch2 2f envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 2]
ax.plot(ch2_rise["t"], np.abs(env_ch2_2f_norm_complex)[rise_start:rise_end],
        "-", linewidth=0.5, color="C0", alpha=0.7)
ax.plot(t_fine_2f, curve_2f_model, "--", color="C3", linewidth=1.2,
        label=r"$\tau_2$ = %.1f \textmu s ($Q_{2f}$ = %d)"
        % (tau_2f, tau_to_Q(f1, tau_2f, harmonic=2)))
ax.plot(t_fine_2f, curve_2f_source, ":", color="0.6", linewidth=0.8, alpha=0.7,
        label=r"source $(1-e^{-t/\tau_1})^2$")
ax.set_ylabel(r"Normalized $P_{2f} / P_{ss,2f}$")
ax.set_xlabel(r"Time from burst ON + %.0f \textmu s" % FIT_SKIP_US)
ax.set_title("Ch2 2f ring-up")
ax.legend(fontsize=5, frameon=False)

ax = axes[2, 2]
if has_fall:
    ax.plot(ch2_fall["t"], np.abs(env_ch2_2f_norm_complex)[fall_start:end_f],
            "-", linewidth=0.5, color="C0", alpha=0.7)
else:
    plot_no_data(ax)
ax.set_ylabel(r"Normalized $P_{2f} / P_{ss,2f}$")
ax.set_title("Ch2 2f ring-down")

# Col 3: 2f phase
plot_phase_column(
    axes[:, 3], t_us, burst_on_us, burst_off_us,
    env_ch2_2f_norm_complex,
    ch2_rise["t"], env_ch2_2f_norm_complex[rise_start:rise_end],
    ch2_fall["t"] if has_fall else None,
    env_ch2_2f_norm_complex[fall_start:end_f] if has_fall else None,
    "Ch2 2f")

plt.tight_layout()
save_fig(fig, f"transient_ch2_{td.stem}", OUT_DIR)

# %%
# =============================================================================
# Exploratory: 3f envelope at best point
# =============================================================================

if env_ch2_3f_norm_complex is not None and p_ss_3f > 0:
    env_3f_mean = np.abs(env_ch2_3f_norm_complex)

    fig3f, axes3f = plt.subplots(2, 1, figsize=figsize_for_layout(1, 2))

    # Full envelope (averaged)
    ax = axes3f[0]
    ax.plot(t_us, env_3f_mean, linewidth=0.5, color="C2")
    if env_3f_std is not None:
        ax.fill_between(t_us, env_3f_mean - env_3f_std, env_3f_mean + env_3f_std,
                         alpha=0.15, color="C2")
    add_burst_markers(ax, burst_on_us, burst_off_us)
    ax.set_ylabel(r"$\langle P_{3f} / P_{ss,3f} \rangle$")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_title(f"3f envelope (averaged {n_used} points)")

    # Normalized ring-up
    rise_3f_e = env_3f_mean[rise_start:rise_end]
    rise_t = np.arange(rise_end - rise_start) * dt * 1e6

    ax = axes3f[1]
    ax.plot(rise_t, rise_3f_e, "-", linewidth=0.5, color="C2", label="3f averaged")
    if env_3f_std is not None:
        rise_3f_std = env_3f_std[rise_start:rise_end]
        ax.fill_between(rise_t, rise_3f_e - rise_3f_std, rise_3f_e + rise_3f_std,
                         alpha=0.15, color="C2")

    try:
        (tau_3f,), _ = curve_fit(rise_simple, rise_t, rise_3f_e,
                                  p0=[5], bounds=([0.1], [500]))
        Q_3f = tau_to_Q(f1, tau_3f, harmonic=3)
        t_fine = np.linspace(0, rise_t[-1], 500)
        ax.plot(t_fine, rise_simple(t_fine, tau_3f), "--", color="C3", linewidth=0.8,
                label=r"$\tau$ = %.1f $\mu$s ($Q_{3f}$ = %d)" % (tau_3f, Q_3f))
        print(f"\n--- Ch2 3f (averaged {n_used} points) ---")
        print(f"  tau_3f = {tau_3f:.2f} us  ->  Q_3f = {Q_3f:.0f}")
    except RuntimeError:
        print(f"\n--- Ch2 3f: fit failed ---")

    ax.set_ylabel(r"$\langle P_{3f} / P_{ss,3f} \rangle$")
    ax.set_xlabel(r"Time from burst ON [$\mu$s]")
    ax.legend(fontsize=6, frameon=False)
else:
    fig3f, axes3f = plt.subplots(1, 1, figsize=figsize_for_layout())
    axes3f.text(0.5, 0.5, "3f envelope not available",
                transform=axes3f.transAxes, ha="center")

plt.tight_layout()
save_fig(fig3f, f"transient_3f_{td.stem}", OUT_DIR)

# %%
print(f"\n=== Done ===")
