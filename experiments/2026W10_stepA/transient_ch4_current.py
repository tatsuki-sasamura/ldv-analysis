# %%
"""Ch4 current transient analysis: complex beat fit.

Fits the electrical current envelope using a complex sliding DFT with
damped-beat model to extract the PZT electrical resonance detuning.

Usage:
    python transient_ch4_current.py <path_to_tdms>
    python transient_ch4_current.py
"""

from ldv_analysis.transient import (
    DEFAULT_TDMS, DF_THRESHOLD, FIT_SKIP_US,
    beat_model_complex, compute_fit_windows, decay_beat_residual,
    detect_burst, estimate_beat_freq, load_transient_data,
    sliding_dft_envelope, smooth_envelope, tau_to_Q,
    add_burst_spans, plot_no_data, plot_phase_column,
)
from ldv_analysis.plotting import save_fig
from ldv_analysis.io_utils import load_tdms_file
from ldv_analysis.fft_cache import load_point_waveforms
from ldv_analysis.config import (
    CURRENT_SCALE, figsize_for_layout, get_output_dir,
)
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# %%
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tdms", nargs="?", default=None)
parser.add_argument("--fresh", action="store_true", help="Recompute envelopes")
_args = parser.parse_args()

OUT_DIR = get_output_dir(__file__)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

tdms_path = Path(_args.tdms) if _args.tdms else DEFAULT_TDMS
td = load_transient_data(tdms_path, CACHE_DIR)

f1 = td.f1
dt = td.dt
n_samples = td.n_samples
t_us = td.t_us
ss_start = td.ss_start
ss_end = td.ss_end
valid = td.valid
best_i = td.best_i

# %%
# =============================================================================
# Envelope computation (plain average — current is spatially uniform)
# =============================================================================

env_cache_path = CACHE_DIR / f"_transient_env_ch4_{td.stem}.npz"
if env_cache_path.exists() and not _args.fresh:
    print(f"  Loading envelope cache: {env_cache_path.name}")
    _ec = np.load(env_cache_path)
    env_ch4_norm_complex = _ec["env_ch4_complex"]
    i_ss_ch4 = float(_ec["i_ss_ch4"])
    env_ch4_std = _ec["env_ch4_std"] if "env_ch4_std" in _ec else None
    env_ch1 = _ec["best_ch1"]
    n_used = int(_ec["n_used"])
    del _ec
else:
    print(f"  Computing averaged Ch4 envelope (per-point)...")
    tdms_file, _ = load_tdms_file(tdms_path)
    wf_group = tdms_file["Waveforms"]
    ch4_channels = [ch for ch in wf_group.channels()
                    if ch.name.startswith("WFCh4")]

    env_ch4_sum = np.zeros(n_samples, dtype=complex)
    env_ch4_mag_sq_sum = np.zeros(n_samples)
    n_used = 0
    for idx in np.where(valid)[0]:
        wf4 = ch4_channels[idx][:]
        env_ch4_c = sliding_dft_envelope(
            wf4, dt, f1, return_complex=True) * CURRENT_SCALE * 1e3  # mA
        env_ch4_sum += env_ch4_c
        env_ch4_mag_sq_sum += np.abs(env_ch4_c)**2
        n_used += 1
    env_ch4_avg = env_ch4_sum / n_used
    _ch4_ss = np.mean(env_ch4_avg[ss_start:ss_end])
    env_ch4_norm_complex = env_ch4_avg / _ch4_ss
    i_ss_ch4 = float(np.abs(_ch4_ss))
    # Std of magnitude in mA
    env_ch4_mag_mean = np.abs(env_ch4_avg)
    env_ch4_std = np.sqrt(np.maximum(
        env_ch4_mag_sq_sum / n_used - env_ch4_mag_mean**2, 0))
    print(f"  Averaged {n_used} Ch4 envelopes (plain)")

    # Ch1 for burst detection
    wfs_best, _ = load_point_waveforms(tdms_path, best_i, channels=(1,))
    env_ch1 = smooth_envelope(wfs_best[1])
    del tdms_file

    np.savez(env_cache_path,
             env_ch4_complex=env_ch4_norm_complex,
             env_ch4_std=env_ch4_std,
             i_ss_ch4=i_ss_ch4, best_ch1=env_ch1, n_used=n_used)
    print(f"  Saved: {env_cache_path.name}")

burst_on, burst_off = detect_burst(env_ch1, dt)
fw = compute_fit_windows(burst_on, burst_off, dt, n_samples)
rise_start, rise_end = fw["rise_start"], fw["rise_end"]
fall_start, end_f, has_fall = fw["fall_start"], fw["end_f"], fw["has_fall"]
print(f"  Burst ON: {burst_on * dt * 1e6:.1f}--{burst_off * dt * 1e6:.1f} us")

# %%
# =============================================================================
# Fitting: complex beat (deviation from steady state)
# =============================================================================

I_ss_ch4 = i_ss_ch4

# Rise: fit (E - 1) as A*exp(-t/tau)*exp(j*(2πdf*t + φ))
ch4_rise_beat = {
    "t": np.arange(rise_end - rise_start) * dt * 1e6,
    "ec": env_ch4_norm_complex[rise_start:rise_end],
}
_dev_r = ch4_rise_beat["ec"] - 1.0
_A0_r = float(np.abs(_dev_r[0]))
_df_guess_r = estimate_beat_freq(
    _dev_r, ch4_rise_beat["t"][1] - ch4_rise_beat["t"][0])
ch4_rise_beat["beat"] = least_squares(
    decay_beat_residual,
    x0=[_A0_r, 10.0, _df_guess_r, 0.0],
    args=(ch4_rise_beat["t"], _dev_r.real, _dev_r.imag),
    bounds=([0, 0.1, -1e6, -np.pi], [10, 500, 1e6, np.pi]),
    method="trf",
)
A_ch4b_r, tau_ch4b_r, df_ch4b_r, phi_ch4b_r = ch4_rise_beat["beat"].x
ch4_beat_sig_r = abs(df_ch4b_r) > DF_THRESHOLD

# Fall
ch4_fall_beat = None
if has_fall:
    ch4_fall_beat = {
        "t": np.arange(end_f - fall_start) * dt * 1e6,
        "ec": env_ch4_norm_complex[fall_start:end_f],
    }
    _A0_f = float(np.abs(ch4_fall_beat["ec"][0]))
    _df_guess_f = estimate_beat_freq(
        ch4_fall_beat["ec"],
        ch4_fall_beat["t"][1] - ch4_fall_beat["t"][0])
    ch4_fall_beat["beat"] = least_squares(
        decay_beat_residual,
        x0=[_A0_f, 10.0, _df_guess_f, 0.0],
        args=(ch4_fall_beat["t"],
              ch4_fall_beat["ec"].real, ch4_fall_beat["ec"].imag),
        bounds=([0, 0.1, -1e6, -np.pi], [10, 500, 1e6, np.pi]),
        method="trf",
    )

A_ch4b_f = tau_ch4b_f = df_ch4b_f = phi_ch4b_f = None
ch4_beat_sig_f = False
if ch4_fall_beat is not None:
    A_ch4b_f, tau_ch4b_f, df_ch4b_f, phi_ch4b_f = ch4_fall_beat["beat"].x
    ch4_beat_sig_f = abs(df_ch4b_f) > DF_THRESHOLD

# %%
# =============================================================================
# Results
# =============================================================================

print(f"\n--- Ch4 (current, complex beat) ---")
print(f"  I_ss = {I_ss_ch4:.2f} mA")
print(f"  Rise (beat):  tau = {tau_ch4b_r:.2f} us"
      f"  ->  Q = {tau_to_Q(f1, tau_ch4b_r):.0f}"
      f"  df = {df_ch4b_r:.0f} Hz, A = {A_ch4b_r:.4f}"
      + ("" if ch4_beat_sig_r else "  (not significant)"))
if ch4_fall_beat is not None:
    print(f"  Fall (beat):  tau = {tau_ch4b_f:.2f} us"
          f"  ->  Q = {tau_to_Q(f1, tau_ch4b_f):.0f}"
          f"  df = {df_ch4b_f:.0f} Hz, A = {A_ch4b_f:.4f}"
          + ("" if ch4_beat_sig_f else "  (not significant)"))

# %%
# =============================================================================
# Visualization: 3x2 (amplitude, phase)
# =============================================================================

# Precompute model curves
curve_ch4_rise_beat_c = (
    1.0 + A_ch4b_r * beat_model_complex(
        np.linspace(0, ch4_rise_beat["t"][-1], 500),
        tau_ch4b_r, df_ch4b_r, phi_ch4b_r))
t_fine_rise = np.linspace(0, ch4_rise_beat["t"][-1], 500)
curve_ch4_rise_beat = np.abs(curve_ch4_rise_beat_c) * I_ss_ch4

curve_ch4_fall_beat_c = None
t_fine_fall = None
curve_ch4_fall_beat = None
if ch4_fall_beat is not None:
    t_fine_fall = np.linspace(0, ch4_fall_beat["t"][-1], 500)
    curve_ch4_fall_beat_c = (
        A_ch4b_f * beat_model_complex(
            t_fine_fall, tau_ch4b_f, df_ch4b_f, phi_ch4b_f))
    curve_ch4_fall_beat = np.abs(curve_ch4_fall_beat_c) * I_ss_ch4

ss_start_us = ss_start * dt * 1e6
ss_end_us = ss_end * dt * 1e6
burst_on_us = burst_on * dt * 1e6
burst_off_us = burst_off * dt * 1e6

fig, axes = plt.subplots(3, 2, figsize=figsize_for_layout(3, 2))

# Col 0: amplitude
ax = axes[0, 0]
ax.plot(t_us, np.abs(env_ch4_norm_complex) * i_ss_ch4, linewidth=0.6, color="C1",
        label=f"avg ({n_used} pts)")
add_burst_spans(ax, burst_on_us, burst_off_us, ss_start_us, ss_end_us)
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time [\textmu s]")
ax.set_title("Ch4 (current) envelope")
ax.legend(fontsize=5, frameon=False)

ax = axes[1, 0]
ax.plot(ch4_rise_beat["t"] + FIT_SKIP_US,
        np.abs(ch4_rise_beat["ec"]) * I_ss_ch4,
        "-", linewidth=0.5, color="C1", alpha=0.7)
ax.plot(t_fine_rise + FIT_SKIP_US, curve_ch4_rise_beat,
        "--", color="C3", linewidth=1.2,
        label=(r"beat: $\tau$ = %.1f \textmu s (Q = %d), $\Delta f$ = %.1f kHz"
               % (tau_ch4b_r, tau_to_Q(f1, tau_ch4b_r), df_ch4b_r / 1e3)))
ax.legend(fontsize=5, frameon=False)
ax.set_ylabel("Current [mA]")
ax.set_xlabel(r"Time from burst ON [\textmu s]")
ax.set_title("Ch4 ring-up (beat)")

ax = axes[2, 0]
if has_fall and ch4_fall_beat is not None:
    ax.plot(ch4_fall_beat["t"] + FIT_SKIP_US,
            np.abs(ch4_fall_beat["ec"]) * I_ss_ch4,
            "-", linewidth=0.5, color="C1", alpha=0.7)
    ax.plot(t_fine_fall + FIT_SKIP_US, curve_ch4_fall_beat,
            "--", color="C3", linewidth=1.2,
            label=(r"beat: $\tau$ = %.1f \textmu s, $\Delta f$ = %.1f kHz"
                   % (tau_ch4b_f, df_ch4b_f / 1e3)))
    ax.legend(fontsize=5, frameon=False)
elif not has_fall:
    plot_no_data(ax)
ax.set_ylabel("Current [mA]")
ax.set_title("Ch4 ring-down (beat)")

# Col 1: phase
plot_phase_column(
    axes[:, 1], t_us, burst_on_us, burst_off_us,
    env_ch4_norm_complex,
    ch4_rise_beat["t"], ch4_rise_beat["ec"],
    ch4_fall_beat["t"] if ch4_fall_beat is not None else None,
    ch4_fall_beat["ec"] if ch4_fall_beat is not None else None,
    "Ch4",
    beat_model_rise=curve_ch4_rise_beat_c,
    beat_model_fall=curve_ch4_fall_beat_c)

plt.tight_layout()
save_fig(fig, f"transient_ch4_{td.stem}", OUT_DIR)

# %%
print(f"\n=== Done ===")
