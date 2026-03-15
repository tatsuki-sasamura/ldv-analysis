# %%
"""Generate manuscript figures (Figs 5–9, A1) for the nonlinear harmonics paper.

Produces publication-ready .eps and .png figures from existing experimental data.
Each figure's plot data is cached as figN.npz for cross-referencing and fast
regeneration.

Usage:
    python manuscript_figures.py           # use caches if available
    python manuscript_figures.py --fresh   # recompute all from TDMS
"""

import argparse

from ldv_analysis.mode_fit import fit_columns, fit_mode_1f, fit_mode_2f
from ldv_analysis.grid_utils import make_channel_grid
from ldv_analysis.fft_cache import (
    detect_velocity_scale, load_or_compute, load_point_waveforms,
)
from ldv_analysis.config import (
    CHANNEL_WIDTH,
    C_SOUND,
    RHO,
    RSSI_THRESHOLD,
    CURRENT_SCALE,
    SENSITIVITY,
    VELOCITY_SCALE,
    channel_centre_func,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.filters import make_transient_valid_mask, make_valid_mask, make_voltage_mask
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fresh", action="store_true",
                    help="Recompute all figure data from TDMS (ignore caches)")
_args, _ = parser.parse_known_args()
FRESH = _args.fresh

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 0.75,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


def tex_mu(text: str) -> str:
    """Convert μ to LaTeX \\textmu when LaTeX is active."""
    if plt.rcParams.get("text.usetex", False):
        return text.replace("μ", r"\textmu ")
    return text


# %%
# =============================================================================
# Configuration
# =============================================================================
DATA_DIR_B = get_data_dir("20260307experimentB")
DATA_DIR_A = get_data_dir("20260306experimentA")

# test10 voltage sweep files (Step B)
VOLTAGE_FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",  5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

CACHE_DIR = get_output_dir(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("D:/OneDrive - Lund University/Publications"
               "/nonlinearphysics/manuscript/figures")


def save_fig(fig, name):
    """Save figure as both .eps and .png."""
    fig.savefig(FIG_DIR / f"{name}.eps", dpi=1200)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=1200)
    print(f"Saved: {FIG_DIR / name}.eps/.png")


def _fig_cache(name):
    """Return path to figure data cache."""
    return FIG_DIR / f"{name}.npz"


def _has_cache(name):
    """Check if figure cache exists and --fresh was not requested."""
    return not FRESH and _fig_cache(name).exists()


# %%
# =============================================================================
# Load test10 voltage sweep data (shared by Figs 5–8)
# =============================================================================

hw = CHANNEL_WIDTH / 2  # m

# Channel geometry from calibration
_geom_B = load_channel_geometry("20260307experimentB", CACHE_DIR)
_centre_fn_B = channel_centre_func(_geom_B)

# Check if we need the full processing loop
_need_processing = FRESH or not all(
    _fig_cache(n).exists() for n in ["Fig5", "Fig6", "Fig7", "Fig8"]
)

results = []

if _need_processing:
    for fname, vpp in VOLTAGE_FILES:
        tdms_path = DATA_DIR_B / fname
        if not tdms_path.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        cache = load_or_compute(tdms_path, CACHE_DIR)
        pos_x = cache["pos_x"]
        pos_y = cache["pos_y"]
        n_x_meta = int(cache["n_x_meta"])
        n_y_meta = int(cache["n_y_meta"])
        f_drive = float(cache["f_drive"])
        pressure_1f = cache["pressure_1f"]

        # Channel mask
        pos_x_c = pos_x - _centre_fn_B(pos_y)
        inside = np.abs(pos_x_c) <= hw

        # Build grid
        rssi = cache["rssi"] if "rssi" in cache else None
        cg = make_channel_grid(
            pos_x_c, pos_y, n_x_meta, n_y_meta,
            CHANNEL_WIDTH, pos_x.max() - pos_x.min(), inside,
            rssi=rssi, rssi_threshold=RSSI_THRESHOLD,
        )

        # 1f fit — axial profile of p0(x)
        grid_prs_1f = cg.to_grid(pressure_1f)
        p0_1f_y = fit_columns(grid_prs_1f,
                              cg.width_grid, CHANNEL_WIDTH, harmonic=1)
        # Whole-channel average E_ac (Baasch et al. 2024): ⟨E_ac⟩ = ⟨p0²⟩/(4ρc²)
        Eac_1f = np.nanmean(p0_1f_y**2) / (4 * RHO * C_SOUND**2)

        # 2f fit
        pressure_2f = cache["pressure_2f"]
        grid_prs_2f = cg.to_grid(pressure_2f)
        p0_2f_y = fit_columns(grid_prs_2f,
                              cg.width_grid, CHANNEL_WIDTH, harmonic=2)
        Eac_2f = np.nanmean(p0_2f_y**2) / (4 * RHO * C_SOUND**2)

        # Electrical power: P_in = ½ V I cos(φ)
        V = cache["voltage_1f"]
        I = cache["current_1f"]
        phase_vi = cache["phase_vi"]
        valid_e = make_voltage_mask(V)
        P_in = 0.5 * float(np.median(V[valid_e])) * float(np.median(I[valid_e])) \
            * np.cos(np.radians(float(np.median(phase_vi[valid_e]))))

        # Peak p0 (for Fig 8 harmonic panels)
        p0_1f_peak = np.nanmax(p0_1f_y)  # Pa
        p0_2f_peak = np.nanmax(p0_2f_y)  # Pa

        # Ch1 drive voltage harmonics: 2f/1f ratio (for Fig 8c)
        # Use median point for Ch1 spectrum (signal is spatially uniform)
        n_points = len(pos_x)
        mid_pt = n_points // 2
        wf_ch1, dt = load_point_waveforms(tdms_path, mid_pt, channels=(1,))
        ch1 = wf_ch1[1]
        ss_start = int(cache["ss_start"])
        ss_end = int(cache["ss_end"])
        ss_n = ss_end - ss_start
        ch1_ss = ch1[ss_start:ss_end]
        tone_1f = np.exp(-2j * np.pi * f_drive * np.arange(ss_n) * dt)
        tone_2f = np.exp(-2j * np.pi * 2 * f_drive * np.arange(ss_n) * dt)
        ch1_1f_amp = np.abs(ch1_ss @ tone_1f) * 2 / ss_n
        ch1_2f_amp = np.abs(ch1_ss @ tone_2f) * 2 / ss_n
        ch1_ratio = ch1_2f_amp / ch1_1f_amp * 100  # %

        results.append(dict(
            vpp=vpp, Eac_1f=Eac_1f, Eac_2f=Eac_2f, P_in=P_in,
            p0_1f_peak=p0_1f_peak, p0_2f_peak=p0_2f_peak,
            p0_1f_y=p0_1f_y, p0_2f_y=p0_2f_y,
            ch1_ratio=ch1_ratio,
            f_drive=f_drive, tdms_path=tdms_path,
            cache=cache, cg=cg,
        ))
        print(f"  {fname}: p0_1f = {p0_1f_peak/1e3:.0f} kPa, "
              f"p0_2f = {p0_2f_peak/1e3:.0f} kPa, "
              f"Ch1 2f/1f = {ch1_ratio:.2f}%, "
              f"P_in = {P_in*1e3:.1f} mW")

# %%
# =============================================================================
# Fig 5: E_ac,avg vs P_in  (Baasch et al. 2024 convention)
# =============================================================================

if _has_cache("Fig5"):
    d = np.load(_fig_cache("Fig5"))
    P_in_mW, Eac_1f_arr, a_eac = d["P_in_mW"], d["E_ac"], float(d["a_eac"])
    Vpp = d["Vpp"]
    print("Fig 5: loaded from cache")
else:
    Vpp = np.array([r["vpp"] for r in results])
    Eac_1f_arr = np.array([r["Eac_1f"] for r in results])
    P_in_arr = np.array([r["P_in"] for r in results])
    P_in_mW = P_in_arr * 1e3
    a_eac = np.sum(P_in_arr * Eac_1f_arr) / np.sum(P_in_arr**2)
    np.savez(_fig_cache("Fig5"), Vpp=Vpp, P_in_mW=P_in_mW,
             E_ac=Eac_1f_arr, a_eac=a_eac)
    print(f"Saved: {_fig_cache('Fig5')}")

P_fine = np.linspace(0, P_in_mW.max() / 1e3 * 1.1, 100)

fig, ax = plt.subplots(figsize=(3.375, 2.5))
ax.plot(P_in_mW, Eac_1f_arr, "ko", markersize=4, zorder=3)
ax.plot(P_fine * 1e3, a_eac * P_fine, "k:", linewidth=0.5,
        label=f"Linear fit")
ax.set_xlabel(r"$P_\mathrm{in}$ [mW]")
ax.set_ylabel(r"$\langle E_\mathrm{ac,1f} \rangle$ [J/m$^3$]")
ax.legend(frameon=False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
save_fig(fig, "Fig5")
plt.close()

# Print E_ac values
print("\n--- Fig 5: <E_ac> vs P_in ---")
for i in range(len(Vpp)):
    print(f"  {Vpp[i]:2.0f} Vpp: <E_ac> = {Eac_1f_arr[i]:.1f} J/m³, "
          f"P_in = {P_in_mW[i]:.1f} mW")
print(f"  Linear slope: {a_eac/1e3:.1f} kJ/(m³ W)")

# %%
print("\n=== Fig 5 Done ===")

# %%
# =============================================================================
# Fig 8: Drive-resolved harmonics
# =============================================================================

if _has_cache("Fig8"):
    d = np.load(_fig_cache("Fig8"))
    Vpp = d["Vpp"]
    p0_1f_peak_arr = d["p0_1f"]
    p0_2f_peak_arr = d["p0_2f"]
    ch1_ratio_arr = d["ch1_ratio_pct"]
    a_1f, b_2f = float(d["a_1f"]), float(d["b_2f"])
    ratio_slope = float(d["ratio_slope"])
    print("Fig 8: loaded from cache")
else:
    Vpp = np.array([r["vpp"] for r in results])
    Eac_1f_arr = np.array([r["Eac_1f"] for r in results])
    P_in_arr = np.array([r["P_in"] for r in results])
    p0_1f_peak_arr = np.array([r["p0_1f_peak"] for r in results])
    p0_2f_peak_arr = np.array([r["p0_2f_peak"] for r in results])
    ch1_ratio_arr = np.array([r["ch1_ratio"] for r in results])

    # Fits through origin: P_1f = a*V, P_2f = b*V²
    Vpp_0 = np.array([0] + list(Vpp))
    p0_1f_0 = np.array([0] + list(p0_1f_peak_arr))
    p0_2f_0 = np.array([0] + list(p0_2f_peak_arr))
    a_1f = np.sum(Vpp_0 * p0_1f_0) / np.sum(Vpp_0**2)
    b_2f = np.sum(Vpp_0**2 * p0_2f_0) / np.sum(Vpp_0**4)
    ratio_slope = b_2f / a_1f

    np.savez(_fig_cache("Fig8"), Vpp=Vpp,
             p0_1f=p0_1f_peak_arr, p0_2f=p0_2f_peak_arr,
             E_ac=Eac_1f_arr, P_in=P_in_arr,
             ratio=p0_2f_peak_arr / p0_1f_peak_arr,
             ch1_ratio_pct=ch1_ratio_arr,
             ch1_2f_1f=ch1_ratio_arr / 100,
             a_1f=a_1f, b_2f=b_2f, ratio_slope=ratio_slope)
    print(f"Saved: {_fig_cache('Fig8')}")

V_fine = np.linspace(0, Vpp.max() * 1.15, 100)
ratio = p0_2f_peak_arr / p0_1f_peak_arr

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.375, 4.2))

# (a) P_1f and P_2f vs Vpp
ax1.plot(Vpp, p0_1f_peak_arr / 1e6, "o", markersize=4, color="tab:blue",
         label=r"$P_{1f}$")
ax1.plot(V_fine, a_1f * V_fine / 1e6, ":", linewidth=0.5, color="tab:blue")
ax1.plot(Vpp, p0_2f_peak_arr / 1e6, "s", markersize=3, color="tab:red",
         label=r"$P_{2f}$")
ax1.plot(V_fine, b_2f * V_fine**2 / 1e6, ":", linewidth=0.5, color="tab:red")
ax1.set_ylabel(r"Pressure amplitude [MPa]")
ax1.legend(frameon=False)
_lbl_kw = dict(va="bottom", ha="left", fontweight="bold")
ax1.text(-0.22, 1.05, "(a)", transform=ax1.transAxes, **_lbl_kw)

# (b) P_2f / P_1f ratio vs Vpp
ax2.plot(Vpp, ratio, "D", markersize=4, color="tab:blue")
ax2.plot(V_fine[1:], ratio_slope * V_fine[1:], ":", linewidth=0.5,
         color="tab:blue")
ax2.set_ylabel(r"$P_{2f}/P_{1f}$")
ax2.set_xlabel(r"Drive voltage $V_\mathrm{pp}$ [V]")
ax2.text(-0.22, 1.05, "(b)", transform=ax2.transAxes, **_lbl_kw)

plt.tight_layout()
save_fig(fig, "Fig8")
plt.close()

# Print summary
print("\n--- Fig 8: Harmonics vs Vpp ---")
print(f"  Fit: P_1f = {a_1f/1e3:.1f} kPa/V")
print(f"  Fit: P_2f = {b_2f/1e3:.3f} kPa/V²")
print(f"  Ratio slope: {ratio_slope:.4f} /V")
for i in range(len(Vpp)):
    print(f"  {Vpp[i]:2.0f} Vpp: P_1f = {p0_1f_peak_arr[i]/1e3:.0f} kPa, "
          f"P_2f = {p0_2f_peak_arr[i]/1e3:.0f} kPa, "
          f"ratio = {ratio[i]:.3f}, "
          f"Ch1 2f/1f = {ch1_ratio_arr[i]:.2f}%")

# %%
print("\n=== Fig 8 Done ===")

# %%
# =============================================================================
# Fig A1: Transient ring-up envelopes (P_1f, P_2f, driving current)
# =============================================================================
#
# Three panels at 25 Vpp showing ring-up envelopes:
#   (a) P_1f with exponential fit → Q_1f
#   (b) P_2f with driven-resonator fit → Q_2f
#   (c) Driving current (raw) — reaches steady state much faster than P,
#       justifying step-input assumption for acoustic transient model

from ldv_analysis.transient import (
    sliding_dft_envelope, smooth_envelope, detect_burst, compute_fit_windows,
    rise_simple, rise_2f, tau_to_Q, RISE_FIT_WINDOW_US, FIT_SKIP_US,
)
from ldv_analysis.io_utils import load_tdms_file
from scipy.optimize import curve_fit

FIGA1_TDMS = DATA_DIR_B / "test10_1907_25Vpp_5m_s_max.tdms"

if _has_cache("FigA1"):
    d7 = np.load(_fig_cache("FigA1"))
    print("Fig A1: loaded from cache")
    tau_ch2_7 = float(d7["tau_ch2"])
    Q_1f = float(d7["Q_1f"])
    tau_2f_7 = float(d7["tau_2f"])
    Q_2f = float(d7["Q_2f"])
else:
    # --- Compute envelopes directly from TDMS ---
    _f7_cache = load_or_compute(FIGA1_TDMS, CACHE_DIR)
    _f7_vel = _f7_cache["velocity_1f"]
    _f7_prs = _f7_cache["pressure_1f"]
    _f7_rssi = _f7_cache["rssi"] if "rssi" in _f7_cache else None
    _f7_valid = make_transient_valid_mask(_f7_rssi, _f7_prs)
    _f7_best = int(np.where(_f7_valid)[0][np.argmax(_f7_vel[_f7_valid])])

    _f1 = float(_f7_cache["f_drive"])
    _n_samples = int(_f7_cache["n_samples"])
    _dt = float(_f7_cache["dt"])
    _ss_start = int(_f7_cache["ss_start"])
    _ss_end = int(_f7_cache["ss_end"])
    _to_kPa = VELOCITY_SCALE / (2 * np.pi * _f1 * SENSITIVITY) / 1e3
    _to_kPa_2f = VELOCITY_SCALE / (2 * np.pi * 2 * _f1 * SENSITIVITY) / 1e3
    print(f"  Computing averaged envelopes for Fig A1 ({_f7_valid.sum()} points)...")
    _tdms_f7, _ = load_tdms_file(FIGA1_TDMS)
    _wfg = _tdms_f7["Waveforms"]
    _ch2s = [ch for ch in _wfg.channels() if ch.name.startswith("WFCh2")]
    _ch4s = [ch for ch in _wfg.channels() if ch.name.startswith("WFCh4")]

    _env1f_wsum = np.zeros(_n_samples, dtype=complex)
    _env1f_sq_wsum = np.zeros(_n_samples)
    _env2f_wsum = np.zeros(_n_samples, dtype=complex)
    _env2f_sq_wsum = np.zeros(_n_samples)
    _env4_sum = np.zeros(_n_samples, dtype=complex)
    _env4_sq_sum = np.zeros(_n_samples)
    _wsum = 0.0
    _n7 = 0
    for _idx in np.where(_f7_valid)[0]:
        _wf2 = _ch2s[_idx][:]
        _ec = sliding_dft_envelope(_wf2, _dt, _f1, return_complex=True) * _to_kPa
        _ec2f = sliding_dft_envelope(_wf2, _dt, 2 * _f1, return_complex=True) * _to_kPa_2f
        _pss_c = np.mean(_ec[_ss_start:_ss_end])
        _pss_m = np.abs(_pss_c)
        if _pss_m > 0:
            _w = _pss_m
            _norm = np.abs(_ec) / _pss_m
            _env1f_wsum += _w * (_ec / _pss_c)
            _env1f_sq_wsum += _w * _norm**2
            _p2f_c = np.mean(_ec2f[_ss_start:_ss_end])
            _p2f_m = np.abs(_p2f_c)
            if _p2f_m > 0:
                _norm2f = np.abs(_ec2f) / _p2f_m
                _env2f_wsum += _w * (_ec2f / _p2f_c)
                _env2f_sq_wsum += _w * _norm2f**2
            _wf4 = _ch4s[_idx][:]
            _ec4 = sliding_dft_envelope(
                _wf4, _dt, _f1, return_complex=True) * CURRENT_SCALE * 1e3
            _env4_sum += _ec4
            _env4_sq_sum += np.abs(_ec4)**2
            _wsum += _w
            _n7 += 1

    _env1f_c = _env1f_wsum / _wsum
    _env2f_c = _env2f_wsum / _wsum
    _env1f_mean = np.abs(_env1f_c)
    _env1f_std = np.sqrt(np.maximum(_env1f_sq_wsum / _wsum - _env1f_mean**2, 0))
    _env2f_mean = np.abs(_env2f_c)
    _env2f_std = np.sqrt(np.maximum(_env2f_sq_wsum / _wsum - _env2f_mean**2, 0))
    _env4_avg = _env4_sum / _n7
    _ch4_ss = np.mean(_env4_avg[_ss_start:_ss_end])
    _i_ss = float(np.abs(_ch4_ss))
    _env4_mag = np.abs(_env4_avg)
    _env4_std = np.sqrt(np.maximum(_env4_sq_sum / _n7 - _env4_mag**2, 0))

    # Ch1 for burst detection
    _wfs_best, _ = load_point_waveforms(FIGA1_TDMS, _f7_best, channels=(1,))
    _env_ch1 = smooth_envelope(_wfs_best[1])
    del _tdms_f7
    print(f"  Averaged {_n7} points")

    _burst_on, _burst_off = detect_burst(_env_ch1, _dt)
    _fw = compute_fit_windows(_burst_on, _burst_off, _dt, _n_samples)
    _rs, _re = _fw["rise_start"], _fw["rise_end"]

    # -- (a) P_1f fit --
    _rise_t = np.arange(_re - _rs) * _dt * 1e6
    _rise_e = _env1f_mean[_rs:_re]
    _rise_std = _env1f_std[_rs:_re]
    (tau_ch2_7,), _ = curve_fit(rise_simple, _rise_t, _rise_e,
                                p0=[10], bounds=([0.1], [500]))
    Q_1f = tau_to_Q(_f1, tau_ch2_7)
    _t_fine = np.linspace(0, _rise_t[-1], 500)
    _rise_fit = rise_simple(_t_fine, tau_ch2_7)

    # -- (b) P_2f fit --
    _rise_2f_e = _env2f_mean[_rs:_re]
    _rise_2f_std = _env2f_std[_rs:_re]
    _pre = _env2f_mean[max(0, _burst_on - 100):_burst_on]
    _nf = float(np.mean(_pre)) if len(_pre) > 0 else 0.0
    (tau_2f_7, _A2f, _bl), _ = curve_fit(
        lambda t, tau2, A, b: b + A * rise_2f(t, tau2, tau_ch2_7),
        _rise_t, _rise_2f_e,
        p0=[tau_ch2_7 * 0.3, 1.0 - _nf, _nf],
        bounds=([0.1, 0, 0], [500, 5, 1]))
    Q_2f = tau_to_Q(_f1, tau_2f_7, harmonic=2)
    _rise_2f_fit = _bl + _A2f * rise_2f(_t_fine, tau_2f_7, tau_ch2_7)

    # -- (c) Current --
    _rise_n = int(RISE_FIT_WINDOW_US * 1e-6 / _dt)
    _ch4_t = np.arange(_rise_n) * _dt * 1e6
    _ch4_e = _env4_mag[_burst_on:_burst_on + _rise_n] * _i_ss / np.abs(np.mean(
        _env4_avg[_ss_start:_ss_end]))  # scale to mA
    _ch4_e = np.abs(_env4_avg)[_burst_on:_burst_on + _rise_n] / np.abs(
        _ch4_ss) * _i_ss
    _ch4_std_r = _env4_std[_burst_on:_burst_on + _rise_n]

    d7 = dict(
        rise_t=_rise_t, rise_e=_rise_e, rise_1f_std=_rise_std,
        rise_t_fine=_t_fine, rise_fit=_rise_fit,
        tau_ch2=tau_ch2_7, Q_1f=Q_1f,
        rise_2f_e=_rise_2f_e, rise_2f_std=_rise_2f_std,
        rise_2f_fit=_rise_2f_fit,
        tau_2f=tau_2f_7, Q_2f=Q_2f,
        ch4_t=_ch4_t, ch4_e=_ch4_e, ch4_std=_ch4_std_r,
    )
    np.savez(_fig_cache("FigA1"), **d7)
    print(f"Saved: {_fig_cache('FigA1')}")

fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(3.375, 5.5))
_lbl_kw7 = dict(va="bottom", ha="left", fontweight="bold")

# (a) P_1f
_t_a, _e_a = d7["rise_t"], d7["rise_e"]
ax_a.plot(_t_a, _e_a, "-", lw=0.75, color="C0", label="Averaged envelope")
if "rise_1f_std" in d7:
    ax_a.fill_between(_t_a, _e_a - d7["rise_1f_std"], _e_a + d7["rise_1f_std"],
                       alpha=0.15, color="C0")
ax_a.plot(d7["rise_t_fine"], d7["rise_fit"], "--", color="C3", lw=0.75,
          label=r"$\tau$ = %.1f $\mu$s ($Q_{1f}$ = %d)" % (tau_ch2_7, Q_1f))
ax_a.set_ylabel(r"$\langle P_{1f} / P_{ss,1f} \rangle$")
ax_a.legend(fontsize=6, frameon=False)
ax_a.text(-0.15, 1.00, "(a)", transform=ax_a.transAxes, **_lbl_kw7)

# (b) P_2f
_t_b, _e_b = d7["rise_t"], d7["rise_2f_e"]
ax_b.plot(_t_b, _e_b, "-", lw=0.75, color="C0", label="Averaged envelope")
if "rise_2f_std" in d7:
    ax_b.fill_between(_t_b, _e_b - d7["rise_2f_std"], _e_b + d7["rise_2f_std"],
                       alpha=0.15, color="C0")
ax_b.plot(d7["rise_t_fine"], d7["rise_2f_fit"], "--", color="C3", lw=0.75,
          label=r"$\tau_2$ = %.1f $\mu$s ($Q_{2f}$ = %d)" % (tau_2f_7, Q_2f))
ax_b.set_ylabel(r"$\langle P_{2f} / P_{ss,2f} \rangle$")
ax_b.legend(fontsize=6, frameon=False)
ax_b.text(-0.15, 1.00, "(b)", transform=ax_b.transAxes, **_lbl_kw7)

# (c) Current
_t_c, _e_c = d7["ch4_t"], d7["ch4_e"]
ax_c.plot(_t_c, _e_c, "-", lw=0.75, color="C1", label="Averaged envelope")
if "ch4_std" in d7:
    ax_c.fill_between(_t_c, _e_c - d7["ch4_std"], _e_c + d7["ch4_std"],
                       alpha=0.15, color="C1")
ax_c.set_ylabel(r"$\langle I \rangle$ [mA]")
ax_c.set_xlabel(tex_mu("Time from burst ON [μs]"))
ax_c.legend(fontsize=6, frameon=False)
ax_c.text(-0.15, 1.00, "(c)", transform=ax_c.transAxes, **_lbl_kw7)

print(f"  Q_1f = {Q_1f:.0f}, Q_2f = {Q_2f:.0f}")

plt.tight_layout()
save_fig(fig, "FigA1")
plt.close()

# %%
print("\n=== Fig A1 Done ===")

# %%
# =============================================================================
# Fig 6: Time-domain waveform distortion (10 Vpp vs 25 Vpp)
# =============================================================================
#
# 2 rows (drive levels) × 3 columns (positions across channel width).
# Each panel: raw pressure waveform with reconstructed 1f+2f overlay.

# --- Configuration ---
FIG6_FILES = [
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]
FIG6_TARGET_XC = [0, hw / 2, hw]  # 0, +W/4, +W/2
FIG6_POS_LABELS = [
    r"$y = 0$", r"$y = +W/4$", r"$y = +W/2$",
]
FIG6_WINDOW_US = 1.0  # display window length (µs)


def _find_best_shared_row(file_list):
    """Find the axial (y) row that passes RSSI and voltage checks at all
    target positions for every file, ranked by highest mean pressure."""
    targets = FIG6_TARGET_XC

    # Collect caches
    caches = []
    for fname, _ in file_list:
        caches.append(load_or_compute(DATA_DIR_B / fname, CACHE_DIR))

    # Common y-values (rounded to 10 µm)
    y_sets = [set(np.round(c["pos_y"], 5)) for c in caches]
    common_y = sorted(y_sets[0].intersection(*y_sets[1:]))

    best = None
    for yval in common_y:
        p_sum = 0.0
        ok = True
        for c in caches:
            pos_x_c = c["pos_x"] - _centre_fn_B(c["pos_y"])
            rssi = c["rssi"] if "rssi" in c else None
            V = c["voltage_1f"]
            valid = make_valid_mask(V, rssi)
            y_mask = np.abs(c["pos_y"] - yval) < 1e-5
            row_idx = np.where(y_mask)[0]
            if len(row_idx) < len(targets):
                ok = False
                break
            for target in targets:
                dists = np.abs(pos_x_c[row_idx] - target)
                bi = np.argmin(dists)
                idx = row_idx[bi]
                if dists[bi] > 10e-6:
                    ok = False
                    break
                if not valid[idx]:
                    ok = False
                    break
            if not ok:
                break
            # Mean pressure across the row (all points, not just targets)
            p_sum += np.mean(c["pressure_1f"][row_idx])
        if not ok:
            continue
        if best is None or p_sum > best[1]:
            best = (yval, p_sum)

    return best[0]


if _has_cache("Fig6"):
    d = np.load(_fig_cache("Fig6"))
    n_rows, n_cols = int(d["n_rows"]), int(d["n_cols"])
    fig6_y = float(d.get("fig6_y", d.get("fig8_y")))
    print(f"\n--- Fig 6: y = {fig6_y*1e3:.3f} mm (from cache) ---")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 3.0))
    for row in range(n_rows):
        vpp = int(d[f"vpp_{row}"])
        for col in range(n_cols):
            t_us = d[f"t_us_{row}_{col}"]
            raw_mpa = d[f"raw_mpa_{row}_{col}"]
            recon_12f = d[f"recon_12f_{row}_{col}"]
            plabel = str(d[f"plabel_{col}"])

            ax = axes[row, col]
            ax.plot(t_us, raw_mpa, linewidth=0.5, color="C0")
            ax.plot(t_us, recon_12f, linewidth=0.5, color="C3", ls="--")
            ax.axhline(0, color="0.85", lw=0.25)
            ax.set_ylim(-3, 3)

            if row == 0:
                ax.set_title(plabel, pad=4)
            if col == 0:
                ax.set_ylabel(f"{vpp} $V_\\mathrm{{pp}}$\nPressure [MPa]")

            a1f = float(d[f"a1f_{row}_{col}"]) / 1e3
            a2f = float(d[f"a2f_{row}_{col}"]) / 1e3
            ratio_pct = a2f / a1f * 100 if a1f > 0 else 0
            print(f"  {vpp:2d} Vpp {plabel:>6s}: p1f={a1f:.0f}, "
                  f"p2f={a2f:.0f} kPa ({ratio_pct:.1f}%)")
else:
    fig6_y = _find_best_shared_row(FIG6_FILES)
    print(f"\n--- Fig 6: y = {fig6_y*1e3:.3f} mm ---")

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.0))
    cache_data = {"n_rows": 2, "n_cols": 3, "fig6_y": fig6_y}

    for row, (fname, vpp) in enumerate(FIG6_FILES):
        tdms_path = DATA_DIR_B / fname
        cache = load_or_compute(tdms_path, CACHE_DIR)
        pos_x_c = cache["pos_x"] - _centre_fn_B(cache["pos_y"])
        f_dr = float(cache["f_drive"])
        ss_start = int(cache["ss_start"])
        ss_end = int(cache["ss_end"])
        ss_n = ss_end - ss_start
        vel_scale = detect_velocity_scale(tdms_path)

        y_mask = np.abs(cache["pos_y"] - fig6_y) < 1e-5
        row_idx = np.where(y_mask)[0]
        cache_data[f"vpp_{row}"] = vpp

        for col, (target, plabel) in enumerate(
                zip(FIG6_TARGET_XC, FIG6_POS_LABELS)):
            dists = np.abs(pos_x_c[row_idx] - target)
            pt_idx = row_idx[np.argmin(dists)]

            # Raw waveform (Ch2 velocity) → pressure
            wf, dt = load_point_waveforms(tdms_path, pt_idx, channels=(2,))
            vel = wf[2] * vel_scale  # m/s
            prs = vel / (2 * np.pi * f_dr * SENSITIVITY)  # Pa

            # Display window: fixed 1 µs centred in steady state
            n_show = int(FIG6_WINDOW_US * 1e-6 / dt)
            t0 = (ss_start + ss_end) // 2 - n_show // 2
            t1 = t0 + n_show
            t_local = np.arange(t1 - t0) * dt
            t_us = t_local * 1e6

            # Local DFT at 1f and 2f over display window
            raw_seg = prs[t0:t1]
            n_loc = t1 - t0
            c1f = (raw_seg @ np.exp(-2j * np.pi * f_dr * t_local)) * 2 / n_loc
            c2f = (raw_seg @ np.exp(-2j * np.pi * 2 * f_dr * t_local)) * 2 / n_loc

            raw_mpa = raw_seg / 1e6  # MPa
            recon_1f = np.real(c1f * np.exp(2j * np.pi * f_dr * t_local)) / 1e6
            recon_12f = recon_1f + np.real(
                c2f * np.exp(2j * np.pi * 2 * f_dr * t_local)) / 1e6

            ax = axes[row, col]
            ax.plot(t_us, raw_mpa, linewidth=0.5, color="C0")
            ax.plot(t_us, recon_12f, linewidth=0.5, color="C3", ls="--")
            ax.axhline(0, color="0.85", lw=0.25)
            ax.set_ylim(-3, 3)

            if row == 0:
                ax.set_title(plabel, pad=4)
            if col == 0:
                ax.set_ylabel(f"{vpp} $V_\\mathrm{{pp}}$\nPressure [MPa]")

            a1f = abs(c1f) / 1e3  # kPa for print
            a2f = abs(c2f) / 1e3
            ratio_pct = a2f / a1f * 100 if a1f > 0 else 0
            print(f"  {vpp:2d} Vpp {plabel:>6s}: p1f={a1f:.0f}, "
                  f"p2f={a2f:.0f} kPa ({ratio_pct:.1f}%)")

            cache_data[f"t_us_{row}_{col}"] = t_us
            cache_data[f"raw_mpa_{row}_{col}"] = raw_mpa
            cache_data[f"recon_12f_{row}_{col}"] = recon_12f
            cache_data[f"plabel_{col}"] = plabel
            cache_data[f"a1f_{row}_{col}"] = abs(c1f)
            cache_data[f"a2f_{row}_{col}"] = abs(c2f)

    np.savez(_fig_cache("Fig6"), **cache_data)
    print(f"Saved: {_fig_cache('Fig6')}")

# Shared x-label
for ax in axes[-1]:
    ax.set_xlabel(tex_mu("Time [μs]"))

# Legend on first panel
handles = [
    Line2D([], [], color="C0", lw=0.7, label="Raw"),
    Line2D([], [], color="C3", lw=0.7, ls="--", label="$1f+2f$"),
]
axes[0, 0].legend(handles=handles, loc="lower left",
                  frameon=True, fancybox=False, edgecolor="0.8")

plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
save_fig(fig, "Fig6")
plt.close()

# %%
print("\n=== Fig 6 Done ===")

# %%
# =============================================================================
# Fig 7: Spatial mode profiles (1f and 2f across channel width)
# =============================================================================
#
# 2×2: (a) P_1f cross-section, (b) P_2f cross-section,
# (c) 2D P_1f map 25 Vpp, (d) 2D P_2f map 25 Vpp

if _has_cache("Fig7"):
    d = np.load(_fig_cache("Fig7"))
    y_best = float(d["y_best"])
    j_best = int(d["j_best"])
    print(f"\n--- Fig 7: axial antinode y = {y_best*1e3:.3f} mm "
          f"(col {j_best}, from cache) ---")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"height_ratios": [1.2, 1]})
    (ax_a, ax_b), (ax_c, ax_d) = axes
    _lbl_kw8 = dict(va="bottom", ha="left", fontweight="bold")

    y_th_um = d["y_th_um"]
    max_p0_2f = 0.0
    n_vpp = int(d["n_vpp"])
    colors_8 = ["C0", "C3"]

    for i in range(n_vpp):
        vpp = int(d[f"vpp_{i}"])
        y_um = d[f"y_um_{i}"]
        p1f_mpa = d[f"p1f_mpa_{i}"]
        p2f_mpa = d[f"p2f_mpa_{i}"]
        fit_1f_mpa = d[f"fit_1f_mpa_{i}"]
        fit_2f_mpa = d[f"fit_2f_mpa_{i}"]
        p0_1f = float(d[f"p0_1f_{i}"])
        p0_2f = float(d[f"p0_2f_{i}"])
        r2_1f = float(d[f"r2_1f_{i}"])
        r2_2f = float(d[f"r2_2f_{i}"])
        max_p0_2f = max(max_p0_2f, p0_2f)

        label = f"{vpp} $V_\\mathrm{{pp}}$"
        ax_a.plot(y_um, p1f_mpa, "o", markersize=1,
                  color=colors_8[i], label=label)
        ax_b.plot(y_um, p2f_mpa, "o", markersize=1,
                  color=colors_8[i], label=label)
        ax_a.plot(y_th_um, fit_1f_mpa, "--", linewidth=0.5, color=colors_8[i])
        ax_b.plot(y_th_um, fit_2f_mpa, "--", linewidth=0.5, color=colors_8[i])

        print(f"  {vpp:2d} Vpp: p0_1f = {p0_1f/1e3:.0f} kPa (R²={r2_1f:.3f}), "
              f"p0_2f = {p0_2f/1e3:.0f} kPa (R²={r2_2f:.3f})")

    # 2D maps
    grid_1f_25 = d["grid_1f_25_mpa"]
    grid_2f_25 = d["grid_2f_25_mpa"]
    w_mm = d["w_mm"]
    l_mm = d["l_mm"]
else:
    # Pick the axial antinode from mode-fit profile (25 Vpp)
    r_peak = results[-1]  # 25 Vpp
    j_best = int(np.nanargmax(r_peak["p0_1f_y"]))
    cg_peak = r_peak["cg"]
    y_best = cg_peak.length_grid[j_best]
    print(f"\n--- Fig 7: axial antinode y = {y_best*1e3:.3f} mm (col {j_best}) ---")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"height_ratios": [1.2, 1]})
    (ax_a, ax_b), (ax_c, ax_d) = axes
    _lbl_kw8 = dict(va="bottom", ha="left", fontweight="bold")

    # --- Top row: 1D cross-sections at antinode ---
    y_th = np.linspace(-hw, hw, 200)
    mode_1f_th = np.abs(np.sin(np.pi * y_th / CHANNEL_WIDTH))
    mode_2f_th = np.abs(np.cos(2 * np.pi * y_th / CHANNEL_WIDTH))

    FIG7_VPP = [10, 25]
    fig7_results = [r for r in results if r["vpp"] in FIG7_VPP]
    colors_8 = ["C0", "C3"]
    max_p0_2f = 0.0

    cache_data = {"y_best": y_best, "j_best": j_best,
                  "y_th_um": y_th * 1e6, "n_vpp": len(fig7_results)}

    for i, r in enumerate(fig7_results):
        cg_i = r["cg"]
        w_grid = cg_i.width_grid

        grid_1f = cg_i.to_grid(r["cache"]["pressure_1f"])
        grid_2f = cg_i.to_grid(r["cache"]["pressure_2f"])
        grid_ph1 = cg_i.to_grid(r["cache"]["phase_1f"])
        grid_ph2 = cg_i.to_grid(r["cache"]["phase_2f"])

        # Extract width cross-section at antinode (complex pressure)
        p1f_row = grid_1f[:, j_best]  # Pa
        p2f_row = grid_2f[:, j_best]
        ph1_row = grid_ph1[:, j_best]
        ph2_row = grid_ph2[:, j_best]
        p1f_complex = p1f_row * np.exp(1j * np.radians(ph1_row))
        p2f_complex = p2f_row * np.exp(1j * np.radians(ph2_row))

        # Mode-shape fits (complex LSQ, same method as freq_sweep_25vpp.py)
        valid = ~np.isnan(p1f_row)
        res_1f = fit_mode_1f(w_grid[valid], p1f_complex[valid], CHANNEL_WIDTH,
                             centre=0.0)
        res_2f = fit_mode_2f(w_grid[valid], p2f_complex[valid], CHANNEL_WIDTH,
                             centre=0.0)
        p0_1f = abs(res_1f.p0)  # Pa
        p0_2f = abs(res_2f.p0)

        y_um = w_grid * 1e6
        label = f"{r['vpp']} $V_\\mathrm{{pp}}$"

        ax_a.plot(y_um, p1f_row / 1e6, "o", markersize=1,
                  color=colors_8[i], label=label)
        ax_b.plot(y_um, p2f_row / 1e6, "o", markersize=1,
                  color=colors_8[i], label=label)

        fit_1f_mpa = p0_1f / 1e6 * mode_1f_th
        fit_2f_mpa = p0_2f / 1e6 * mode_2f_th
        ax_a.plot(y_th * 1e6, fit_1f_mpa, "--", linewidth=0.5,
                  color=colors_8[i])
        ax_b.plot(y_th * 1e6, fit_2f_mpa, "--", linewidth=0.5,
                  color=colors_8[i])
        max_p0_2f = max(max_p0_2f, p0_2f)

        print(f"  {r['vpp']:2d} Vpp: p0_1f = {p0_1f/1e3:.0f} kPa (R²={res_1f.r2:.3f}), "
              f"p0_2f = {p0_2f/1e3:.0f} kPa (R²={res_2f.r2:.3f})")

        cache_data[f"vpp_{i}"] = r["vpp"]
        cache_data[f"y_um_{i}"] = y_um
        cache_data[f"p1f_mpa_{i}"] = p1f_row / 1e6
        cache_data[f"p2f_mpa_{i}"] = p2f_row / 1e6
        cache_data[f"fit_1f_mpa_{i}"] = fit_1f_mpa
        cache_data[f"fit_2f_mpa_{i}"] = fit_2f_mpa
        cache_data[f"p0_1f_{i}"] = p0_1f
        cache_data[f"p0_2f_{i}"] = p0_2f
        cache_data[f"r2_1f_{i}"] = res_1f.r2
        cache_data[f"r2_2f_{i}"] = res_2f.r2

    # 2D maps
    grid_1f_25 = cg_peak.to_grid(r_peak["cache"]["pressure_1f"]) / 1e6  # MPa
    grid_2f_25 = cg_peak.to_grid(r_peak["cache"]["pressure_2f"]) / 1e6
    w_mm = cg_peak.width_grid * 1e3
    l_mm = cg_peak.length_grid * 1e3

    cache_data["grid_1f_25_mpa"] = grid_1f_25
    cache_data["grid_2f_25_mpa"] = grid_2f_25
    cache_data["w_mm"] = w_mm
    cache_data["l_mm"] = l_mm

    np.savez(_fig_cache("Fig7"), **cache_data)
    print(f"Saved: {_fig_cache('Fig7')}")

ax_a.set_ylabel("Pressure [MPa]")
ax_a.set_xlabel(tex_mu("$y$ [μm]"))
ax_a.set_ylim(bottom=0)
ax_a.text(-0.15, 1.05,
          "(a) $p_{1f}(y)$, $x$ = %.1f mm" % (y_best * 1e3),
          transform=ax_a.transAxes, **_lbl_kw8)
ax_a.legend(fontsize=5, frameon=False, loc="upper right")

ax_b.set_ylabel("Pressure [MPa]")
ax_b.set_xlabel(tex_mu("$y$ [μm]"))
ax_b.set_ylim(0, 1.2 * max_p0_2f / 1e6)
ax_b.text(-0.15, 1.05,
          "(b) $p_{2f}(y)$, $x$ = %.1f mm" % (y_best * 1e3),
          transform=ax_b.transAxes, **_lbl_kw8)
ax_b.legend(fontsize=5, frameon=False, loc="upper right")

# --- Bottom row: 2D pressure maps at 25 Vpp ---
for ax_map, grid, lbl, cmap_label in [
    (ax_c, grid_1f_25, r"(c) $p_{1f}(x,y)$, 25 $V_\mathrm{pp}$", "Pressure [MPa]"),
    (ax_d, grid_2f_25, r"(d) $p_{2f}(x,y)$, 25 $V_\mathrm{pp}$", "Pressure [MPa]"),
]:
    lo, hi = np.nanpercentile(grid, [5, 95])
    im = ax_map.pcolormesh(l_mm, w_mm, grid, shading="nearest",
                           cmap="viridis", vmin=lo, vmax=hi)
    ax_map.axvline(y_best * 1e3, color="red", linewidth=0.8, ls="--")
    ax_map.set_xlabel("Length, $x$ [mm]")
    ax_map.set_ylabel("Width, $y$ [mm]")
    ax_map.set_aspect("auto")
    ax_map.text(-0.15, 1.05, lbl, transform=ax_map.transAxes, **_lbl_kw8)
    cb = fig.colorbar(im, ax=ax_map, pad=0.02)
    cb.set_label(cmap_label)

plt.tight_layout()
save_fig(fig, "Fig7")
plt.close()

# %%
print("\n=== Fig 7 Done ===")

# %%
# =============================================================================
# Fig 9: Simulation vs experiment overlay (p₂f/p₁f vs E_ac)
# =============================================================================

_sim_cache = FIG_DIR / "fig3.npz"
if _sim_cache.exists():
    sim = np.load(_sim_cache)

    # Simulation E_ac from 1f only (self-consistent): use p1f_sc peak amplitude
    # with the simulation's water constants (rho=1000, c=1500).
    # This matches the experiment convention: 1f-only p0^2/(4*rho*c^2).
    _SIM_RHO = 1000.0
    _SIM_C = 1500.0
    sim_Eac_1f = sim["p1f_sc"]**2 / (4 * _SIM_RHO * _SIM_C**2)

    # Experiment data from Fig8 cache or computed above
    if _has_cache("Fig8"):
        exp = np.load(_fig_cache("Fig8"))
        exp_Eac = exp["E_ac"]
        exp_ratio = exp["ratio"]
    else:
        exp_Eac = Eac_1f_arr
        exp_ratio = p0_2f_peak_arr / p0_1f_peak_arr

    fig, ax = plt.subplots(figsize=(3.375, 2.5))

    # Simulation curve (self-consistent, 1f energy only)
    ax.plot(sim_Eac_1f, sim["ratio_sc"], "-", color="C0", linewidth=0.75,
            label="Simulation (self-consistent)")

    # Experiment markers
    ax.plot(exp_Eac, exp_ratio, "ko", markersize=4, zorder=3,
            label="Experiment")

    ax.set_xlabel(r"$\langle E_\mathrm{ac,1f} \rangle$ [J/m$^3$]")
    ax.set_ylabel(r"$P_{2f}/P_{1f}$")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_fig(fig, "Fig9")
    plt.close()

    print("\n--- Fig 9: Simulation vs experiment ---")
    print(f"  Experiment: E_ac = {exp_Eac[0]:.0f}–{exp_Eac[-1]:.0f} J/m³, "
          f"ratio = {exp_ratio[0]:.3f}–{exp_ratio[-1]:.3f}")
    print(f"  Simulation: E_ac(1f) up to {sim_Eac_1f[-1]:.0f} J/m³, "
          f"ratio up to {sim['ratio_sc'][-1]:.3f}")
    print("\n=== Fig 9 Done ===")
else:
    print("\n--- Fig 9: SKIPPED (simulation cache fig3.npz not found) ---")
    print(f"  Expected at: {_sim_cache}")
    print("  Run generate_manuscript_figures.py in harmonic_model first.")
