# %%
"""1D voltage sweep: p0(1f), p0(2f), drive current harmonics vs Vpp.

Processes test12 line scans (101×2) at y=8.9mm across multiple voltages.
Fits 1f and 2f mode shapes, extracts V, I, phase, and checks Ch1 2f/1f
ratio for drive harmonics.

Usage:
    python voltage_sweep_1d.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    figsize_for_layout,
    get_output_dir,
)
from ldv_analysis.fft_cache import load_or_compute, load_point_waveforms
from ldv_analysis.filters import make_valid_mask
from ldv_analysis.mode_fit import fit_mode_1f, fit_mode_2f

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("G:/My Drive/260318_ldv")

FILES = [
    ("test12_5Vpp1907kHz.tdms", 5),
    ("test12_10Vpp1907kHz.tdms", 10),
    ("test12_15Vpp1907kHz.tdms", 15),
    ("test12_20Vpp1907kHz.tdms", 20),
    ("test12_25Vpp1907kHz.tdms", 25),
    ("test12_30Vpp1907kHz.tdms", 30),
    ("test12_35Vpp1907kHz_5m_s_max.tdms", 35),
    ("test12_40Vpp1907kHz_5m_s_max.tdms", 40),
    ("test12_45Vpp1907kHz_5m_s_max.tdms", 45),
]

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# Process each file
# =============================================================================

results = []

for fname, vpp in FILES:
    tdms_path = DATA_DIR / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    cache = load_or_compute(tdms_path, CACHE_DIR)

    pos_y = cache["pos_x"]  # width direction
    pressure_1f = cache["pressure_1f"]
    pressure_2f = cache["pressure_2f"]
    phase_1f = cache["phase_1f"]
    phase_2f = cache["phase_2f"]
    voltage_1f = cache["voltage_1f"]
    rssi = cache["rssi"] if "rssi" in cache else None

    valid = make_valid_mask(voltage_1f, rssi)

    # Electrical parameters
    V_med = float(np.median(voltage_1f[valid]))
    has_ch4 = "current_1f" in cache
    I_med = float(np.median(cache["current_1f"][valid]) * 1e3) if has_ch4 else 0
    ph_med = float(np.median(cache["phase_vi"][valid])) if "phase_vi" in cache else 0

    # Drive current harmonics from raw Ch4 waveform at strongest point
    f_drive = float(cache["f_drive"])
    dt = float(cache["dt"])
    ss_start = int(cache["ss_start"])
    ss_end = int(cache["ss_end"])
    ss_n = ss_end - ss_start
    best_pt = int(np.argmax(cache["velocity_1f"]))

    current_harmonics = {}
    try:
        wfs, _ = load_point_waveforms(tdms_path, best_pt, channels=(1, 4))
        ch4_ss = wfs[4][ss_start:ss_end]
        ch1_ss = wfs[1][ss_start:ss_end]
        for h in range(1, 6):
            tone_h = np.exp(-2j * np.pi * h * f_drive * np.arange(ss_n) * dt)
            current_harmonics[h] = np.abs(ch4_ss @ tone_h) * 2 / ss_n
        ch1_1f_amp = np.abs(ch1_ss @ np.exp(-2j * np.pi * f_drive * np.arange(ss_n) * dt)) * 2 / ss_n
        ch1_2f_amp = np.abs(ch1_ss @ np.exp(-2j * np.pi * 2 * f_drive * np.arange(ss_n) * dt)) * 2 / ss_n
        ch1_2f_ratio = ch1_2f_amp / ch1_1f_amp if ch1_1f_amp > 0 else 0
    except Exception:
        for h in range(1, 6):
            current_harmonics[h] = 0
        ch1_2f_ratio = 0

    # 3f pressure from cache
    pressure_3f = cache["pressure_3f"]

    # 1f mode-shape fit
    p1f_complex = pressure_1f[valid] * np.exp(1j * np.radians(phase_1f[valid]))
    res_1f = fit_mode_1f(pos_y[valid], p1f_complex, CHANNEL_WIDTH)
    p0_1f = abs(res_1f.p0)

    # 2f mode-shape fit (use 1f centre)
    p2f_complex = pressure_2f[valid] * np.exp(1j * np.radians(phase_2f[valid]))
    res_2f = fit_mode_2f(pos_y[valid], p2f_complex, CHANNEL_WIDTH, res_1f.centre)
    p0_2f = abs(res_2f.p0)

    # 3f mode-shape fit: p(y) = p0 * |sin(3*pi*y_c/W)| — LSQ projection
    centre = res_1f.centre
    y_c_valid = pos_y[valid] - centre
    k_3f = 3 * np.pi / CHANNEL_WIDTH
    mode_3f = np.abs(np.sin(k_3f * y_c_valid))
    denom_3f = np.sum(mode_3f**2)
    p0_3f = float(np.sum(pressure_3f[valid] * mode_3f) / denom_3f) if denom_3f > 0 else 0

    I_h = {h: current_harmonics.get(h, 0) for h in range(1, 6)}
    results.append({
        "vpp": vpp,
        "V": V_med,
        "I_mA": I_med,
        "phase": ph_med,
        "p0_1f": p0_1f,
        "p0_2f": p0_2f,
        "p0_3f": p0_3f,
        "r2_1f": res_1f.r2,
        "r2_2f": res_2f.r2,
        "ratio_2f_1f": p0_2f / p0_1f if p0_1f > 0 else 0,
        "ch1_2f_1f": ch1_2f_ratio,
        "I_harmonics": I_h,
        "centre": res_1f.centre,
        "pos_y_valid": pos_y[valid],
        "prs_1f_valid": pressure_1f[valid],
        "prs_2f_valid": pressure_2f[valid],
        "prs_3f_valid": pressure_3f[valid],
    })

    I_h_str = " ".join(f"I{h}f={I_h[h]*1e3:.2f}" for h in range(2, 5))
    print(f"  {vpp:2d} Vpp: V={V_med:.2f}V I={I_med:.1f}mA "
          f"p0_1f={p0_1f/1e3:.0f}kPa p0_2f={p0_2f/1e3:.0f}kPa "
          f"2f/1f={p0_2f/p0_1f:.4f} {I_h_str}")

# %%
# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'Vpp':>5} {'V':>7} {'I(mA)':>7} {'phase':>7} "
      f"{'p0_1f':>9} {'p0_2f':>9} {'2f/1f':>8} {'Ch1 2f/1f':>10} "
      f"{'R2_1f':>6} {'R2_2f':>6}")
print("=" * 85)
for r in results:
    print(f"{r['vpp']:5d} {r['V']:7.2f} {r['I_mA']:7.1f} {r['phase']:7.1f} "
          f"{r['p0_1f']/1e3:9.0f} {r['p0_2f']/1e3:9.0f} {r['ratio_2f_1f']:8.4f} "
          f"{r['ch1_2f_1f']:10.4f} {r['r2_1f']:6.3f} {r['r2_2f']:6.3f}")

# %%
# =============================================================================
# Fits through origin
# =============================================================================

vpps = np.array([r["vpp"] for r in results])
p1f = np.array([r["p0_1f"] for r in results]) / 1e3  # kPa
p2f = np.array([r["p0_2f"] for r in results]) / 1e3
p3f = np.array([r["p0_3f"] for r in results]) / 1e3
ratios = np.array([r["ratio_2f_1f"] for r in results])
ch1_ratios = np.array([r["ch1_2f_1f"] for r in results])
I_arr = np.array([r["I_mA"] for r in results])

# p0_1f = a * Vpp (linear)
a_1f = float(np.sum(vpps * p1f) / np.sum(vpps**2))

# p0_2f = b * Vpp^2 (quadratic)
a_2f = float(np.sum(vpps**2 * p2f) / np.sum(vpps**4))

# p0_3f = c * Vpp^3 (cubic)
a_3f = float(np.sum(vpps**3 * p3f) / np.sum(vpps**6))

# 2f/1f ratio = c * Vpp (linear)
a_ratio = float(np.sum(vpps * ratios) / np.sum(vpps**2))

print(f"\nFits through origin:")
print(f"  p0_1f = {a_1f:.1f} kPa/Vpp")
print(f"  p0_2f = {a_2f:.4f} kPa/Vpp²")
print(f"  p0_3f = {a_3f:.6f} kPa/Vpp³")
print(f"  2f/1f = {a_ratio:.5f} /Vpp")

# %%
# =============================================================================
# Plot: 4 panels (sweep summary)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=figsize_for_layout(2, 2))
vpp_fine = np.linspace(0, vpps.max() * 1.05, 100)

# Panel 1: p0_1f vs Vpp
ax = axes[0, 0]
ax.plot(vpps, p1f, "o", markersize=4)
ax.plot(vpp_fine, a_1f * vpp_fine, "--", linewidth=0.8, alpha=0.6,
        label=f"${a_1f:.1f}$ kPa/Vpp")
ax.set_xlabel("$V_\\mathrm{pp}$ [V]")
ax.set_ylabel("$p_{0,\\mathrm{1f}}$ [kPa]")
ax.set_title("1f pressure amplitude")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3)

# Panel 2: p0_2f and p0_3f vs Vpp
ax = axes[0, 1]
ax.plot(vpps, p2f, "s", markersize=4, color="C1",
        label=f"2f: ${a_2f:.3f}$ kPa/Vpp$^2$")
ax.plot(vpp_fine, a_2f * vpp_fine**2, "--", linewidth=0.8, alpha=0.6, color="C1")
ax.plot(vpps, p3f, "^", markersize=4, color="C2",
        label=f"3f: ${a_3f:.5f}$ kPa/Vpp$^3$")
ax.plot(vpp_fine, a_3f * vpp_fine**3, "--", linewidth=0.8, alpha=0.6, color="C2")
ax.set_xlabel("$V_\\mathrm{pp}$ [V]")
ax.set_ylabel("$p_0$ [kPa]")
ax.set_title("2f and 3f pressure")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3)

# Panel 3: 2f/1f ratio vs Vpp
ax = axes[1, 0]
ax.plot(vpps, ratios, "^", markersize=4, color="C2")
ax.plot(vpp_fine, a_ratio * vpp_fine, "--", linewidth=0.8, alpha=0.6,
        color="C2", label=f"slope = {a_ratio:.5f} /Vpp")
ax.set_xlabel("$V_\\mathrm{pp}$ [V]")
ax.set_ylabel("$p_{\\mathrm{2f}} / p_{\\mathrm{1f}}$")
ax.set_title("2f/1f ratio")
ax.legend(fontsize=6, frameon=False)
ax.grid(True, alpha=0.3)

# Panel 4: Drive current harmonics vs Vpp
ax = axes[1, 1]
for h in range(2, 5):
    I_h_arr = np.array([r["I_harmonics"].get(h, 0) for r in results])
    I_1f_arr = np.array([r["I_harmonics"].get(1, 1) for r in results])
    ax.plot(vpps, I_h_arr / I_1f_arr * 100, "o-", markersize=3,
            label=f"$I_{{\\mathrm{{{h}f}}}} / I_{{\\mathrm{{1f}}}}$")
ax.set_xlabel("$V_\\mathrm{pp}$ [V]")
ax.set_ylabel("Current harmonic ratio [\\%]")
ax.set_title("Drive current harmonics")
ax.legend(fontsize=5, frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "voltage_sweep_1d_test12.png"
fig.savefig(out_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Mode shape plots per voltage (1f and 2f side by side)
# =============================================================================

n_files = len(results)
fig, axes = plt.subplots(n_files, 3, figsize=figsize_for_layout(n_files, 3, sharex=True),
                         sharex=True)
if n_files == 1:
    axes = axes[np.newaxis, :]

y_fine = np.linspace(-CHANNEL_WIDTH / 2, CHANNEL_WIDTH / 2, 200) * 1e3  # mm
k_1f = np.pi / CHANNEL_WIDTH
k_2f = 2 * np.pi / CHANNEL_WIDTH
k_3f = 3 * np.pi / CHANNEL_WIDTH

for i, r in enumerate(results):
    centre = r["centre"]
    y_c = (r["pos_y_valid"] - centre) * 1e3  # mm, centred
    p1f_data = r["prs_1f_valid"] / 1e3  # kPa
    p2f_data = r["prs_2f_valid"] / 1e3
    p3f_data = r["prs_3f_valid"] / 1e3

    # 1f mode shape
    ax = axes[i, 0]
    ax.plot(y_c, p1f_data, "o", markersize=2, alpha=0.5, color="C0")
    fit_1f = r["p0_1f"] / 1e3 * np.abs(np.sin(k_1f * y_fine * 1e-3))
    ax.plot(y_fine, fit_1f, "--", linewidth=1, color="C3",
            label=f"$p_0$ = {r['p0_1f']/1e3:.0f} kPa")
    ax.set_ylabel(f"{r['vpp']} Vpp")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_title("1f mode shape")

    # 2f mode shape
    ax = axes[i, 1]
    ax.plot(y_c, p2f_data, "o", markersize=2, alpha=0.5, color="C1")
    fit_2f = r["p0_2f"] / 1e3 * np.abs(np.cos(k_2f * y_fine * 1e-3))
    ax.plot(y_fine, fit_2f, "--", linewidth=1, color="C3",
            label=f"$p_0$ = {r['p0_2f']/1e3:.0f} kPa")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_title("2f mode shape")

    # 3f mode shape
    ax = axes[i, 2]
    ax.plot(y_c, p3f_data, "o", markersize=2, alpha=0.5, color="C2")
    fit_3f = r["p0_3f"] / 1e3 * np.abs(np.sin(k_3f * y_fine * 1e-3))
    ax.plot(y_fine, fit_3f, "--", linewidth=1, color="C3",
            label=f"$p_0$ = {r['p0_3f']/1e3:.0f} kPa")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_title("3f mode shape")

axes[-1, 0].set_xlabel("Channel width, $y$ [mm]")
axes[-1, 1].set_xlabel("Channel width, $y$ [mm]")
axes[-1, 2].set_xlabel("Channel width, $y$ [mm]")

plt.tight_layout()
out_path2 = OUT_DIR / "voltage_sweep_1d_mode_shapes.png"
fig.savefig(out_path2, dpi=FIG_DPI)
plt.close()
print(f"Saved: {out_path2}")

# %%
# =============================================================================
# Linearity check
# =============================================================================

print(f"\n=== Linearity check (p0_1f / linear fit) ===")
for r in results:
    expected = a_1f * r["vpp"]
    ratio = r["p0_1f"] / 1e3 / expected
    flag = " ***" if abs(ratio - 1) > 0.1 else ""
    print(f"  {r['vpp']:2d} Vpp: {r['p0_1f']/1e3:.0f} / {expected:.0f} = {ratio:.3f}{flag}")

# %%
print("\n=== Done ===")
