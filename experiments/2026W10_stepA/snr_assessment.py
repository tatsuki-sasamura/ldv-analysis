# %%
"""SNR assessment from post-burst noise.

Uses per-point noise RMS from the post-burst segment (cached in fft_cache)
to compute signal-to-noise ratio.  Produces:
  1. SNR histogram for a representative burst-mode dataset
  2. SNR vs voltage (from voltage sweep files)
  3. Spatial map of SNR (inside-channel vs edge)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ldv_analysis.config import (
    CHANNEL_WIDTH,
    FIG_DPI,
    channel_centre_func,
    figsize_for_layout,
    get_data_dir,
    get_output_dir,
    load_channel_geometry,
)
from ldv_analysis.fft_cache import load_or_compute

# %%
# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = get_data_dir("20260307experimentB")

# Voltage sweep files (burst mode, same frequency/position)
FILES = [
    ("test10_1907_5Vpp_1m_s_max.tdms",  5),
    ("test10_1907_10Vpp_2m_s_max.tdms", 10),
    ("test10_1907_15Vpp_2m_s_max.tdms", 15),
    ("test10_1907_20Vpp_2m_s_max.tdms", 20),
    ("test10_1907_25Vpp_5m_s_max.tdms", 25),
]

OUT_DIR = get_output_dir(__file__)
CACHE_DIR = OUT_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_geom = load_channel_geometry("20260307experimentB", CACHE_DIR)
_centre_fn = channel_centre_func(_geom)

# %%
# =============================================================================
# Load caches
# =============================================================================

hw = CHANNEL_WIDTH / 2  # m

results = []
ref_cache = None  # highest-voltage burst-mode file for spatial plot

for fname, vpp in FILES:
    tdms_path = DATA_DIR / fname
    if not tdms_path.exists():
        print(f"  SKIP (not found): {fname}")
        continue

    cache = load_or_compute(tdms_path, CACHE_DIR)

    noise_vel = cache["noise_rms_velocity"]
    has_noise = not np.all(np.isnan(noise_vel))

    if not has_noise:
        print(f"  {fname}: continuous mode, no noise data")
        results.append(dict(vpp=vpp, has_noise=False))
        continue

    pressure_1f = cache["pressure_1f"]
    noise_prs = cache["noise_rms_pressure"]
    pos_x = cache["pos_x"]
    pos_y = cache["pos_y"]

    snr = pressure_1f / noise_prs
    snr_db = 20 * np.log10(snr)

    # Inside-channel mask
    inside = np.abs(pos_x - _centre_fn(pos_y)) <= hw

    results.append(dict(
        vpp=vpp,
        has_noise=True,
        snr_db_median=float(np.median(snr_db)),
        snr_db_inside=float(np.median(snr_db[inside])),
        snr_db_outside=float(np.median(snr_db[~inside])),
        noise_prs_median=float(np.median(noise_prs)),
    ))
    ref_cache = dict(
        vpp=vpp, pos_x=pos_x, pressure_1f=pressure_1f,
        noise_prs=noise_prs, snr_db=snr_db, inside=inside,
    )
    print(f"  {fname}: SNR = {np.median(snr_db):.1f} dB (all), "
          f"{np.median(snr_db[inside]):.1f} dB (inside), "
          f"noise floor = {np.median(noise_prs)/1e3:.1f} kPa")

# %%
# =============================================================================
# Plot 1: SNR histogram (highest-voltage burst-mode file)
# =============================================================================

if ref_cache is not None:
    fig, ax = plt.subplots(figsize=figsize_for_layout())
    bins = np.arange(-10, 60, 2)
    ax.hist(ref_cache["snr_db"][ref_cache["inside"]], bins=bins,
            alpha=0.7, label="Inside channel", color="C0")
    ax.hist(ref_cache["snr_db"][~ref_cache["inside"]], bins=bins,
            alpha=0.7, label="Outside channel", color="C1")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Count")
    ax.set_title(f"SNR distribution --- {ref_cache['vpp']} Vpp")
    ax.legend(frameon=False, fontsize=5)
    plt.tight_layout()
    out_path = OUT_DIR / "snr_histogram.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"\nSaved: {out_path}")

# %%
# =============================================================================
# Plot 2: SNR vs voltage
# =============================================================================

burst_results = [r for r in results if r.get("has_noise")]
if len(burst_results) > 1:
    vpp_arr = np.array([r["vpp"] for r in burst_results])
    snr_inside = np.array([r["snr_db_inside"] for r in burst_results])
    snr_all = np.array([r["snr_db_median"] for r in burst_results])
    noise_floor = np.array([r["noise_prs_median"] for r in burst_results])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize_for_layout(2, 1, sharex=True), sharex=True,
    )

    ax1.plot(vpp_arr, snr_inside, "o-", markersize=4, label="Inside channel")
    ax1.plot(vpp_arr, snr_all, "s--", markersize=3, alpha=0.6, label="All points")
    ax1.set_ylabel("Median SNR [dB]")
    ax1.set_title("SNR vs drive voltage")
    ax1.legend(frameon=False, fontsize=5)

    ax2.plot(vpp_arr, noise_floor / 1e3, "D-", markersize=4, color="C2")
    ax2.set_xlabel("Drive voltage [V]")
    ax2.set_ylabel("Noise floor [kPa]")

    plt.tight_layout()
    out_path = OUT_DIR / "snr_vs_voltage.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {out_path}")

# %%
# =============================================================================
# Plot 3: SNR spatial map (highest-voltage burst-mode file)
# =============================================================================

if ref_cache is not None:
    pos_y_ref = None
    for fname, vpp in reversed(FILES):
        if vpp == ref_cache["vpp"]:
            c = load_or_compute(DATA_DIR / fname, CACHE_DIR)
            pos_y_ref = c["pos_y"]
            n_x = int(c["n_x_meta"])
            n_y = int(c["n_y_meta"])
            break

    if pos_y_ref is not None and n_x > 1 and n_y > 1:
        # Grid the data using metadata dimensions
        pos_x_ref = ref_cache["pos_x"]
        x_grid = np.linspace(pos_x_ref.min(), pos_x_ref.max(), n_x)
        y_grid = np.linspace(pos_y_ref.min(), pos_y_ref.max(), n_y)

        # Bin each point to nearest grid cell
        xi = np.argmin(np.abs(pos_x_ref[:, None] - x_grid[None, :]), axis=1)
        yi = np.argmin(np.abs(pos_y_ref[:, None] - y_grid[None, :]), axis=1)

        snr_grid = np.full((n_x, n_y), np.nan)
        snr_grid[xi, yi] = ref_cache["snr_db"]

        fig, ax = plt.subplots(figsize=figsize_for_layout(1, 1, ax_w_scale=1.5))
        pcm = ax.pcolormesh(
            y_grid * 1e3, x_grid * 1e3, snr_grid,
            shading="nearest", cmap="RdYlGn", vmin=0, vmax=40,
        )
        _c_mean = (_geom["centre_left_m"] + _geom["centre_right_m"]) / 2
        ax.axhline((_c_mean - hw) * 1e3, color="w", ls=":", lw=0.5)
        ax.axhline((_c_mean + hw) * 1e3, color="w", ls=":", lw=0.5)
        ax.set_xlabel("Axial position [mm]")
        ax.set_ylabel("Width position [mm]")
        ax.set_title(f"SNR map --- {ref_cache['vpp']} Vpp")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("SNR [dB]")
        plt.tight_layout()
        out_path = OUT_DIR / "snr_spatial_map.png"
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close()
        print(f"Saved: {out_path}")

# %%
# =============================================================================
# Summary table
# =============================================================================

print("\n| Vpp | SNR all (dB) | SNR inside (dB) | SNR outside (dB) | Noise (kPa) |")
print("|-----|-------------|----------------|-----------------|-------------|")
for r in results:
    if r.get("has_noise"):
        print(f"| {r['vpp']:3d} | {r['snr_db_median']:11.1f} | "
              f"{r['snr_db_inside']:14.1f} | {r['snr_db_outside']:15.1f} | "
              f"{r['noise_prs_median']/1e3:11.1f} |")
    else:
        print(f"| {r['vpp']:3d} |     (continuous, no noise data)    "
              f"              |             |")

# %%
print("\n=== Done ===")
