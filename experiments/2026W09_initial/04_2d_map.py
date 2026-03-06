# %%
"""2D spatial maps of apparent velocity and acoustic pressure from area scans.

Generates pcolormesh heatmaps (velocity, pressure, phase, RSSI) for every
converted file in the active dataset.  Channel boundaries are detected by
minimising pressure² outside a strip of known width (375 µm), then data
is displayed in centred channel coordinates.

FFT results are cached as _fft.npz files alongside the converted data for
fast subsequent runs (~65s → <1s).

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute


def _ts(label, t0):
    elapsed = time.perf_counter() - t0
    print(f"  [{elapsed:6.2f}s] {label}")
    return time.perf_counter()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ldv_analysis.config import (
    CONVERTED_DIR,
    EXCLUDED_FILES,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    figsize_for_layout,
    get_output_dir,
)

OUT_DIR = get_output_dir(__file__)

CHANNEL_WIDTH = 0.375  # mm (known physical width)

# Particle parameters (polystyrene, 5 um diameter — same as PIV repo)
RHO_P = 1.05e3        # kg/m^3
KAPPA_P = 2.49e-10     # Pa^-1
RADIUS = 2.5e-6        # m
RHO_F = 1.003586e3     # kg/m^3 (water at 25 C)
C_F = 1.50757e3        # m/s
VIS = 0.8882e-3        # Pa.s (dynamic viscosity)
F_US = 1.97e6          # Hz (ultrasound frequency)

# Acoustic contrast factors (viscous-corrected)
kappa_f = 1 / (RHO_F * C_F**2)
kappa_hat = KAPPA_P / kappa_f
rho_hat = RHO_P / RHO_F
th_bou = np.sqrt(2 * VIS / RHO_F / (2 * np.pi * F_US))
delta_hat = th_bou / RADIUS
Gamma = -3 / 2 * (1 + (1 + delta_hat) * 1j) * delta_hat
f1 = 1 - kappa_hat
f2 = 2 * (1 - Gamma) * (rho_hat - 1) / (2 * rho_hat + 1 - 3 * Gamma)
V_p = 4 / 3 * np.pi * RADIUS**3


# %%
# =============================================================================
# FFT cache
# =============================================================================

def _compute_fft_results(npz_path):
    """Compute FFT-derived quantities from raw waveforms and cache to disk."""
    data = np.load(npz_path)

    ch_length = data["scan_pos_y"]
    ch_width = data["scan_pos_x"]
    n_length = int(data["meta_n_y"])
    n_width = int(data["meta_n_x"])

    wf1 = data["wf_ch1"]
    wf2 = data["wf_ch2"]
    dt = float(data["wf_dt"])
    n_points = wf1.shape[0]
    n_samples = wf1.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    rssi = data["scan_rssi"] if "scan_rssi" in data else None

    fft_v = np.fft.rfft(wf1, axis=1)
    fft_vel = np.fft.rfft(wf2, axis=1)
    peak_idx = np.argmax(np.abs(fft_v[:, 1:]), axis=1) + 1
    pts = np.arange(n_points)
    drive_freqs = freqs[peak_idx]

    df = freqs[1] - freqs[0]
    peak_idx_2f = np.clip(np.round(2 * drive_freqs / df).astype(int), 0, len(freqs) - 1)

    velocity_1f = np.abs(fft_vel[pts, peak_idx]) * 2 / n_samples * VELOCITY_SCALE
    velocity_2f = np.abs(fft_vel[pts, peak_idx_2f]) * 2 / n_samples * VELOCITY_SCALE

    pressure_1f = velocity_1f / (2 * np.pi * drive_freqs * SENSITIVITY)
    pressure_2f = velocity_2f / (2 * np.pi * 2 * drive_freqs * SENSITIVITY)

    diff_1f = np.degrees(np.angle(fft_vel[pts, peak_idx]) - np.angle(fft_v[pts, peak_idx]))
    phase_1f = (diff_1f + 180) % 360 - 180
    diff_2f = np.degrees(np.angle(fft_vel[pts, peak_idx_2f]) - 2 * np.angle(fft_v[pts, peak_idx]))
    phase_2f = (diff_2f + 180) % 360 - 180

    arrays = dict(
        ch_length=ch_length, ch_width=ch_width,
        n_length=np.array(n_length), n_width=np.array(n_width),
        drive_freqs=drive_freqs,
        velocity_1f=velocity_1f, velocity_2f=velocity_2f,
        pressure_1f=pressure_1f, pressure_2f=pressure_2f,
        phase_1f=phase_1f, phase_2f=phase_2f,
    )
    if rssi is not None:
        arrays["rssi"] = rssi

    cache_dir = npz_path.parent.parent / "fft_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (npz_path.stem + "_fft.npz")
    np.savez(cache_path, **arrays)
    return cache_path


def _load_fft_cache(npz_path):
    """Load FFT cache, creating it if missing."""
    cache_dir = npz_path.parent.parent / "fft_cache"
    cache_path = cache_dir / (npz_path.stem + "_fft.npz")
    if not cache_path.exists():
        print("  Cache not found, computing FFT...")
        _compute_fft_results(npz_path)
        print("  Cache saved.")
    else:
        print("  Using FFT cache.")
    return np.load(cache_path)


# %%
# =============================================================================
# Process each converted file
# =============================================================================

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
npz_files = [f for f in npz_files if f.stem + ".tdms" not in EXCLUDED_FILES]
npz_files = [f for f in npz_files if "101x101" in f.stem]

if not npz_files:
    print("No converted files found.")
    sys.exit(0)

print(f"Found {len(npz_files)} files to process\n")

for npz_path in npz_files:
    stem = npz_path.stem
    print(f"--- {stem} ---")

    t0 = time.perf_counter()

    cache = _load_fft_cache(npz_path)

    ch_length = cache["ch_length"]
    ch_width = cache["ch_width"]
    n_length = int(cache["n_length"])
    n_width = int(cache["n_width"])
    drive_freqs = cache["drive_freqs"]
    velocity_1f = cache["velocity_1f"]
    velocity_2f = cache["velocity_2f"]
    pressure_1f = cache["pressure_1f"]
    pressure_2f = cache["pressure_2f"]
    phase_1f = cache["phase_1f"]
    phase_2f = cache["phase_2f"]
    rssi = cache["rssi"] if "rssi" in cache else None
    n_points = len(ch_length)

    # Skip 1D line scans
    if n_length <= 2:
        print(f"  Skipping (1D line scan: {n_length} × {n_width})")
        continue

    print(f"  {n_points} scan points, grid {n_length} × {n_width}")
    t0 = _ts("Load data", t0)

    print(f"  Drive freq: {drive_freqs.mean()/1e6:.4f} MHz")
    print(f"  Velocity 1f: {np.nanmean(velocity_1f):.4e} m/s, 2f: {np.nanmean(velocity_2f):.4e} m/s")
    print(f"  Pressure 1f: {np.nanmean(pressure_1f)/1e3:.1f} kPa, 2f: {np.nanmean(pressure_2f)/1e3:.1f} kPa")

    # %%
    # =================================================================
    # Channel boundary detection: minimise pressure² outside the strip
    # =================================================================

    hw = CHANNEL_WIDTH / 2
    x_min, x_max = ch_length.min(), ch_length.max()

    # Parameter bounds: centre ± w/2 must stay inside scan range
    w_min, w_max = ch_width.min(), ch_width.max()
    y_lo = w_min + hw
    y_hi = w_max - hw

    prs_sq = pressure_1f**2

    def outside_pressure_sum(params):
        y_left, y_right = params
        centre = y_left + (y_right - y_left) / (x_max - x_min) * (ch_length - x_min)
        outside = np.abs(ch_width - centre) > hw
        return np.nansum(prs_sq[outside])

    result = brute(outside_pressure_sum,
                   ranges=((y_lo, y_hi), (y_lo, y_hi)),
                   Ns=100, finish=None)
    y_left_opt, y_right_opt = result

    a_opt = (y_right_opt - y_left_opt) / (x_max - x_min)
    b_opt = y_left_opt - a_opt * x_min
    tilt_deg = np.degrees(np.arctan(a_opt))

    t0 = _ts("Boundary detection", t0)

    print(f"  Channel: y_left={y_left_opt:.4f}, y_right={y_right_opt:.4f}")
    print(f"  Tilt: {tilt_deg:.3f} deg, width: {CHANNEL_WIDTH} mm (fixed)")

    # %%
    # =================================================================
    # Build 2D grid in centred channel coordinates
    # =================================================================

    ch_width_c = ch_width - (a_opt * ch_length + b_opt)
    inside_c = np.abs(ch_width_c) <= hw

    length_grid = np.linspace(ch_length.min(), ch_length.max(), n_length)
    l_idx = np.argmin(np.abs(ch_length[:, None] - length_grid[None, :]), axis=1)

    width_span = ch_width.max() - ch_width.min()
    scan_step = width_span / max(n_width - 1, 1)
    n_width_c = max(int(round(CHANNEL_WIDTH / scan_step)), 2)
    half_step = CHANNEL_WIDTH / n_width_c / 2
    width_c_grid = np.linspace(-hw + half_step, hw - half_step, n_width_c)

    w_c_idx = np.argmin(np.abs(ch_width_c[:, None] - width_c_grid[None, :]), axis=1)

    def to_grid(values):
        grid = np.full((n_width_c, n_length), np.nan)
        mask = inside_c & ~np.isnan(values)
        grid[w_c_idx[mask], l_idx[mask]] = values[mask]
        return grid

    grid_vel_1f = to_grid(velocity_1f)
    grid_prs_1f = to_grid(pressure_1f / 1e3)  # kPa
    grid_phase_1f = to_grid(phase_1f)
    grid_vel_2f = to_grid(velocity_2f)
    grid_prs_2f = to_grid(pressure_2f / 1e3)  # kPa
    grid_phase_2f = to_grid(phase_2f)
    grid_rssi = to_grid(rssi) if rssi is not None else None
    t0 = _ts("Build grid", t0)

    # figsize
    length_span = length_grid[-1] - length_grid[0]
    fig_w = figsize_for_layout(ax_w_scale=2.5)[0]
    fig_h = fig_w * (CHANNEL_WIDTH / length_span) * 3  # 3× vertical exaggeration
    fig_h = max(fig_h, 1.5)

    def map_plot(grid_data, cmap, title, cb_label, output_name,
                 vmin=None, vmax=None, pclip=None, xlim=None):
        if xlim is not None:
            x_frac = (xlim[1] - xlim[0]) / length_span
        else:
            x_frac = 1.0
        fw = fig_w * x_frac + 1.2
        fh = fig_h + 1.0
        fig, ax = plt.subplots(figsize=(fw, fh))
        kwargs = dict(shading="nearest", cmap=cmap)
        if pclip is not None:
            lo, hi = np.nanpercentile(grid_data, [pclip, 100 - pclip])
            kwargs["vmin"] = lo
            kwargs["vmax"] = hi
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        im = ax.pcolormesh(length_grid, width_c_grid, grid_data, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel("Channel length, x (mm)")
        ax.set_ylabel("Channel width, y (mm)")
        ax.set_title(f"{title}\n{stem}")
        ax.set_aspect("auto")
        plt.colorbar(im, ax=ax, label=cb_label)
        plt.tight_layout()
        output_path = OUT_DIR / output_name
        plt.savefig(output_path, dpi=FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {output_path.name}")

    # %%
    # =================================================================
    # Plot: 1f maps
    # =================================================================

    map_plot(grid_vel_1f, "viridis", "Apparent Velocity at 1f",
             "Apparent velocity (m/s)", f"{stem}_velocity_1f.png",
             pclip=5)

    map_plot(grid_prs_1f, "viridis", "Acoustic Pressure at 1f",
             "Acoustic pressure (kPa)", f"{stem}_pressure_1f.png",
             pclip=5)

    map_plot(grid_phase_1f, "twilight", "Phase at 1f (rel. to Ch1)",
             "Phase (deg)", f"{stem}_phase_1f.png", vmin=-180, vmax=180)

    # %%
    # =================================================================
    # Plot: 2f maps
    # =================================================================

    map_plot(grid_vel_2f, "viridis", "Apparent Velocity at 2f",
             "Apparent velocity (m/s)", f"{stem}_velocity_2f.png",
             pclip=5)

    map_plot(grid_prs_2f, "viridis", "Acoustic Pressure at 2f",
             "Acoustic pressure (kPa)", f"{stem}_pressure_2f.png",
             pclip=5)

    map_plot(grid_phase_2f, "twilight", "Phase at 2f (rel. to 2×Ch1)",
             "Phase (deg)", f"{stem}_phase_2f.png", vmin=-180, vmax=180)

    # %%
    # =================================================================
    # Plot: zoomed 1f maps (x = 0–1 mm)
    # =================================================================

    zoom = (0, 1)

    map_plot(grid_vel_1f, "viridis", "Apparent Velocity at 1f (zoom)",
             "Apparent velocity (m/s)", f"{stem}_velocity_1f_zoom.png",
             pclip=5, xlim=zoom)

    map_plot(grid_prs_1f, "viridis", "Acoustic Pressure at 1f (zoom)",
             "Acoustic pressure (kPa)", f"{stem}_pressure_1f_zoom.png",
             pclip=5, xlim=zoom)

    map_plot(grid_phase_1f, "twilight", "Phase at 1f (zoom)",
             "Phase (deg)", f"{stem}_phase_1f_zoom.png",
             vmin=-180, vmax=180, xlim=zoom)

    map_plot(grid_vel_2f, "viridis", "Apparent Velocity at 2f (zoom)",
             "Apparent velocity (m/s)", f"{stem}_velocity_2f_zoom.png",
             pclip=5, xlim=zoom)

    map_plot(grid_prs_2f, "viridis", "Acoustic Pressure at 2f (zoom)",
             "Acoustic pressure (kPa)", f"{stem}_pressure_2f_zoom.png",
             pclip=5, xlim=zoom)

    map_plot(grid_phase_2f, "twilight", "Phase at 2f (zoom)",
             "Phase (deg)", f"{stem}_phase_2f_zoom.png",
             vmin=-180, vmax=180, xlim=zoom)

    # %%
    # =================================================================
    # Plot: RSSI heatmap
    # =================================================================

    if grid_rssi is not None:
        map_plot(grid_rssi, "viridis", "RSSI",
                 "RSSI (V)", f"{stem}_rssi_2d.png")

    # =================================================================
    # Gor'kov potential map
    # =================================================================

    W_m = CHANNEL_WIDTH * 1e-3  # 0.375e-3 m
    k = np.pi / W_m
    wc_m = width_c_grid * 1e-3  # mm -> m
    # wc=0 is channel centre = pressure node, so p ∝ |sin(ky_c)|
    sin_profile = np.abs(np.sin(k * wc_m))

    # Extract p0(x) from gridded pressure via least-squares projection
    p0_x = np.full(len(length_grid), np.nan)
    for j in range(len(length_grid)):
        col = grid_prs_1f[:, j] * 1e3  # kPa -> Pa
        valid = ~np.isnan(col)
        if valid.sum() > 3:
            p0_x[j] = np.sum(col[valid] * sin_profile[valid]) / np.sum(sin_profile[valid]**2)

    # Compute U(x, y_c) on the grid
    sin2 = np.sin(k * wc_m)**2
    cos2 = np.cos(k * wc_m)**2
    U = (V_p / (4 * RHO_F * C_F**2)) * p0_x[None, :]**2 \
        * (f1 * sin2[:, None] - 1.5 * np.real(f2) * cos2[:, None])
    U_fJ = U * 1e15  # J -> fJ

    print(f"  Gorkov U range: {np.nanmin(U_fJ):.3f} to {np.nanmax(U_fJ):.3f} fJ")

    map_plot(U_fJ, "viridis", "Gor'kov Potential",
             "U (fJ)", f"gorkov_potential_{stem}.png", pclip=5)

    # E_pot = p^2 / (4 rho c^2)  — directly from measured pressure field
    E_pot = (grid_prs_1f * 1e3)**2 / (4 * RHO_F * C_F**2)  # kPa -> Pa, then J/m³

    print(f"  E_pot range: {np.nanmin(E_pot):.2f} to {np.nanmax(E_pot):.2f} J/m³")

    map_plot(E_pot, "viridis", r"$E_\mathrm{pot}$",
             r"$E_\mathrm{pot}$ (J/m³)", f"energy_potential_{stem}.png", pclip=5)

    _ts("All plots", t0)
    print()

# %%
print(f"=== Done ===")
print(f"Output directory: {OUT_DIR}")
