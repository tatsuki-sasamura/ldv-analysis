# %%
"""Harmonic spectrum of Ch2 (apparent velocity, refracto-vibrometry) — RMS across all scan points.

For each TDMS file, computes the amplitude at each harmonic (1f–5f) for
every scan point, then reports the RMS across the scan line.  Ch2 measures
apparent velocity from refractive-index modulation in the water-filled
microchannel, not actual surface velocity.

Requires: Run 00_convert_tdms.py first to generate .npz files.
"""

from config import (
    CONVERTED_DIR,
    DEFAULT_FIGSIZE,
    DISPLACEMENT_SCALE,
    EXCLUDED_FILES,
    FIG_DPI,
    SENSITIVITY,
    VELOCITY_SCALE,
    get_output_dir,
)
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# %%
# =============================================================================
# Configuration
# =============================================================================

# LDV bandwidth is 6 MHz; only 1f (~2 MHz) and 2f (~4 MHz) are reliable
MAX_HARMONIC = 2
MAX_HARMONIC_PROFILE = 2  # For per-file spatial plots
OUT_DIR = get_output_dir(__file__)

npz_files = sorted(CONVERTED_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

# %%
# =============================================================================
# Extract harmonic amplitudes at every scan point for each file
# =============================================================================

file_harmonics_rms = {}  # stem -> array of shape (MAX_HARMONIC,)
file_harmonics_per_point = {}  # stem -> (pos_x, all_harmonics)
file_representative = {}  # stem -> dict with waveform/spectrum of best point

for npz_path in npz_files:
    tdms_name = npz_path.stem + ".tdms"
    if tdms_name in EXCLUDED_FILES:
        continue

    print(f"Processing {npz_path.stem}...")
    data = np.load(npz_path)
    wf1 = data["wf_ch1"]
    wf2 = data["wf_ch2"]
    wf3 = data["wf_ch3"]
    dt = float(data["wf_dt"])
    n_points = wf1.shape[0]
    n_samples = wf1.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    # Vectorised FFT: (n_points, n_freq_bins)
    fft_v = np.fft.rfft(wf1, axis=1)
    fft_vel = np.fft.rfft(wf2, axis=1)
    fft_disp = np.fft.rfft(wf3, axis=1)

    # Drive frequency per point from Ch1 peak (skip DC)
    peak_idx = np.argmax(np.abs(fft_v[:, 1:]), axis=1) + 1  # (n_points,)

    # Harmonic bin indices: (n_points, MAX_HARMONIC)
    # Use searchsorted (O(n log m)) instead of argmin broadcast (O(n*m) memory)
    pts = np.arange(n_points)
    harmonics_mul = np.arange(1, MAX_HARMONIC + 1)  # [1, 2, ...]
    drive_freqs = freqs[peak_idx]  # (n_points,)
    h_freqs = drive_freqs[:, None] * \
        harmonics_mul[None, :]  # (n_points, MAX_HARMONIC)
    df = freqs[1] - freqs[0]  # uniform frequency spacing
    h_idx = np.clip(np.round(h_freqs / df).astype(int), 0, len(freqs) - 1)

    # Amplitude at each harmonic
    spectrum = np.abs(fft_vel) * 2 / n_samples * \
        VELOCITY_SCALE  # (n_points, n_freq_bins)
    all_harmonics = spectrum[pts[:, None], h_idx]  # (n_points, MAX_HARMONIC)

    # Phase relative to Ch1 at each harmonic
    phase_v1f = np.angle(fft_v[pts, peak_idx])  # (n_points,) radians
    # (n_points, MAX_HARMONIC)
    phase_vel_h = np.angle(fft_vel[pts[:, None], h_idx])
    diff = np.degrees(phase_vel_h) - \
        np.degrees(phase_v1f[:, None]) * harmonics_mul[None, :]
    all_phases = (diff + 180) % 360 - 180

    # Ch3 displacement amplitude at each harmonic
    spectrum_disp = np.abs(fft_disp) * 2 / n_samples * \
        DISPLACEMENT_SCALE  # (n_points, n_freq_bins) in m
    all_disp = spectrum_disp[pts[:, None], h_idx]  # (n_points, MAX_HARMONIC)

    # Pressure from Ch2 (velocity): p = v / (2*pi*f * SENSITIVITY)
    pressure_ch2 = all_harmonics / (2 * np.pi * h_freqs * SENSITIVITY)  # Pa

    # Pressure from Ch3 (displacement): p = d / SENSITIVITY
    pressure_ch3 = all_disp / SENSITIVITY  # Pa

    # Store per-point data
    rssi = data["scan_rssi"] if "scan_rssi" in data else None
    file_harmonics_per_point[npz_path.stem] = (
        data["scan_pos_x"], all_harmonics, all_phases, pressure_ch2, pressure_ch3, rssi,
    )

    # Select representative point: high 1f+2f with good RSSI
    score = all_harmonics[:, 0] + all_harmonics[:, 1]  # 1f + 2f amplitude
    if rssi is not None:
        rssi_norm = rssi / rssi.max() if rssi.max() > 0 else np.ones(n_points)
        score = score * rssi_norm
    best_i = int(np.argmax(score))
    best_drive = drive_freqs[best_i]
    file_representative[npz_path.stem] = {
        "point_idx": best_i,
        "pos_x": data["scan_pos_x"][best_i],
        "drive_freq": best_drive,
        "wf_ch2": wf2[best_i],
        "dt": dt,
        "fft_spectrum": spectrum[best_i],  # full |FFT| in m/s
        "freqs": freqs,
        "harmonics": all_harmonics[best_i],
        "rssi": rssi[best_i] if rssi is not None else None,
    }

    # RMS across scan points
    rms = np.sqrt(np.mean(all_harmonics**2, axis=0))
    rms_prs_ch2 = np.sqrt(np.mean(pressure_ch2**2, axis=0))
    rms_prs_ch3 = np.sqrt(np.mean(pressure_ch3**2, axis=0))
    file_harmonics_rms[npz_path.stem] = (rms, rms_prs_ch2, rms_prs_ch3)

    labels = [f"{h+1}f" for h in range(MAX_HARMONIC)]
    print(f"  RMS vel:    " +
          ", ".join(f"{l}={v:.2e}" for l, v in zip(labels, rms)))
    print(f"  RMS p(Ch2): " +
          ", ".join(f"{l}={v:.1f} kPa" for l, v in zip(labels, rms_prs_ch2 / 1e3)))
    print(f"  RMS p(Ch3): " +
          ", ".join(f"{l}={v:.1f} kPa" for l, v in zip(labels, rms_prs_ch3 / 1e3)))

# %%
# =============================================================================
# Plot: harmonic RMS amplitudes per file
# =============================================================================

stems = list(file_harmonics_rms.keys())
n_files = len(stems)
harmonic_labels = [f"{h+1}f" for h in range(MAX_HARMONIC)]
x = np.arange(MAX_HARMONIC)
width = 0.8 / n_files

# --- Apparent velocity bar chart ---
w, h = DEFAULT_FIGSIZE
fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
for j, stem in enumerate(stems):
    rms_vel, _, _ = file_harmonics_rms[stem]
    offset = (j - n_files / 2 + 0.5) * width
    ax.bar(x + offset, rms_vel, width, label=stem)

ax.set_xticks(x)
ax.set_xticklabels(harmonic_labels)
ax.set_ylabel("Apparent velocity RMS (m/s)")
ax.set_xlabel("Harmonic")
ax.set_title("Harmonic Content --- RMS Across Scan Points (Ch2)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
output_path = OUT_DIR / "harmonics_comparison.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# --- Pressure bar chart (Ch2 velocity-derived and Ch3 displacement-derived) ---
fig, axes = plt.subplots(1, 2, figsize=(2 * w, h), sharey=True)
for j, stem in enumerate(stems):
    _, rms_prs_ch2, rms_prs_ch3 = file_harmonics_rms[stem]
    offset = (j - n_files / 2 + 0.5) * width
    axes[0].bar(x + offset, rms_prs_ch2 / 1e3, width, label=stem)
    axes[1].bar(x + offset, rms_prs_ch3 / 1e3, width, label=stem)

for ax, title in zip(axes, ["From Ch2 (velocity)", "From Ch3 (displacement)"]):
    ax.set_xticks(x)
    ax.set_xticklabels(harmonic_labels)
    ax.set_xlabel("Harmonic")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
axes[0].set_ylabel("Acoustic pressure RMS (kPa)")
fig.suptitle("Acoustic Pressure --- RMS Across Scan Points", fontsize=13)
plt.tight_layout()
output_path = OUT_DIR / "pressure_comparison.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: harmonic ratio (nf / 1f) per file — velocity
# =============================================================================

fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
for j, stem in enumerate(stems):
    rms_vel, _, _ = file_harmonics_rms[stem]
    if rms_vel[0] > 0:
        ratio = rms_vel / rms_vel[0]
        offset = (j - n_files / 2 + 0.5) * width
        ax.bar(x + offset, ratio, width, label=stem)

ax.set_xticks(x)
ax.set_xticklabels(harmonic_labels)
ax.set_ylabel("Amplitude ratio (nf / 1f)")
ax.set_xlabel("Harmonic")
ax.set_title("Harmonic Ratio Relative to Fundamental --- Velocity (RMS)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
output_path = OUT_DIR / "harmonics_ratio.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: pressure ratio (nf / 1f) per file
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(2 * w, h), sharey=True)
for j, stem in enumerate(stems):
    _, rms_prs_ch2, rms_prs_ch3 = file_harmonics_rms[stem]
    offset = (j - n_files / 2 + 0.5) * width
    if rms_prs_ch2[0] > 0:
        axes[0].bar(x + offset, rms_prs_ch2 /
                    rms_prs_ch2[0], width, label=stem)
    if rms_prs_ch3[0] > 0:
        axes[1].bar(x + offset, rms_prs_ch3 /
                    rms_prs_ch3[0], width, label=stem)

for ax, title in zip(axes, ["From Ch2 (velocity)", "From Ch3 (displacement)"]):
    ax.set_xticks(x)
    ax.set_xticklabels(harmonic_labels)
    ax.set_xlabel("Harmonic")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
axes[0].set_ylabel("Pressure ratio (nf / 1f)")
fig.suptitle("Pressure Ratio Relative to Fundamental (RMS)", fontsize=13)
plt.tight_layout()
output_path = OUT_DIR / "pressure_ratio.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Plot: harmonic amplitude vs position — one plot per file
# =============================================================================

print("\n=== Harmonic Profiles Along Scan Line ===")

for stem, (pos_x, all_harmonics, all_phases, prs_ch2, prs_ch3, rssi) in file_harmonics_per_point.items():
    order = np.argsort(pos_x)

    n_rows = 5 if rssi is not None else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(w, n_rows * h), sharex=True)

    # Apparent velocity
    for hi in range(MAX_HARMONIC_PROFILE):
        axes[0].plot(pos_x[order], all_harmonics[order, hi],
                     label=f"{hi+1}f", marker=".", markersize=2, linewidth=0.8)
    axes[0].set_ylabel("Apparent velocity (m/s)")
    axes[0].set_title(f"Harmonic Distribution --- {stem}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Pressure from Ch2 (velocity)
    for hi in range(MAX_HARMONIC_PROFILE):
        axes[1].plot(pos_x[order], prs_ch2[order, hi] / 1e3,
                     label=f"{hi+1}f", marker=".", markersize=2, linewidth=0.8)
    axes[1].set_ylabel("Pressure from Ch2 (kPa)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Pressure from Ch3 (displacement)
    for hi in range(MAX_HARMONIC_PROFILE):
        axes[2].plot(pos_x[order], prs_ch3[order, hi] / 1e3,
                     label=f"{hi+1}f", marker=".", markersize=2, linewidth=0.8)
    axes[2].set_ylabel("Pressure from Ch3 (kPa)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Phase
    for hi in range(MAX_HARMONIC_PROFILE):
        axes[3].plot(pos_x[order], all_phases[order, hi],
                     label=f"{hi+1}f", marker=".", markersize=2, linewidth=0.8)
    axes[3].set_ylabel("Phase rel. to Ch1 (deg)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # RSSI
    if rssi is not None:
        axes[4].plot(pos_x[order], rssi[order], color="C7",
                     marker=".", markersize=2, linewidth=0.8)
        axes[4].set_ylabel("RSSI (V)")
        axes[4].grid(True, alpha=0.3)

    axes[-1].set_xlabel("X position (mm)")
    plt.tight_layout()
    output_path = OUT_DIR / f"harmonics_profile_{stem}.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {output_path.name}")

# %%
# =============================================================================
# Plot: representative waveform and spectrum for each file
# =============================================================================

print("\n=== Representative Waveforms ===")

N_PERIODS = 5  # number of drive periods to show in time-domain plot

for stem, rep in file_representative.items():
    wf = rep["wf_ch2"]
    dt = rep["dt"]
    f_drive = rep["drive_freq"]
    period = 1.0 / f_drive
    t_show = N_PERIODS * period
    n_show = min(int(t_show / dt), len(wf))
    t = np.arange(n_show) * dt * 1e6  # µs

    fft_spec = rep["fft_spectrum"]  # velocity amplitude spectrum (m/s)
    freqs = rep["freqs"]

    # Convert velocity waveform to pressure in frequency domain
    # p(f) = v(f) / (2*pi*f * SENSITIVITY), skip DC
    fft_vel_complex = np.fft.rfft(wf * VELOCITY_SCALE)
    fft_prs_complex = np.zeros_like(fft_vel_complex)
    fft_prs_complex[1:] = fft_vel_complex[1:] / \
        (2 * np.pi * freqs[1:] * SENSITIVITY)
    wf_pressure = np.fft.irfft(fft_prs_complex, n=len(wf))  # Pa

    # Pressure amplitude spectrum
    prs_spec = np.zeros_like(fft_spec)
    prs_spec[1:] = fft_spec[1:] / (2 * np.pi * freqs[1:] * SENSITIVITY)

    info = f"point \\#{rep['point_idx']}, X={rep['pos_x']:.3f} mm"
    if rep["rssi"] is not None:
        info += f", RSSI={rep['rssi']:.2f} V"
    info += f"\n1f={rep['harmonics'][0]:.3e}, 2f={rep['harmonics'][1]:.3e} m/s"

    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 2 * h))

    # Time-domain: velocity
    axes[0, 0].plot(t, wf[:n_show] * VELOCITY_SCALE, linewidth=0.8)
    axes[0, 0].set_xlabel("Time (µs)")
    axes[0, 0].set_ylabel("Apparent velocity (m/s)")
    axes[0, 0].set_title(f"Waveform --- {stem}\n{info}")
    axes[0, 0].grid(True, alpha=0.3)

    # Time-domain: pressure
    axes[0, 1].plot(t, wf_pressure[:n_show] / 1e3, linewidth=0.8, color="C1")
    axes[0, 1].set_xlabel("Time (µs)")
    axes[0, 1].set_ylabel("Acoustic pressure (kPa)")
    axes[0, 1].set_title("Pressure (from Ch2)")
    axes[0, 1].grid(True, alpha=0.3)

    # Spectrum: velocity
    freq_mask = freqs <= 10e6
    axes[1, 0].plot(freqs[freq_mask] / 1e6, fft_spec[freq_mask], linewidth=0.8)
    for hi in range(MAX_HARMONIC):
        h_freq = f_drive * (hi + 1)
        axes[1, 0].axvline(h_freq / 1e6, color="red", ls=":", alpha=0.5)
        axes[1, 0].text(h_freq / 1e6, fft_spec.max() * 0.9, f" {hi+1}f",
                        fontsize=9, color="red", va="top")
    axes[1, 0].set_xlabel("Frequency (MHz)")
    axes[1, 0].set_ylabel("Apparent velocity (m/s)")
    axes[1, 0].set_title("FFT Spectrum --- velocity")
    axes[1, 0].grid(True, alpha=0.3)

    # Spectrum: pressure
    axes[1, 1].plot(freqs[freq_mask] / 1e6, prs_spec[freq_mask] / 1e3,
                    linewidth=0.8, color="C1")
    for hi in range(MAX_HARMONIC):
        h_freq = f_drive * (hi + 1)
        axes[1, 1].axvline(h_freq / 1e6, color="red", ls=":", alpha=0.5)
        axes[1, 1].text(h_freq / 1e6, prs_spec[freq_mask].max() / 1e3 * 0.9,
                        f" {hi+1}f", fontsize=9, color="red", va="top")
    axes[1, 1].set_xlabel("Frequency (MHz)")
    axes[1, 1].set_ylabel("Acoustic pressure (kPa)")
    axes[1, 1].set_title("FFT Spectrum --- pressure")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUT_DIR / f"waveform_{stem}.png"
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()
    print(f"Saved: {output_path.name}")

# %%
print(f"\n=== Done ===")
print(f"Output directory: {OUT_DIR}")
