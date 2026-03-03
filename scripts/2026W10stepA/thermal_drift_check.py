# %%
"""Quick check: Ch1/Ch4 stability across the scan to confirm no thermal drift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DPI, figsize_for_layout
from ldv_analysis.io_utils import load_tdms_file, extract_waveforms

# %%
DEFAULT_TDMS = Path("G:/My Drive/20260303experimentA/stepA1967.tdms")
OUT_DIR = Path(__file__).parent.parent.parent / "output" / "2026W10stepA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tdms_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TDMS
stem = tdms_path.stem

tdms_file, metadata = load_tdms_file(tdms_path)
wf_ch1, dt = extract_waveforms(tdms_file, channel=1)
wf_ch4, _ = extract_waveforms(tdms_file, channel=4)
n_points = wf_ch1.shape[0]

# Steady-state window
ss_start = int(15e-6 / dt)
ss_end = int(502e-6 / dt)
ss_n = ss_end - ss_start
freqs = np.fft.rfftfreq(ss_n, d=dt)

# %%
# Vectorised FFT
fft_ch1 = np.fft.rfft(wf_ch1[:, ss_start:ss_end], axis=1)
fft_ch4 = np.fft.rfft(wf_ch4[:, ss_start:ss_end], axis=1)

pts = np.arange(n_points)
peak_idx = np.argmax(np.abs(fft_ch1[:, 1:]), axis=1) + 1

V_1f = np.abs(fft_ch1[pts, peak_idx]) * 2 / ss_n * 10    # x10 atten
I_1f = np.abs(fft_ch4[pts, peak_idx]) * 2 / ss_n * 0.2   # A/V
Z_mag = V_1f / I_1f
phase_VI = np.degrees(np.angle(fft_ch1[pts, peak_idx]) - np.angle(fft_ch4[pts, peak_idx]))
phase_VI = (phase_VI + 180) % 360 - 180

# %%
# Stats
print(f"Ch1 voltage:  mean={V_1f.mean():.3f} V,  std={V_1f.std():.4f} V  "
      f"({V_1f.std()/V_1f.mean()*100:.2f}%)")
print(f"Ch4 current:  mean={I_1f.mean()*1e3:.2f} mA,  std={I_1f.std()*1e3:.3f} mA  "
      f"({I_1f.std()/I_1f.mean()*100:.2f}%)")
print(f"|Z|:          mean={Z_mag.mean():.1f} ohm,  std={Z_mag.std():.2f} ohm  "
      f"({Z_mag.std()/Z_mag.mean()*100:.2f}%)")
print(f"V-I phase:    mean={phase_VI.mean():.2f} deg,  std={phase_VI.std():.3f} deg")

# Drift: first 50 vs last 50 points
n_check = 50
print(f"\nDrift (first {n_check} vs last {n_check} points):")
for name, arr, unit in [("V", V_1f, "V"), ("I", I_1f * 1e3, "mA"),
                         ("|Z|", Z_mag, "ohm"), ("phase", phase_VI, "deg")]:
    early = arr[:n_check].mean()
    late = arr[-n_check:].mean()
    shift = late - early
    if name == "phase":
        print(f"  {name:>6}: {early:.4f} -> {late:.4f} {unit}  (shift = {shift:+.4f} deg)")
    else:
        print(f"  {name:>6}: {early:.4f} -> {late:.4f} {unit}  "
              f"(shift = {shift:+.4f}, {shift/early*100:+.2f}%)")

# %%
# Plot
fig, axes = plt.subplots(4, 1, figsize=figsize_for_layout(4, 1, sharex=True), sharex=True)

axes[0].plot(pts, V_1f, ".", markersize=1, color="C0")
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title(f"Electrical stability --- {stem}")
axes[0].grid(True, alpha=0.3)

axes[1].plot(pts, I_1f * 1e3, ".", markersize=1, color="C1")
axes[1].set_ylabel("Current (mA)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(pts, Z_mag, ".", markersize=1, color="C2")
axes[2].set_ylabel(r"$|Z|$ ($\Omega$)")
axes[2].grid(True, alpha=0.3)

axes[3].plot(pts, phase_VI, ".", markersize=1, color="C3")
axes[3].set_ylabel("V--I phase (deg)")
axes[3].set_xlabel("Scan point index")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / f"thermal_drift_{stem}.png"
plt.savefig(output_path, dpi=FIG_DPI)
plt.close()
print(f"\nSaved: {output_path}")
