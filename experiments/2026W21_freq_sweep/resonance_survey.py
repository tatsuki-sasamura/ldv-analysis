# %%
"""Resonance-frequency survey across three remount campaigns on the
SAME microchannel chip (March / April / May 2026).

Datasets are organised by *campaign* (W10 / W16 / W21), not by chip —
A and C and D below are all the same physical sample, mounted and
de-mounted three times.  The script tracks how the 1f resonance
frequency and peak pressure shift between mounts.

For each dataset, runs ``load_or_compute`` per file, fits a ``|sin(pi y / W)|``
mode shape to the 1f pressure, and writes a per-dataset summary
(f_mhz, p0_kpa, r2) as a ``.npz`` cache.  Plots are regenerated from
the summaries — re-running just the plotting block is near-instantaneous.

Datasets covered (12 total; trimmed 2026-05-21 to comparable sweeps):

  W10  (TDMS v1, Mar 6 2026, post-flush, "stepA" filenames)
     - test5_*  fine 10 Vpp,   ~57 freqs @ 1 kHz steps (continuous)
     - test9_*  narrow 25 Vpp, ~21 freqs @ 1 kHz steps (burst)

  W16  (TDMS v1, Apr 13 2026)
     - W16test2_20Vpp*kHz      ~61 freqs @ 1 kHz steps (continuous)

  W21  (HDF5 v2, May 18 2026, burst, "sample" filenames)
     - sample_101x1_fsweep_{10,20,30,40}Vpp        4 × 141 wide-range freqs
     - sample_101x1_fsweep_narrow_{10,20,30,40}Vpp 5 × 41 narrow-range freqs
       (10 Vpp ×2 — with-splitter + no-splitter calibration pair)

Dropped from earlier versions (kept here for provenance): stepA pre-flush /
test3 coarse, a different chip (newchip, Mar 20), an 11×1 sparse grid, wide
50 Vpp (nonlinear peak shift), narrow 50/60 Vpp, and an air-filled run.

Outputs (all under ``experiments/2026W21_freq_sweep/output/resonance_survey/``):
  * summaries/<dataset_id>.npz                 -- per-dataset summary cache
  * fft_cache/<dataset_id>/_fft_cache_*.npz    -- per-file FFT cache
  * W10.png, W16.png, W21_wide.png, W21_narrow.png  -- per-campaign sweeps
  * cross_campaign_summary.png                  -- one scatter point per dataset
  * summary.txt                                  -- tabular dump
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ldv_analysis.config import (  # noqa: E402
    CHANNEL_WIDTH, FIG_DPI, figsize_for_layout,
)
from ldv_analysis.fft_cache import load_or_compute  # noqa: E402
from ldv_analysis.filters import make_valid_mask  # noqa: E402
from ldv_analysis.mode_fit import fit_mode_1f  # noqa: E402

DATA_ROOT = Path(r"D:/OneDrive - Lund University/Data")
OUT_DIR = ROOT / "experiments" / "2026W21_freq_sweep" / "output" / "resonance_survey"
SUMMARY_DIR = OUT_DIR / "summaries"
CACHE_ROOT = OUT_DIR / "fft_cache"
FIG_DIR = OUT_DIR


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """One frequency-sweep ready to analyze."""
    campaign: str             # 'W10' / 'W16' / 'W21' — same chip, three mounts
    dataset_id: str           # stable filename-safe id
    label: str                # short display label
    drive_label: str          # e.g. "10 Vpp" or "30 Vpp (~64 Vpp at PZT)"
    files: list[Path] = field(default_factory=list)
    fmt: str = "tdms"         # 'tdms' or 'hdf5'
    notes: str = ""           # e.g. "with splitter" — annotated in plot legend


def _build_datasets() -> list[Dataset]:
    out: list[Dataset] = []

    # --- W10 campaign (TDMS v1, Mar 6 2026, post-flush state) ---
    # Trimmed 2026-05-21: keep only post-flush fine (test5) and narrow (test9).
    # Pre-flush (stepA_sweep_*) and coarse (test3) dropped — different chip
    # state and too sparse / wrong range for the comparison.
    d = DATA_ROOT / "20260306experimentA"
    for prefix, label, drive, did in [
        ("test5", "Post-flush fine (Mar 6, 10 Vpp)",  "10 Vpp", "stepA_test5_fine10V"),
        ("test9", "Post-flush narrow (Mar 6, 25 Vpp)", "25 Vpp", "stepA_test9_narrow25V"),
    ]:
        pat = re.compile(rf"^{prefix}_(\d+)\.tdms$")
        fs = sorted([p for p in d.iterdir() if pat.match(p.name)],
                    key=lambda p: int(pat.match(p.name).group(1)))
        # test3, test5 include the 2f files (3700+ kHz) we want only 1f
        # — keep f in [1.8, 2.1] MHz for the 1f sweeps.
        fs = [p for p in fs if 1800 <= int(pat.match(p.name).group(1)) <= 2100]
        if fs:
            out.append(Dataset(
                campaign="W10", dataset_id=did, label=label,
                drive_label=drive, files=fs, fmt="tdms",
            ))

    # --- W16 campaign (TDMS v1, Apr 13 2026) ---
    d = DATA_ROOT / "260413_ldv"
    pat = re.compile(r"^W16test2_20Vpp(\d{4})kHz_5m_s_max\.tdms$")
    fs = sorted([p for p in d.iterdir() if pat.match(p.name)],
                key=lambda p: int(pat.match(p.name).group(1)))
    if fs:
        out.append(Dataset(
            campaign="W16", dataset_id="W16_test2",
            label="W16 test2 (Apr 13, 20 Vpp)", drive_label="20 Vpp",
            files=fs, fmt="tdms",
        ))

    # --- W21 campaign (HDF5 v2, May 18 2026) ---
    # Trimmed 2026-05-21: keep wide + narrow ladders up to (filename) 40 Vpp.
    # Dropped: 11×1 grid (sparse sampling), wide 50 Vpp (anomalous +8 kHz peak
    # shift), wide 20 Vpp AIR-filled (different physics), narrow 50/60 Vpp
    # (60 V nonlinear peak shift; 50 paired).
    #
    # *** Vpp relabel ***: W21 HDF5 attribute / filename Vpp is the AFG-set
    # amplitude in Vp (peak), not Vpp.  Actual drive Vpp = 2 × filename value
    # — so what the filename calls "10 Vpp" is really 20 Vpp into the amp.
    # ``dataset_id`` keeps the filename digit (cache-key stability); ``label``
    # and ``drive_label`` show the corrected Vpp.
    W21_VPP_FACTOR = 2
    out_root = DATA_ROOT / "output"

    # Wide ladder (filename 10/20/30/40 Vpp → true 20/40/60/80 Vpp)
    for fn_vpp, dirname in [
        (10, "sample_101x1_fsweep_10Vpp_20260518_173341"),
        (20, "sample_101x1_fsweep_20Vpp_20260518_185107"),
        (30, "sample_101x1_fsweep_30Vpp_20260518_191818"),
        (40, "sample_101x1_fsweep_40Vpp_20260518_195418"),
    ]:
        d = out_root / dirname
        if not d.exists():
            continue
        true_vpp = fn_vpp * W21_VPP_FACTOR
        fs = sorted(p for p in d.glob("*.h5") if not p.name.endswith(".inprogress"))
        out.append(Dataset(
            campaign="W21", dataset_id=f"sample_wide_{fn_vpp}V",
            label=f"Wide {true_vpp} Vpp", drive_label=f"{true_vpp} Vpp",
            files=fs, fmt="hdf5",
        ))

    # Narrow ladder (filename 10/20/30/40 Vpp → true 20/40/60/80 Vpp)
    for fn_vpp, dirname, note in [
        (40, "sample_101x1_fsweep_narrow_40Vpp_20260518_210751", ""),
        (30, "sample_101x1_fsweep_narrow_30Vpp_20260518_211632", ""),
        (20, "sample_101x1_fsweep_narrow_20Vpp_20260518_212516", ""),
        (10, "sample_101x1_fsweep_narrow_10Vpp_20260518_213357", "with splitter"),
        (10, "sample_101x1_fsweep_narrow_10Vpp_20260518_214800", "no splitter (calibration)"),
    ]:
        d = out_root / dirname
        if not d.exists():
            continue
        true_vpp = fn_vpp * W21_VPP_FACTOR
        fs = sorted(p for p in d.glob("*.h5") if not p.name.endswith(".inprogress"))
        out.append(Dataset(
            campaign="W21", dataset_id=f"sample_narrow_{fn_vpp}V_{dirname.split('_')[-1]}",
            label=f"Narrow {true_vpp} Vpp" + (f" ({note})" if note else ""),
            drive_label=f"{true_vpp} Vpp", files=fs, fmt="hdf5", notes=note,
        ))

    return out


# ---------------------------------------------------------------------------
# Per-dataset analysis with summary cache
# ---------------------------------------------------------------------------

def _drive_freq_of(path: Path, fmt: str) -> float:
    if fmt == "hdf5":
        with h5py.File(path, "r") as f:
            return float(f.attrs["drive_frequency_hz_nominal"])
    m = re.search(r"(\d{4})", path.stem)
    if not m:
        raise ValueError(f"cannot parse kHz from {path.name}")
    return int(m.group(1)) * 1e3


def _analyze_dataset(ds: Dataset) -> dict:
    """Return dict(f_mhz, p0_kpa, r2). Cached in OUT_DIR/summaries/<id>.npz."""
    summary_path = SUMMARY_DIR / f"{ds.dataset_id}.npz"
    if summary_path.exists():
        try:
            d = np.load(summary_path)
            return dict(f_mhz=d["f_mhz"], p0_kpa=d["p0_kpa"], r2=d["r2"])
        except Exception:  # noqa: BLE001 — rerun if cache unreadable
            pass

    cache_dir = CACHE_ROOT / ds.dataset_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {ds.dataset_id}: {len(ds.files)} files ({ds.label}) ===")
    rows: list[tuple[float, float, float]] = []
    for i, p in enumerate(ds.files, 1):
        try:
            f_nom = _drive_freq_of(p, ds.fmt)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        try:
            c = load_or_compute(p, cache_dir, velocity_scale=None)
        except Exception as e:  # noqa: BLE001
            print(f"  FFT error {p.name}: {e}")
            continue
        pos = np.asarray(c["pos_x"])
        V = np.asarray(c["voltage_1f"])
        rssi = np.asarray(c["rssi"]) if "rssi" in c.files else None
        valid = make_valid_mask(V, rssi)
        if valid.sum() < 3:
            print(f"  {f_nom/1e3:.1f} kHz: only {valid.sum()} valid pts -- skip")
            continue
        P_abs = np.asarray(c["pressure_1f"])
        P_phase = np.asarray(c["phase_1f"])
        P_cmplx = P_abs * np.exp(1j * np.radians(P_phase))
        res = fit_mode_1f(pos[valid], P_cmplx[valid], CHANNEL_WIDTH)
        rows.append((float(c["f_drive"]) / 1e6,
                     float(abs(res.p0)) / 1e3,
                     float(res.r2)))
        if i % 20 == 0:
            print(f"  {i}/{len(ds.files)}  f={rows[-1][0]:.4f} MHz, "
                  f"p0={rows[-1][1]:.0f} kPa")

    if not rows:
        return dict(f_mhz=np.array([]), p0_kpa=np.array([]), r2=np.array([]))
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows)
    out = dict(f_mhz=arr[:, 0], p0_kpa=arr[:, 1], r2=arr[:, 2])
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(summary_path, **out)
    print(f"  Saved summary -> {summary_path.name}  ({len(arr)} pts)")
    return out


# ---------------------------------------------------------------------------
# Plotting (one figure per campaign + cross-campaign drift summary)
# ---------------------------------------------------------------------------

def _plot_campaign(datasets: list[tuple[Dataset, dict]], campaign_title: str,
                   fname: str, colors=None) -> None:
    """One landscape figure: every same-campaign sweep, normalized + absolute panels."""
    if not datasets:
        return
    fw, fh = 9.0, 5.0
    fig, (ax_abs, ax_norm) = plt.subplots(
        2, 1, figsize=(fw, fh), sharex=True,
    )

    cmap = plt.get_cmap("viridis")
    if colors is None:
        n = len(datasets)
        colors = [cmap(0.1 + 0.75 * i / max(n - 1, 1)) for i in range(n)]

    for (ds, d), c in zip(datasets, colors):
        f = d["f_mhz"]
        p = d["p0_kpa"]
        if f.size == 0:
            continue
        i_pk = int(np.argmax(p))
        f_pk, p_pk, r2_pk = f[i_pk], p[i_pk], d["r2"][i_pk]
        lbl = f"{ds.label}  →  {f_pk:.4f} MHz, {p_pk:.0f} kPa  (R²={r2_pk:.2f})"
        ax_abs.plot(f, p, marker="o", markersize=3.5, linewidth=1.2,
                    color=c, label=lbl)
        ax_norm.plot(f, p / p.max(), marker="o", markersize=3.5,
                     linewidth=1.2, color=c)
        ax_norm.axvline(f_pk, color=c, linestyle=":", linewidth=0.7, alpha=0.5)

    ax_abs.set_ylabel(r"$P_{1f}$ peak [kPa]")
    ax_abs.set_title(f"{campaign_title} — resonance sweeps")
    ax_abs.legend(fontsize=7, loc="upper right", frameon=True,
                  facecolor="white", framealpha=0.9)
    ax_abs.grid(True, alpha=0.3)
    ax_abs.set_ylim(bottom=0)

    ax_norm.set_xlabel("Drive frequency [MHz]")
    ax_norm.set_ylabel(r"$P_{1f}$ / max")
    ax_norm.grid(True, alpha=0.3)
    ax_norm.set_ylim(0, 1.1)

    plt.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out}")


def _short_annot(ds: Dataset) -> str:
    """Compact annotation for the cross-campaign plot.

    The cross-campaign plot tries to show every dataset in one row per
    campaign; the full ``ds.label`` is too verbose for this density.
    """
    note = ""
    if "with splitter" in ds.notes:
        note = " (splitter)"
    elif "no splitter" in ds.notes:
        note = " (cal)"
    if "wide" in ds.dataset_id:
        return f"W {ds.drive_label}{note}"
    if "narrow" in ds.dataset_id:
        return f"N {ds.drive_label}{note}"
    if "test5" in ds.dataset_id:
        return f"fine {ds.drive_label}"
    if "test9" in ds.dataset_id:
        return f"narrow {ds.drive_label}"
    return f"{ds.drive_label}"


def _plot_cross_campaign(all_results: list[tuple[Dataset, dict]]) -> None:
    """Scatter: peak frequency per dataset, grouped by campaign (one row each).

    All datasets are the same physical chip; the rows visualise mount-to-mount
    drift between the three remount campaigns (W10 → W16 → W21).  Datasets
    that share a peak frequency (within 0.5 kHz) have their annotations
    stacked vertically below the marker to keep labels legible.
    """
    campaigns = list(dict.fromkeys(ds.campaign for ds, _ in all_results))
    y_of_campaign = {c: i for i, c in enumerate(campaigns)}
    campaign_color = {"W10": "C0", "W16": "C2", "W21": "C3"}

    # Collect (campaign, f_pk, p_pk, short_label), sorted within each campaign
    pts_by_campaign: dict[str, list[tuple[float, float, str]]] = {c: [] for c in campaigns}
    for ds, d in all_results:
        if d["f_mhz"].size == 0:
            continue
        i_pk = int(np.argmax(d["p0_kpa"]))
        pts_by_campaign[ds.campaign].append((
            float(d["f_mhz"][i_pk]),
            float(d["p0_kpa"][i_pk]),
            _short_annot(ds),
        ))
    for camp in pts_by_campaign:
        pts_by_campaign[camp].sort(key=lambda t: t[0])

    n_camp = len(campaigns)
    fig, ax = plt.subplots(figsize=(12, 2.0 + 1.5 * n_camp))

    F_GROUP_TOL = 5e-4   # 0.5 kHz: closer than this → stack labels
    LABEL_FONT = 14      # annotation font size
    STACK_PT = 18        # vertical point spacing between stacked labels

    for camp, pts in pts_by_campaign.items():
        y = y_of_campaign[camp]
        # Group consecutive points (sorted by f_pk) within F_GROUP_TOL
        groups: list[list[tuple[float, float, str]]] = []
        for fp, pp, lb in pts:
            if groups and abs(fp - groups[-1][-1][0]) < F_GROUP_TOL:
                groups[-1].append((fp, pp, lb))
            else:
                groups.append([(fp, pp, lb)])
        for grp in groups:
            for stack_idx, (fp, pp, lb) in enumerate(grp):
                ms = 6 + 0.6 * np.log10(max(pp, 10))
                ax.plot(fp, y, "o", color=campaign_color.get(camp, "0.5"),
                        markersize=ms, alpha=0.75)
                ax.annotate(lb, (fp, y),
                            xytext=(0, -18 - STACK_PT * stack_idx),
                            textcoords="offset points",
                            fontsize=LABEL_FONT, ha="center", alpha=0.9)

    ax.set_yticks(list(y_of_campaign.values()))
    ax.set_yticklabels(list(y_of_campaign.keys()), fontsize=14)
    # Extra room below the bottom row for stacked annotations
    ax.set_ylim(n_camp - 1 + 0.9, -0.7)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(r"Peak $P_{1f}$ frequency [MHz]", fontsize=13)
    ax.set_title(r"Same chip across three remounts: peak $P_{1f}$ frequency "
                 r"per dataset (marker size $\propto$ log $P_{1f}$).  "
                 r"W = wide ladder, N = narrow ladder.", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "cross_campaign_summary.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = _build_datasets()
    print(f"Built {len(datasets)} datasets across "
          f"{len(set(d.campaign for d in datasets))} campaigns:")
    for ds in datasets:
        print(f"  {ds.campaign:6s}  {ds.dataset_id:32s}  "
              f"{len(ds.files):>4d} files  ({ds.label})")

    # Analyze (cached per dataset)
    results = [(ds, _analyze_dataset(ds)) for ds in datasets]

    # Per-campaign plots
    by_campaign: dict[str, list[tuple[Dataset, dict]]] = {}
    for ds, d in results:
        by_campaign.setdefault(ds.campaign, []).append((ds, d))

    if "W10" in by_campaign:
        _plot_campaign(by_campaign["W10"], "W10 (Mar 6 2026)", "W10.png")
    if "W16" in by_campaign:
        _plot_campaign(by_campaign["W16"], "W16 (Apr 13 2026)", "W16.png")
    if "W21" in by_campaign:
        # W21 has wide-range and narrow-range ladders — plot separately
        w21 = by_campaign["W21"]
        wide   = [t for t in w21 if "wide"   in t[0].dataset_id]
        narrow = [t for t in w21 if "narrow" in t[0].dataset_id]
        _plot_campaign(wide,   "W21 (May 18 2026) — wide ladder",   "W21_wide.png")
        _plot_campaign(narrow, "W21 (May 18 2026) — narrow ladder", "W21_narrow.png")

    # Cross-campaign drift summary
    _plot_cross_campaign(results)

    # Tabular summary
    lines = []
    lines.append(f"{'campaign':10s}  {'dataset_id':32s}  {'n':>5s}  "
                 f"{'peak f (MHz)':>13s}  {'peak P (kPa)':>13s}  {'R^2':>5s}")
    lines.append("-" * 90)
    for ds, d in results:
        if d["f_mhz"].size == 0:
            lines.append(f"{ds.campaign:10s}  {ds.dataset_id:32s}  "
                         f"{'0':>5s}  -- no data --")
            continue
        i = int(np.argmax(d["p0_kpa"]))
        lines.append(f"{ds.campaign:10s}  {ds.dataset_id:32s}  "
                     f"{len(d['f_mhz']):>5d}  {d['f_mhz'][i]:>13.4f}  "
                     f"{d['p0_kpa'][i]:>13.1f}  {d['r2'][i]:>5.2f}")
    summary = "\n".join(lines)
    print()
    print(summary)
    (OUT_DIR / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
