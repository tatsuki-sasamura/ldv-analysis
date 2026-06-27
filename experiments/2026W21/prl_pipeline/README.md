# PRL figure pipeline (W21)

Minimal scripts that turn the W21 voltage cascade and frequency sweeps
into the figures (Fig 1, 2, 3, S1, S2) required by
`nonlinearphysics-manuscript/prl/analysis_contract.md`.

Replaces the 929-line enterprise plan in `PLAN.md` — same outputs, one
researcher, one week.

## Files

| File | Role |
|---|---|
| `conventions.py` | Single source of truth for §3 contract thresholds (LOW_DRIVE_P2_OVER_P1, BETA, water constants, adoption thresholds). Re-exports from `ldv_analysis.config`. |
| `_data.py` | Three shared loaders: `low_drive_mask`, `load_ladder`, `compute_pin_time` (the only genuinely new low-level primitive). |
| `fig1.py` | Spatial maps of P_1f, P_2f, P_3f at 120 Vpp with mode-shape fits. |
| `fig2.py` | 4-panel: (a) cascade scaling, (b) K_exp slope, (c) E_1 vs P_in, (d) deviation. **Decisional** (panel c). |
| `fig3.py` | Force reconstruction + central-stiffness ratio R_κ. **Decisional**. |
| `figS1.py` | Ruleouts: drive purity, per-harmonic SNR, P_in time-vs-spectral, summary table. |
| `figS2.py` | One-pole fit on 1f sweep, two-pole on 2f sweep. `--selftest` for synthetic recovery. |
| `run_all.py` | Runs every step in order. `--skip figS2` available for fast iteration. |
| `handover.py` | Copies `fig*.{png,pdf,npz,json}` to `$MANUSCRIPT_DIR/prl/figures/`. |

## Prerequisites

Three things must exist before this pipeline can run:

1. The repo's `.venv` Python with `numpy`, `scipy`, `matplotlib`, `h5py`,
   `scienceplots` installed (project standard).
2. Cascade FFT caches under
   `experiments/2026W21/output/sample_101x21_fsweep_peak_*Vpp_*/fft_cache/*.npz`,
   built by running `experiments/2026W21/harmonic_ladder.py` once. Each
   cascade scan has 11 frequency caches; harmonic_ladder builds them
   all on first run.
3. `experiments/2026W21/output/harmonic_ladder.npz` — produced by the
   same script.

Optional (for Fig S1 panel a): the AFG drive-purity CSV produced by
`experiments/2026W21/drive_purity_b1.py`, dropped at
`experiments/2026W21/output/drive_purity_b1/drive_purity_b1.csv`.

## Usage

```bash
# Regenerate everything (~ a few minutes; figS2 cache build hours
# the first time, instant on warm caches)
.venv/Scripts/python experiments/2026W21/prl_pipeline/run_all.py

# Skip the slow figS2 cache build during iteration
.venv/Scripts/python experiments/2026W21/prl_pipeline/run_all.py --skip figS2

# Run only one figure
.venv/Scripts/python experiments/2026W21/prl_pipeline/run_all.py --only fig3

# Verify figS2 fit code with synthetic data (instant, no W21 reads)
.venv/Scripts/python experiments/2026W21/prl_pipeline/figS2.py --selftest

# Copy outputs to the manuscript repo
.venv/Scripts/python experiments/2026W21/prl_pipeline/handover.py
```

Outputs land in `experiments/2026W21/output/prl_pipeline/`:
- `figN.png`, `figN.pdf` — the figures themselves
- `figN.npz` — every array needed to regenerate the figure
- `figN.json` — human-readable decision-criterion summary
  (only for fig2, fig3, figS1, figS2)

## Conventions

All thresholds and constants live in `conventions.py`. Editing them
requires regenerating every dependent figure. Lint target: any literal
threshold appearing in `fig*.py` outside the `import` block is a bug
(use `git grep -nE '0\.10|0\.1\b' experiments/2026W21/prl_pipeline/fig*.py`).

## Decision criteria (contract Outcome Matrix)

Decisional figures print their decision to stdout and store
`decision_pass` (bool) + `decision_msg` (str) in their NPZ. A failing
decision is **not** a code failure — it triggers the contract's
fallback wording for that figure (chosen manually by inspecting the
printed result).

- **Fig 2c**: `|E_1/E_1^lin − 1| > 3·σ` at top drive
- **Fig 3**: `R_κ < 1` and `(1 − R_κ) > 3·σ` at top drive

Descriptive figures (1, 2a, 2b magnitude, S1, S2 magnitude) verify
numerically against the pre-pipeline `prl_draft_fig*.npz` outputs.

## R2 risk

R2 = pressure-calibration audit of complex `C_{p←d}(f)`. Status: open.
Affects:

- Fig 2b *sign* of K_exp (magnitude unaffected)
- Fig S2 complex residues / two-pole phase decomposition

If R2 fails, magnitude-only fallback figures stay publishable.
