# `ldv-analysis` PRL pipeline — status

Implementation status: **complete on the LA side** (2026-06-27). Every
contract-required figure (Fig. 1, 2, 3, S1, S2 + the diagnostic
`figS1_decoder_check`) is generated, contract conventions are enforced
through `conventions.py`, and decision criteria are written as
`figN.json` sidecars.

For usage see `README.md`. For the figure-centered contract this
pipeline implements see
`nonlinearphysics-manuscript/prl/analysis_contract.md`.

This file replaces the 928-line implementation plan that described the
to-be-built pipeline. The historical version lives in the git log
(`51f0d6d`); the current document is short on purpose because most of
it is now code, not prose.

---

## Remaining LA work

Two small items, both unblocked the moment `harmonic_model` writes its
mechanism-guide overlay file.

1. **HM mechanism-guide overlay.** `fig2.py` and/or `figS2.py` should
   read a small `figS2_overlay.npz` written by `harmonic_model` and add
   the contained curve as a dashed reference line on Fig. 2b (Coppens
   slope) and/or Fig. S2. Expected payload:
   ```
   m_axis            (M,)        # Mach-number axis used by fig2.py
   k2_singlepole     (M,)        # (β/4) · Q_2 · cosθ · M
   ```
   The overlay is optional: if the file is absent or its M-axis doesn't
   match, the figure renders the cascade points and `K_exp` fit alone.

2. **figS3 in handover.** Fig. S3 is `harmonic_model`'s deliverable,
   not LA's. If HM places its outputs under
   `experiments/2026W21/output/prl_pipeline/` (or any path
   `handover.py` knows about), `handover.py` can copy them to
   `$MANUSCRIPT_DIR/prl/figures/` in the same pass as the LA figures.
   Adding `"figS3"` to the `STEMS` tuple is the entire change required.

Neither item is a hard dependency on HM existing yet — both fail soft
(missing file → no overlay; missing stem → nothing to copy).

---

## What LA delivers to HM

Two files, by the existing `np.load(..., allow_pickle=False)` interface.

| File | Fields HM consumes | Purpose |
|---|---|---|
| `figS2.npz` | `q_eff_1`, `q_eff_2`, `cascade_f_op_hz`, `two_pole_f_a`, `two_pole_f_b` | Single-pole reference `K_2(M) = (β/4) Q_2 cosθ M` and any higher-mode mechanism-guide curves. `cosθ` derives from `f_op` and the 2f pole locations. |
| `fig3.npz` | `e_n_over_e1` (N×5) | Fig. S3 dimensionless force-shape, equilibria, central stiffness, local-linearity widths, R_t trajectory map. |

The rest of every `figN.npz` exists for LA's own plot regeneration and
contract decisions; HM does not need it.

---

## Open conventions (was §11 in the original plan)

| ID | Status |
|---|---|
| R0 ([HYP-F2b] tag) | **resolved** (manuscript commit `f05f5b9`, 2026-06-26). |
| R1 (dn/dp(water) cross-repo) | **resolved** (manuscript `651fafa` + ldv-analysis `3adeb35`, 2026-06-26: IAPWS-derived `1.45 × 10⁻¹⁰ Pa⁻¹`). |
| R2 (pressure-calibration audit) | **documentation note** (`54e4dcb`). Pipeline is Ch2-only; Polytec velocity-decoder phase response is trusted per vendor spec. No published number is gated. |
| R3 (4f/5f adoption sanity) | **effectively resolved** by the actual data. `fig3.json` shows `ever_adopted_n = [T,T,T,T,T]`, `global_series_depth = 5`; the contract's `E_foc/E_1 = 1 − 4 E_2/E_1 + 9 E_3/E_1 − 16 E_4/E_1 + 25 E_5/E_1` is the right form, no truncation. |
| R4 (logged water temperature) | **open, low priority**. Contract permits the logged-or-nominal fallback; no published claim depends on the difference. |

---

## Architecture summary

```text
W21 cascade + frequency sweeps
        │
        ▼
LA pipeline (conventions, _data, figN scripts, run_all)
        │
        ├─ figS2.npz  (Q_n, f_op, pole locations) ───┐
        ├─ fig3.npz   (e_n_over_e1)              ────┤
        │                                            ▼
        │                                    harmonic_model
        │                                            │
        ├──────────────────────── figS2_overlay.npz ─┘
        │                            ┌── figS3.{pdf,png,npz} ──→ manuscript/prl/figures/
        ▼                            ▼
fig{1,2,3,S1,S2}.{pdf,png,npz}  + overlay reference line on Fig 2 / Fig S2
        │
        ▼
$MANUSCRIPT_DIR/prl/figures/  (via handover.py)
```

HM does not import `ldv-analysis`. Communication is via the npz files
only.
