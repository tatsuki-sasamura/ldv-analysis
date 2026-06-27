# `ldv-analysis` PRL pipeline — status

Implementation status: **complete on the LA side** (2026-06-27). Every
contract-required figure (Fig. 1, 2, 3, S1, S2 + the diagnostic
`figS1_decoder_check`) is generated, contract conventions are enforced
through `conventions.py`, and decision criteria are written as
`figN.json` sidecars.

For usage see `README.md`. For the figure-centered contract this
pipeline implements see
`nonlinearphysics-manuscript/prl/analysis_contract.md`.

This file replaces the 925-line implementation plan that described the
to-be-built pipeline. The historical version lives in the git log
(`51f0d6d`); the current document is short on purpose because most of
it is now code, not prose.

---

## Remaining LA work

Two small items, both unblocked the moment `harmonic_model` writes its
mechanism-guide overlay file.

1. **HM mechanism-guide overlay (cascade curves, n = 1…5).**
   `fig2.py` and/or `figS2.py` should read a small `figS2_overlay.npz`
   written by `harmonic_model` and overlay the predicted cascade
   curves on top of the measured cascade points. Expected payload:

   ```text
   v_drive_vpp        (M,)        # drive-voltage axis (model grid)
   p_nf_pred_kpa      (M, 5)      # predicted |P_nf| for n = 1..5
   low_drive_anchor   dict-like   # (V, P_1f) anchor used to normalize
   q_n_used           (5,)        # Q_n fed to the model (provenance)
   model_role_note    str         # verbatim copy of HM's scope statement
   ```

   The overlay is descriptive — the contract permits it to be moved to
   Fig. S2 only if the dashed lines look misleading on Fig. 2. If the
   file is absent, the figure renders the cascade points + `K_exp` fit
   alone. Add a small caption hook so the `model_role_note` can be
   rendered next to the overlay (so the reader sees the scope
   limitation in the same panel).

   **This supersedes the earlier "single-pole reference line"
   payload** (`m_axis`, `k2_singlepole`) that an earlier draft of this
   plan specified. The richer five-harmonic cascade overlay matches
   the contract's "mechanism guide" framing better and uses the
   existing `harmonic_model` multi-harmonic solver instead of a single
   inline arithmetic.

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
| `figS2.npz` | `q_eff_1`, `q_eff_2`, `cascade_f_op_hz`, `two_pole_f_a`, `two_pole_f_b` | Parameters for the multi-harmonic-solver cascade prediction. $Q_3$, $Q_4$, $Q_5$ are not exported; HM reuses $Q_2$ as a documented mechanism-guide default per its own plan §4. |
| `fig3.npz` | `e_n_over_e1` (N×5), `adopted` (N×5), `pzt_vpp` (N,) | Fig. S3 dimensionless force-shape, equilibria, central stiffness, local-linearity widths, $R_t$ trajectory map. |

The rest of every `figN.npz` exists for LA's own plot regeneration and
contract decisions; HM does not need it.

---

## Known phase-convention caveat (LDV side)

The `fft_cache.py` pressure inversion stores magnitudes correctly but
the stored "phase" is the velocity-vs-`V_1f` phase, not the
pressure-vs-`V_1f` phase. The conversion in
`src/ldv_analysis/config.py:velocity_to_pressure` returns the real
scalar $-1 / (2\pi f \cdot H \cdot dn/dp)$ — physically the pressure
phasor relative to the velocity phasor needs an additional $1 / (i\omega)$
factor (since $v_\mathrm{apparent} \propto dp/dt$ in phasor form
implies $V = i\omega P \cdot \text{const}$, so $P = V / (i\omega) \cdot
\text{const}$).

The net effect is a **global $-90°$ phase rotation** between every
stored complex phasor and the true pressure phasor. What is affected
vs. what is not:

| Quantity | Affected by the $-90°$ offset? |
|---|---|
| All magnitudes (Fig. 1 maps, Fig. 2a scaling, $K_\mathrm{exp}$, $E_n$, Fig. 3 force shape) | No |
| $Q_n$ from $|H|^2$ linewidth fit | No (linewidth invariant under rigid rotation) |
| Two-pole residue *magnitudes* | No |
| Two-pole residue *absolute phases* | Yes — rotated by $-90°$ |
| Stored `phase_{n}f` field in caches | Yes — stores velocity-vs-V_1f rather than pressure-vs-V_1f |

Nothing the PRL publishes currently consumes absolute complex-phase
information. The contract and the manuscript do not depend on this
fix being in. **Recommended treatment for now:** document the offset
explicitly (this section + a parallel note in the manuscript contract
if needed); fix the conversion only if a future paper needs absolute
phase. A code-level fix is straightforward but requires regenerating
`figS2.npz` and any cascade caches that store `phase_{n}f`.

---

## Open conventions (was §11 in the original plan)

| ID | Status |
|---|---|
| R0 ([HYP-F2b] tag) | **resolved** (manuscript commit `f05f5b9`, 2026-06-26). |
| R1 (dn/dp(water) cross-repo) | **resolved** (manuscript `651fafa` + ldv-analysis `3adeb35`, 2026-06-26: IAPWS-derived `1.45 × 10⁻¹⁰ Pa⁻¹`). |
| R2 (pressure-calibration audit) | **documentation note** (`54e4dcb`). Pipeline is Ch2-only; Polytec velocity-decoder phase response is trusted per vendor spec. No published number is gated. |
| R3 (4f/5f adoption sanity) | **effectively resolved** by the actual data. `fig3.json` shows `ever_adopted_n = [T,T,T,T,T]`, `global_series_depth = 5`; the contract's `E_foc/E_1 = 1 − 4 E_2/E_1 + 9 E_3/E_1 − 16 E_4/E_1 + 25 E_5/E_1` is the right form, no truncation. |
| R4 (logged water temperature) | **open, low priority**. Contract permits the logged-or-nominal fallback; no published claim depends on the difference. |
| R5 (phase-convention offset) | **documentation note** (this PLAN, "Known phase-convention caveat"). $-90°$ global rotation in stored complex phasors. Magnitudes and $Q_n$ unaffected; no published claim depends on absolute complex phase. Fix deferred until a future paper requires it. |

---

## Architecture summary

```text
W21 cascade + frequency sweeps
        │
        ▼
LA pipeline (conventions, _data, figN scripts, run_all)
        │
        ├─ figS2.npz  (Q_n, f_op, pole locations) ───┐
        ├─ fig3.npz   (e_n_over_e1, adopted, vpp)────┤
        │                                            ▼
        │                                    harmonic_model
        │                          (multi-harmonic solver, β = 3.5,
        │                           low-drive normalization, n = 1..5)
        │                                            │
        ├──────────────────────── figS2_overlay.npz ─┘
        │                            ┌── figS3.{pdf,png,npz} ──→ manuscript/prl/figures/
        ▼                            ▼
fig{1,2,3,S1,S2}.{pdf,png,npz}  + cascade overlay on Fig 2 / Fig S2
        │
        ▼
$MANUSCRIPT_DIR/prl/figures/  (via handover.py)
```

HM does not import `ldv-analysis`. Communication is via the npz files
only.
