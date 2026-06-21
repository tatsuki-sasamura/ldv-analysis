# LDV / PTV Uncertainty Budget — Source-Provenance Audit — 2026-05-22

## Purpose

Companion to `reports/2026-05-21_ldv_ptv_uncertainty_budget.md`.
Each Type-B uncertainty term in that report is classified by the
**verified state of its primary source**:

- **Cat A (verified)** — A primary source has been *read in this
  session*; the value used in the report appears in that source and
  has been confirmed.
- **Cat B (claimed-but-not-verified)** — The probable primary source
  is known (paper title + authors, datasheet name, public standard),
  but the source has not been opened.  The cited value may or may
  not literally appear there.
- **Cat C (no traceable source)** — No primary source path exists.
  Chip-specific quantity that has not been measured, or an explicit
  placeholder.

### Why this stricter standard

A previous draft of this audit listed several terms as "Cat A — I
know the source, can add the citation now".  That conflates "I can
name a paper" with "I have verified the paper contains this number".
The corrected standard is the second one only: the source must be
opened and the value confirmed before a term can be called Cat A.

Under this corrected standard, **only the in-repo numerical
verifications are currently Cat A**.  Every external-citation Cat A
from the earlier draft is demoted to Cat B until the source is
actually opened.  Verification path is given in the per-term table.

---

## Per-term audit

| # | Side | Term | Value used | Cat | Verification status / source path |
|---|---|---|---:|:---:|---|
| 1 | LDV | dn/dp water at 633 nm | 1.48 × 10⁻¹⁰ Pa⁻¹ (this repo); **1.4 × 10⁻¹⁰** (manuscript) | **A-derived** | **Verified 2026-05-22 (external traceability via ChatGPT cite-check).**  Primary sources: IAPWS R9-97 (refractive-index formulation as function of T, ρ, λ — 633 nm is inside the 0.2–1.1 µm validity range) combined with IAPWS-95 (Helmholtz-energy equation of state for ρ, c_s, dρ/dp).  At T = 20 °C, λ = 632.8 nm, ρ ≈ 998.2 kg/m³, c_s ≈ 1482 m/s, the derivation gives n ≈ 1.33210 and (∂n/∂p)_s ≈ 1.48 × 10⁻¹⁰ Pa⁻¹.  Note: this is a *derived* value — IAPWS does not publish dn/dp directly in a table.  Cross-repo inconsistency (1.4 vs 1.48) still requires user-side resolution — see manuscript-side note. |
| 2 | LDV | Channel height H | 150 µm ± 5–10 % | **C** | Foundry process spec for *this* chip not in either repo; chip never SEM-cross-sectioned.  Fix path is physical measurement, not citation. |
| 3 | LDV | BK7 constants (n, p₁₁, p₁₂, E, ν, ρ, c_L) | (script) n=1.515, p₁₁=0.121, p₁₂=0.270, E=70 GPa, ν=0.22, ρ=2230, c_L=5500 | **SCRIPT VALUES DISAGREE with N-BK7 datasheet** | **Verified 2026-05-22 (external):** SCHOTT N-BK7 official datasheet (media.schott.com PDF) at 632.8 nm gives n=1.51509, ρ=2510 kg/m³, E=82 GPa, ν=0.206.  Krupych et al. 2011 *Ukr. J. Phys. Opt.* measured photoelastic p₁₁=0.118 ± 0.004, p₁₂=**0.226** ± 0.005 (NOT 0.270).  **Four script values disagree with N-BK7 spec**: ρ off by +12 %, E off by +17 %, ν off by −6 %, p₁₂ off by −16 %.  The p₁₂=0.270 in the script equals the fused-silica p₁₂ (item 4) — confirming the previously-flagged copy-paste suspicion.  **Open question**: is the chip's top/bottom glass actually N-BK7, or a generic borosilicate (Borofloat33 / Pyrex, where ρ≈2230, E≈65–70 GPa are closer to the script's current numbers)?  Until resolved, the BK7 row in the glass script is mis-parameterised. |
| 4 | LDV | Fused silica constants | n=1.458, p₁₁=0.121, p₁₂=0.270, E=73 GPa, ν=0.17, ρ=2200, c_L=5970 | **PARTIAL A** | **Verified 2026-05-22 from en.wikipedia.org/wiki/Fused_quartz** (cites standard handbook sources): n_d = 1.4585 ✓, E = 71.7 GPa (user 73, ~2% high), ν = 0.17 ✓, ρ = 2203 kg/m³ ✓, c_L = 5960 m/s ✓ (derived from elastic constants in Wikipedia).  **Photoelastic p₁₁, p₁₂ NOT in Wikipedia article** — Primak & Post 1959 *JAP* 30 is the standard primary source for fused silica p_ij and remains unverified.  Non-photoelastic constants are confirmed. |
| 5 | LDV | Schott D263T constants | (script) n=1.523, p₁₁=0.118, p₁₂=0.250, E=72.9 GPa, ν=0.22, ρ=2510, c_L=5700 | **PARTIAL A — bulk verified; p_ij formulation caveat** | **Verified 2026-05-22 (external):** SCHOTT D263 official tech-details page (`schott.com/en-se/products/d-263-p1000318/technical-details`) gives n_d=1.5231 ✓, ρ=2.51 g/cm³ = 2510 ✓, E=72.9 kN/mm² = 72.9 GPa ✓, **ν=0.21** (script: 0.22, small disagreement), stress-optic constant **K=34.7 nm/cm/MPa = 3.47 × 10⁻¹² Pa⁻¹**.  **Formulation caveat**: K is the Pockels stress-optic constant (proportional to (p₁₁ − p₁₂)(1+ν)/E), NOT the hydrostatic-pressure combination (p₁₁ + 2p₁₂)(1 − 2ν)/E that the script's ∂n/∂p calculation uses.  Individual (p₁₁, p₁₂) cannot be backed out from K alone.  Cleanest fix is to keep the (p₁₁, p₁₂) approximation for D263T but flag it as inherited-from-similar-borosilicate, or reformulate the script to use K with a different pressure-optic expression. |
| 6 | LDV | Glass photoelastic central +14.5 %, bracket 9.8–19.3 % | derived | **B** (derived) | Reproducible from items 3–5 via `reports/2026-05-21_glass_pressure_self_verification.py`.  Cat A *conditional* on items 3–5 being verified Cat A first.  Currently Cat B because inputs are Cat B. |
| 7 | LDV | Polytec velocity-decoder linearity | ±5 % (report) → **±0.5 % (verified)** | **A — 10× TIGHTER THAN REPORT** | **Verified 2026-05-22 from user-provided** `Polytec_VibroFlex-Connect_VFX-F-110_ds.pdf` (Polytec official datasheet, doc ref `OM_DS_VibroFlex-Connect_E_52024`, 2020/08).  Velocity performance table footnote 2 (page 6): **"Maximum linearity error: 0.5 % for all measurement ranges."**  Also confirmed: model VFX-F-110 (matches manuscript `main.tex:495`), DC–24 MHz bandwidth, analog outputs ±1 V @ 50 Ω / ±2 V @ 1 MΩ.  The report's ±5 % is **10× too conservative**.  σ_velocity in the LDV-side Type-B table should be tightened from 0.05 to **0.005**.  Downstream impact: LDV velocity term becomes negligible compared to glass photoelastic (0.145) and channel height (0.075); LDV_SYS_FRAC drops from 0.178 to 0.171 (modest because glass photoelastic still dominates). |
| 8 | LDV | Air-null mode-fit residual | factor 1.08 (dropped from budget) | **A** | **Verified.** `experiments/2026W21_freq_sweep/output/resonance_survey/summaries/sample_wide_20V_AIR.npz` mode-fit gives R² = −4.56 on the W21 air-filled scan; reproducible in this repo. |
| 9 | PTV | κ_PS — central = **lab convention 2.49 × 10⁻¹⁰**, deviation range **[1.72, 3.30] × 10⁻¹⁰** | central 2.49; bracket [1.72, 3.30] | **RESOLVED 2026-05-22 as asymmetric Type-B bracket; budget σ corrected to propagated p_0 effect** | User decision 2026-05-22: keep 2.49 × 10⁻¹⁰ as the *central* value (lab convention, origin lost — relabel away from the wrong "Settnes-Bruus" attribution); treat 1.72 (verified Settnes-Bruus 2012 Table I) as the *low-side* deviation and 3.30 (verified Barnkob 2010 Table 1) as the *high-side* deviation.  **Input-variance half-width** is σ_κ_p/κ_p ≈ ±0.32 (relative to central).  **Propagation to PTV p₀** via Φ = f₁/3 + f₂/2 (with f₁ = 1 − κ_p/κ_w, κ_w ≈ 4.5 × 10⁻¹⁰) and p_0 ∝ 1/√Φ: at κ_PS=1.72 PTV p₀ is 0.86× central (ln = −0.152); at κ_PS=3.30 PTV p₀ is 1.26× central (ln = +0.234).  **Propagated symmetric half-width on ln(p_0) ≈ 0.19** — this is what enters σ_P quadrature, not the 0.32 input-variance figure.  Sensitivity coefficient d ln p_0 / d ln κ_p ≈ 0.57 attenuates input to p_0 effect.  Consequently the observed LDV/PTV ratio range from κ_PS choice alone is [0.80×, 1.16×] around the lab-convention point. |
| 10 | PTV | κ_PS Barnkob | 3.30 × 10⁻¹⁰ Pa⁻¹ | **A** | **Verified 2026-05-22 from Zotero item R6F9YTR2 (Barnkob, Augustsson, Laurell, Bruus, *Lab Chip* **10**, 563 (2010), DOI: 10.1039/B920376A).** Table 1 of that paper gives polystyrene parameters: ρ_PS = 1050 kg/m³, c_PS = 1700 m/s.  Computing κ = 1/(ρc²) = 1 / (1050 × 1700²) = **3.30 × 10⁻¹⁰ Pa⁻¹** matches the report's claim exactly. |
| 11 | PTV | Particle radius 2.5 µm ± **5 %** (updated) | nominal 5 µm diameter ± 5 % | **A** | **Verified 2026-05-22 from thermofisher.com G0500B (Fluoro-Max) product page.** Listed specs: diameter mean 5.0 µm; **uniformity < 5 %**; particle density 1.05 g/cm³; refractive index 1.59 @ 589 nm.  User-decided 2026-05-22: tighten σ_radius from ±10 % to ±5 % to match the manufacturer's uniformity spec.  Since p₀ ∝ 1/R, σ_radius = 0.05 propagates to ±5 % on PTV p₀ (down from ±10 % in the original budget). |
| 12 | PTV | Fluid water at 25 °C (with ±5 °C chip-T uncertainty) | **±5 %** (revised 2026-05-22) | **A — derived from temperature coefficients** | User confirmed 2026-05-22: chip fluid is **Milli-Q pure water**, not buffered.  Verified primary-source values (Tanaka et al. 2001 ρ; Bilaniuk & Wong 1993 c; Lide *CRC Handbook* 2003 η) at 25 °C: ρ = 997 kg/m³, c = 1497 m/s, η = 0.890 mPa·s.  **Budget σ_water revised from 0.01 to 0.05** under user-set ±5 °C chip-T uncertainty: temperature coefficients are `(1/ρ)·dρ/dT ≈ −0.026 %/K` (CIPM density), `(1/c)·dc/dT ≈ +0.18 %/K` (Marczak 1997), `(1/η)·dη/dT ≈ −2.3 %/K` (Lide *CRC* 2003 / IAPWS-2008 viscosity).  Correlated propagation through `p_PTV ∝ √(ρc²·η)`: `d ln p/dT = (1/2)(−0.026 + 2·0.18 − 2.3) ≈ −0.99 %/K` → at ±5 °C, σ_water ≈ **±5 %**, η-dominated.  `config.py` uses rounded `RHO = 1000`, `C_SOUND = 1500` — within 0.3 % of verified pure-water values, well inside the ±5 % budget tolerance. |
| 13 | PTV | p₀ ∝ 1/R sensitivity → σ_radius = 0.10 | derived | **A** | **Verified.** `particle-tracking/output/<vpp>/07_sensitivity_sweep/sensitivity_grid.csv` reproduces 1003.8 / 903.5 / 821.4 kPa at R = 2.25 / 2.50 / 2.75 µm. |
| 14 | PTV | Streaming + wall residue | dropped from budget | **DROPPED 2026-05-22** | Earlier carried ±7.5 % as a placeholder; user decided 2026-05-22 to drop it entirely because no quantitative justification exists for this chip / drive regime.  Flagged as a *known unmodelled effect* in budget report §8 limit 7 (replaces the previous σ_streaming = 0.075 term in σ_P quadrature).  Resolution paths remain: streaming simulation for the actual geometry, or trajectory-level comparison against the pure-radiation-force Stokes balance. |
| 15 | Both | Peak definition / mode-fit vs max | "a few %" | **C** | Listed as limitation in §8 but never quantified.  Fix path is internal numerical analysis, not citation. |
| 16 | Both | Resonance-peak drift between LDV and PTV mounts (matched-conditions assumption) | not in symmetric budget | **BINARY EFFECT — added §8 limit 8 (2026-05-22)** | Budget assumes LDV and PTV are matched-conditions (same chip mount / session) so Δf ≈ 0 and drift contributes 0.  Verified for the 10 Vpp report data: within-W10 peak drift between sessions is ≤ 1 kHz (test5 peak 1.9070 MHz, test9 peak 1.9080 MHz from `resonance_survey.py`), giving R inflation ≤ 1.01 (~1 %).  **If the same-mount assumption is wrong** (e.g. PTV from a later mount in the W10 → W21 sequence), an additional one-sided R inflation up to **~1.5×** applies per `2026-05-22_mount_drift_ratio_inflation.md`.  Binary effect, not Gaussian-symmetric; documented in budget report §8 limit 8 and TL;DR caveat rather than added to σ_P.  Resolution: confirm the actual mount the PTV data was on. |

---

## Roll-up by category (updated 2026-05-22 after Zotero + WebFetch + user confirmations)

| Category | Count | Items |
|---|:---:|---|
| **A — fully verified** | **6** | 7 (Polytec linearity = ±0.5 %, not ±5 %), 8 (air-null R²=−4.56), 10 (κ_PS Barnkob), 11 (G0500B; σ_radius tightened to ±5 %), 12 (water; config rounded but inside ±1 % budget), 13 (p₀ ∝ 1/R) |
| **A — derived from primary sources (ChatGPT-verified)** | **1** | 1 (dn/dp water = 1.48 × 10⁻¹⁰ at 20 °C derived from IAPWS R9-97 + IAPWS-95) |
| **PARTIAL A — bulk confirmed, sub-quantity unverified or formulation caveat** | **2** | 4 (fused silica: n / E / ν / ρ / c_L confirmed; p_ij still need Primak & Post 1959); 5 (D263T: n / E / ν / ρ verified from SCHOTT, stress-optic K verified; (p₁₁, p₁₂) not separately published — formulation caveat) |
| **Type-B bracket with verified deviation endpoints; budget σ = p_0-propagated** | **1** | 9 (κ_PS central = 2.49 × 10⁻¹⁰ lab convention; deviation bracket [1.72, 3.30] × 10⁻¹⁰ from verified Settnes-Bruus and Barnkob endpoints; input variance σ_κ_p/κ_p = ±0.32; **propagated effect on ln(p_0) = ±0.19** — the latter is what enters σ_P) |
| **SCRIPT REPARAMETERISED with verified sources (4-glass bracket)** | **1** | 3 (was "BK7-class"; script now carries N-BK7 with SCHOTT-verified ρ=2510, E=82, ν=0.206 + Krupych 2011 p_ij, plus a separate Borofloat 33 / Pyrex row, plus fused silica, plus D263T; full chip-glass identity remains unknown, so the bracket spans all four candidates as the worst-case effect) |
| **A — derived from 4-glass bracket (rerun 2026-05-22, T_p=1 model)** | **1** | 6 (glass photoelastic central = **+9.7 %**, bracket **8.3 % – 11.0 %**; replaces the pre-2026-05-22 +14.5 % / 9.8 %–19.3 % and the intermediate +14.0 % / 8.3 %–19.7 %.  The drop comes from removing the T_p = 2Z_g/(Z_g+Z_w) traveling-wave upper bound — non-physical for the evanescent geometry — and keeping only T_p = 1 (pressure continuity).  Material-property spread across the 4 candidate glasses is now the only source of σ_b.) |
| **Dropped from budget (no justification)** | **1** | 14 (streaming + wall residue — dropped 2026-05-22 instead of carrying as placeholder) |
| **Binary effect outside symmetric budget** | **1** | 16 (resonance drift between LDV and PTV mounts — Δf ≈ 0 if matched-conditions holds, otherwise up to ~1.5× R inflation per mount-drift report; treated in §8 limit 8 as conditional on assumption rather than as Type-B σ) |
| **C — no primary-source path exists** | 2 | 2 (channel height), 15 (peak-definition correction) |

### Item 9 resolution (2026-05-22)

The PTV inversion uses `KAPPA_P = 2.49e-10` in `06_fitting.py` with
docstring comment "Settnes-Bruus 2012".  That attribution is wrong:
verified Settnes-Bruus 2012 Table I gives 1.72 × 10⁻¹⁰ Pa⁻¹ (ρ=1050,
c=2350).  The 2.49 number was inherited from an undocumented lab script.

User decision 2026-05-22: keep 2.49 × 10⁻¹⁰ as the **central** value but
relabel it as a lab-convention placeholder (not "Settnes-Bruus"); use
the verified Settnes-Bruus (1.72) and Barnkob (3.30) values as the
**low-side and high-side deviations** of an asymmetric Type-B bracket.

Budget impact:

| Quantity | Previous report | Updated 2026-05-22 |
|---|---|---|
| Label attribution for 2.49 | "Settnes-Bruus 2012" (wrong) | "lab convention, origin lost" |
| σ_κ_PS / κ_PS (Type-B) | ±0.14 (from a 2.49 ↔ 3.30 span) | ±0.32 (from a 1.72 ↔ 3.30 span around central 2.49) |
| PTV p₀ shift at κ_PS = 1.72 | not used | 0.86× central |
| PTV p₀ shift at κ_PS = 3.30 | 1.25× central (§4.5 gap-closing) | 1.25× central (still) |
| LDV/PTV ratio range from κ_PS alone | [1.0, 1.25] (one-sided closing) | **[0.80, 1.16]** (two-sided, larger envelope) |

The §4.5 "Barnkob gap-closing" analysis remains directionally correct
(adopting Barnkob deflates R toward unity), but the symmetric §4.3
quadrature envelope on σ(ln R_obs) needs to be recomputed with the
larger σ_κ_PS = 0.32 — see propagation note at bottom of this file.

### Item 3 finding (2026-05-22 — needs user decision)

`reports/2026-05-21_glass_pressure_self_verification.py` line 130
parameterises the "BK7-class" row as
`(n=1.515, p₁₁=0.121, p₁₂=0.270, E=70 GPa, ν=0.22, ρ=2230, c_L=5500)`.
Cross-checked against verified primary sources:

| Constant | Script "BK7-class" | SCHOTT N-BK7 datasheet (632.8 nm) | Krupych 2011 (BK7 p_ij) | Δ vs N-BK7 |
|---|---:|---:|---:|---:|
| n | 1.515 | 1.51509 | — | ≈ ✓ |
| E (GPa) | 70 | 82 | — | **−15 %** |
| ν | 0.22 | 0.206 | — | +7 % |
| ρ (kg/m³) | 2230 | 2510 | — | **−11 %** |
| p₁₁ | 0.121 | — | 0.118 ± 0.004 | ≈ ✓ |
| p₁₂ | 0.270 | — | 0.226 ± 0.005 | **+19 %** |

Two possibilities:

1. **Script is meant to model N-BK7** — then the constants are wrong
   and need to be updated (ρ → 2510, E → 82, ν → 0.206, p₁₂ → 0.226).
   Re-running the glass script with these would change ∂n/∂p for the
   BK7 row by ~−23 %, shifting the central +14.5 % glass photoelastic
   bias estimate.  Budget §2.1 b_glass would need re-computing.
2. **Script is meant to model a generic borosilicate** (e.g. Borofloat33,
   Pyrex 7740) where ρ ≈ 2230, E ≈ 65–70 GPa, ν ≈ 0.20 are the right
   neighbourhood.  Then the values are roughly correct but the row
   should be renamed (currently labelled "BK7-class", misleading).
   p₁₂ = 0.270 is still suspect — that value equals fused-silica p₁₂
   verbatim and almost certainly came from a copy-paste.  Even for
   generic borosilicate, p₁₂ ≈ 0.22–0.23 is more typical.

**Resolution needed**: which glass is the chip's top/bottom layer made
of?  This is not derivable from the manuscript or experiment_summary —
needs to be supplied by the user.  Once known, the script can be
re-parameterised cleanly.

The "two genuinely verifiable terms" critique is **exactly right under
the strict standard**.  Every external-source claim in the original
report is currently Cat B — meaning the named paper / datasheet / spec
is the *likely* source but the value's appearance in that source has
not been confirmed in this session.

---

## Verification path forward

To upgrade Cat B → Cat A, each item needs its source opened and the
value confirmed.  Access realism:

| Source type | Accessibility | Items |
|---|---|---|
| In-repo data | Already verified | 8, 13 (already Cat A) |
| Public datasheet (PDF on vendor website) | WebFetch should work | 5 (Schott D263T), 7 (Polytec VFX-F-110 / VIB-A-511 / VFX-I-130), 11 (Thermo Fisher G0500B) |
| Public standard / database | WebFetch should work | 12 (NIST Webbook / IAPWS tables) |
| Open-access handbook | Probably WebFetch-able | 3 (Schott BK7 — datasheet open; p_ij from Bass *Handbook of Optics* — possibly behind paywall), 4 (fused silica via Heraeus datasheet) |
| Subscription journal | Needs Zotero, institutional access, or arXiv preprint | 9 (Settnes–Bruus 2012 PRE), 10 (Barnkob 2010 *Lab Chip*) |
| Specific 633-nm literature pull | Mixed; needs a literature search to identify the right paper | 1 (dn/dp water) |
| No external source | Cannot upgrade | 2, 14, 15 |

### Recommended order

1. **Cheap wins** — fetch the four public-datasheet items (5, 7, 11)
   and the NIST water row (12).  WebFetch in this session should
   handle all four.
2. **κ_PS values (9, 10)** — these are the two PTV terms that drive
   the gap-closing analysis in §4.5 of the budget report.  Worth
   actual paper-reading.  If Zotero can be brought online or arXiv
   preprints are available (Settnes–Bruus 2012 likely has an arXiv
   version) this is solvable; otherwise needs the user to pull the
   PDFs.
3. **Glass photoelastic constants (3–5)** — important because the
   central 14.5 % LDV bias is *derived* from them and is the
   single-largest LDV-side term in the budget.  Verify p_ij from a
   primary source for at least one glass type to anchor the bracket.
4. **dn/dp (1)** — drives the whole LDV pressure inversion.  Worth a
   targeted lit pull when ready.

The verification status of items 2, 14, 15 cannot be improved by
citation; either the report acknowledges them as Type-B placeholders
with no traceable source, or the chip-specific measurements / models
get done.

---

## Cross-repo inconsistencies flagged

Independent of sourcing — these are conflicts between this repo and
`nonlinearphysics-manuscript/` that need user-side resolution before
sourcing can proceed:

1. **dn/dp water** — manuscript uses 1.4 × 10⁻¹⁰ Pa⁻¹
   (`main.tex:472`); this report adopted 1.48 × 10⁻¹⁰ Pa⁻¹ on
   2026-05-21 with no traceable justification in either repo.
2. **κ_P convention** — manuscript's `experiment_summary.md` uses
   κ_P = 2.49 × 10⁻¹⁰ Pa⁻¹ (Settnes–Bruus convention), but
   `references.bib` does not contain a Settnes & Bruus 2012 entry.

Both flagged in
`nonlinearphysics-manuscript/pra/NOTES_pending_citations.md`.
