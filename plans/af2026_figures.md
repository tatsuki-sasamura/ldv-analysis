# AF2026 abstract figure generation

## Context

The Acoustofluidics 2026 conference abstract
(`acoustofluidics/abstract.tex` in the manuscript repo) is a 2-page
A4 paper, single column, Calibri 11 pt, 2 cm margins. It needs **two**
figures, sourced from PRA caches but adapted for AF layout and the
post-2026-04-23 notation convention.

Implemented by `experiments/2026W10_stepA/af2026_figures.py`, which
reads cached NPZs from PRA and writes new figures into the AF folder.

## Notation convention (settled in `abstract.tex`)

Figures must follow the same symbols used by the abstract prose.

| Symbol | Meaning | In figures |
|---|---|---|
| `$\hat{P}_{nf}$` | scalar peak pressure amplitude (mode-fit coefficient or its max over $x$) | y-axis labels of drive-sweep ratio plots; in-panel annotations of fitted peak amplitude |
| `$P_{nf}(y)$`, `$P_{nf}(x,y)$` | position-dependent amplitude (DFT output, mode profile) | optional in-figure equation form; not strictly needed since "Pressure [MPa]" axis labels are generic |
| `$p(x,y,t)$` | instantaneous pressure (lowercase `p` + time argument) | not used inside figures (only abstract prose) |
| `$E_\mathrm{ac}$` | acoustic energy density. **No** `\langle\cdot\rangle` brackets, **no** `1f` subscript. | x-axis label of drive-sweep ratio plot |
| `$Q_1$`, `$Q_2$` | quality factors; numeric subscripts. **Not** `$Q_{1f}, Q_{2f}$`. | only in captions; not in axis labels of these two figures |
| `$V_\mathrm{drive}$` in `$V_\mathrm{pp}$` | drive voltage | (Methods context only; not in these AF figures) |
| `$W$, $H$, $f$, $\beta$, $M$, $y$, $x$` | as in PRA figures (no change) | y-axis label `$y$ [μm]`, mode-fit equation forms |

Most-impacting update vs. PRA figures: **the hat on scalar peak
amplitudes**. The y-axis label of the drive-sweep ratio is
`$\hat{P}_{2f}/\hat{P}_{1f}$`, not `$P_{2f}/P_{1f}$`.

## Strict AF figure widths

A4 page (21 × 29.7 cm) with 2 cm margins → text width = **17.0 cm =
6.6929 in**.

| Layout | Width (cm) | figsize_w (in) |
|---|---|---|
| Full `\linewidth` | 17.0 | **6.7** |
| Half (`0.5\linewidth`) | 8.5 | **3.35** |
| Two-thirds (`0.67\linewidth`) | 11.4 | **4.5** |

Use these as natural widths so figure fonts render at their authored
size (no scaling). PRA double-column (7.0 in) and PRA single-column
(3.375 in) are very close to AF full and AF half respectively, so
content authored for PRA generally translates to AF with negligible
font scaling.

## Output

Save to `$MANUSCRIPT_DIR/acoustofluidics/figures/` — a **separate**
figure folder from `$MANUSCRIPT_DIR/pra/figures/`. AF and PRA do not
share figures.

| File | Purpose |
|---|---|
| `Fig1.eps`, `Fig1.png`, `Fig1.npz` | Spatial mode shapes (R1) |
| `Fig2.eps`, `Fig2.png`, `Fig2.npz` | Drive-sweep harmonic ratio with theory (R2) |

The npz cache for each AF figure is also written next to the image,
so the AF figures are self-contained re-renderable artefacts (don't
need to re-load PRA caches if only the AF layout changes).

`dpi=1200` for both `.eps` and `.png`, matching `manuscript_figures.py`.

## Figure 1 — Spatial mode shapes (R1, full-width)

**Source data:** `pra/figures/Fig7.npz` (existing).

**Layout:** **2×2 panel grid, full-width. Top row = 2D maps,
bottom row = line profiles** (panel order matches data flow:
2D measurement first, then 1D slice).

| Panel | Content |
|---|---|
| (a) | 2D map of $P_{1f}(x,y)$ at 25 V$_\mathrm{pp}$ |
| (b) | 2D map of $P_{2f}(x,y)$ at 25 V$_\mathrm{pp}$ |
| (c) | $P_{1f}(y)$ line profile (slice at $x = x_0$) + $\sin(\pi y/W)$ fit |
| (d) | $P_{2f}(y)$ line profile (slice at $x = x_0$) + $\cos(2\pi y/W)$ fit |

The 2D maps in (a, b) are the raw spatial measurement and convey the
cavity-mode pattern at a glance; the line profiles in (c, d) are the
1D slice at $x = y_{\text{best}} \approx 8$ mm used for the mode-shape
fit. Reading order (a) → (b) → (c) → (d) traces the analysis pipeline.

**figsize:** **`(6.7, 3.0)`** for strict AF full-width. Renders 17.0 ×
7.6 cm in the abstract. Compact aspect to keep the 2-page budget under
control; the 2D maps are correspondingly thin but still legible.
(`gridspec_kw={"height_ratios": [1.2, 1]}` gives top row slightly
more vertical space than bottom — line profiles in (a, b) get more
height than 2D maps in (c, d).)

**Inclusion in `abstract.tex`:**
```latex
\includegraphics[width=\linewidth]{figures/Fig1}
```

**Labels and annotations:**
- (a, b) y-axis: `"Pressure [MPa]"` (generic, no notation change)
- (a, b) x-axis: `"$y$ [μm]"`
- (a, b) panel header (above plot): `"(a) $p_{1f}(y)$, $x = ...$ mm"`
- (a, b) legend: data marker `"Measured"` + fit curve labeled with
  the equation form, e.g.
  `r"$p_{1f}(y) = \hat{P}_{1f}\sin(\pi y/W)$, $\hat{P}_{1f} = 4.0$ MPa"`
  (combines the equation + fitted peak value in a single line)
- (c, d) x-axis: `"Length, $x$ [mm]"`
- (c, d) y-axis: `"Width, $y$ [mm]"`
- (c, d) red dashed vertical line at the chosen line-cut $x$
  (`y_best` from the npz cache)
- (c, d) colorbars labeled `"Pressure [MPa]"`

## Figure 2 — Drive-sweep harmonic ratio with theory (R2, half-width)

**Source data:** `pra/figures/Fig8.npz` (experiment) + `pra/figures/fig3.npz`
(simulation; perturbative theory line via `ratio_pert`).

**Layout:** **single panel, half-width.**

**figsize:** **`(3.35, 2.0)`** for strict AF half-width. Renders 8.5 ×
5.1 cm in the abstract. Matches PRA single-column tradition (3.375 ×
2.0) almost exactly.

**Inclusion in `abstract.tex`:**
```latex
\includegraphics[width=0.5\linewidth]{figures/Fig2}
```

**Required label updates** (relative to PRA `Fig9` in
`manuscript_figures.py`):

1. **Y-axis hat:** `r"$P_{2f}/P_{1f}$"` → `r"$\hat{P}_{2f}/\hat{P}_{1f}$"`
2. **X-axis simplify:** `r"$\langle E_\mathrm{ac,1f} \rangle$ [J/m$^3$]"`
   → `r"$E_\mathrm{ac}$ [J/m$^3$]"` (drop angle brackets and `1f`
   subscript)
3. **Legend wording:** `"Simulation (self-consistent)"` →
   `"Perturbative theory"`. The AF abstract uses the perturbative
   form ($\hat{P}_{2f}/\hat{P}_{1f} = \beta Q_2 M / 4$) and explicitly
   drops the self-consistent framing per the 2026-04-23 strategy
   decision.
4. **X-axis range:** truncate to the experimental range +
   slight buffer. `ax.set_xlim(0, 700)` (J/m³). The perturbative line
   should be masked to the same range to avoid an empty curve
   extension to ~5000 J/m³.
5. **MRE annotation:** keep, scoped to the perturbative comparison.
   Current MRE = ~10% (with $Q_1 = 121, Q_2 = 100$ from FigA1 ground
   truth, fed through `harmonic_model/scripts/generate_manuscript_figures.py`
   which produces `fig3.npz`).

## Q-value provenance (important)

The perturbative theory line in Fig 2 depends on $Q_1$ and $Q_2$. The
canonical source is `pra/figures/FigA1.npz`:
- $Q_1 = 121$ (from $\tau_1 = 20.2$ µs, drive at $f \approx 1.907$ MHz)
- $Q_2 = 100$ (effective Q at $2f_1 = 3814$ kHz, includes detuning
  loss to the 3845 kHz cavity eigenmode → cos θ ≡ 1 in the Coppens
  form)

These values are now used in
`harmonic_model/scripts/generate_manuscript_figures.py`
(`Q_1F = 121`, `Q_2F = 100`), which writes `pra/figures/fig3.npz` —
the cache that AF Fig 2 reads for the perturbative theory curve.

If $Q_1$ or $Q_2$ change in the future, the chain to update is:
1. `pra/figures/FigA1.npz` (regenerated by burst-Q analysis)
2. Constants in `harmonic_model/scripts/generate_manuscript_figures.py`
3. Re-run `generate_manuscript_figures.py` to refresh
   `pra/figures/fig3.npz`
4. Re-run `experiments/2026W10_stepA/af2026_figures.py` to refresh
   AF Fig 2

## Caption brevity

Captions live in `abstract.tex`, not the figure image. Plan only:
captions should be **2–3 sentences** for AF (vs. PRA's 5–10-line
captions).

## Notation pitfalls to avoid

- Do **not** use lowercase `$p_{nf}(y)$` or `$p_{nf}(x,y)$` in panel
  headers, axis labels, or legends. **Lowercase `$p$` is reserved for
  instantaneous pressure** (function of $x, y, t$). Figures plot
  amplitudes — use uppercase `$P_{nf}(y)$` and `$P_{nf}(x,y)$` always.
- Do **not** use `$Q_{1f}, Q_{2f}$` (PRA-style). Use `$Q_1, Q_2$`.
- Do **not** use `\langle E_\mathrm{ac} \rangle` (PRA-style). Use
  `$E_\mathrm{ac}$`.
- Do **not** use bare `$P_{nf}$` for a scalar. Always `$\hat{P}_{nf}$`.
- Do **not** label the theory curve "Simulation (self-consistent)".
  Use `"Perturbative theory"`. The AF abstract is perturbative-regime.

## Verification

After regenerating, the figures should fit the abstract via:

```latex
\begin{figure}[h!]
    \centering
    \includegraphics[width=\linewidth]{figures/Fig1}
    \caption{...}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\linewidth]{figures/Fig2}
    \caption{...}
\end{figure}
```

Build the abstract with `latexmk -xelatex` and check:
- Both figures fit on the 2-page budget without overflow
- Fonts in figures are legible at the rendered size (especially Fig 2
  at half-width — should have authored fonts at ~8–9 pt)
- 2D map y-axis covers the full channel width $\pm W/2 = \pm 187$ µm
  (data in `Fig7.npz`'s `w_mm` reaches that range; should not be
  clipped by `set_ylim` or auto-scaling)
