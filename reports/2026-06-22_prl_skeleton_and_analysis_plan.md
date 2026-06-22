# PRL Skeleton & Analysis Plan — Integrated Snapshot (2026-06-22)

*Translated from the handover note `PRL_handover_2026-06-22.md`. Below,
**settled items**, **provisional decisions**, and **items to be decided
by the analysis results** are kept distinct.*

---

## 1. Central thesis and terminology

### Central thesis

In standard microchannel acoustophoresis, the primary acoustic field is
treated as the solution of a linear, single-frequency wave equation, and
the time-averaged second-order quantities — the acoustic radiation force
and streaming — are computed from that field. For a half-wavelength
standing wave,

$$P_{1f}\propto V_{\mathrm{drive}},\qquad
E_{\mathrm{ac}}\propto P_{1f}^{2}\propto V_{\mathrm{drive}}^{2}$$

is used as the standard scaling, and the focusing time and the required
channel length are tied to this $E_{\mathrm{ac}}$.

The central thesis of this paper is:

> **Even with ordinary single-frequency drive and an ordinary
> half-wavelength microchannel resonator, finite-amplitude nonlinearity
> self-generates harmonic cavity modes. As a result, the approximation
> treating the primary field as monochromatic and single-mode does not
> necessarily hold, and the input scaling of acoustophoretic focusing
> and throughput changes.**

What matters is not that the radiation force is itself a second-order
effect. In the conventional framework, the time-averaged second-order
transport quantities are computed from a **linear, monochromatic**
primary field. In this work, the **spectrum of the oscillatory acoustic
field itself is reconstructed by the harmonics**.

### Two mechanisms of performance degradation

#### A. Harmonic force-field reshaping

This exists even in the perturbative regime.

For the ideal $n$-th transverse mode, define

$$E_n=\frac{\hat P_{nf}^{2}}{4\rho c^2}.$$

Let the quantity corresponding to the near-center radiation-force
stiffness be

$$E_{\mathrm{foc}}^{(N)}=\sum_{n=1}^{N}(-1)^{n+1}n^2\frac{\Phi_n}{\Phi_1}E_n.$$

When $\Phi_n\simeq\Phi_1$ holds,

$$E_{\mathrm{foc}}^{(N)}\simeq E_1-4E_2+9E_3-16E_4+25E_5-\cdots .$$

Therefore:

* even-order harmonics **weaken** central focusing;
* odd-order harmonics **strengthen** central focusing;
* the total acoustic energy $E_{\mathrm{tot}}=\sum_n E_n$ and the
  quantity effective for particle focusing $E_{\mathrm{foc}}$ **do not
  coincide**.

#### B. Fundamental suppression

In the finite-amplitude regime, nonlinear back-coupling gives

$$P_{1f}\not\propto V_{\mathrm{drive}}.$$

That is, not only do the harmonics reshape the radiation-force field, but
the available fundamental itself falls below the low-voltage linear
extrapolation.

The two can be separated as

$$\frac{\kappa_{\mathrm{foc}}}{\kappa_{\mathrm{lin}}}=
\underbrace{\left(\frac{P_{1f}}{P_{1f}^{\mathrm{lin}}}\right)^2}_{\text{fundamental suppression}}
\underbrace{\left[1+\sum_{n=2}^{N}(-1)^{n+1}n^2\frac{\Phi_n}{\Phi_1}\left(\frac{P_{nf}}{P_{1f}}\right)^2\right]}_{\text{harmonic force-field reshaping}} .$$

### Terminology

* physical modal acoustic energy density: $E_n$
* total acoustic energy within the measurement band:
  $E_{\mathrm{tot}}^{(N)}=\sum_{n=1}^N E_n$
* near-center focusing stiffness:
  $\kappa_{\mathrm{foc}}=-\left.\dfrac{\partial F_y}{\partial y}\right|_{y=0}$
* focusing-performance index reduced to the $E_{\mathrm{ac}}$ of a single
  fundamental: $E_{\mathrm{foc}}$
* the whole phenomenon: **self-generated harmonic cascade**
* the high-amplitude region: **finite-amplitude regime** or **departure
  from leading-order perturbative scaling**

"non-perturbative regime" is **not** adopted as a central term, because
it can be read as a claim that the entire perturbation expansion fails.

### Subject of the title / abstract

In the title and abstract, foreground **throughput scaling** (easier for
a broad readership) rather than the in-house-defined $E_{\mathrm{foc}}$.

Provisional title candidate:

> **Self-Generated Harmonics Limit Throughput Scaling in Microchannel
> Acoustophoresis**

However, because the actual flow rate is not directly measured,
"throughput scaling" is safer than "throughput limit".

---

## 2. Gap vs prior work

### What is already known

In macroscopic finite-amplitude resonators, the following have been
known since Coppens et al.:

* harmonic generation by single-tone drive;
* harmonic amplification by the resonator;
* amplitude/phase changes due to detuning;
* back-coupling to the fundamental via higher-order corrections.

Therefore this paper does **not** claim as novel:

* that harmonics arise in a resonator per se;
* the Kuznetsov equation or Coppens scaling themselves;
* the iterative computation method itself;
* the concept of back-coupling itself.

Past self-consistent calculations have also confirmed the fundamental
suppression that perturbation solutions cannot capture, and the
formation of a harmonic cascade.

### Standard assumptions on the microchannel side

In standard microchannel acoustophoresis theory, the radiation force is
derived from a linear, monochromatic primary field, and for a
half-wavelength standing wave $E_{\mathrm{ac}}$ is tied directly to
focusing performance.

### Working gap

The current gap is defined by the following combination:

> **In a single-frequency-driven MHz standing-wave microchannel
> resonator, directly observe and quantify the self-generated harmonics
> as spatial modes, and thereby show that the correspondence between the
> single-mode $E_{\mathrm{ac}}$ and focusing/throughput performance
> breaks down. Furthermore, connect the fundamental suppression added on
> the high-amplitude side to a self-consistent nonlinear model.**

In AF2026, the self-generated 2f mode, $P_{2f}\propto V^2$, and
consistency with Coppens scaling were shown over 5–25 Vpp.

### Literature review still required

Carry out a targeted review in the microchannel context on these 5
points:

1. self-generated harmonic cavity modes from single-tone drive;
2. direct measurement of the spatial distribution of harmonics;
3. radiation-force reshaping by harmonic parity;
4. quantitative impact on focusing stiffness or throughput;
5. simultaneous observation of harmonic generation and fundamental
   suppression.

Depending on the search results, set the strength of "first direct
observation" / "first demonstration" / "to our knowledge".

---

## 3. Claim–evidence correspondence table

| Claim | Required evidence | Status |
|---|---|---|
| Harmonics are self-generated under single-frequency drive | 1f–3f spatial maps, mode profiles, drive-channel harmonic purity | Spatial maps acquired. Finalize the 120 Vpp drive-purity table |
| At low amplitude it follows known finite-amplitude acoustics | $P_1\propto V$, $P_2\propto V^2$, and if needed $P_3\propto V^3$; Coppens scaling | Holds to 2f in AF2026. Redefine a common calibration window for W21 |
| At high amplitude it departs from leading-order scaling | W21 10–120 Vpp $P_1,P_2,P_3$; $P_2/P_1^2$ | Observed experimentally |
| Harmonics reshape the radiation-force field | measured 1f–5f, parity/mode shape of each harmonic, $E_{\mathrm{foc}}$ | Formula settled. Whether to adopt 4f/5f pending the uncertainty analysis |
| The fundamental is suppressed | $P_1$ drop from low-voltage extrapolation, sublinearity vs $P_{\mathrm{in}}$ | Experimental sublinearity observed. Wording of the cause depends on model results |
| Back-coupling explains the fundamental suppression | self-consistent model including detuning phase | Incomplete. Aim for a semi-quantitative mechanistic account |
| Throughput scaling decreases | $E_{\mathrm{foc}}/E_{\mathrm{lin}}$; comparison of local stiffness and trajectory | Computable from the measured field; not a direct flow measurement |
| Performance is sublinear in input power | $P_{\mathrm{in}}=\langle vi\rangle$, $E_1(P_{\mathrm{in}})$, $E_{\mathrm{foc}}(P_{\mathrm{in}})$ | Raw voltage/current exist; not yet analyzed |
| A 1D model explains the phenomenon | agreement in onset, sign, order of magnitude | Not done. Precise device prediction not required |

---

## 4. Rough structure of the paper

### 1. Introduction

* the standard picture of microchannel acoustophoresis;
* linear, monochromatic primary field;
* $P\propto V$, $E_{\mathrm{ac}}\propto V^2$;
* relation to focusing time, admissible flow speed, throughput;
* finite-amplitude harmonics are known in macroscopic resonators;
* direct observation in a microchannel and its performance impact are
  not established.

### 2. Experiment and field reconstruction

* glass–Si–glass chip;
* $W=375~\mu\mathrm{m}$, $H=150~\mu\mathrm{m}$;
* single-tone burst drive;
* LDV refracto-vibrometry;
* pointwise DFT;
* 1f–5f complex amplitudes;
* transverse mode fitting;
* electrical voltage/current acquisition.

### 3. Direct observation of the harmonic cascade

* spatial maps of 1f, 2f, 3f;
* comparison with the expected transverse modes;
* comparison with drive harmonic impurity;
* confirmation that the harmonics are self-generated.

### 4. Perturbative-to-finite-amplitude crossover

* $P_{nf}\sim V^n$ at low amplitude;
* $P_2/P_1$, $P_3/P_1$;
* $P_2/P_1^2$;
* departure from leading-order scaling at high amplitude;
* sublinearity of $P_1$.

### 5. Self-consistent reduced model

* 1D Kuznetsov model;
* minimal phenomenological boundary including detuning;
* perturbative limit at low amplitude;
* back-coupling via iteration;
* comparison of onset, sign, order of magnitude;
* state explicitly that this is not exact device modeling.

### 6. Consequence for focusing and throughput scaling

* difference between $E_{\mathrm{tot}}$ and $E_{\mathrm{foc}}$;
* parity-dependent contributions;
* decomposition with fundamental suppression;
* $E_1$, $E_{\mathrm{foc}}$ vs $P_{\mathrm{in}}$;
* reduction rate relative to the linear low-power extrapolation;
* throughput ratio under standard Stokes / dilute-particle conditions.

### 7. Discussion and conclusion

* it occurs even in ordinary single-tone devices;
* force-field reshaping exists even in the perturbative regime;
* fundamental suppression is added in the finite-amplitude regime;
* precise prediction of higher-order chip modes is out of scope;
* a single $E_{\mathrm{ac}}$ alone is insufficient for design and
  performance assessment.

---

## 5. Figure list

### Fig. 1 — Spatial emergence of self-generated harmonics

For now, full-width 1f–3f in a 2×3 layout.

* (a) $|P_{1f}(x,y)|$
* (b) $|P_{2f}(x,y)|$
* (c) $|P_{3f}(x,y)|$
* (d–f) transverse profile and mode fit at the same axial position
* room for a small device/LDV schematic inset

The current draft outputs 1f–3f maps and profiles at 120 Vpp,
1.902 MHz.

Keep 3f at this stage; consider removing it only if space is short or
the mode-fit reliability is insufficient.

### Fig. 2 — Scaling and nonlinear crossover

Candidate panels:

* (a) $P_{1f},P_{2f},P_{3f}$ vs $V_{\mathrm{drive}}$;
* $V,V^2,V^3$ guides from a common low-amplitude window;
* (b) $P_{2f}/P_{1f}$, $P_{3f}/P_{1f}$ vs Mach number;
* (c) $P_{2f}/P_{1f}^2$ vs Mach number;
* overlay the perturbative prediction and the self-consistent model.

The current script fits each harmonic using "the first up-to-3 points
that reach SNR ≥ 3", so switch to a common perturbative window in the
final version.

### Fig. 3 — Input power, focusing and throughput scaling

Candidate central panel — compare

$$E_1(P_{\mathrm{in}}),\qquad E_{\mathrm{tot}}(P_{\mathrm{in}}),\qquad
E_{\mathrm{foc}}(P_{\mathrm{in}})$$

against a linear fit in the low-input-power range.

Decomposition panels:

$$\frac{E_1}{E_{\mathrm{lin}}}\quad\text{(fundamental suppression)},\qquad
\frac{E_{\mathrm{foc}}}{E_1}\quad\text{(harmonic force-field reshaping)},\qquad
\frac{E_{\mathrm{foc}}}{E_{\mathrm{lin}}}\quad\text{(total performance penalty)} .$$

Inset candidates:

* convergence of $E_{\mathrm{foc}}^{(1)}$ to $E_{\mathrm{foc}}^{(5)}$;
* comparison of local stiffness ratio and full-trajectory throughput
  ratio;
* the model's semi-quantitative prediction.

---

## 6. Quantitative metrics and analysis conventions

### Canonical dataset

* W21 peak-band cascade;
* 10–120 Vpp, 12 canonical scans;
* exclude `failed_*` runs;
* use the 40 V and 60 V extra runs for repeatability checks;
* at each voltage, choose the drive frequency that maximizes $P_1$;
* extract $P_1$–$P_5$ from that **same-frequency file and same axial
  position**.

W21 has the 12-point cascade, synchronized drive voltage/current/LDV
waveforms, a 1f–5f cache, and 1f / 2f sweeps.

### Voltage axis

* run names / bookkeeping: nominal PZT Vpp;
* quantitative plots: prefer measured PZT Vpp;
* keep the nominal/measured difference in the metadata.

### W21 frequency values

Current observed peaks:

$$f_1=1.902~\mathrm{MHz},\qquad
2f_1=3.804~\mathrm{MHz},\qquad
f_{2a}=3.794~\mathrm{MHz},\qquad
f_{2b}\simeq3.817~\mathrm{MHz}.$$

The generated 2f lies between the two nearby 2f peaks.

These are currently **observed peak values**, not pole parameters from a
complex fit.

The AF2026 values

$$f_1=1.907~\mathrm{MHz},\quad
f_2=3.845~\mathrm{MHz},\quad
\cos\theta\simeq0.53$$

are isolated as old values from a different dataset and are **not**
carried over into the W21 PRL analysis. The AF2026 low-amplitude result
itself can still be used as an independent prior result for the
perturbative regime.

### Perturbative calibration range

Take the primary operational criterion as

$$\frac{P_{2f}}{P_{1f}}\le 0.10 .$$

This is a practical criterion for selecting the range where the
nonlinear correction is sufficiently smaller than the fundamental and
recursive back-coupling is broadly a higher-order correction.

Also confirm

$$P_{1f}/V\simeq\text{const.},\qquad
P_{2f}/P_{1f}^{2}\simeq\text{const.}$$

As a sensitivity analysis, use thresholds 0.08 and 0.12 as well.

Use the **same voltage window** for all harmonics and fit

$$P_{nf}=a_n V^n .$$

If 3f does not have sufficient SNR within that window, do not force a
$V^3$ fit of 3f.

### Input power

Primary definition: over the steady on-state interval of the burst,

$$P_{\mathrm{in}}=\langle v(t)\,i(t)\rangle .$$

As a cross-check, compute

$$P_{\mathrm{in},1f}=\tfrac12\operatorname{Re}\!\left[V_{1f}I_{1f}^{*}\right] .$$

Use the **burst on-state real power** (corresponding to the acoustic
field amplitude), not the long-time-average power including duty cycle.

### Harmonic inclusion criterion for $E_{\mathrm{foc}}$

The signed contribution of the $n$-th harmonic is

$$\Delta E_{\mathrm{foc},n}=(-1)^{n+1}n^2\frac{\Phi_n}{\Phi_1}\frac{P_{nf}^{2}}{4\rho c^2}.$$

The relevant comparison is the stiffness contribution $\propto
n^2P_{nf}^{2}$, **not** $n^2P_{nf}$.

Conditions for inclusion in the central value:

1. the harmonic is statistically detected;
2. the sign of the contribution is supported by the spatial mode or the
   central curvature;
3. $\left|\Delta E_{\mathrm{foc},n}\right| > u\!\left(E_{\mathrm{foc}}^{(n-1)}\right)$;
4. $\left|\Delta E_{\mathrm{foc},n}\right| > u\!\left(\Delta E_{\mathrm{foc},n}\right)$,
   adopting $2u$ where possible.

If 4f and 5f satisfy these, take

$$E_{\mathrm{foc}}^{(5)}$$

as the band-limited best estimate.

If not, take

$$E_{\mathrm{foc}}^{(3)}$$

as the robust central estimate, with 4f–5f as a sensitivity band.

### Uncertainty

Include at minimum:

* harmonic amplitude fit uncertainty;
* noise bias;
* axial-column variability;
* mode-shape mismatch;
* calibration uncertainty;
* 40 V / 60 V repeat-scan variability;
* higher-harmonic truncation sensitivity.

### Throughput ratio

For a Newtonian fluid, dilute particles, Stokes drag, and the same
channel geometry / particle / focusing criterion,

$$\frac{\mathrm{throughput}_{\max}}{\mathrm{throughput}_{\max}^{\mathrm{lin}}}
=\frac{\kappa_{\mathrm{foc}}}{\kappa_{\mathrm{lin}}}
=\frac{E_{\mathrm{foc}}}{E_{\mathrm{lin}}} .$$

In standard theory too, the focusing time is $\propto E_{\mathrm{ac}}^{-1}$
and the admissible flow speed for a given length is $\propto
E_{\mathrm{ac}}$.

However, also confirm with a trajectory that uses the full force field,
not only the local stiffness.

---

## 7. Theory-model scope and parameter strategy

### Role of the model

The theory model does **not** aim at:

* precise full-device simulation of the whole chip;
* reproducing the structural origin of the detuning;
* few-percent-accuracy prediction of 3f–5f.

The goal is to **explain semi-quantitatively the mechanisms** of:

* harmonic generation;
* detuning phase;
* leading back-coupling;
* fundamental suppression;
* finite-amplitude crossover.

### Current model capabilities and constraints

The current `harmonic_model` has:

* 1D linear Kuznetsov;
* 1D nonlinear iterative Kuznetsov;
* multi-harmonic decomposition;
* complex phasor field;
* Gor'kov–Stokes trajectory.

It lacks:

* a 2D/3D solver;
* an impedance boundary;
* anything beyond steady state;
* anything beyond a fixed velocity boundary.

### Primary model candidate

The current first choice is:

> **a 1D multiharmonic self-consistent model with a minimal
> phenomenological dispersive impedance boundary added.**

Reasons:

* it can retain the cascade beyond 3f;
* the detuning phase can be put inside the iteration;
* it can handle interference among multiple generation paths;
* it connects to the full experiment better than a 1f–2f coupled-mode
  model.

This is **not** claimed to be a physical model of the chip structure.

Appropriate phrasing:

> a one-dimensional self-consistent nonlinear model with a minimal
> dispersive boundary chosen to reproduce the measured 1f–2f anharmonicity

### Impedance-model complexity

Start with one parameter, or a minimal parameter set. Candidate: a
mass-type reactive boundary such as

$$Z_w(\omega)=i\omega m_{\mathrm{eff}} .$$

However, if the two 2f modes contribute comparably, a single branch
cannot represent the complex 2f response. In that case use the fallback:

* limit the model to a detuning-sensitivity illustration; or
* introduce a two-pole susceptibility for 2f only; or
* do not demand precise prediction beyond 3f.

### Analytical reduction

The 1f–2f coupled-mode equations can be used, separately from the main
simulation, as an analytical model explaining the leading back-coupling:

$$A_2\sim\frac{g_2A_1^2}{\gamma_2+i\Delta_2},$$

$$\left[\gamma_1+i\Delta_1+\frac{g_1g_2|A_1|^2}{\gamma_2+i\Delta_2}\right]A_1=F .$$

This makes explicit:

* 2f generation;
* detuning phase;
* the 2f-mediated cubic correction to 1f.

### Parameter strategy

| Parameter | How determined |
|---|---|
| $W,\rho,c,\beta$ | geometry / material properties |
| input scale | low-drive $P_1/V$ or $P_1/\sqrt{P_{\mathrm{in}}}$ |
| $f_1,Q_1$ | W21 1f complex sweep / transient |
| $f_{2a},Q_{2a},f_{2b},Q_{2b}$ | W21 2f complex sweep |
| boundary reactance parameter | measured 1f–2f anharmonicity |
| nonlinear coupling scale | low-drive $P_2/P_1^2$, and if needed a 1D Kuznetsov projection |
| $G_2$ validation | comparison of independent $Q_2$, detuning, low-drive scaling |
| beyond 3f | no independent fit; semi-quantitative comparison only |

### Calibration and prediction

* calibration: perturbative window only;
* comparison: finite-amplitude region;
* do **not** re-tune parameters to fit the high-drive data;
* do **not** use a "no fitted parameters" claim;
* model success criterion: onset, sign, order of magnitude.

---

## 8. Procedure

### Step 1 — Prior-art review

Search the microchannel literature on the 5 gap items and fix the
strength of the first-claim language.

### Step 2 — Fix the canonical W21 dataset

* 12 canonical scans;
* exclude failed scans;
* same-frequency / same-position $P_1$–$P_5$;
* measured Vpp;
* tabulate the selected file, frequency, and axial position;
* complete separation from the AF2026 old values.

### Step 3 — Decide the perturbative calibration window

For each voltage, compute $P_2/P_1$, $P_1/V$, $P_2/P_1^2$.

Primary criterion: $P_2/P_1\le0.10$. Sensitivity: 0.08, 0.12.

Re-fit $P_n=a_nV^n$ using the common window.

### Step 4 — Drive purity

For all Vpp, especially 120 Vpp:

* electrical $V_{2f}/V_{1f}$;
* $V_{3f}/V_{1f}$;
* comparison with the acoustic harmonics.

### Step 5 — Input real power

From the raw voltage/current, compute $P_{\mathrm{in}}=\langle vi\rangle$.
Also check apparent power, power factor, fundamental phasor power, and
the electrical higher-harmonic contribution.

### Step 6 — $E_{\mathrm{foc}}$ reconstruction and uncertainty

For each voltage compute $\Delta E_{\mathrm{foc},n}$ and
$E_{\mathrm{foc}}^{(N)}$, $N=1,\ldots,5$.

Outputs: the signed contribution at each order; the cumulative value;
the uncertainty; the ratio of added contribution to existing
uncertainty; the difference between 1f–3f and 1f–5f. This decides
whether 4f/5f are included in the best estimate.

### Step 7 — 1f resonance parameter extraction

Complex-fit $f_1,Q_1$ from the W21 low-drive response. Compare the
transient, linewidth, and per-voltage apparent $Q$ of the cascade.

### Step 8 — 2f complex response extraction

Build the drive-referenced complex response from the raw voltage and LDV
waveforms.

Fits to perform:

* one-pole complex fit;
* two-pole complex fit;
* treat the fixed channel delay as a nuisance parameter;
* one- vs two-pole comparison via AIC/BIC, etc.

Outputs: $f_{2a},Q_{2a},f_{2b},Q_{2b}$ and the combined complex
susceptibility at 3.804 MHz.

### Step 9 — Implement the minimal detuning model

Add a minimal reactive / impedance boundary to `harmonic_model`:

* put the detuning phase inside each iteration;
* calibrate the low-drive scaling;
* fix parameters at high drive;
* start from a one-parameter model.

### Step 10 — Model comparison

Compare $P_1/P_1^{\mathrm{lin}}$, $P_2/P_1$, $P_3/P_1$,
$E_{\mathrm{foc}}/E_{\mathrm{lin}}$.

Judge on onset, sign, order of magnitude — exact amplitude not required.

### Step 11 — Throughput interpretation

Reconstruct the full radiation-force field from the measured 1f–5f and
compute particle trajectories. Compare the local-stiffness-derived
throughput ratio with the full-trajectory throughput ratio. If they
agree, keep "throughput scaling" in the title.

### Step 12 — Finalize figures and claims

Fix Fig. 1–3, the title, abstract, and the claim–evidence table
according to the analysis results.

---

## 9. Decision branches based on results

### A. Treatment of 4f / 5f

* contribution exceeds the cumulative uncertainty in Step 6 and the mode
  parity is supported → $E_{\mathrm{foc}}^{(5)}$ as best estimate;
* amplitude detected but the signed stiffness contribution is uncertain
  → $E_{\mathrm{foc}}^{(3)}$ as the central value, 4f–5f as a band;
* contribution below the uncertainty → ignore in the main text, report
  only as sensitivity.

### B. Perturbative window

* $P_1/V$ and $P_2/P_1^2$ constant within $P_2/P_1\le0.1$ → use that
  range as the calibration window;
* not constant → lower the threshold, or conclude W21 lacks sufficient
  perturbative points;
* 3f below the detection limit → do not claim a $V^3$ fit.

### C. 2f response

* one-pole fit sufficient → minimal impedance branch;
* two-pole fit clearly better → two-pole $\chi_2$ or an uncertainty band;
* phase calibration unstable → present the model as a phase-sensitivity
  band.

### D. Model success

* reproduces onset, sign, order of magnitude → "self-consistent model
  explains the observed suppression";
* reproduces sign only → "captures the leading mechanism";
* does not reproduce → weaken to "fundamental sublinearity accompanies
  harmonic growth";
* does not reproduce exact 3f–5f → within scope as planned; not a model
  failure.

### E. Input power

* $E_1(P_{\mathrm{in}})$ also sublinear → strong acoustic-field-side
  fundamental suppression;
* $E_1(P_{\mathrm{in}})$ linear while only $E_1(V)$ is sublinear →
  electrical-impedance / PZT-response effects dominate;
* only $E_{\mathrm{foc}}/E_1$ drops → harmonic force-field reshaping is
  the main effect;
* both drop → the two penalties coexist.

### F. Throughput

* local stiffness ratio and trajectory ratio close → "throughput
  scaling" in the title;
* markedly different → change the title to focusing performance;
* off-center equilibria form → use the trajectory, not just local
  stiffness, as the primary metric.

### G. 3f in Fig. 1

* mode fit robust in the unified pipeline → keep in the main figure;
* map clear but ideal-mode fit weak → keep as cascade evidence, but do
  not call it precise mode identification;
* space or reliability insufficient → drop the 3f profile and
  consolidate into Fig. 2.

### H. Prior art

* no comparable direct microchannel observation → "to our knowledge,
  first spatially resolved observation";
* prior reports exist → move the novelty to the performance consequence
  and the finite-amplitude crossover.

---

## 10. Explicitly out of scope

1. identifying the structural origin of the detuning;
2. a full structural model including Si, glass, PZT, adhesive, tubing,
   and support conditions;
3. 2D/3D nonlinear fluid–structure simulation;
4. complete determination of independent modal susceptibilities for
   3f–5f;
5. precise prediction of the absolute amplitude and phase of 3f–5f;
6. measurement beyond 6f (difficult due to the 12 MHz LDV bandwidth
   ceiling);
7. true 99% closure of the infinite harmonic sum;
8. direct throughput measurement using real flow rate;
9. experimental verification of device-to-device generality;
10. complete separation of all contributions (thermal, streaming,
    cavitation, PZT nonlinearity, etc.);
11. high particle concentration, particle–particle interaction;
12. generalization to non-Newtonian samples;
13. design/optimization of a harmonic-suppression device;
14. elucidating the physical origin of the chip structural modes.

However, out of scope is not the same as ignored. To protect the central
claim, confirm at minimum:

* drive harmonic impurity;
* input electrical power;
* resonance frequency drift;
* drive dependence of $Q$;
* the range the model can explain;
* wording adjustments where an alternative explanation cannot be ruled
  out.

---

## Current overall picture

The main thread of the paper is:

> **self-generated harmonics arise in a single-tone microchannel
> → the single-mode acoustic-field assumption breaks down
> → parity-dependent force-field reshaping exists from the perturbative
> regime
> → fundamental suppression is added in the finite-amplitude regime
> → the input scaling of focusing and throughput decreases.**

The theory model is positioned as:

> **not a device model that fully reproduces the origin of the
> phenomenon, but a semi-quantitative reduced model that demonstrates the
> nonlinear coupling and back-coupling including the detuning phase.**

The remaining undetermined items are governed mainly by the analysis
results of Steps 6, 8, 9, 10, and 11.
