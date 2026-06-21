# 06 — Photophysics & a realistic ME-HSI simulation (why variance ≠ informativeness)

This documents the **research** behind a more realistic synthetic generator, and the generator
itself. Motivation: doc 05 showed that on the *current* synthetic cubes a **trivial variance-ranking
ties the best learned autoencoder** (0.480 vs 0.484). That is not a failure of the AE — it is a
property of the *data*: the benchmark plants signal as clean, bright, well-separated Gaussian peaks,
so **band variance is essentially proportional to informativeness** and a variance ranker is already
near-optimal. To tell whether a learned selector adds value, we need data where the physics
**decouples variance from informativeness** — which is exactly what real fluorescence does.

So before changing anything we did the photophysics: *what actually happens to an electron under
excitation, how a chemical composition turns into a 4D excitation–emission cube, and where reality
departs from the clean model.*

---

## 1. From electrons to an emission spectrum (the Jablonski/Perrin picture)

A fluorophore has electronic states — a ground singlet **S₀**, excited singlets **S₁, S₂, …**, and
triplets **T₁** — each with a ladder of **vibrational** sublevels. Fluorescence is a four-step cycle:

1. **Absorption (~10⁻¹⁵ s).** A photon of energy *hν* promotes an electron from S₀(v=0) to a
   vibrational sublevel of S₁/S₂. By the **Franck–Condon principle** the nuclei are frozen during the
   (much faster) electronic jump, so the transition is "vertical" and lands on whichever excited
   vibrational level has the best wavefunction overlap — this gives the **absorption band its width
   and vibronic shape**. The probability vs wavelength is the **extinction coefficient ε(λ)**; the
   measured *excitation spectrum* ≈ the absorption spectrum.
2. **Vibrational relaxation + internal conversion (~10⁻¹²–10⁻¹¹ s).** The molecule cascades
   non-radiatively to **S₁(v=0)**, dumping the excess as heat. This is **Kasha's rule**: emission
   almost always starts from the lowest vibrational level of the lowest excited singlet.
3. **Emission / fluorescence (~10⁻⁹ s).** The electron drops from S₁(v=0) to various vibrational
   sublevels of S₀, emitting a photon. Franck–Condon again sets the **emission band shape**; because
   it ends on *excited* vibrational levels of S₀, the emitted photon is **red-shifted** (the
   **Stokes shift**) and the emission band is ~a **mirror image** of absorption.
4. **Return to S₀.** Ready to cycle again (unless it bleached or crossed to a triplet).

Two consequences are the backbone of all EEM modelling:

- **Kasha + Vavilov ⇒ the emission *shape* is independent of the excitation wavelength.** Exciting
  at 350 or 480 nm changes *how many* molecules emit, not *what colour* — only the **amplitude**
  scales (with ε(λ_ex)·Φ), the **emission profile** is fixed.
- **Quantum yield** Φ = k_r/(k_r+k_nr) (radiative vs all decay) and **brightness = ε·Φ** set the
  detected photon count per molecule. Two fluorophores can have equal concentration but 100× different
  brightness.

### 1.1 …which forces the mathematical forward model

For one fluorophore *k* in the dilute regime, the Jablonski physics factorizes its 4D signal exactly:

```
S_k(x, y, λex, λem) = c_k(x,y) · [ε_k · Φ_k] · a_k(λex) · e_k(λem)
                       └ amount ┘  └ brightness ┘  └ exc. ┘ └ emission ┘
```

`a_k` = excitation profile (≈ normalized absorption), `e_k` = **area-normalized** emission profile.
This separable structure (amplitude in λex, fixed shape in λem) is the **trilinear / PARAFAC**
signature of EEM data — it *is* Kasha's rule written as algebra. Multiple fluorophores add **linearly**
in the dilute limit:

```
cube(x,y,λex,λem) = Σ_k S_k          (the Linear Mixing Model)
```

**SpectraForge already implements exactly this** (`forward.render`: `amp = ε·Φ·exc(λex)`, times the
area-normalized `emission(λem)`, summed over fluorophores). So the *core* engine is photophysically
correct. The realism gap is not the trilinear core — it is everything that sits *on top* of it.

---

## 2. Where reality departs from the clean model (the confounds)

| Effect | Physics | Effect on the cube | Variance vs info |
|--------|---------|--------------------|------------------|
| **Rayleigh scatter** | elastic photon scatter | a bright line at λem = λex (+ a grating ghost at 2·λex) | **high variance, zero chemical info** |
| **Raman scatter (water)** | inelastic, O–H stretch ~3300–3600 cm⁻¹ | a line at a fixed *wavenumber* offset: 1/λ_R = 1/λex − Δν̄ | **high variance, zero info** |
| **Autofluorescence** | endogenous collagen/elastin/NADH/FAD/lipofuscin | broad, spatially-structured background, often dominant | **high variance, usually class-irrelevant** |
| **Inner-filter (IFE)** | excitation absorbed (primary) + emission reabsorbed (secondary): F = F_obs·10^((A_ex+A_em)/2) | signal saturates/distorts with concentration | breaks linearity & peak∝conc |
| **FRET** | dipole–dipole transfer, E = R₀⁶/(R₀⁶+r⁶), when donor em overlaps acceptor abs and r≲R₀ (2–6 nm) | donor quenched, acceptor sensitized | nonlinear, breaks additivity |
| **Quenching / bleaching / solvatochromism** | dynamic/static quenching, photodestruction, pH/polarity shifts | scales/shifts peaks over space & time | variance unrelated to identity |
| **Spectral overlap** | broad bands (FWHM 40–100 nm) overlap heavily | discrimination hides in a shoulder/tail | **discriminative signal is LOW variance** |
| **Shot (Poisson) noise** | photon counting: Var = mean | bright bands are noisy *because* they are bright | inflates bright-band variance |
| **Read / dark / fixed-pattern** | detector electronics | additive, band-dependent | adds non-informative variance |

### 2.1 The central insight

On the clean trilinear model with bright separated peaks, **band variance ∝ informativeness**, so a
variance ranker is near-optimal and a learned method can only tie it. Real physics **decouples** the
two in four independent ways:

1. **High variance, zero info** — Rayleigh/Raman scatter lines.
2. **High variance, class-irrelevant** — bright, spatially-varying autofluorescence nuisances.
3. **Inflated variance** — shot noise makes bright (possibly redundant) bands look "important".
4. **Low-variance signal** — the discriminating information sits in dim fluorophores or in the
   subtle *shape* difference between overlapping bands.

A selector that only looks at per-band variance walks straight into (1)–(3) and misses (4). A method
that models **inter-band structure** (reconstruct a band from the others — the masked/spectral
autoencoder) can tell a scatter spike (uncorrelated with any chemical pattern) from a dim peak that
co-varies with a whole emission band. **That is the regime where a learned selector should finally
beat the trivial baseline — and the regime a fair benchmark must contain.**

---

## 3. What we added to the engine (physically-faithful, all opt-in)

Backward-compatible: the linear invariant `render(A+B)==render(A)+render(B)` is unchanged with the
new effects off.

- **Water Raman line** (`artifacts.add_scatter_lines`, `raman_strength`): at
  `λ_R = 1/(1/λex − Δν̄·10⁻⁷)`, Δν̄ ≈ 3400 cm⁻¹ — the missing high-variance, no-info band.
- **Spatially-varying scatter** — Rayleigh/Raman scaled by a per-pixel **turbidity/reflectance field**
  (`render(..., scatter_field=…)`), so the scatter bands carry large spatial *variance* but no class
  information.
- **Vibronic / asymmetric emission** (`Fluorophore.em_skew`, `vibronic`): a red-tailed band with an
  optional vibronic shoulder (Franck–Condon progression) instead of a pure Gaussian — so overlapping
  fluorophores differ in *shape*, not just peak position.
- **Confounded scene generator** (`scenegen.make_confounded_scene`): classes set by **dim, overlapping**
  fluorophores; **bright nuisance** autofluorophores with their own varying fields; a turbidity field
  driving scatter. Labels come from the *discriminative* components only.

**Specified, not yet wired (next increment — the formulas are in §2):** the *secondary* inner-filter
(emission-axis reabsorption `10^(−A_em/2)`) and **FRET** (`E = R₀⁶/(R₀⁶+r⁶)`, donor quenched /
acceptor sensitized for co-localized pairs). Both need the per-fluorophore contributions inside
`render`, so they belong in a follow-up that adds an opt-in non-linear path while keeping the linear
invariant. They add *nonlinearity* (favouring a learned nonlinear selector over linear unmixing) but
are not needed for the variance≠informativeness result below, which is driven by scatter + bright
nuisances + dim/overlapping signal + shot noise.

---

## 4. Results — does the realistic regime finally separate the methods?

`reports/realistic_benchmark.py` builds the confounded regime: 3 **dim, overlapping,
excitation-differentiated** discriminative dyes (emission 515/535/555 nm, ε·Φ ≈ 0.22) set the class;
2 **bright nuisance** autofluorophores (NADH-like 445 nm, lipofuscin-like 665 nm, ε·Φ ≈ 0.8, with
their own spatial fields) plus **spatially-varying Rayleigh + water-Raman scatter** and
signal-dependent shot noise sit on top. 4 excitations (405/470/488/506 nm), 12-band budget, mean over
3 scenes.

The construction works: **corr(per-band variance, per-band discriminability F) = −0.09** — the
high-variance bands are *not* the informative ones (on the clean doc-05 data this correlation is
strongly positive).

| selection | KNN macro-F1 | % bands on the discriminative window |
|-----------|-------------:|-------------------------------------:|
| all bands | 0.504 | 28% |
| **variance-ranking** | **0.344** | 33% |
| discriminability-oracle | 0.491 | 100% |
| random | 0.372 | 28% |
| C0 standard-CAE | 0.364 | 17% |
| **C2 spectral-AE** | **0.431** | 69% |
| **C3 masked-spectral-AE** | **0.414** | 61% |
| C4 variational | 0.345 | 19% |

**The result the clean benchmark could not produce.** On this realistic regime the **trivial
variance-ranking collapses to 0.344 ≈ random** (0.372) and far below all-bands (0.504): it spends its
budget on the bright nuisance/scatter bands (only 33% land on the discriminative window). The
**per-pixel spectral autoencoders C2 (0.431) and C3 (0.414) clearly beat variance-ranking** and put
**69%/61%** of their bands on the dim discriminative window — they recover the low-variance signal that
variance ranking cannot see. The discriminability oracle (0.491) shows the ceiling; C2/C3 reach ~85%
of the way there.

Two candidates do **not** clear it here: the **C0 spatial CAE (0.364)** stays degenerate (17% on the
window — consistent with doc 05), and the **C4 VAE (0.345)** fails on this harder, dimmer signal —
its latent does not capture the subtle discriminative structure (a lead to revisit: lower β / longer
warmup / larger latent). C2 (the simplest spectral AE) is the most robust here.

**Bottom line.** Doc 05's worry — "a trivial variance baseline ties the learned method, so where is
the value?" — was an artifact of a benchmark where, by construction, variance = informativeness. Under
physically-grounded confounds (scatter, bright nuisances, shot noise, dim/overlapping signal) the two
**diverge**, and a learned selector that models inter-band structure delivers a **real, measurable
advantage over the trivial baseline (+0.07–0.09 macro-F1, ~2× the discriminative-band hit rate)**.
This is the regime in which the method should be evaluated, and it is now reproducible on demand. The
remaining gate is unchanged: confirm the same advantage on the **real** Lichens/Collagen cubes.
