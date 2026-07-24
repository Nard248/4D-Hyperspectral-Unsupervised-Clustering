# SpectraForge — Expert Research Opinion: Review, New Experiments, and Theory

**Scope.** A full, adversarial expert review of `paper/paper.md` + `paper/summary.md` + the simulator
(`src/spectraforge/`) + the experiment suite (`reports/`), **plus seven new experiments I ran** to test
whether the conclusions can be improved, broken, or hardened, **plus** a physics-fidelity audit and a
theoretical formalization. Produced by a multi-agent review (5 dimensions, adversarially verified) that I
then cross-checked and extended by running code. All my new numbers are 3-seed, use the *exact* paper
harness (`build_dataset`, `roi_mask`, `feature_matrix`, `_clfs`, few-shot 30/class, best-NL F1, k=24), and
use calibrated random nulls (60–180 draws + empirical p-values). New code: `reports/improve_experiments.py`.

---

## 0. TL;DR — the five things that matter most

1. **One stated conclusion does not survive scrutiny: "only supervised selection beats random under
   clutter."** It rests on a **test-label leak** — `mutInfo*` is computed on the *full ROI labels
   including the held-out test pixels* (`beat_random.py:67` and 8 other scripts). I re-ran it both ways
   (`improve_experiments.py --mi-leak`): under clutter the *leaked* MI beats random (p≈.01) but the
   *leak-free* MI **does not** (p=.21–.46) — it is statistically indistinguishable from blind PCA. The
   program's own leak-free round-3 run already showed this (`OVERNIGHT-LOG.md:229`); the **paper quotes the
   leaked numbers**. This is the only change that alters a conclusion, and it must be fixed.

2. **Every attempt to *rescue* the method fails — which makes the negative result much stronger.** I built
   and ran the obvious "you should have tried X" methods: a **supervised end-to-end differentiable
   (concrete/Gumbel) k-subset selector** (the methods reviewer's #1 missing experiment) → it **overfits and
   is the worst supervised method everywhere**; a **clutter-structure-aware blind selector** (low-rank
   clutter removal + unsupervised cluster-discriminability) → the **best blind attempt on record but still
   never crosses random**; a **smooth/calibratable-clutter** regime (the physics reviewer's conclusion-flip
   test) → blind **still** doesn't beat random. The thesis survives the strongest adversaries.

3. **But the physics reviewer found the deepest scoping issue: the headline regime is *linear and ≤5-dim by
   construction*.** Frozen Gaussian spectra + linear concentration mixing, with reabsorption/FRET **off** in
   `build_dataset`. On exactly-linear data "an AE cannot beat PCA" is closer to a *theorem* than an
   experimental finding about autoencoders. The paper must scope "AE ≈ PCA" to the linear forward model and
   (ideally) add nonlinear per-pixel physics (solvatochromic shift, self-quenching) before claiming the AE
   has no spectral advantage.

4. **The paper artifact under-sells and partly mis-reports a genuinely rigorous program.** Zero tables (the
   significance table and the 66-method leaderboard exist in the repo but not in the paper), no Related Work
   / Problem Statement / Limitations, a Figure-5 caption that **contradicts its own data**, and a
   reproducibility overclaim (the headline figure is hard-coded, not re-run). All fixable; none threaten the
   science.

5. **The blind-selection science is real, conservative, and well-controlled — keep it, and reframe it.** The
   strongest version of this paper is **"a physics-grounded benchmark + a characterization theorem of *when*
   blind band selection can/can't work + the spatial-necessity result,"** not "an honest appraisal of
   autoencoders." Three formalizable propositions (§8) turn the hand-wavy "three mechanisms" into claims a
   reviewer can't attack.

---

## 1. What I did

- **Reproduced the headline.** Fresh `beat_random` run: L1 PCA 0.694 / AE 0.696 beat random-95th 0.548; L5
  below random. The qualitative result reproduces exactly.
- **Ran a 5-dimension adversarial review** (methodology, methods, theory, writing, physics), each finding
  verified against the code. Full synthesis in the companion notes; the physics dimension I ran separately
  after the first pass dropped it.
- **Ran 7 new experiments** (`reports/improve_experiments.py`), summarized in §9.
- **Formalized the theory** (§8) and **cross-checked the literature** (concrete-AE = Balın et al. ICML 2019;
  BS-Net/DARecNet/TAttMSRecNet are reconstruction-based "fidelity over utility"; the 2024 field trend is
  toward *supervised embedded* selection — which is exactly this program's recommendation).

---

## 2. Overall assessment

The science is fundamentally sound and the central **negative** result is robust and correctly scoped:
under realistic confounds, blind selection (AE *and* PCA) ≈ random; the AE has no distinct advantage over
PCA; selection is a compression tool, not an accuracy tool; and per-pixel selection is provably blind to
spatial-texture classes. These survive every validity attack because the biases mostly favor the foils
(the random control is *handicapped* by lacking the diversity constraint; cross-scene pooling *raises* the
beats-random bar; best-of-panel is applied symmetrically). Several of the toughest concerns turned out to
be things the program had already diagnosed and re-run correctly in round 3. That is the signature of a
rigorous program.

The real problems are concentrated **(a) on the supervised side** (the leak, which inflates the one
positive claim) and **(b) in the paper as a written object** (no tables, no related work, an
internally-contradictory figure, an overclaim) and **(c) in the disclosure of how much the negative result
depends on a linear, adversarially-confounded generator.** None threaten the core negative science; all are
fixable.

---

## 3. The one result that changes a conclusion — the supervised leak (CRITICAL)

**Finding.** `mutual_info_classif(Xr, yr)` is computed on the **entire ROI** (`beat_random.py:67`,
`mixnoise_experiment.py:115`, `compression_curve.py:68`, `what_works.py:83`, `resolution_effect.py:68`,
`phase_map.py:59`, `clutter_robust.py:83`, `supervised_nonlinear.py:67`, `semisup_experiment.py:75`), then
the same `Xr` is split into a 30/class few-shot train + held-out test. The selector therefore sees the test
pixels' labels (select-before-CV leakage) **and** the stated "few-shot" budget is violated for the selector
(thousands of labels for selection, 30/class for the classifier). The *correct* form is already used in
`supervised_ae.py:119` and `sparse_regime.py:86`.

**My direct before/after (`--mi-leak`, 3 seeds, 180 random draws, empirical p):**

| level | random-95th | MI **leaked** (full ROI) | MI **fixed** (train-only) | blind PCA |
|---|---|---|---|---|
| L1-pristine | 0.567 | 0.677 ✓ (p.00) | 0.630 ✓ (p.00) | 0.694 ✓ (p.00) |
| L2-low | 0.476 | 0.506 ✓ (p.00) | 0.480 ✓ (p.06) | 0.479 ✓ (p.05) |
| **L3-moderate** | 0.570 | **0.578 ✓ (p.01)** | **0.552 ✗ (p.32)** | 0.542 ✗ (p.36) |
| **L4-high** | 0.590 | **0.603 ✓ (p.01)** | **0.567 ✗ (p.46)** | 0.555 ✗ (p.66) |
| **L5-severe** | 0.600 | **0.612 ✓ (p.01)** | **0.586 ✗ (p.21)** | 0.550 ✗ (p.83) |

Under every clutter level, the leak inflates MI by +0.026 to +0.036 — *exactly enough* to cross the random
threshold. **Leak-free, supervised MI does not beat random under clutter; it is indistinguishable from blind
PCA.** The "only supervised beats random under clutter" and "supervised mutInfo beats full under clutter"
sub-claims (paper.md:173–179, 239; summary.md:43–48) must be re-derived or retracted.

**What survives:** the *blind* comparisons (AE/PCA/random/full) use no labels and are intact. The leak is
in the *secondary* supervised claims only. The honest restatement: *with abundant labels to choose bands
(a fully-labeled pilot scene), supervised MI selection helps; at a realistic few-shot budget for selection,
no method — blind or supervised — beats random under clutter, and full-data wins.*

**Fix (P0):** one-line change in 9 scripts (`Xr,yr` → `Xr[L],yr[L]`), re-run, revise the supervised numbers.

---

## 4. Physics fidelity — the deeper scoping (and what my E7 settles)

**The trilinear core is physically correct** (verified): exact `render(A+B)=render(A)+render(B)` with
physics off; consistent peak/area normalization; reabsorption correctly suppresses the *blue* emission edge
(D1 cross-section ≈0.41 at the blue edge vs ≈5e-7 at the red); FRET is energy-sensible; water-Raman lands at
the right wavelength (ex470→559 nm). This is a faithful generator.

**Critical scoping issue #1 — the headline regime is linear & ≤5-dim by construction.** The clean cube is a
sum of only 5 frozen rank-1 trilinear terms (`fluorophore.py:11` `frozen=True`); the only per-pixel
variation is *linear* concentration mixing; reabsorption/FRET are **off** in `build_dataset`
(`realistic_benchmark.py:59,81`). On an exactly-linear, low-rank manifold, **PCA is near-optimal by
construction**, so "AE ≈ PCA" is partly a property of the data model, not a discovery about autoencoders.
→ **Scope the claim** to the linear forward model; to test the AE fairly, add a curved manifold
(solvatochromic peak shift / FWHM jitter driven by a microenvironment field; self-quenching), which is the
one regime a nonlinear selector *could* win.

**Critical scoping issue #2 — the clutter is spectrally white.** `add_cube_clutter`
(`realistic_benchmark.py:105–109`) imprints i.i.d. per-band gains — un-regressable by a spectral AE, unlike
the smooth, flat-fieldable clutter real instruments produce. **I tested this directly (E7).**

**E7 result — the negative result is *robust to clutter type*:**

| clutter | random-95th | full | pca_load | resid-clusterF | variance |
|---|---|---|---|---|---|
| none (nuisance only) | 0.384 | 0.415 | 0.372 (p.34) | 0.350 (p.92) | 0.387 ✓ (p.06) |
| **white** (benchmark) | 0.583 | 0.592 | 0.530 (p.90) | 0.561 (p.38) | 0.574 (p.08) |
| **smooth** (realistic) | 0.565 | 0.573 | 0.526 (p.68) | 0.524 (p.68) | 0.533 (p.67) |

Under *smooth, calibratable* clutter, blind selection **still doesn't beat random** — so the conclusion is
**not** an artifact of adversarial white clutter (this *answers and partly refutes* the physics critique on
this specific point). The deeper reason: even with the clutter removable, the **bright nuisance
fluorophores remain** — physical, class-irrelevant, high-variance, and not blindly separable from the
discriminative dyes. *The barrier is the nuisances, not the clutter spectral shape.* (Caveat: E7 tested
pca_load/resid-clusterF/variance as proxies for the blind family — established as AE≈PCA — not the conv-AE
itself; worth a one-off AE confirmation.)

**Other physics findings worth fixing:** per-pixel evaluation uses a **random train/test split on
spatially-autocorrelated smooth fields** (`classification_experiment.py:79`; also the few-shot draws) →
neighbor leakage inflates absolute F1 and perturbs the random margins (block-CV is used only in the spatial
regime — extend it to the per-pixel benchmark); **emission area-normalization is grid-dependent** so
switching `em_step` 5↔2 nm silently changes per-band SNR (`fluorophore.py:55`); **code defaults ≠ paper
Appendix A** (`nuisance_amp=2.0, photon_scale=600` vs the paper's 1.2 / 2000) — pull figure params from
`params.json`.

---

## 5. How to improve the RESULTS — prioritized

1. **[P0] Fix the supervised leak and re-run** (§3). The only change that alters a conclusion.
2. **[P0] Raise headline scripts to ≥5–6 seeds with scene-level CIs** (`beat_random`/`compression`/`spatial`
   use 2 seeds; REPEATS are same-scene resamples = pseudoreplication). The "significance-tested" language is
   currently true only for `metric_suite`.
3. **[P1] Calibrated beats-random null** (≥200 random subsets/scene, permutation p-value, Holm/BH across the
   ~32 experiments). *I already adopt this style in `improve_experiments.py` (60–180 draws + empirical p) —
   port it back.*
4. **[P1] All-pixel + impure-50% robustness pass.** Few-shot results currently evaluate only the purest 50%
   of pixels (`roi_mask`); re-run on all pixels and the hard 50% (the `frac` arg already exists).
5. **[P1] Extend spatial-block CV to the per-pixel benchmark** (physics #14) — removes spatial-autocorrelation
   leakage from the very margins the conclusions rest on.
6. **[P2] random-diverse control** (the random null is denied the 10 nm dedup that every structured selector
   gets — `beat_random.py:73`); and per-classifier verdict tables (not just `max(fs)`).

---

## 6. How to improve the PAPER — prioritized

1. **[P0] Add the significance table** (AE vs pca_load per regime: best-NL F1, bootstrap 95% CI, Wilcoxon p,
   Cohen-d, random floor + full ceiling). It already exists at `docs/spectraforge/18-metric-suite.md:94–129`;
   transplant it. The paper currently has **zero tables**.
2. **[P0] Add the leaderboard** (top ~12 of the 66-method `grand_sweep.csv`). State plainly: the best blind
   config is `pca_load[k=8]` (0.4614), just above the best AE (`ae_masked_conv` 0.4403). **Do not** write
   "PCA outranks every AE" — some `pca_load` configs rank *below* the AEs and the sweep is single-seed; label
   "within ~0.02 = seed noise."
3. **[P0] Fix the reproducibility overclaim** (paper.md:235): Figs 1–4 re-render from the model; **Fig 5 is
   hard-coded** (`paper_figures.py:123–135`). Reword to "Fig 5 plots committed numbers from beat_random.py /
   compression_curve.py / spatial_regime.py," or wire it to read the CSVs.
4. **[P0] Fix the Figure-5 spatial-panel caption — it contradicts its own data.** The caption says "only a
   spatial model can find the discriminative bands," but the panel shows **random selection = 0.710 beats
   both spatial selectors** (texture-var 0.657, texture-MI 0.688) and the spatial-CNN selector (0.534). The
   true story (which matches `OVERNIGHT-LOG.md:149–157`): a spatial *classifier* is necessary; *selection*
   mostly can't help above random and per-pixel selection falls below it. Also: the legend in the left panel
   **occludes the clean PCA/AE/mutInfo bars** — the single most important visual — move it.
5. **[P1] Add Related Work** (paper has 8 refs, none on band selection, though the code benchmarks
   concrete-AE/STG/BS-Net/mRMR/PARAFAC). Cite them; reuse `publications/generalization/paper/main.tex` +
   `publications/tpami/revision/SOTA_METHODS_EXPLAINED.md` (not the older commsai related_work, which carries
   the pre-pivot framing). Position against the *limits-of-unsupervised-feature-selection* literature
   (Dy & Brodley 2004; He-Cai-Niyogi Laplacian Score 2005; Guyon & Elisseeff 2003; Peng-Long-Ding mRMR 2005).
6. **[P1] Add a formal Problem Statement / Notation** (X ∈ R^{H×W×E×B}; per-pixel feature; S* =
   argmax_{|S|=k} A(f_S); random null A_95; full ceiling A_all) and a **Limitations** section (all-synthetic,
   linear-by-construction, 2–3 seeds for the headlines, best-of-panel bias, easy-pixel ROI).
7. **[P2] Reframe.** Drop the AE-centric second title clause; pitch as **benchmark + characterization theorem
   + spatial-necessity**. Move "the SpectraForge research agent" from the byline to Acknowledgments; replace
   "full freedom over the generator" with the safeguard framing. Pair every ratio with the absolute (e.g.
   "0.414 macro-F1; chance 0.33; oracle 0.486 → 85% of oracle"). **Venue:** as written this is a NeurIPS
   Datasets-&-Benchmarks / reproducibility paper, not TGRS/TPAMI (synthetic-only, negative-leaning).

---

## 7. How to improve the SYSTEM/METHOD — ranked, with my experimental verdicts

| # | Idea | Status / my result | Verdict |
|---|---|---|---|
| 1 | **Supervised end-to-end differentiable (concrete/Gumbel) k-subset selector** | **I ran it (E1).** Overfits the selection logits; **worst supervised method** at 30 *and* 100 labels, on clutter *and* on the XOR+FRET joint regime (concrete-sup 0.674 vs RFimp 0.750). | Negative — closes the methods reviewer's #1 gap. Pre-empts "did you try differentiable selection?" |
| 2 | **RF-importance / supervised-AE on joint (XOR/FRET) structure** | **I ran it (E1).** At 100 labels on XOR+FRET, **RFimp 0.750 beats full 0.737 and random**, supAE 0.733 (>marginal MI 0.715). | Positive (known) — confirms "RF-importance for nonlinear/joint" and a real selection-beats-full corner. **State it.** |
| 3 | **Clutter-structure-aware blind** (low-rank clutter removal → unsupervised cluster-F-ratio) | **I ran it (E3, 3 seeds, calibrated).** Best blind attempt on record: lifts blind from ~30th→~80th percentile of random; `resid-clusterF` > `resid-variance` (after de-cluttering, rank by *discriminability*, not variance). **Never crosses random-95th.** | Negative but informative — hardens the wall; the residual barrier is the nuisance fluorophores. |
| 4 | **Smooth/calibratable clutter** (is the negative an artifact of white clutter?) | **I ran it (E7).** Blind still doesn't beat random under smooth clutter. | Negative — the conclusion is robust to clutter spectral shape. |
| 5 | **Low-variance/high-Fisher counterexample** (constructive proof of mechanism (c)) | **I ran it (E6/E6b).** Oracle finds the variance-buried marker and **beats full**, but doesn't cross random-95th even when engineered. | Partial — decoupling is real at the oracle level; selection is still a weak lever at k=24/564. |
| 6 | **Add nonlinear per-pixel physics** (solvatochromic shift, self-quenching) then re-test AE vs PCA | Not run (needs generator work). The physics audit shows this is the regime where AE *could* beat PCA. | **The highest-value untried experiment** — directly tests whether "AE ≈ PCA" survives a non-linear manifold. |
| 7 | Faithful **supervised STG** + **SupCon** (replacing the failed NT-Xent contrastive) | Not run. Predicted ties (supAE already answers the substantive question). | Low payoff; do it only to remove "you strawmanned STG" objections. |
| 8 | Vanilla **robust PCA** for clutter removal | — | Won't work: clutter *and* signal are both low-rank; RPCA separates low-rank from *sparse*. State this. |

---

## 8. Theory — three propositions that turn the "mechanisms" into claims

**Prop A — Selection ≠ classification (disjoint optimizations).** S*(k)=argmax_{|S|=k} I(X_S;Y); blind
proxies maximize a label-free functional (Var(X_S), reconstruction, top-k subspace energy). With physics
off (render exactly linear), a pure-nuisance band x_j=Σ_{n∈N} L_{jn}c_n is a deterministic function of
nuisance concentrations ⊥ Y, so I(x_j;Y)=0 *exactly* while Var(x_j)∝nuisance_amp²→∞ ⇒ the top-k variance
bands are all nuisance and S_var ∩ S* = ∅. *Corollary:* a nonlinear classifier raises best-NL on a **fixed**
S, but S* is classifier-independent (I(X_S;Y) is) — so the AE's nonlinearity moves the *classifier*, never
the *argmax*. (Empirical anchor: `metric_suite` `relevance`=Σ_j I(X_j;Y), realistic variance relevance 0.203
/ gt_precision 0.000 vs mutInfo 0.545.)

**Prop B — Spiked-covariance theorem (replaces "unsupervised ≡ variance").** x = D u + N v + ε with
I(u;Y)>0, v ⊥ Y (independent seeding makes this exact). Variance score s_j = a²‖D_{j·}‖² + b²‖N_{j·}‖² + σ².
As clutter mass m·b² grows relative to a²·max‖D_{j·}‖², Kendall-τ(s, relevance) → 0 — variance ranking
becomes asymptotically random w.r.t. class. *Corollary (PCA):* pca_load(k) recovers supp(D) iff the r signal
eigendirections sit in the top-k eigenvectors of Cov(x); under m clutter modes with eigenvalue > a² and
m≥k, all top-k PCs are clutter (⇒ random). Proof: Davis–Kahan. This explains pca_load>variance on low-rank
nuisance, the collapse under m=36 clutter, the H15 partial rescue at K≈m, **and my E3** (de-clutter then
discriminability is the right move, but the nuisance subspace is class-relevant-looking and survives).
*Resolves the `cluster_fratio` apparent contradiction:* it is **not** blind-symmetric — it scores
I(X_j; ŷ) for a data-driven partition ŷ; it can exceed variance iff ŷ correlates with Y, which under clutter
it does not.

**Prop C — Impossibility of unsupervised feature selection (replaces the (c) biconditional).** If (j,j') are
exchangeable under P_X but I(X_j;Y) > I(X_{j'};Y), every *blind* selector ranks j above j' with probability
exactly ½ — it cannot use an asymmetry invisible in P_X. The clutter regime constructs exactly such pairs
(a dim dye band and an equal-variance nuisance band). **The spatial-blindness result is the sharpest instance
and is a genuine theorem:** in `spatial_regime.py:38–40` each class's texture is renormalized to identical
per-pixel marginals ⇒ P(X_pixel|Y=c) ⊥ c ⇒ I(X_pixel;Y)=0 ⇒ every per-pixel selector is at chance; class
information lives only in neighborhood functionals. **Restatement of (c):** "extractable-*blind* ⇒
variance-prominent (under clutter the only label-free statistic is decoupled from relevance); extractable-
*with-labels* does **not**." My E6/E6b is the constructive witness (oracle finds the variance-buried marker;
variance/pca don't). *Bonus:* "selection never beats full" = the data-processing inequality (X_S=f(X) ⇒
I(X_S;Y) ≤ I(X;Y)); the lone exception is a finite-sample bias-variance effect, which the program already
attributes to overfitting.

---

## 9. New experiments — summary table

All via `reports/improve_experiments.py`, 3 seeds, paper harness, calibrated nulls.

| ID | Question | Headline result |
|---|---|---|
| **P0** `--mi-leak` | Does the supervised-MI test-label leak inflate the one positive claim? | **Yes, decisively.** Leak-free MI does **not** beat random under clutter (p.21–.46); leaked MI does (p.01). |
| **E1** `--sup-select` | Does a supervised differentiable k-subset selector help? | **No.** Overfits; worst supervised method everywhere. On XOR+FRET, RFimp (0.750) beats full; concrete-sup (0.674) is worst. |
| **E3** `--clutter-blind` | Can a clutter-structure-aware *blind* selector beat random? | **No, but best-on-record.** `resid-clusterF` reaches ~80th pct of random; never crosses 95th. De-clutter then rank by *discriminability*, not variance. |
| **E6/E6b** `--typeb` | Is "extractable ⇔ variance-prominent" a law? | **Decoupled at the oracle level** (oracle finds the variance-buried marker, beats full) but the margin over random is too small to be decisive — selection stays a weak lever. |
| **E7** `--smooth-clutter` | Is the clutter-negative an artifact of white clutter? | **No.** Blind doesn't beat random under smooth (realistic) clutter either; the barrier is the nuisance fluorophores. |

---

## 10. Prioritized TODO

**P0 — before any submission (correctness + integrity)**
- [ ] Fix the supervised leak in the 9 scripts; re-run; re-derive paper.md:173–179, 239; retract/scope
      "supervised beats full/random under clutter."
- [ ] Fix the reproducibility claim (Fig 5 is hard-coded) and the Figure-5 spatial caption (it contradicts
      its data); move the occluding legend in the left panel.
- [ ] Add Table 1 (significance, from doc 18) and Table 2 (leaderboard, corrected).

**P1 — archival quality**
- [ ] ≥5–6 seeds + scene-level CIs on the headline scripts; calibrated permutation null (port from
      `improve_experiments.py`).
- [ ] All-pixel + impure-50% robustness pass; extend spatial-block CV to the per-pixel benchmark.
- [ ] Related Work + Problem Statement + Limitations; state the three Propositions (§8).
- [ ] Scope "AE ≈ PCA" to the linear forward model; **run the nonlinear-physics regime** (solvatochromic /
      self-quenching) — the highest-value science experiment remaining.

**P2 — strengthening / polish**
- [ ] random-diverse control; per-classifier tables; reconcile code defaults vs Appendix A (read from
      `params.json`); grid-invariant emission normalization.
- [ ] Re-frame title/Discussion; byline fix; pair ratios with absolutes; Reproducibility/Availability
      paragraph; choose venue (NeurIPS D&B).

---

*Bottom line: the negative science is real, conservative, and now stress-tested against the strongest
adversaries I could build — it holds. Spend effort on (1) the supervised-leak re-run (the only thing that
changes a stated conclusion), (2) turning a rigorous program into an archival paper (tables, formal
statements, an honest Figure 5, the three propositions), and (3) the one experiment that could still
surprise you: a genuinely nonlinear per-pixel forward model, where "AE ≈ PCA" is finally a fair test rather
than a near-tautology.*
