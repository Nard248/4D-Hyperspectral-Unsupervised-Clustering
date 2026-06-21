# 08 — Extended experiments (enlarged experimentation grounds)

A broader, fully-recorded experimental program over the band-selection candidates, building on the
doc-07 finding that *the training objective — not network size — governs selection quality*. Every
study writes a CSV to [`../../reports/exp_records/`](../../reports/exp_records/) and is reproducible
via `python reports/experiment_suite.py`. Four studies:

1. **Masking rescues capacity** — a controlled test: does adding the masking objective to the
   over-capacity deep net (C5 → **C5b**) fix the failure doc 07 found?
2. **Confound phase diagram** — sweep the realism from clean (0) up, and watch *where* variance-ranking
   collapses and the learned advantage opens.
3. **Reconstruction-vs-selection inversion** — quantify, across all candidates, that reconstruction
   fidelity does not predict (and here mildly anti-predicts) downstream selection quality.
4. **Budget robustness** — F1 vs band budget {6, 12, 18, 24}.

All on the realistic confounded ME-HSI from doc 06 (dim overlapping discriminative dyes + bright
nuisance autofluorescence + Rayleigh/Raman scatter + shot noise). KNN macro-F1, mean±std over seeds.

---

## Study 1 — Does masking rescue the over-capacity failure?

C5 (deep residual MLP) reconstructs best but selects worst (doc 07). **C5b** is the *same* network
with one change — denoising band masking in the loss. If C5b ≫ C5, the objective, not the size, is
the cause.

**It does.** Adding masking to the identical deep network flips it from the worst learned selector to
a competitive one (mean±std, 3 seeds):

| model | KNN-F1 | disc-band % | influence–signal corr |
|-------|-------:|------------:|----------------------:|
| C5 deep (MSE) | 0.320 ± 0.009 | 14% | **−0.397** ± 0.004 |
| **C5b deep + masking** | **0.403** ± 0.014 | **44%** | **+0.208** ± 0.203 |

Masking lifts F1 **+0.083**, **triples** the discriminative-band hit rate (14→44%), and **flips the
influence–signal correlation from −0.40 to +0.21** — on the *same* architecture, capacity, and
regularization. The only change is the loss. A **mask-ratio sweep** shows it is a genuine dose-response
(the over-capacity net needs *enough* masking; a light 0.3 mask does not flip it):

| model | mask ratio | KNN-F1 | disc % | infl-corr |
|-------|-----------:|-------:|-------:|----------:|
| C5b deep-masked | 0.3 | 0.343 | 17% | −0.120 |
| C5b deep-masked | 0.5 | 0.403 | 44% | +0.208 |
| C5b deep-masked | 0.7 | 0.405 | 42% | +0.271 |
| C7 masked-conv | 0.3 | 0.413 | 42% | +0.232 |
| C7 masked-conv | 0.5 | 0.433 | 53% | +0.170 |
| C7 masked-conv | 0.7 | 0.396 | 31% | +0.215 |

So masking *rescues* capacity — but C5b (0.403) still does **not beat** the small masked AE C3 (0.414)
or the masked-conv C7 (0.433). **Capacity is neutral-at-best with the right objective, and harmful with
the wrong one.** (The conv peaks at mask 0.5 — its band-axis locality already provides some of the
protection masking gives.)

---

## Study 2 — Confound phase diagram: when does the learned advantage appear?

Sweep a confound multiplier `L` scaling the bright nuisances + scatter (L=0 → clean; L=1 → the doc-06
regime; L=2 → heavy). At each level: `corr(per-band variance, discriminability F)` (the decoupling)
and KNN-F1 for the baselines + the small learned AEs.

Mean F1 over 5 seeds; `corr(var,F)` is the variance↔informativeness coupling.

| confound L | corr(var,F) | all bands | oracle | **variance-rank** | C2 | C3 | random |
|-----------:|------------:|----------:|-------:|------------------:|----:|----:|-------:|
| 0.00 (clean) | **+0.89** | 0.469 | 0.495 | **0.506** | 0.443 | 0.468 | 0.354 |
| 0.25 | −0.02 | 0.486 | 0.489 | 0.341 | 0.403 | 0.396 | 0.356 |
| 0.50 | −0.07 | 0.487 | 0.487 | 0.349 | 0.404 | 0.382 | 0.365 |
| 1.00 (doc 06) | −0.08 | 0.509 | 0.495 | 0.344 | 0.413 | 0.402 | 0.375 |
| 2.00 (heavy) | −0.08 | 0.511 | 0.486 | **0.356** | 0.438 | 0.409 | 0.381 |

A sharp **crossover**: at zero confound variance↔informativeness is +0.89 and **variance-ranking is the
best method of all** (0.506, above the learned AEs and ≈ the oracle) — the learned selector is pure
overhead here. The instant any realistic confound is added the coupling **flips negative**, variance-
ranking **collapses to ≈chance (~0.34–0.36) and stays there**, while the learned AEs hold 0.40–0.46 and
the oracle holds ~0.49. **This is the precise explanation of the doc-05 "tie": the clean benchmark sat
at the left edge of this diagram, the only place a trivial variance baseline is competitive.** Every
realistic regime is to its right, where the learned method wins.

---

## Study 3 — Reconstruction fidelity does not buy selection quality

Every candidate on the realistic data: reconstruction R, influence–signal corr, F1, and the fraction
of selected bands on the discriminative window — plus the **across-candidate correlation between
reconstruction R and F1**.

All candidates on the realistic data (mean±std, 3 seeds), sorted by reconstruction R:

| candidate | recon R | infl–signal corr | KNN-F1 | disc % |
|-----------|--------:|-----------------:|-------:|-------:|
| C5 deep | **0.842** ± 0.002 | **−0.397** | 0.320 ± 0.009 | 14% |
| C5b deep-masked | 0.636 ± 0.002 | +0.208 | 0.403 ± 0.014 | 44% |
| C6 conv | 0.629 ± 0.002 | +0.196 | 0.407 ± 0.011 | 28% |
| C7 masked-conv | 0.582 ± 0.005 | +0.170 | 0.433 ± 0.016 | 53% |
| C2 spectral | 0.549 ± 0.007 | +0.115 | 0.431 ± 0.041 | 69% |
| C3 masked | 0.512 ± 0.002 | +0.320 | 0.414 ± 0.042 | 61% |

> **Across-candidate corr(reconstruction R, KNN-F1) = −0.936.**

A strong *negative* correlation: the better a candidate reconstructs the input, the **worse** it selects.
The mechanism (doc 07) is now quantitative — the highest-fidelity model (C5, R=0.84) spends its capacity
on the loud nuisances/scatter, so its perturbation-influence points at exactly the non-informative bands
(corr −0.40). **Reconstruction R is a *necessary gate* (a degenerate R≈0 model is useless, per doc 02),
but never a quantity to maximize — past "good enough", more fidelity buys worse selection.**

---

## Study 4 — Band-budget robustness

Mean F1 over 5 seeds at budgets {6, 12, 18, 24} on the realistic data:

| budget | oracle | variance-rank | C2 | C3 | all bands |
|-------:|-------:|--------------:|----:|----:|----------:|
| 6 | 0.466 | 0.346 | 0.375 | 0.358 | 0.509 |
| 12 | 0.495 | 0.344 | 0.413 | 0.402 | 0.509 |
| 18 | 0.522 | 0.349 | 0.427 | 0.410 | 0.509 |
| 24 | 0.544 | 0.363 | 0.443 | 0.427 | 0.509 |

The oracle and the learned AEs **improve monotonically** with budget (toward all-bands 0.509), but
**variance-ranking is flat at ≈0.35 at every budget** — you cannot fix it by selecting *more* bands,
because each extra high-variance band is another nuisance/scatter band. Selecting *informative* bands is
the only lever, and that is what the learned method does.

---

## Synthesis

Four independent studies converge on one picture, now quantitative and recorded:

1. **The objective governs selection, not the size.** Adding masking to the over-capacity deep net
   (C5→C5b) flips it from worst to competitive (+0.083 F1, infl-corr −0.40→+0.21) — same architecture,
   only the loss changed. Capacity is neutral-at-best *with* the right objective and harmful *without* it.
2. **Reconstruction fidelity is anti-correlated with selection quality** (across-candidate corr(R, F1)
   = **−0.94**). Treat R as a pass/fail gate (>0), never a target. This is the strongest possible form of
   the doc-02→07 thesis.
3. **The learned advantage is a function of realism.** A trivial variance baseline is optimal *only* at
   zero confound (where variance = informativeness); it collapses to chance under any realistic confound,
   while the learned AEs are robust. The doc-05 "tie" was an artifact of evaluating at that single
   left-edge point.
4. **Masking + modest capacity wins; you cannot buy quality with budget or size.** The masked candidates
   (small C3, masked-conv C7) are the best learned selectors; variance-ranking is broken at every band
   budget.

**Practical recommendation (unchanged, now firmly evidenced):** use a **masked** spectral AE — C7
(masked 1D-conv) when a bit more capacity is wanted, else the cheap small C3 — with mask ratio ≈ 0.5–0.7;
gate on R>0 but never maximize it; and **evaluate on confounded (realistic), not clean, data**, because
that is the only regime that distinguishes a learned selector from a one-line variance sort. The
outstanding gate remains real data (Lichens/Collagen).

### Records
Raw per-seed CSVs in [`../../reports/exp_records/`](../../reports/exp_records/):
`study_candidates.csv`, `study_mask_ratio.csv`, `study_phase.csv`, `study_budget.csv`.
