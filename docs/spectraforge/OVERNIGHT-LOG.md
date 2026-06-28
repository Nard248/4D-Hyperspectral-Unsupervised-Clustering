# Overnight Autonomous Research Log — live (read this in the morning)

Newest entries at the top. Each: hypothesis → experiment → result → verdict → new questions. Plan in
`OVERNIGHT-PLAN.md`.

## Morning summary — read this first

### ★ FINAL VERDICT (after two autonomous rounds) ★

**The AE+perturbation — and the AE in general — provides no distinctive value over PCA in any role
tested (selection *or* denoising), on this synthetic ME-HSI.** Concretely, with significance testing and
a random baseline:

1. **Selection:** blind AE ≈ PCA ≈ **random** under realistic clutter (only beats random clutter-free);
   never beats full-data; the lone doc-18 nonlinear edge didn't reproduce at 3 seeds.
2. **Denoising:** classifying on the AE reconstruction helps **+0.13 on clean data**, but that is
   **linear** low-rank denoising — **PCA-reconstruction(8) does it as well or better** (the AE's
   nonlinearity adds nothing); and it helps *only* on clean data.
3. **What to actually use:** **clean data →** low-rank PCA denoising + (any) blind selection or just full
   data; **cluttered/realistic data →** regime-matched **supervised** selection (marginal MI for clutter,
   RF-importance for nonlinear) — the only thing that beats random; selection is a **compression** tool,
   not an accuracy tool, and is **unstable** under clutter (use regions/consensus).
4. **Unifying cause:** clutter is high-variance, so every unsupervised (variance/reconstruction-driven)
   method — variance, PCA-loadings, AE-recon, AE-perturbation — is defeated by it. Only labels escape.
5. **The one thing that could change this verdict: real Lichens/Collagen data** (the decisive, still-open
   gate), where informative structure may not be variance-prominent. Evaluate it with this battery —
   always including the **random** and **full-data** baselines.

*(Round-1 summary below is preserved; this box is the current, complete bottom line.)*

---


The night was spent **rigorously stress-testing the value of blind band selection** (the AE+perturbation
idea and PCA) under realistic conditions, with significance testing throughout. The findings are honest
and consequential — several over-optimistic earlier claims did **not** survive, and one big new result
emerged.

**Headline findings**

1. **Blind selection never beats full-data.** Across label budgets (6→100/class) on noisy data, full-564
   accuracy ≥ every blind selection; it only *ties* at ≤6 labels/class and pulls ahead with more labels.
   Only the *all-label* oracle beats full, and only below ~40 labels/class. (label-budget crossover)
2. **Semi-supervised few-label AE is a dead end.** Few-label re-rank/fusion ≈ the few-label F-test, and
   estimating relevance on the AE-denoised reconstruction gives **no** label-efficiency gain. (H1, H1b)
3. **THE BIG ONE — on realistic high-res cluttered data, blind selection (AE *and* PCA) does NOT beat
   RANDOM band selection.** In every cluttered regime (L2–L5) neither AE nor PCA exceeds the 95th
   percentile of random 24-band subsets; **only the supervised `mutInfo` selector beats random.** Blind
   selection beats random *only* in the pristine, clutter-free L1 case (where PCA is best). (beat-random)
   - **Mechanism (confirmed, doc-test below): it's the CLUTTER, not resolution.** Re-tested at 564/228/116
     bands — blind selection fails to beat random at *every* resolution under clutter. Fixed-pattern
     clutter creates high-variance class-irrelevant structure that variance / PCA-loadings / AE-recon all
     rank highly, so unsupervised selection picks clutter bands ≈ randomly w.r.t. class. Only labels escape it.
4. **AE ≈ PCA everywhere.** PCA is better on clean (significantly), they tie under clutter (the AE−PCA
   margin improves with clutter but never reaches a significant win), and the AE's *only* distinctive
   significant advantage remains the **nonlinear reabsorption regime** (doc 18). (H2, H4, beat-random)
5. **Compression:** blind selectors reach 95% of full accuracy at k≈64/564; supervised `mutInfo` at
   k≈24 — ~2.7× better compression from using labels. (H4)

**Bottom line (honest).** For *realistic, noisy, high-resolution* ME-HSI, **blind/unsupervised band
selection — whether the AE or PCA — adds little over random; LABEL-AWARE (supervised) selection is
necessary and clearly effective.** The AE+perturbation is competitive-with-PCA but not distinctly
better except on nonlinear structure. This is the strongest, most decision-relevant result of the whole
program.

**WHAT TO ACTUALLY USE (the constructive answer, `what_works.py`).** A clear winner emerged:
**supervised mutual-information selection** beats random in *every* regime and **beats full-data under
clutter** (0.604 vs 0.594) — simple, robust, label-aware. Blind PCA/AE beat random *only* clutter-free
(PCA ≥ AE there); even supervised RF-importance fails under clutter (only the robust marginal `mutInfo`
survives). **The AE+perturbation is never distinctly the best selector in the few-shot framework.**

**Recommendation / next moves.** (a) For real noisy ME-HSI, **use supervised `mutInfo` band selection
on the ROI labels you already have** — it's the robust winner and the only method that beats both random
and full-data under clutter. (b) The blind AE+perturbation is *not* the recommended tool; its one
rigorously-significant edge is the **nonlinear reabsorption regime under the fair-panel CV metric**
(doc 18) — worth a *supervised-AE* (latent + head) test specifically there, but the few-shot evidence
here is unpromising. (c) **Real-data validation** remains the decisive gate — evaluate with this honest
battery, and **always include the random baseline** (the night's biggest lesson: without it, you can
mistake clutter-fitting for selection). (d) Mechanism confirmed: **clutter, not resolution**, is what
collapses blind selection to random.

---

## Round 3 — failure analysis + mitigation campaign (synthetic-only; full model+data freedom)

Goal: for each documented failure, instrument the **why**, design a **mitigation** that attacks that
cause, and test it (keeping **random** + **full-data** as honesty controls). Root cause across all
failures: **unsupervised objectives (reconstruction/variance) are misaligned with discriminability** —
they rank by variance, and clutter is high-variance-but-irrelevant.

| failure | why (mechanism) | mitigation (this round) |
|---------|-----------------|-------------------------|
| **F1** blind ≈ random under clutter | variance/recon rank clutter; objective ≠ discriminability | **M1 contrastive clutter-INVARIANT encoder** (augment with known nuisance model → invariance), select by influence on the invariant rep. Flagship. Also M1b improved disentangling. |
| **F2** AE ≈ PCA | synthetic discriminative structure is ~linear → nothing nonlinear to exploit | **M2** physically-grounded *dominant*-nonlinear regime (strong inner-filter/FRET/saturation manifolds) + a manifold-capturing AE |
| **F3** selection unstable | many equiv. bands + clutter variation | **M3** consensus/ensemble + region selection |
| **F4** AE-denoise = PCA-denoise | AE nonlinearity unused on ~linear data | folded into M2 (nonlinear regime) |
| **F5** selection ≯ full-data | full has all info | reframe as compression; quantify cost/accuracy frontier |

Newest entries at the top of this Round-3 block.

### M1 — contrastive clutter-invariant selection: FAILED, with a deep mechanism

Result (L4-clutter): contrastive 0.552 < random 0.593 (no better than pca_load). **Why:** instance-
discrimination *rewards* using the data's intrinsic clutter (clutter distinguishes pixels), and the
augmentation only removes invariance to *added* clutter — so the encoder still uses clutter; occlusion
highlights clutter bands.

**Deep barrier (the honest core):** under clutter, signal and clutter are *both* low-rank spatial-spectral
factors, and "which factor is class-relevant" is **defined by the labels** — so no purely unsupervised
criterion separates them **unless it exploits a property the clutter lacks.** There is one: the signal
has **physical structure** (smooth fluorophore emission; trilinear EEM) the random clutter does not. →
two mechanism-grounded mitigations:
- **M-phys:** enforce physical structure (spectral smoothness / trilinearity) to remove non-physical
  clutter, *then* select. Attacks F1 with a property clutter lacks.
- **M-repr:** the AE's genuine theoretical edge — **nonlinear representation beats linear PCA on a
  nonlinear manifold.** Test AE-embedding vs PCA-projection classification on strongly nonlinear data
  (where H16b's linear-denoising win should *not* hold).

### M-repr — nonlinear AE embedding vs linear PCA: FAILED

| feature (reabsorb-strong) | best-NL |
|---------------------------|--------:|
| raw | 0.513 |
| **PCA-proj16 (linear)** | **0.586** |
| AE-latent16 (nonlinear) | 0.513 (−0.073 vs PCA) |
| AE-rec | 0.514 |

The AE's nonlinear 16-D embedding is *no better than raw* and far below PCA's linear 16-D projection,
even on nonlinear data. **Why:** the AE's *reconstruction* objective spends latent capacity on dominant
variance (nuisances), not the class-relevant (often low-variance) structure; the nonlinearity has nothing
to grip because the class signal isn't the dominant structure.

### THE UNIFYING FINDING (all of Round 3 so far)

**Every unsupervised method — PCA, AE-recon, AE-latent, contrastive — captures the DOMINANT VARIANCE, not
the CLASS-RELEVANT structure.** When class ≠ dominant variance (clutter, nuisances, subtle nonlinearity —
the realistic case), they all fail; only label-using methods find class-relevant structure. This is not a
model deficiency; it is the definition of "unsupervised." → The only honest way to make the AE+perturbation
idea *work* is to make it **supervised**: an AE with a classification head whose latent is forced to encode
class-relevant structure, then perturbation finds class-relevant bands (and can capture nonlinear/joint
structure marginal `mutInfo` misses). → **M-sup** (running).

### M-sup — supervised AE + perturbation (the working version)

**Smoke (clutter, 1 seed, undertrained):** supAE* 0.578 ≈ mutInfo* 0.583 (tie), > RFimp* 0.553, < full
0.613. On clutter (≈linear) supAE just matches marginal MI — expected. **The decisive test is FRET**
(nonlinear/XOR): the supAE's classifier head should learn the interaction and its perturbation should
highlight *both* interacting dyes' bands — which marginal `mutInfo` cannot see (it failed on FRET; only
`RFimp` won). If supAE ≥ RFimp on FRET *and* ties mutInfo on clutter, it's a **unified supervised
selector that wins where each baseline fails** — the genuine "it works" result.

**Full run (3 seeds):** clutter — supAE* 0.556 ≈ mutInfo 0.556 ≈ random 0.559 (no one beats random here;
full 0.595 wins). FRET — **supAE* 0.526 is the best selector** (> RFimp 0.520 > mutInfo 0.519 > random
0.515): the classifier head *does* capture the XOR the marginal MI misses. **But margins are tiny
(+0.007, within noise) and nothing beats full-data.**

**Why selection barely matters here (the key realization):** the discriminative signal is **spread across
a broad emission window (many redundant bands)**, so selecting 24 ≈ using all 564. **Band selection only
matters when the signal is SPARSE/concentrated.** → build a **sparse-nonlinear regime** (narrow
fluorophore peaks at a few (ex,em) points + FRET + heavy clutter/noise) where full overfits the noise,
random misses the signal, and a method that *finds the few nonlinear signal bands wins decisively*. This
is where the supervised AE should win *big*, honestly (sparse concentrated signal is physically real).

### M-sparse — does selection (and supAE) win BIG when the signal is sparse? (building)

---

## Round 2 — continuing (constructive follow-ups to round 1)

Round 1 closed the negatives (blind ≈ random under clutter; use supervised mutInfo). Round 2 chases the
*constructive* leads, newest first.

**Round-2 hypotheses:**
- **H15 (running): clutter-robust UNSUPERVISED selection.** Project out the top-K (clutter) PCA subspace,
  select bands on the residual (signal subspace). Can unsupervised selection be *rescued* to beat random
  under clutter? — the most valuable lead (revives the label-free use case).
- **H10: supervised NONLINEAR selector vs marginal mutInfo on nonlinear data.** mutInfo is per-band
  (misses interactions); on reabsorption/FRET a nonlinear supervised selector (RFE, mutInfo-on-AE-latent,
  supervised-AE) should win. The AE's last best shot — made supervised.
- **H12: when is supervision worth it?** Supervised−blind gap vs (label budget × clutter) — a phase map.
- **H13: stability** of supervised vs blind selection (Kuncheva/Jaccard) — does mutInfo also select
  consistently, or is it noisy at few labels?
- **H14: spectral-angle (SAM) classifier** — does the real-world distance metric change the picture?

_(Round-2 entries appended below as they complete.)_

### H16 — the AE as a DENOISER (a genuine positive!) + H16b control running

| regime | raw-full | AErec-full | raw-PCA | AErec-PCA | Δfull |
|--------|---------:|-----------:|--------:|----------:|------:|
| clean | 0.755 | **0.879** | 0.699 | **0.901** | **+0.124** |
| clutter | 0.594 | 0.544 | 0.549 | 0.516 | −0.050 |

Used as a **denoiser** (classify on the reconstruction, not select bands), the AE lifts accuracy +0.124
on clean data. **But the control (H16b) defeats the AE claim:** it's **linear low-rank denoising**, done
as well or better by PCA-reconstruction.

**H16b — nonlinear (AE) vs linear (PCA-recon) denoising, full-data best-NL:**

| regime | raw | **pcaR8 (linear)** | AErec (nonlinear) | AErec − pcaR8 |
|--------|----:|-------------------:|------------------:|--------------:|
| clean | 0.755 | **0.888** | 0.879 | −0.009 |
| low | 0.569 | 0.525 | 0.547 | +0.021 |
| moderate | 0.583 | 0.531 | 0.559 | +0.028 |
| clutter | 0.594 | 0.521 | 0.544 | +0.023 |

**Verdict:** the clean-data denoising win is real but **linear** — PCA-reconstruction(8) (0.888) ≥
AE-reconstruction (0.879); the AE's nonlinearity adds *nothing* (−0.009 on clean). Denoising helps **only
on clean** data (under noise/clutter it hurts vs raw, both methods). **Useful positive: low-rank
(PCA-8) reconstruction denoising adds +0.13 accuracy on clean data** — a simple, linear preprocessing
step. **The AE provides no distinctive value over PCA in *any* role — selection or denoising.**

### ROUND-2 CONCLUSION — the decision map + the honest verdict

Combining H15 (clutter-robust), H10 (supervised-nonlinear), H12 (phase map):

**The decision map (H12, when is each approach worth it):**
- **Blind selection (PCA/AE) beats random ONLY when clutter ≈ 0** (+0.04…+0.09); under *any* realistic
  clutter it collapses to ≈ random. So **blind selection is only useful on clean instruments.**
- **Supervised selection beats full-data only in one corner: clutter present + very few labels
  (~10/class)** (+0.02…+0.03). With ≥25 labels, full-data wins — so selection is then for
  **compression/cost**, not accuracy.
- **Full data is the best accuracy choice across most of the map** (any moderate label budget).

**The selector verdict (H10):** the best selector is **regime-matched supervision** — **marginal
mutInfo** for clutter/linear confounds, **RF-importance** for nonlinear (interaction) structure (marginal
MI fails on FRET/XOR). **The AE+perturbation is not distinctly best in any regime**, and its lone doc-18
nonlinear edge **did not reproduce** at 3 seeds (AE *below* PCA on reabsorption). H15: clutter-suppression
helps blind selection but cannot rescue it (labels still needed).

**Stability (H13):** under clutter, **every selector is highly unstable** — mean pairwise Jaccard of the
24-band sets across scenes is ~0 (PCA 0.021, AE 0.000, mutInfo* 0.014, RFimp* 0.036). The sets are nearly
disjoint because the clutter differs per scene and selectors track it. *Even supervised selection doesn't
pick a consistent band set.* Practical implication: select **spectral regions / consensus across scenes**,
not exact bands, for a fixed-band instrument.

**Bottom line of the whole program (rounds 1+2):** on realistic ME-HSI, **band selection is a
compression tool, not an accuracy tool**; when it matters, **use regime-matched *supervised* selection**;
the unsupervised AE+perturbation idea, while sound, is **dominated by simpler methods everywhere** and is
**indistinguishable from random under clutter**. Selection is also **unstable** under clutter (use regions
/ consensus). The unifying cause: **clutter is high-variance, so every variance/reconstruction-driven
(unsupervised) method is defeated by it; only label-aware methods escape.** Always benchmark against
**random** and **full-data**.

### H13 — selection stability (L4-high, 3 seeds, mean pairwise Jaccard)

| selector | Jaccard |
|----------|--------:|
| PCA | 0.021 |
| AE | 0.000 |
| mutInfo* | 0.014 |
| RFimp* | 0.036 |

All near-zero → selected band-sets are nearly disjoint across scenes; no selector (blind or supervised)
is stable under clutter. → testing H16 (AE as a *denoiser*, a new role) next.

### H12 — phase map (full, 2 seeds; few-shot best-NL F1)

A) `mutInfo − full` (supervised selection minus full-data):
```
clutter\labels   10      25      50     100
   0.0        -0.032  -0.039  -0.083  -0.112
   1.5        +0.023  -0.016  -0.065  -0.093
   3.0        +0.031  -0.001  -0.036  -0.074
   4.5        +0.020  -0.002  -0.049  -0.072
```
B) `PCA − random` (blind selection minus random):
```
   0.0        +0.039  +0.070  +0.080  +0.088
   1.5        -0.003  -0.006  +0.001  +0.002
   3.0        +0.002  -0.017  -0.021  -0.013
   4.5        -0.011  -0.012  -0.036  -0.016
```
**Verdict:** supervised selection beats full only at **clutter>0 + ~10 labels/class**; blind beats random
only at **clutter=0**. A clean, actionable decision map.

### H10 — supervised/nonlinear selectors on nonlinear data (reabsorption, CV-panel best-NL; smoke)

| method | best-NL |
|--------|--------:|
| pca_load | 0.560 |
| AE | 0.566 (+0.006 vs pca) |
| RFimp* | 0.586 |
| **mutInfo*** | **0.595 (best)** |
| mRMR* | 0.529 |

**Full run (3 seeds + FRET, CV-panel best-NL):**

| selector | reabsorb | fret |
|----------|---------:|-----:|
| pca_load | 0.597 | 0.656 |
| AE | 0.569 (**−0.028** vs pca) | 0.660 (+0.004) |
| mutInfo* | 0.579 | 0.584 |
| **RFimp*** | **0.614** | **0.667** |

**Verdict (revises doc 18):** (1) **the AE's doc-18 "nonlinear edge over PCA" does NOT robustly
reproduce** — at 3 seeds it is *below* PCA on reabsorption (−0.028) and a tie on FRET. The lone AE win
was config/seed-sensitive. (2) **`RF-importance` (supervised *nonlinear*) is the best selector on
nonlinear data**, beating PCA, AE, and marginal `mutInfo`; marginal `mutInfo` is *poor* here (esp.
FRET/XOR — per-band MI can't see interactions). **So the best selector is regime-matched supervision:
marginal MI for clutter/linear, RF-importance for nonlinear.** The AE+perturbation is **not distinctly
best in any regime** — the honest, near-final verdict of the whole program.

### H15 — clutter-robust unsupervised selection (near-miss)

| L4-high (random µ/95th = 0.563/0.587) | | L5-severe (0.569/0.592) | |
|---|---:|---|---:|
| pca_load (vanilla) | 0.559 | pca_load | 0.567 |
| **pcaRes10** | **0.581** | **pcaRes20** | **0.576** |
| pcaRes40 | 0.542 | pcaRes40 | 0.556 |
| mutInfo* | 0.604 ✓ | mutInfo* | 0.615 ✓ |

**Verdict:** projecting out the top ~10–20 (clutter) PCs and selecting on the residual **lifts blind PCA
selection meaningfully** (0.559→0.581, 0.567→0.576) — confirming the clutter mechanism (remove clutter →
selection improves) — **but it does NOT cross random's 95th percentile.** Removing too many PCs (40–80)
*hurts* (takes signal with the clutter). So clutter-suppression is a **partial rescue, not a fix**: under
clutter, unsupervised selection still can't beat random; labels remain necessary. Optimal K ≈ clutter
mode count. → H10 (the AE's last shot: supervised/nonlinear selectors on nonlinear data) next.

---

## Detailed log — Round 1 (newest first)

### beat-random — does blind selection beat random? (k=24, few-shot 30/class, 12 random draws)

| level | random µ / 95th | PCA | AE | mutInfo* | beats random |
|-------|-----------------|----:|---:|---------:|--------------|
| L1-pristine | 0.506 / 0.573 | **0.699** | **0.655** | 0.696 | AE & PCA **YES** |
| L2-low | 0.448 / 0.479 | 0.470 | 0.449 | **0.504** | only mutInfo |
| L3-moderate | 0.544 / 0.574 | 0.539 | 0.548 | **0.576** | only mutInfo |
| L4-high | 0.563 / 0.587 | 0.549 | 0.575 | **0.604** | only mutInfo |
| L5-severe | 0.569 / 0.592 | 0.549 | 0.569 | **0.615** | only mutInfo |

**Verdict:** blind AE/PCA beat random **only** in pristine L1; under any clutter they collapse to random,
while supervised `mutInfo` beats random throughout. The single most important honesty result of the
night. (Machinery validated: L1 shows PCA clearly above random, so the test detects a good selection.)
→ resolution-effect test launched to confirm the mechanism.

### resolution-effect — is it redundancy (resolution) or clutter? (L4-high, k=24)

| em_step | nbands | random | PCA | AE | mutInfo* | blind>random |
|--------:|-------:|-------:|----:|---:|---------:|--------------|
| 2 | 564 | 0.563 | 0.549 | 0.575 | 0.604 | no |
| 5 | 228 | 0.563 | 0.544 | 0.558 | 0.601 | no |
| 10 | 116 | 0.567 | 0.564 | 0.554 | 0.573 | no |

**Verdict: REFUTES the resolution-redundancy hypothesis.** Blind selection fails to beat random at every
resolution under clutter → the cause is the **clutter itself** (class-irrelevant high-variance structure
that misleads every unsupervised criterion), not high resolution. Honest: hypothesis tested and rejected;
correct mechanism identified. → launched `what_works.py` (the constructive counterpart: which selector to
actually use per regime; does a supervised *nonlinear* selector win on nonlinear data?).

### what_works — the constructive answer (k=24, few-shot 30/class, best-NL F1)

| regime | random / 95th | full-564 | PCA | AE | mutInfo* | RFimp* | beats random |
|--------|---------------|---------:|----:|---:|---------:|-------:|--------------|
| clean | 0.508 / 0.576 | 0.755 | **0.699** | 0.655 | 0.696 | 0.668 | all |
| clutter | 0.563 / 0.588 | 0.594 | 0.549 | 0.575 | **0.604** | 0.569 | **only mutInfo** |
| nonlinear | 0.365 / 0.381 | 0.419 | **0.396** | 0.388 | 0.384 | 0.382 | all |

**Verdict:** **supervised `mutInfo` is the robust winner** — beats random everywhere and beats full-data
under clutter. Blind PCA/AE work only clutter-free (PCA ≥ AE). Even supervised RF-importance fails under
clutter (30-label/564-feature importances are too noisy; marginal MI is robust). The AE is never
distinctly best. This is the constructive close to the night: **use label-aware MI selection.**

---

## Session status: CONVERGED

The hypotheses converged to a clear, consistent, honest conclusion (above). Remaining agenda items
(H3 AE-config tuning, H5 redundancy, H6 consensus) are **deprioritised** — the core finding (blind
selection ≈ random under clutter; the AE is not distinctly better) undercuts their value. The one
worthwhile future test is a **supervised-AE on the nonlinear regime with the fair-panel metric**. I have
**stopped launching new runs** rather than burn compute on marginal confirmations; the picture is solid.

---

## Session start — 2026-06-26

Kicking off with **H1 (semi-supervised AE selector)** — the highest-value open lead from doc 19: blind
selection can't beat full-data, but supervised selection beats it by up to +0.056 at severe noise. Can a
few ROI labels guiding the AE's perturbation ranking capture that, beating PCA and full-data while
keeping the AE's denoising robustness?

### H1 (semi-supervised AE) — first result + pivot

**Setup:** AE (unsupervised) + few-ROI-label guidance (rerank / fuse) vs full/PCA/AE/few-label-F/oracle,
high-res 564-band, 20 labels/class. **Result (1-seed smoke):**

| level | full | PCA | AE | AE+rerank | AE+fuse | fewF | oracle* |
|-------|-----:|----:|---:|----------:|--------:|-----:|--------:|
| L1 | 0.691 | 0.668 | 0.679 | 0.656 | 0.644 | 0.631 | 0.678 |
| L5 | 0.545 | 0.501 | 0.500 | 0.498 | 0.519 | 0.521 | **0.574** |

**Verdict:** the simple few-label rerank/fuse **does not beat full-data**, and `AE+fuse ≈ fewF` — so
fusing AE influence adds little. The bottleneck is that **20-label ANOVA-F relevance is too noisy** to
pick the winning bands; only the oracle (*all* labels) beats full. The AE still beats PCA at L5 (+0.018).

**New hypothesis (H1b):** the AE's real lever in the supervised setting is **label efficiency** —
estimate few-label relevance on the **AE-denoised reconstruction** (clutter removed) rather than raw
data, so good bands are found with *fewer* labels. Test: does AE-denoised relevance beat raw few-label
relevance and full-data at a *smaller* label budget, under clutter? → `semisup_denoise.py`,
label-budget sweep {10,25,50,100}/class on the cluttered L4/L5.

### H1b (AE-denoised relevance / label efficiency) — honest negative + reframing

**Result (1-seed smoke, L5-severe):** labels/class=25 → full=0.566 beats all selections (oracle 0.595
edges it); **fewF-den ≈ fewF-raw** (0.527 vs 0.524 — denoising the relevance estimate does NOT help).
labels/class=100 → full=**0.857**, dominating everything (even oracle 0.790).

**Verdict (reshapes the agenda):** full-data accuracy **scales strongly with label count**. The
"selection beats full-data" phenomenon (doc 19) is **narrow** — it needs *very few* labels (≤12/class)
+ high clutter. At a reasonable budget full-data wins, and AE-denoised relevance buys no label
efficiency. **So the AE's defensible value is COMPRESSION (match full-data accuracy from a few % of
bands) + NOISE-ROBUSTNESS vs PCA — not beating full-data.** The "semi-supervised AE beats full" lead
(H1) is a **dead end at realistic label budgets**; recorded as an honest negative.

**Next:** map the **label-budget crossover** precisely (budgets 6→40, L4/L5, 2 seeds) — where does
full-data overtake selection, and is the AE the best selector in the few-label regime? Then pivot the
night to the *real* value props: **H2** (AE−PCA margin vs noise, monotonic) and **H4** (compression:
accuracy vs #bands — how few bands can the AE use while matching full?).

### Label-budget crossover map (L5-severe, 2 seeds × 5 repeats) — closes the "beat full" question

| labels/class | full | PCA | AE | fewF-raw | fewF-den | oracle* |
|-------------:|-----:|----:|---:|---------:|---------:|--------:|
| 6  | 0.407 | 0.404 | 0.408 | 0.400 | 0.391 | **0.454** |
| 10 | 0.443 | 0.441 | 0.427 | 0.426 | 0.437 | **0.488** |
| 15 | 0.485 | 0.471 | 0.470 | 0.470 | 0.470 | **0.521** |
| 25 | 0.571 | 0.528 | 0.529 | 0.534 | 0.536 | **0.579** |
| 40 | 0.677 | 0.610 | 0.608 | 0.612 | 0.615 | 0.658 |

**Definitive verdict:** **blind selection NEVER beats full-data** — it only *ties* at the very-few-label
edge (≤6/class), and full-data pulls *further ahead as labels grow*. Only the **all-label oracle** beats
full, and only below ~40 labels/class. `fewF-den ≈ fewF-raw` everywhere (denoised relevance is a dead
end). So the AE/selection value proposition is **NOT accuracy improvement over full data.**

**Reframe for the rest of the night → the real, defensible value props:**
1. **Compression** (H4, now central): how few bands can a selector use while *retaining* full-data
   accuracy? Selection lets you acquire 24/564 bands at ~equal accuracy — quantify the accuracy-vs-#bands
   curve and which selector compresses best.
2. **Noise-robustness vs PCA** (H2, running): is the AE the best *blind* selector as clutter grows?
3. **Nonlinear-regime edge** (doc 17/18): the AE's significant win on reabsorption.

### H2 — AE−PCA margin vs clutter (base fixed, few-shot 25/class, bootstrap CI)

| clutter | AE | PCA | margin | 95% CI |
|--------:|---:|----:|-------:|--------|
| 0.0 | 0.412 | 0.449 | −0.037 | [−0.049, −0.025] (PCA wins) |
| 0.5 | 0.420 | 0.424 | −0.005 | [−0.016, +0.008] |
| 1.0 | 0.476 | 0.467 | +0.009 | [−0.005, +0.025] |
| 2.0 | 0.508 | 0.503 | +0.005 | [−0.008, +0.018] |
| 3.0 | 0.522 | 0.520 | +0.002 | [−0.013, +0.016] |
| 4.5 | 0.527 | 0.533 | −0.006 | [−0.027, +0.013] |

**Verdict:** the AE−PCA margin **improves monotonically** as clutter rises (−0.037 → ~0), confirming the
*direction* (clutter relatively helps the AE) — **but the AE never *significantly* beats PCA** (CI
excludes 0 only at clean, *against* the AE). So under clutter it is a **tie**, not a win. The doc-19
"+0.020" was within noise. The AE's distinctive significant win remains the **nonlinear reabsorption
regime** (doc 18).

### Mid-session synthesis (honest)

The night has rigorously **closed over-optimistic claims**: (a) selection does **not** beat full-data
(any budget); (b) semi-supervised few-label AE is a **dead end**; (c) AE **ties** PCA under noise
(better only directionally), wins significantly only on **nonlinear** data. The AE's real value prop is
**compression + the nonlinear edge**, not accuracy dominance. Pivoting the rest of the night to nail
the compression story (H4) and the mechanism (H5 redundancy), plus consensus stability (H6).

### H4 — Compression curve (L3-moderate, 50 labels/class; full-564 = 0.683)

| k | PCA | AE | mutInfo* | random | AE %full |
|--:|----:|---:|---------:|-------:|---------:|
| 12 | 0.552 | 0.552 | 0.604 | 0.547 | 81% |
| 24 | 0.606 | 0.616 | **0.659** | 0.638 | 90% |
| 32 | 0.616 | 0.632 | 0.658 | 0.618 | 93% |
| 64 | 0.656 | 0.657 | 0.680 | 0.662 | 96% |

**Findings:** (a) **AE ≈ PCA** at every k (AE marginally ahead mid-range); both hit 95% of full at
**k=64**. (b) **Supervised `mutInfo` compresses ~2.7× better — 95% of full at k=24.** (c) **Red flag:
random is competitive** (k=24: random 0.638 ≈ AE 0.616 ≈ PCA 0.606) — in this high-res, redundant,
cluttered, few-label regime a random 24-band subset captures most of the signal, so **blind selection
barely beats random; only supervised selection clearly helps.**

→ Launched a rigorous **beat-random** check across all 5 levels (does AE/PCA exceed the 95th percentile
of random subsets, and does the AE beat random where PCA doesn't?).

_(beat-random running)_

> Note: the chain stalled after H2 (launched as a detached process → no completion wake-up). Fixed:
> all experiments now run as tracked background tasks.
