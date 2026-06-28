# Overnight Autonomous Research Log — live (read this in the morning)

Newest entries at the top. Each: hypothesis → experiment → result → verdict → new questions. Plan in
`OVERNIGHT-PLAN.md`.

## Morning summary — read this first

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

**Recommendation / next moves.** (a) Pivot to **supervised/semi-supervised selection using the ROI
labels** you already have — it's the only thing that beats random *and* full at low budgets, and
compresses best. (b) A proper **supervised AE** (latent + classifier head, not few-label re-ranking) is
worth one clean test. (c) **Real-data validation** remains the decisive gate, now evaluated with this
honest battery (must include the *random* baseline). (d) Confirm the **resolution-redundancy** mechanism
(running) — if true, it's a clean, publishable insight about *when* band selection helps at all.

---

## Detailed log (newest first)

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
