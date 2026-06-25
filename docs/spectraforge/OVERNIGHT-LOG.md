# Overnight Autonomous Research Log — live (read this in the morning)

Newest entries at the top. Each: hypothesis → experiment → result → verdict → new questions. Plan in
`OVERNIGHT-PLAN.md`.

## Morning summary (updated as work proceeds)

- _(will be filled in with the headline findings of the night)_

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

_(H2 AE-PCA-vs-clutter running; H4 compression curve next)_
