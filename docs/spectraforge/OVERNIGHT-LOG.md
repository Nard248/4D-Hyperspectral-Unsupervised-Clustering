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

_(label-budget crossover sweep running)_
