# SpectraForge & the Band-Selection Investigation — Documentation

This directory documents (1) the **SpectraForge** synthetic ME-HSI generator and its validation
harness, (2) the **investigation** that found why the perturbation-autoencoder band selector
behaves degenerately on synthetic data, and (3) the **plan + runbook** for the next phase
(trying autoencoder architectures that don't "cheat"), which is intended to be executed and
**trained on a separate machine**.

## Read in this order

| Doc | What it covers | For whom |
|-----|----------------|----------|
| [`01-system-overview.md`](01-system-overview.md) | How every part works: SpectraForge engine, GUI, the validation harness, the band-selection pipeline (`spectral_select` + `selection_core`). | Anyone onboarding. |
| [`02-investigation-and-findings.md`](02-investigation-and-findings.md) | What we did and what we found: the corrected validation harness, and the **root cause** — the spatial CAE collapses to predicting the per-band mean on this data. The full evidence chain (5 refuted hypotheses). | Anyone who needs the *why*. |
| [`03-architecture-plan.md`](03-architecture-plan.md) | How to proceed: a research-grounded **ladder of candidate architectures/objectives (C0–C4)** designed so the AE cannot cheat, and the fitness/acceptance protocol. | Whoever designs/reviews the next phase. |
| [`04-training-runbook.md`](04-training-runbook.md) | **The executable handoff.** Exact environment, commands, the model interface contract, per-candidate build specs, smoke-test criteria, and the synthetic-gate / real-accept training+comparison protocol. Written so another machine's **Claude Code agent** can implement and run it. | The agent/operator on the training machine. |
| [`05-training-results.md`](05-training-results.md) | **The report-back.** The C0–C4 ladder built, unit-tested, and trained on synthetic ME-HSI: full comparison tables, the synthetic-gate verdict (**C3 + C4 pass**), C3/C4 hyperparameter ablations, and why real-data acceptance is still pending. | Anyone reviewing the outcome. |
| [`06-photophysics-and-realistic-simulation.md`](06-photophysics-and-realistic-simulation.md) | **The photophysics + a realistic regime.** From electronic transitions (Jablonski/Kasha/Franck–Condon) to the trilinear EEM model, the confounds that decouple variance from informativeness (scatter, nuisances, shot noise, dim/overlap), and a benchmark where the learned AE finally **beats the trivial variance baseline** (0.43 vs 0.34). | Anyone extending the simulator. |
| [`07-enlarging-the-network.md`](07-enlarging-the-network.md) | **Bigger nets + the training bag-of-tricks (C5 deep-MLP, C6 1D-conv, C7 masked-conv).** The counterintuitive result: pure capacity *hurts* perturbation-selection (the deepest net reconstructs best but selects worst); the **masking objective** is the lever — masked-conv (C7) is the best learned method. | Anyone tempted to scale the model. |
| [`08-extended-experiments.md`](08-extended-experiments.md) | **Enlarged, recorded experiment grounds (4 studies).** Masking *rescues* the over-capacity net (C5→C5b); reconstruction R is **anti-correlated** with selection (corr −0.94); a confound **phase diagram** showing the learned advantage emerges only off the clean edge; budget robustness. CSVs in `reports/exp_records/`. | Anyone validating the thesis. |
| [`09-cae-verdict.md`](09-cae-verdict.md) | **Did the published CAE work? No.** The cross-experiment verdict + the decisive **density sweep**: the CAE's R stays ≈0 and it selects 0% informative bands from sparse to dense, refuting the "real data is dense so it works" excuse. The real-data claim now rests on no validated mechanism. | Decision-makers / reviewers. |
| [`10-blind-selection-search.md`](10-blind-selection-search.md) | **The broad blind-selection search (converged).** ~25 method families + 16 worktree-isolated exploration agents + densification + a 7-regime generalization battery (hundreds of evals). Winner: a **PCA-loadings selector (k≈6)** recovering **96% of the labels-using oracle, robustly and generalizably**; reconstruction-based methods rank last (the nuisance trap). | Anyone choosing a selector. |
| [`11-cae-reconstruction-audit.md`](11-cae-reconstruction-audit.md) | **"Verify the autoencoder actually works."** Pipeline verified lossless (round-trip 1e-8, no batch/chunk corruption); reconstruction inspected numerically + visually (flat output, R²<0 — the loss deceives). Exhaustive config × data search: **no setting makes the spatial CAE reconstruct** (fails even a rank-1 field). The principle survives only via the per-pixel spectral AE. | Anyone trusting the CAE. |

## One-paragraph summary of where we are

SpectraForge generates chemically-grounded synthetic 4D ME-HSI cubes (excitation × emission ×
height × width) with perfect ground truth, to validate unsupervised band-selection methods. Using
it we discovered that the published perturbation-autoencoder selector, **on these synthetic cubes**,
selects non-informative (off-peak) bands and classifies worse than random. After ruling out the
validation metric, normalization, and several selection knobs, the **root cause is the spatial
convolutional autoencoder**: on this (sparse, after global normalization) data it converges to a
degenerate "predict the per-band mean" solution — reconstruction correlation R ≈ 0 — so its
perturbation-influence is noise. A simple per-pixel **spectral** autoencoder learns the data
(R ≈ +0.28) and the *same* selection principle then recovers the informative bands. **That next phase
is now done** (see [`05`](05-training-results.md)): the C0–C4 ladder was built, unit-tested, and
trained on synthetic ME-HSI. The masked (C3) and variational+free-bits (C4) spectral autoencoders
**pass the synthetic gate and beat the trivial variance-ranking baseline**; the CAE's
influence-vs-signal correlation (−0.81) flips positive (+0.4…+0.76). Real-data acceptance remains the
open gate (the processed cubes were not available on the training machine), so the published
`"standard"` CAE stays the default.

## Status & branch

Docs 01–04 are on **`main`**. The C0–C4 implementation, comparison/ablation harnesses, tests, and
[`05-training-results.md`](05-training-results.md) are on branch
**`spectraforge/architecture-ladder-training`** (synthetic phase complete; real-data acceptance
pending). The published `"standard"` CAE is untouched and remains the default.
