# Overnight Autonomous Research Session — Plan (2026-06-26 → morning)

**Role:** I act as the main researcher — deploy trainings/experiments, validate hypotheses, form new
ones, iterate. Each experiment runs in the background; on completion I analyse → append to
`OVERNIGHT-LOG.md` → commit → launch the next. The morning deliverable is `OVERNIGHT-LOG.md` (results +
new questions) plus committed code/docs and a refreshed checkpoint.

**Mission.** Establish, with significance-tested multi-metric evidence, *where and why* the
AE+latent-perturbation selector is the method of choice — and push it to **clearly beat PCA and
full-data** in the regimes that matter (noisy, real-instrument, few-label), culminating in a
**semi-supervised AE selector**. Stay honest: report ties and losses, not just wins.

## Standing context (from docs 13–19)

- AE+perturbation is a sound *blind* selector: ties PCA on most regimes, **significantly beats PCA on
  nonlinear reabsorption**, and **beats PCA under noise/clutter**; PCA is strongest on clean/linear.
- Blind selection **can't beat full-data**; **supervised selection beats full-data, growing with noise
  (+0.056 at L5)**. → the semi-supervised AE is the key open lead.
- Both AE and PCA are **low-stability**; the AE matches full-data accuracy from 24/564 bands under noise.

## Hypotheses (each: experiment + success criterion)

| # | hypothesis | experiment | success criterion |
|---|-----------|-----------|-------------------|
| **H1** | A **semi-supervised AE** (few ROI labels guiding the perturbation ranking) beats full-data, PCA, and the unsupervised AE, approaching the supervised oracle. | `semisup_experiment.py`: AE base + {re-rank by few-label F, score-fusion, latent-discriminability weighting} vs full/PCA/AE/few-label-F/oracle, across 5 mix/noise levels. | semi-sup AE Δ vs full > 0 and > PCA and > few-label-F, across ≥3 levels, CI-backed. |
| **H2** | AE−PCA margin grows **monotonically with clutter/noise**; quantify the crossover. | fine clutter sweep (amp 0→5), AE vs PCA best-NL, bootstrap CI. | monotone AE−PCA vs clutter, crossover amp identified. |
| **H3** | **Denoising-tuned AE** (mask ratio, noise injection, latent, depth) maximises clutter robustness. | config sweep on L4/L5 cluttered regime. | a config with AE−PCA materially larger than default; recorded recipe. |
| **H4** | AE's advantage is largest at **small band budgets**; converges at large k. | budget sweep k∈{8,12,16,24,32,48} at L4. | AE−PCA decreasing in k; AE best at small k. |
| **H5** | AE selects **less-redundant / more-complementary** bands than PCA (mechanism). | redundancy (mean|corr|), conditional-MI / complementarity on selected sets. | AE redundancy < PCA at noisy levels; links to accuracy. |
| **H6** | **Consensus/ensemble** AE selection improves stability *and* accuracy. | bagged AE selection over seeds; stability (Jaccard) + accuracy. | consensus stability > single; accuracy ≥ single. |
| **H7** | Semi-sup AE turns the **nonlinear-regime** (reabsorption/FRET) gap into a clear accuracy win. | H1 selector on reabsorption/FRET, panel metric. | semi-sup AE > PCA significantly on nonlinear. |
| **H8** | Selection's value is **larger under spectral-angle (SAM)/cosine** distance (real-world classifier). | re-run full-vs-selected with SAM-KNN. | selection−full larger under SAM than Euclidean. |
| **H9** | AE-selected bands **transfer across noise levels** better than PCA. | select on level i, evaluate on level j. | AE off-diagonal retention > PCA. |

**Priority order:** H1 → H3 → H2 → H4 → H5 → H6 → H8 → H9 → H7, re-prioritised as findings arrive.

## Protocol

- Each experiment: 2 dataset seeds × ≥4 repeats, bootstrap CI on the key margin; high-res 564-band data;
  ROI pixels; few-shot where relevant. Keep each run ≲40 min so I cycle often.
- After each: append result + verdict + **new questions** to `OVERNIGHT-LOG.md`; `git commit`; refresh
  the `checkpoint-synthetic-success` tag.
- Chain stays alive via background-task completion notifications; if a run hangs, a scheduled heartbeat
  re-checks. Failures are logged and skipped, not retried in a loop.
- Honesty gate: every claim needs a fairness precondition met + CI/effect size; revisions flagged.
