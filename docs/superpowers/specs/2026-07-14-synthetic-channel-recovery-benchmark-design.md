# Design Spec — Controlled Synthetic Channel-Recovery Benchmark

- **Author:** Narek Meloyan
- **Date:** 2026-07-14
- **Status:** Approved (design); implementation starting
- **Related:** `src/channel_select/` (engine under test), `src/selection_core/` (shared math),
  `publications/generalization/` (submission 122 — the paper this strengthens),
  `experiments/pamap2/general_pamap2_loso.py` (harness pattern to mirror).

## 1. Motivation

Peer review of submission 122 ("Label-Free, Dependency-Aware Channel Selection") raised two
substantive doubts that the two real domains do not fully close:

- **"Why select at all?"** On PAMAP2 the 27 channels are intrinsically redundant, so *no*
  selector (including supervised MI) beats random. The reviewer correctly reads this as "any
  subset works, so why bother."
- **"Only two domains; is it really general / dependency-aware?"** The paper claims to model
  *conditional* relevance, but on saturated real data that claim cannot be isolated from marginal
  scoring.

A **controlled synthetic benchmark with known ground-truth channels** answers both with hard,
quantitative claims that real data cannot provide, because we control exactly which channels carry
signal. This is standard in the feature-selection literature (XOR / Madelon-style benchmarks).

## 2. Goals / Non-Goals

**Goals**
- Prove **necessity** (beats random): the method recovers the known informative channels at high
  precision while random cannot, and its selected-K downstream accuracy beats random-K by a wide,
  by-construction margin.
- Prove **dependency-awareness**: recover channels informative *only jointly*, which marginal
  scorers (variance/PCA) miss.
- Prove **redundancy handling**: MMR avoids selecting near-duplicate channels.
- Show a **redundancy law**: the method-vs-random gap grows monotonically with data redundancy —
  turning "why select" into "the value of selection scales with redundancy."

**Non-Goals**
- No new model architecture. Reuse `TemporalGroupedAutoencoder` + `selection_core` unchanged
  (the synthetic data matches the PAMAP2 tensor layout `groups × channels × time`).
- No XOR / purely-nonlinear coupling (see §3.1 rationale). No GUI. No SpectraForge in this spec.
- `spectral_select` / TPAMI numbers untouched.

## 3. Generative Model

Produce `N` windows, each shaped `(G groups, C_g channels/group, T time)`, with an integer class
label `y ∈ {1..K_cls}` and a ground-truth channel-role vector. Channels are one of four types.

Per window, `y` is drawn uniformly, then channels are generated conditioned on `y`.

### 3.1 Channel types
Let `template[y, k] ∈ R^T` be fixed class-dependent waveforms drawn once at construction (seeded),
e.g. class-specific low-frequency sinusoids or fixed random smooth curves.

1. **Marginal-signal** (`M` channels): `x = a·template[y, k] + ε`, `ε ~ N(0, σ_m²)`, high SNR.
   Individually discriminative. Every method should find these.
2. **Joint-signal** (`J` channels in sets of size `js`): all channels in a set share one class
   source `g = template[y, k]`; each channel is `x = w·g + ε`, `ε ~ N(0, σ_j²)` with **large
   σ_j** so per-channel SNR is low but the shared factor is recoverable across the set.
   Marginally weak, jointly strong. **Rationale for linear shared factor (not XOR):** the engine
   is an *unsupervised reconstruction* AE — it depends on a channel when that channel carries
   reconstructable shared structure. A shared linear factor is exactly detectable this way; a pure
   XOR has no marginal or pairwise-linear structure and a label-free AE would honestly miss it,
   sabotaging our own claim. This is the honest, detectable form of "informative only in
   combination."
3. **Redundant** (`R` channels): each is a near-duplicate of a specific marginal channel `i`:
   `x = a·template[y, k_i] + ε'`, `ε' ~ N(0, σ_r²)` tiny → high correlation with channel `i`.
   Informative but redundant. MMR should pick one representative, not all.
4. **Noise** (`P` channels): `x = ε`, `ε ~ N(0, 1)`, class-independent. Never informative.

Total channels `= M + J + R + P`, distributed across `G` groups (each group holds a mix of types).

### 3.2 Ground truth returned
`make_grouped_synthetic(config, seed)` returns `(dataset, roles)` where `dataset` is a
`GroupedChannelDataset` (with `labels`; no `subject_ids` — synthetic) and `roles` maps each
`(group, channel)` to one of `{marginal, joint, redundant, noise}` plus, for redundant channels,
the id of the marginal cluster they duplicate.

Define `INFORMATIVE = marginal ∪ joint`.

### 3.3 Config knobs
`SyntheticConfig`: `n_windows`, `n_classes`, `n_groups`, `channels_per_group`, `time`,
`n_marginal`, `n_joint`, `joint_set_size`, `n_redundant`, (`n_noise` = remainder), `snr_marginal`,
`snr_joint`, `noise_dup_sigma`, and `redundancy_level` (a convenience scalar that scales `n_noise`
and `n_redundant` for the sweep). Fully seeded / reproducible.

## 4. Metrics & Protocol

Selection is run **once** at `K = KMAX` (MMR is greedy, so `selected[:k]` is the size-`k`
selection), mirroring `general_pamap2_loso.py`.

- **precision@K** `= |selected[:K] ∩ INFORMATIVE| / K`. Random ≈ `|INFORMATIVE| / total`.
- **recall_joint@K** `= |selected[:K] ∩ JOINT| / |JOINT|` — the dependency claim.
- **redundancy_leak@K** `=` number of redundant duplicates in `selected[:K]` beyond one per cluster.
- **downstream accuracy**: features = per-channel (mean, std) over time (same `feats()` as the HAR
  harness); 70/30 train/test split; KNN (k=5) + StandardScaler; report method-K vs random-K vs full.

**Baselines**: random-K (mean over seeds), variance top-K, PCA-loading top-K, supervised
mutual-information top-K (MI between per-channel features and `y`). Supervised MI included to show
the label-free method matches supervised recovery.

**Redundancy sweep**: vary `noise_fraction` across e.g. `{0.2, 0.4, 0.6, 0.8}`; plot precision@K
and the downstream accuracy gap `(method − random)`; expect a monotone increase.

## 5. Success Criteria (falsifiable)

1. `precision@K ≥ 0.9` at `K = |INFORMATIVE|` for the method, vs random `≈ |INFORMATIVE|/total`.
2. `recall_joint ≥ 0.8` for the method **and** variance `recall_joint ≤ 0.4` (dependency proof).
3. Method downstream acc within a small ε of full-channel acc **and** `≥ random-K + 15pp`.
4. Redundancy sweep: `(method − random)` gap increases monotonically with noise fraction.

If (2) fails (method does not beat variance on joint channels), the dependency claim is not
supported and we report that honestly and revisit `snr_joint` / `joint_set_size`.

## 6. Files

- `src/channel_select/synthetic.py` — `SyntheticConfig`, `make_grouped_synthetic(config, seed) →
  (GroupedChannelDataset, roles)`.
- `tests/channel_select/test_synthetic.py` — TDD (see §7).
- `experiments/synthetic/run_synthetic_recovery.py` — generate → train AE → select → metrics vs
  baselines → redundancy sweep → write `publications/generalization/reports/synthetic_recovery.txt`.
- `experiments/synthetic/make_synthetic_figures.py` — recovery-precision, joint-recall, and
  redundancy-sweep figures into `publications/generalization/figures/`.

## 7. TDD Test List

1. `test_shapes_and_groups` — output dataset has requested `groups`, `channels_per_group`, `time`,
   `axis_type="temporal1d"`, `n_windows`; roles cover every `(g,c)`.
2. `test_seed_reproducible` — same seed → identical tensors; different seed → different.
3. `test_informative_more_class_correlated_than_noise` — marginal channels have higher
   feature/label mutual-information (or class-mean separation) than noise channels.
4. `test_joint_marginally_weak_but_jointly_strong` — each joint channel alone has low class
   separation, but the joint-set average has high class separation.
5. `test_redundant_correlated_with_source` — each redundant channel has high correlation with its
   marginal source and low with unrelated channels.
6. `test_roles_partition` — `marginal + joint + redundant + noise` counts sum to total channels.

## 8. Paper Integration

New results subsection (before the two real domains), framed as a controlled benchmark with known
ground truth: reports precision@K, the joint-channel recall bar (ours vs variance), and the
redundancy-sweep curve. This directly answers reviewer 2's "why select at all" (necessity +
redundancy law) and "is it really dependency-aware" (joint recovery), and it is the domain where
the method demonstrably **beats random**, complementing the honest "ties random" PAMAP2 case.
```
