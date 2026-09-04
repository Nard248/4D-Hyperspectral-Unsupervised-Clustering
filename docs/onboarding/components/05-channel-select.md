# channel_select: domain-agnostic deep-learning channel selection

## Purpose and status

`channel_select` generalises the perturbation-autoencoder method beyond hyperspectral imaging to any **group-structured multi-channel data**: wearable sensors (PAMAP2: hand / chest / ankle IMUs), EEG electrodes grouped by scalp region, and a synthetic benchmark with planted factors. It is the code behind the CODASSCA 2026 short paper on generalisation and the ongoing "when does channel selection matter" study. It reuses `selection_core` for the algorithm and adds grouped datasets, three autoencoders, dataset adapters, a training loop and the diversity-aware selection step.

Status in September 2026:

* Verified transfer: on the **synthetic factor-recovery benchmark** the method reaches accuracy 0.944 at K = 8 (the full-channel ceiling is 0.872, because it drops noise channels) with factor coverage 1.0 and zero noise channels, against 0.244 for variance ranking and 0.254 for random. On **EEG motor imagery** (BCI IV-2a, leave-one-subject-out, CSP + LDA) it matches supervised mutual-information selection and random.
* Negative results: on **PAMAP2**, **Indian Pines** and **Opportunity** it ties or loses to random selection. The diagnostic (`experiments/saturation_diagnostic.py`, `publications/generalization/RESEARCH_LOG.md`) shows these are saturated benchmarks where any K channels work; that is a dataset property, not a method failure, but it also means the method has no demonstrated advantage on correlated, redundant channels.
* The bottleneck autoencoders, the EEG adapter, the synthetic generator, the screening scripts and their tests are **untracked in git** at the time of writing.

## Files

```
src/channel_select/
├── __init__.py            exports SelectionConfig, GroupStructuredModel, GroupedChannelData only
├── protocols.py           SelectionConfig (validated), GroupStructuredModel and GroupedChannelData Protocols
├── data.py                GroupedChannelDataset
├── engine.py              run_selection, SelectionResult, select_channels (top-K / MMR / min-distance), measure_channel_influence
├── synthetic.py           SyntheticConfig, SyntheticRoles, make_grouped_synthetic
├── adapters/
│   ├── pamap2.py          raw UCI PAMAP2 .dat files -> windows (Data/Raw/PAMAP2/Protocol/)
│   ├── pamap2_monster.py  MONSTER PAMAP2 npy/csv (Data/Raw/PAMAP2_MONSTER/)  <- the one the experiments use
│   └── bci_eeg.py         BCI IV-2a through MOABB (downloads to the MNE cache)
└── models/
    ├── temporal.py            TemporalGroupedAutoencoder (Conv1d, full-time latent)
    ├── temporal_bottleneck.py BottleneckTemporalAutoencoder (adaptive pooling, optional signed latent)
    ├── spatial_bottleneck.py  SpatialBottleneckAutoencoder (Conv2d + global pooling, for image patches)
    └── training.py            train_autoencoder
tests/channel_select/      data, engine (dims / perturb / influence / normalize / select / end to end), protocols, adapters,
                           models, training, synthetic
experiments/pamap2/, experiments/eeg/, experiments/synthetic/, experiments/*_screen.py, experiments/saturation_diagnostic.py
docs/superpowers/specs/2026-05-23-general-channel-selection-design.md, 2026-07-14-synthetic-channel-recovery-benchmark-design.md
publications/generalization/   the paper, EXPLAINER.md, RESEARCH_LOG.md, reports/*.txt
```

## Data model

`GroupedChannelDataset(data: {group: Tensor}, axis_type, labels=None, subject_ids=None)`. The container is a dict of per-group tensors with the **channel axis last**:

* `axis_type="temporal1d"`: `(n_windows, time, n_channels_in_group)`
* `axis_type="spatial2d"`: `(n_windows, H, W, n_channels_in_group)`

Properties `groups`, `channels_per_group`, `n_windows`; `get_all_data()`; `subset(indices)`; `loso_split(holdout_subject) -> (train_idx, test_idx)` (needs `subject_ids`). The EEG adapter attaches `session_ids` as an extra attribute that `subset` does not carry along.

Adapters transpose raw `(N, C, T)` arrays into this layout:

| Adapter | Expects on disk | Groups / channels | Notes |
|---|---|---|---|
| `pamap2.load_pamap2(protocol_dir, window=256, step=128, subjects=None)` | `Data/Raw/PAMAP2/Protocol/subject10*.dat` from the UCI zip | hand / chest / ankle, 9 IMU channels each (acc 16 g xyz, gyro xyz, mag xyz) = 27 | majority-vote label per window, transients dropped; not present locally |
| `pamap2_monster.load_pamap2_monster(data_dir)` | `Data/Raw/PAMAP2_MONSTER/PAMAP2_X.npy (N, 52, 100)`, `PAMAP2_y.csv`, `PAMAP2_subject_id.csv` from `huggingface.co/datasets/monster-monash/PAMAP2` | same 27 channels (heart rate dropped) | the one every experiment uses |
| `bci_eeg.load_bci_iv_2a(subjects=None, fmin=8.0, fmax=30.0)` | nothing local: MOABB `BNCI2014_001`, `MotorImagery(n_classes=4)` downloads on first call (`pip install moabb mne`) | frontal (6), central (7), centro-parietal (5), parietal (4) = 22 electrodes; 4 classes, 9 subjects, 250 Hz | groups are montage metadata, not labels, which keeps the method label-free |

## Models

All satisfy `GroupStructuredModel` (`groups`, `channels_per_group`, `encode(batch) -> latent`, `decode(latent) -> {group: recon}`), so `selection_core` can drive them.

* `TemporalGroupedAutoencoder(channels_per_group, time_len, latent_dim=16, hidden=32)`: per group `Conv1d(c -> hidden, k=5) + ReLU + Conv1d(hidden -> latent, k=5) + ReLU`; latents mean-fused across groups to `(B, latent, time)`; per-group decoder `Conv1d(latent -> hidden) + ReLU + Conv1d(hidden -> c)`. Full-time latent. **Known weakness**: enough capacity to memorise per-channel noise, which produces an *inverted* perturbation influence on some data. Used by the older PAMAP2 scripts.
* `BottleneckTemporalAutoencoder(channels_per_group, time_len, latent_dim=8, hidden=32, pool=1, latent_act=True, fusion="mean")`: same convolutions plus `AdaptiveAvgPool1d(pool)` at the end of the encoder (the bottleneck) and an optional **signed** latent (`latent_act=False` removes the final ReLU); decoder re-broadcasts with nearest interpolation. `fusion` is `mean` or `amax`. The locked configuration for the synthetic and EEG results is `latent_dim=8, pool=1, latent_act=False`. Used by every experiment since July 2026.
* `SpatialBottleneckAutoencoder(channels_per_group, spatial_size, latent_dim=8, hidden=32, latent_act=False, fusion="mean")`: the imaging analogue with `Conv2d(k=3)` and `AdaptiveAvgPool2d(1)`; input `(B, H, W, C)` patches. Used by the Indian Pines screen.
* `train_autoencoder(model, data, epochs=25, lr=1e-3, batch_size=32, device="cpu") -> [loss per epoch]`: Adam, loss = **unweighted sum** of per-group MSE (groups with more channels dominate). The model is left on `device`; move it back to CPU before `run_selection`, which does no device handling.

`models/__init__.py` exports only `TemporalGroupedAutoencoder`; import the bottleneck models by full path.

## Selection

```python
from channel_select.protocols import SelectionConfig
from channel_select.engine import run_selection
cfg = SelectionConfig(dimension_selection_method="variance", n_important_dimensions=50,
                      perturbation_method="percentile", perturbation_magnitudes=[50, 70, 90],
                      normalization_method="none", n_channels_to_select=10,
                      diversity_method="mmr", lambda_diversity=0.4)
model = model.to("cpu")
result = run_selection(model, dataset.get_all_data(), cfg)
result.selected        # [(group, channel_index), ...] in MMR order; selected[:k] is the size-k selection
result.influence       # {group: normalised influence vector}
result.important_dims  # [(score, coord), ...]
```

`SelectionConfig` defaults: `variance` dimensions, 50 of them, `percentile` perturbation with magnitudes `[10, 20, 30]`, `bidirectional`, `variance` normalisation, 10 channels, `mmr` with `lambda_diversity=0.5`, `min_distance=0.0`. It rejects `max_per_excitation` (the HSI spelling); use `max_per_group`.

`select_channels(influence, data, K, method="mmr", lambda_diversity=0.5, min_distance=0.0)`: `none` = top-K by influence; `mmr` = greedy, seeded with the top channel, score `relevance - λ · max |cosine|` against the already selected channels' L2-normalised profiles; `min_distance` = greedy with a same-group `|Δchannel| >= min_distance` constraint. Because MMR is greedy, run it once at the largest K and evaluate prefixes.

The shared recipe used by all 2026 experiments: `variance` dimensions, `normalization_method="none"`, MMR with `λ` 0.3 to 0.4, percentile magnitudes `[50, 70, 90]`.

## The synthetic factor-recovery benchmark

`make_grouped_synthetic(SyntheticConfig(...), seed=0) -> (GroupedChannelDataset, SyntheticRoles)`. Defaults: 600 windows, 4 latent factors with 2 states each (16 classes), 4 groups x 16 channels = 64 channels, time 64, `cluster_size=5` informative channels per factor (20 informative, 44 pure-noise channels), amplitude `amp_top=2.0` decaying by `amp_decay=0.9` per factor so that **variance orders the clusters** and a marginal selector piles into the loudest one. Each (factor, state) has a fixed sinusoid template with a state-dependent DC offset; channel slots are shuffled and dealt across groups. `SyntheticRoles` gives `clusters` (per factor), `noise`, `factor_of`, `factor_states`, `informative`. The driver reports **factor coverage** (fraction of factors with at least one selected channel) and **noise fraction** at each K, plus KNN-5 accuracy against variance, PCA-loading, supervised mutual information and random baselines.

The 2026-07-14 design spec describes a four-type generator (marginal / joint / redundant / noise) with different metrics; the implementation is this two-type factor design. Read the code, not the spec.

## Running the experiments

All scripts take no arguments, must run from the repository root, default to `DEVICE = "mps" if available else "cpu"` (edit for CUDA), and need `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon.

| Script | Data | Writes |
|---|---|---|
| `experiments/synthetic/run_synthetic_recovery.py` | generated | `publications/generalization/reports/synthetic_recovery.txt` (25 seeds, K in 2..20) |
| `experiments/synthetic/dump_synthetic_attribution.py`, `make_synthetic_figure.py` | generated | poster data (`.npz`) and `fig_synthetic.png` (numbers hard-coded from the report) |
| `experiments/eeg/run_bci_loso.py` | MOABB download | `publications/generalization/reports/eeg_bci_loso.txt` (LOSO, CSP + LDA, decimate 3, 1 500-window subsample) |
| `experiments/eeg/dump_bci_selections.py`, `make_eeg_figure.py` | MOABB | poster JSON, `fig_eeg.png` |
| `experiments/pamap2/general_pamap2_slice.py` | `Data/Raw/PAMAP2_MONSTER` | the vertical slice (subject 5 held out) |
| `experiments/pamap2/general_pamap2_loso.py` | same | full LOSO, subjects 1 to 8 |
| `experiments/pamap2/general_pamap2_baseline_diag.py`, `_richfeat_diag.py`, `_pca_diag.py` | same | random / variance / supervised MI / full-27 ceiling, rich features + random forest, PCA loading |
| `experiments/pamap2/general_make_figures.py` | none | `publications/generalization/figures/fig_*.png` from hard-coded numbers |
| `experiments/saturation_diagnostic.py` | same | prints `SATURATED` or `HAS HEADROOM` (headroom = supervised best-K minus random-K accuracy) |
| `experiments/hsi_benchmark_screen.py` | Indian Pines via `huggingface_hub` | log in `publications/codassca2026/poster/data/` (ours loses to random at every K) |
| `experiments/opportunity_screen.py` | Opportunity via `huggingface_hub` | log in the same folder (ties random) |
| `experiments/tep_screen.py`, `uea_screen.py` | Tennessee Eastman (live download), UEA archive via `aeon` | stdout only, no recorded verdict |

## Tests

`tests/channel_select/`: dataset container, protocol validation, engine primitives (`test_engine_dims`, `_perturb`, `_influence`, `_normalize`, `_select`, `_end_to_end`), the three models (shape round trip + protocol conformance), training loss decreases, both PAMAP2 adapters on synthetic arrays, the EEG adapter's region partition, and six synthetic-benchmark properties (shapes, seed reproducibility, roles partition, within-cluster correlation, one factor per cluster, noise uninformative).

## Known issues and gotchas

1. **Four PAMAP2 scripts write to a non-existent root-level `generalization/reports/`** (`general_pamap2_slice.py` creates it; `_loso`, `_baseline_diag`, `_richfeat_diag` raise `FileNotFoundError` at the end of a multi-hour run). The correct path is `publications/generalization/reports/`, which only `_pca_diag`, the EEG driver and the synthetic driver use. Fix before running.
2. Relative `Data/Raw/PAMAP2_MONSTER` paths: run from the repository root.
3. `run_selection` has **no device handling** and encodes everything in one pass: move the model to CPU and subsample.
4. `PYTORCH_ENABLE_MPS_FALLBACK=1` is documented in each script's header but not set in code.
5. Undeclared dependencies: `moabb`, `mne`, `huggingface_hub`, `aeon`.
6. Figures are decoupled from data: re-running an experiment does not update `fig_*.png`; the numbers are transcribed by hand.
7. Sibling-script imports via `sys.path.insert` (the PCA diag, the EEG dump, the synthetic dump) execute the sibling's module-level code.
8. Group fusion (`mean`) and the training loss (sum over groups) are unweighted; groups with more channels dominate.
9. `session_ids` on the EEG dataset is lost by `subset()` and `loso_split()`.
10. `Data/Raw/PAMAP2/Protocol/` (the raw `.dat` adapter's input) is not present locally; that adapter is tested only on synthetic arrays.
