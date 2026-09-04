# Dataset Contract

> The full data guide for new team members (raw folder contract, metadata spreadsheets, `.im3` internals, pickle dialects, annotation formats, catalogue with shapes) is [`onboarding/DATA_GUIDE.md`](onboarding/DATA_GUIDE.md). This file is the short inventory.

The `Data/` directory is **gitignored** — datasets are not version-controlled. This file documents
what is canonical, where it lives, and which code consumes it, so the data layer is reproducible
without committing gigabytes of binaries.

## Layout

```
Data/
├── Raw/         # source hyperspectral cubes (.im3) + metadata, and raw arrays (PAMAP2 .npy)
└── processed/   # preprocessed spectra (.pkl), masks, ROI definitions
```

## Canonical datasets (kept)

| Dataset | `Data/` path(s) | Size | Consumed by |
|---|---|---|---|
| **Lichens (canonical)** | `processed/Lichens Dataset 1/` | 2.8G | `experiments/run_master_experiment.py`, `rerun_knn.py`, `export_tiffs.py` — TPAMI-submitted reference |
| **Lichens v2** | `Raw/Lichens_2/`, `processed/Lichens_2/` | 1.1G + 50M | `tests/test_config.py`; matches raw source name |
| **Drop Data** (base) | `Raw/Drop Data/`, `processed/Drop Data/` | 587M | `experiments/_archive/2026_paper_runs/drop_data_preprocess.py`, `drop_data_post_analysis.py` |
| **Drop Data Cropped** | `processed/Drop Data Cropped/` | 250M | `drop_data_cropped_pipeline.py` |
| **Drop Data Radiometric** | `processed/Drop Data Radiometric/` | 200M | `drop_data_radiometric_rerun.py`, `drop_data_radiometric_knn.py` |
| **Collagen (Acetic Acid)** | `Raw/Collagen_Acetic_Acid/`, `processed/Collagen_Acetic_Acid/` | 34M + 304M | `collagen_*` experiments |
| **Collagen (Pepsin)** | `Raw/Collagen_Pepsin/`, `processed/Collagen Pepsin/` | 34M + 215M | `pepsin_*` experiments |
| **Sponges (Acid Group 1)** | `processed/Sponges Acid Group 1/` | 203M | `poster_*` / sponges tuning experiments |
| **PAMAP2 (MONSTER)** | `Raw/PAMAP2_MONSTER/` | 1.5G | `experiments/pamap2/general_pamap2_*.py` (uses `channel_select`) |

Total kept ≈ 7.3 GB.

## Archived (moved 2026-06-15, recoverable)

The following had **zero code references** and were moved to `archive/redundant_data/` (see
`docs/ARCHIVE_MANIFEST.md` for the full table): three redundant Lichens processings
(`lichens_data_cropped`, `LichensProcessed`, `Lichhens 2 - Processed`), orphan sets
(`Raw/Lime`, `Sample Processed`, `Sample Export 2`), and stray exports at the `processed/` root.
Two empty placeholder dirs were removed.

## Notes & future

- Data is intentionally not in git. To work on a fresh clone, place the datasets above under
  `Data/` matching these paths.
- **Future option:** adopt DVC or an external object store (institutional repo / Zenodo / cloud)
  for shareable, checksummed data versioning. The `archive/` directory and this contract are the
  interim mechanism.
