"""BCI Competition IV-2a (BNCI2014_001) motor-imagery EEG -> GroupedChannelDataset.

22 EEG electrodes, 4 classes (left hand / right hand / feet / tongue), 9 subjects, 250 Hz.
Groups are scalp REGIONS from the electrode montage (frontal / central / centro-parietal /
parietal) -- fixed acquisition metadata, not labels -- so the label-free engine applies with
only a Conv1D encoder, exactly as for wearable IMUs. LOSO across the 9 subjects.

Canonical 22-channel order (BNCI2014_001):
  0 Fz | 1 FC3 2 FC1 3 FCz 4 FC2 5 FC4 | 6 C5 7 C3 8 C1 9 Cz 10 C2 11 C4 12 C6
  | 13 CP3 14 CP1 15 CPz 16 CP2 17 CP4 | 18 P1 19 Pz 20 P2 21 POz
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from ..data import GroupedChannelDataset

CANONICAL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
REGION_GROUPS = {
    "frontal": [0, 1, 2, 3, 4, 5],
    "central": [6, 7, 8, 9, 10, 11, 12],
    "centro_parietal": [13, 14, 15, 16, 17],
    "parietal": [18, 19, 20, 21],
}


def dataset_from_arrays(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                        session_ids: np.ndarray = None) -> GroupedChannelDataset:
    """X: (N_trials, 22, time) in CANONICAL_CHANNELS order. Groups -> (N, time, n_region).

    ``session_ids`` (optional) is attached as ``ds.session_ids`` for within-subject
    evaluation (motor imagery is subject-specific; the standard protocol trains and tests
    within a subject across its two recording sessions).
    """
    data = {}
    for region, idxs in REGION_GROUPS.items():
        block = np.nan_to_num(X[:, idxs, :], nan=0.0)         # (N, n_region, time)
        data[region] = torch.tensor(block.transpose(0, 2, 1), dtype=torch.float32)
    ds = GroupedChannelDataset(
        data, axis_type="temporal1d",
        labels=torch.tensor(np.asarray(y).astype(int), dtype=torch.long),
        subject_ids=torch.tensor(np.asarray(subject_ids).astype(int), dtype=torch.long),
    )
    if session_ids is not None:
        ds.session_ids = torch.tensor(np.asarray(session_ids).astype(int), dtype=torch.long)
    return ds


def load_bci_iv_2a(subjects: Optional[list[int]] = None, fmin: float = 8.0, fmax: float = 30.0):
    """Fetch BNCI2014_001 via MOABB (band-passed to the mu/beta band), return a
    GroupedChannelDataset. Requires ``moabb`` (+ mne); downloads on first call.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    subjects = subjects or dataset.subject_list
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax)
    # X: (n_trials, 22, n_times) numpy, channels already in the canonical montage order;
    # labels: class-name strings; meta: DataFrame with a 'subject' column.
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    classes = {c: i for i, c in enumerate(sorted(set(labels)))}
    y = np.array([classes[c] for c in labels])
    subj = meta["subject"].to_numpy()
    sess_names = sorted(meta["session"].unique())
    sess_map = {s: i for i, s in enumerate(sess_names)}
    sess = meta["session"].map(sess_map).to_numpy()
    return dataset_from_arrays(np.asarray(X, dtype=np.float32), y, subj, session_ids=sess)
