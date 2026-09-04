"""TDD for the BCI-IV-2a EEG adapter (motor-imagery, 22 electrodes -> scalp-region groups).

Groups are the electrode montage's scalp regions -- physical acquisition metadata, not
labels -- so the label-free engine applies unchanged (only the encoder is Conv1D, as for
wearable IMUs). LOSO across the 9 subjects mirrors the PAMAP2 protocol.
"""
import numpy as np

from channel_select.adapters.bci_eeg import dataset_from_arrays, REGION_GROUPS


def test_dataset_from_arrays_shapes_and_region_grouping():
    N, C, T = 30, 22, 100
    X = np.zeros((N, C, T), dtype=np.float32)
    X[:, 0, :] = 3.0                                     # Fz -> frontal group, local index 0
    y = np.arange(N) % 4                                 # 4 motor-imagery classes
    subj = np.array([1] * 15 + [2] * 15)

    ds = dataset_from_arrays(X, y, subj)
    assert ds.axis_type == "temporal1d"
    assert set(ds.groups) == set(REGION_GROUPS)          # scalp regions
    # every electrode assigned exactly once
    assert sum(ds.channels_per_group.values()) == 22
    g0 = "frontal"
    assert ds.data[g0].shape[0] == N and ds.data[g0].shape[1] == T
    assert np.allclose(ds.data[g0][:, :, 0].numpy(), 3.0)   # Fz landed at frontal[0]
    tr, te = ds.loso_split(2)
    assert len(te) == 15 and len(tr) == 15


def test_region_groups_partition_22_channels():
    idxs = [i for region in REGION_GROUPS.values() for i in region]
    assert sorted(idxs) == list(range(22))               # disjoint + cover all 22 electrodes
