"""TDD for the controlled synthetic benchmark generator (factor-coverage design).

See docs/superpowers/specs/2026-07-14-synthetic-channel-recovery-benchmark-design.md.
The class label depends on SEVERAL independent factors; each factor is expressed in a
cluster of correlated channels; the rest are noise. To classify you must COVER the
factors, so an unsupervised selector that keeps one channel per factor (and drops noise)
reduces dimensionality without losing accuracy -- while budget-blind marginal selectors
that pile into one high-variance cluster miss factors.
"""
import numpy as np
import torch

from channel_select.synthetic import SyntheticConfig, make_grouped_synthetic


def _cfg():
    return SyntheticConfig(
        n_windows=600, n_factors=4, states_per_factor=2,
        n_groups=4, channels_per_group=16, time=64, cluster_size=5,
    )


def _feat(ds, gc):
    g, c = gc
    return ds.data[g][:, :, c].numpy().mean(axis=1)


def _sep(feat, labels):
    classes = np.unique(labels)
    means = np.array([feat[labels == k].mean() for k in classes])
    within = np.mean([feat[labels == k].var() for k in classes])
    return float(means.var() / (within + 1e-9))


def test_shapes_groups_and_labels():
    ds, roles = make_grouped_synthetic(_cfg(), seed=0)
    assert ds.axis_type == "temporal1d"
    assert len(ds.groups) == 4
    assert all(v == 16 for v in ds.channels_per_group.values())
    assert ds.n_windows == 600
    assert ds.data[ds.groups[0]].shape == (600, 64, 16)
    assert ds.labels.shape == (600,)
    assert int(ds.labels.min()) >= 0 and int(ds.labels.max()) < 2 ** 4  # 16 classes


def test_seed_reproducible():
    a, _ = make_grouped_synthetic(_cfg(), seed=5)
    b, _ = make_grouped_synthetic(_cfg(), seed=5)
    c, _ = make_grouped_synthetic(_cfg(), seed=6)
    g = a.groups[0]
    assert torch.equal(a.data[g], b.data[g]) and torch.equal(a.labels, b.labels)
    assert not torch.equal(a.data[g], c.data[g])


def test_roles_partition_and_clusters():
    ds, roles = make_grouped_synthetic(_cfg(), seed=0)
    total = sum(ds.channels_per_group.values())
    assert len(roles.clusters) == 4                       # one per factor
    assert all(len(cl) == 5 for cl in roles.clusters)     # cluster_size
    assert len(roles.informative) == 20 and len(roles.noise) == total - 20
    allc = roles.informative | roles.noise
    assert len(allc) == total                             # disjoint + covering
    assert roles.informative == set().union(*roles.clusters)
    assert roles.factor_states.shape == (600, 4)


def test_within_cluster_correlated_cross_cluster_not():
    ds, roles = make_grouped_synthetic(_cfg(), seed=1)
    c0 = sorted(roles.clusters[0]); c1 = sorted(roles.clusters[1])
    within = np.corrcoef(_feat(ds, c0[0]), _feat(ds, c0[1]))[0, 1]
    cross = np.corrcoef(_feat(ds, c0[0]), _feat(ds, c1[0]))[0, 1]
    assert within > 0.7
    assert abs(cross) < 0.3


def test_cluster_encodes_only_its_own_factor():
    ds, roles = make_grouped_synthetic(_cfg(), seed=2)
    fs = roles.factor_states
    for f, cl in enumerate(roles.clusters):
        feat = _feat(ds, sorted(cl)[0])
        own = _sep(feat, fs[:, f])
        other = np.mean([_sep(feat, fs[:, g]) for g in range(4) if g != f])
        assert own > 5 * (other + 1e-9)          # channel tracks its factor, not others


def test_noise_is_uninformative():
    ds, roles = make_grouped_synthetic(_cfg(), seed=3)
    y = ds.labels.numpy()
    info = np.mean([_sep(_feat(ds, gc), y) for gc in roles.informative])
    noise = np.mean([_sep(_feat(ds, gc), y) for gc in roles.noise])
    assert info > 5 * noise
