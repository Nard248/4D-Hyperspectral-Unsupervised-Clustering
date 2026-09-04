"""TDD for the compressive-bottleneck temporal CAE variant.

Pools over time so the latent is (B, latent_dim, 1) -- a true bottleneck that forces the
AE to encode only the low-rank class structure and ignore per-channel noise. Drop-in for
the GroupStructuredModel engine protocol.
"""
import torch

from channel_select.models.temporal_bottleneck import BottleneckTemporalAutoencoder
from channel_select.protocols import GroupStructuredModel


def test_bottleneck_roundtrip_shapes_and_protocol():
    cpg = {"a": 3, "b": 4}
    m = BottleneckTemporalAutoencoder(cpg, time_len=16, latent_dim=5, pool=1)
    batch = {"a": torch.randn(7, 16, 3), "b": torch.randn(7, 16, 4)}
    z = m.encode(batch)
    assert z.shape == (7, 5, 1)                       # compressed over time
    out = m.decode(z)
    assert out["a"].shape == (7, 16, 3)               # decode restores full time + channels
    assert out["b"].shape == (7, 16, 4)
    assert isinstance(m, GroupStructuredModel)


def test_bottleneck_forward_runs():
    cpg = {"g0": 6}
    m = BottleneckTemporalAutoencoder(cpg, time_len=32, latent_dim=4, pool=1)
    batch = {"g0": torch.randn(5, 32, 6)}
    out = m(batch)
    assert out["g0"].shape == (5, 32, 6)
