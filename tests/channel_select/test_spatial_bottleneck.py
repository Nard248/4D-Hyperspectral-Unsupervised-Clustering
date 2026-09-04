"""TDD for the spatial (Conv2D) bottleneck AE — the encoder swap for imaging band selection.

Same selection engine as the temporal path; only the encoder changes to a 2-D spatial
convolution (channels = spectral bands, regular axis = the image plane). Pools the spatial
axes to a per-patch bottleneck so the perturbation influence stays a reliable band-importance
signal, exactly as for the temporal variant.
"""
import torch

from channel_select.models.spatial_bottleneck import SpatialBottleneckAutoencoder
from channel_select.protocols import GroupStructuredModel


def test_spatial_bottleneck_roundtrip_and_protocol():
    cpg = {"a": 3, "b": 5}
    m = SpatialBottleneckAutoencoder(cpg, spatial_size=(9, 9), latent_dim=6)
    batch = {"a": torch.randn(4, 9, 9, 3), "b": torch.randn(4, 9, 9, 5)}  # (B, H, W, ch)
    z = m.encode(batch)
    assert z.shape == (4, 6, 1, 1)                       # spatial bottleneck
    out = m.decode(z)
    assert out["a"].shape == (4, 9, 9, 3)                # restores H, W, channels
    assert out["b"].shape == (4, 9, 9, 5)
    assert isinstance(m, GroupStructuredModel)


def test_spatial_bottleneck_forward_runs():
    m = SpatialBottleneckAutoencoder({"g0": 8}, spatial_size=(7, 7), latent_dim=4)
    out = m({"g0": torch.randn(5, 7, 7, 8)})
    assert out["g0"].shape == (5, 7, 7, 8)
