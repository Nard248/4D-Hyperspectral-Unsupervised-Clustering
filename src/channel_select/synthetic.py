"""Controlled synthetic benchmark (factor-coverage design).

Generates grouped 1-D channel data (``groups x channels x time``, matching the PAMAP2
tensor layout) whose class label depends on SEVERAL independent latent factors. Each
factor is expressed in a *cluster* of correlated channels (mutually redundant); the rest
are noise. See
``docs/superpowers/specs/2026-07-14-synthetic-channel-recovery-benchmark-design.md``.

Why this shape demonstrates the claim (unsupervised dimensionality reduction that keeps
accuracy and drops noise):
  - To classify you must COVER every factor -> keeping one channel per cluster and
    dropping duplicates + noise preserves full accuracy at a fraction of the channels.
  - A perturbation CAE gives the informative channels high influence (the latent must
    encode the factors to reconstruct them) and noise low influence -> noise removal.
  - MMR's redundancy penalty spreads the picks across clusters -> factor coverage.
  - Marginal/variance selection instead piles into the single highest-variance cluster
    (clusters have decreasing amplitude), covering few factors -> it needs many more
    channels to reach the same accuracy. Random selection mostly grabs noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
import torch

from .data import GroupedChannelDataset


@dataclass
class SyntheticConfig:
    n_windows: int = 600
    n_factors: int = 4
    states_per_factor: int = 2
    n_groups: int = 4
    channels_per_group: int = 16
    time: int = 64
    cluster_size: int = 5              # correlated channels expressing one factor
    amp_top: float = 2.0              # amplitude of the loudest cluster
    amp_decay: float = 0.9           # each next cluster is quieter -> variance orders them
    sigma_signal: float = 0.3
    sigma_noise: float = 0.15         # low enough that even the loudest (least perturbation-
                                     # sensitive) cluster clears the influence noise floor

    @property
    def total_channels(self) -> int:
        return self.n_groups * self.channels_per_group

    @property
    def n_informative(self) -> int:
        return self.n_factors * self.cluster_size

    @property
    def n_noise(self) -> int:
        return self.total_channels - self.n_informative

    @property
    def n_classes(self) -> int:
        return self.states_per_factor ** self.n_factors


@dataclass
class SyntheticRoles:
    clusters: list = field(default_factory=list)      # list[set[(g,c)]], index == factor
    noise: set = field(default_factory=set)
    factor_of: dict = field(default_factory=dict)     # (g,c) -> factor index
    factor_states: np.ndarray = None                  # (N, n_factors)

    @property
    def informative(self) -> set:
        return set().union(*self.clusters) if self.clusters else set()


def make_grouped_synthetic(
    config: SyntheticConfig, seed: int = 0
) -> tuple[GroupedChannelDataset, SyntheticRoles]:
    cfg = config
    if cfg.n_noise < 0:
        raise ValueError("n_factors * cluster_size exceeds total_channels")

    rng = np.random.default_rng(seed)
    N, T, Ktot, F, S = (cfg.n_windows, cfg.time, cfg.total_channels,
                        cfg.n_factors, cfg.states_per_factor)

    # --- fixed generative structure: one distinct waveform per (factor, state) ---
    t = np.arange(T) / T
    freq = rng.uniform(1.0, 4.0, (F, S))
    phase = rng.uniform(0.0, 2 * np.pi, (F, S))
    dc = rng.normal(0.0, 1.5, (F, S))                 # state-dependent DC -> mean encodes state
    templates = (np.sin(2 * np.pi * freq[:, :, None] * t[None, None, :]
                        + phase[:, :, None]) + dc[:, :, None])       # (F, S, T)
    amp = cfg.amp_top * (cfg.amp_decay ** np.arange(F))             # decreasing per factor

    # --- per-window factor states + composite label ---
    factor_states = rng.integers(0, S, (N, F))                      # (N, F)
    radix = S ** np.arange(F)
    y = (factor_states * radix[None, :]).sum(axis=1)                # mixed-radix encode

    def cluster_signal(f: int) -> np.ndarray:
        """(N, T) signal for factor f: its state-dependent waveform, scaled by amp[f]."""
        return amp[f] * templates[f, factor_states[:, f], :]

    # --- assign flat channel slots to clusters / noise (shuffled, then grouped) ---
    specs: list[dict] = []
    for f in range(F):
        for _ in range(cfg.cluster_size):
            specs.append({"type": "cluster", "factor": f})
    for _ in range(cfg.n_noise):
        specs.append({"type": "noise"})
    assert len(specs) == Ktot
    specs = [specs[i] for i in rng.permutation(Ktot)]

    def slot(p: int) -> tuple[Hashable, int]:
        return (f"g{p // cfg.channels_per_group}", p % cfg.channels_per_group)

    X = np.empty((N, T, Ktot), dtype=np.float32)
    roles = SyntheticRoles(clusters=[set() for _ in range(F)], factor_states=factor_states)
    for p, s in enumerate(specs):
        gc = slot(p)
        if s["type"] == "cluster":
            f = s["factor"]
            X[:, :, p] = cluster_signal(f) + rng.normal(0, cfg.sigma_signal, (N, T))
            roles.clusters[f].add(gc)
            roles.factor_of[gc] = f
        else:
            X[:, :, p] = rng.normal(0, cfg.sigma_noise, (N, T))
            roles.noise.add(gc)

    data = {
        f"g{g}": torch.tensor(
            X[:, :, g * cfg.channels_per_group:(g + 1) * cfg.channels_per_group],
            dtype=torch.float32,
        )
        for g in range(cfg.n_groups)
    }
    ds = GroupedChannelDataset(
        data, axis_type="temporal1d",
        labels=torch.tensor(y.astype(int), dtype=torch.long),
    )
    return ds, roles
