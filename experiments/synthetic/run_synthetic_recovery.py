"""Controlled synthetic benchmark: unsupervised dimensionality reduction that keeps
accuracy and drops noise, via the perturbation CAE.

The class depends on several independent factors, each spread over a correlated channel
cluster (clusters have decreasing amplitude); the rest are noise. Covering every factor
is necessary to classify. We report, for the label-free engine vs baselines:
  - downstream accuracy vs the number of retained channels K (the headline: our method
    reaches the full-channel ceiling at a small K; variance needs far more; random lags),
  - factor coverage @K (clusters hit) and noise fraction @K (noise removal).

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/synthetic/run_synthetic_recovery.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score

from channel_select.synthetic import SyntheticConfig, make_grouped_synthetic
from channel_select.models.temporal_bottleneck import BottleneckTemporalAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 100
LATENT = 8           # compressive bottleneck (pool over time) -> un-inverted influence;
                     # signed latent (latent_act=False) + latent=8 gives the best coverage
KS = [2, 4, 6, 8, 12, 16, 20]
KMAX = max(KS)
REPORT = Path("publications/generalization/reports/synthetic_recovery.txt")


def flat_channels(ds):
    return [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]


def chan_mean_matrix(ds):
    return np.stack([ds.data[g][:, :, c].numpy().mean(axis=1) for (g, c) in flat_channels(ds)], axis=1)


def downstream_feats(ds, selected):
    out = []
    for (g, c) in selected:
        arr = ds.data[g][:, :, c].numpy()
        out.append(arr.mean(axis=1)); out.append(arr.std(axis=1))
    return np.stack(out, axis=1)


def knn_acc(ds, selected, seed=0):
    if not selected:
        return float("nan")
    X, y = downstream_feats(ds, selected), ds.labels.numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
    return float(accuracy_score(yte, clf.predict(sc.transform(Xte))))


def order_by(scores):
    return [gc for gc, _ in sorted(scores.items(), key=lambda x: -x[1])]


def variance_scores(ds):
    return {gc: float(ds.data[gc[0]][:, :, gc[1]].numpy().var()) for gc in flat_channels(ds)}


def pca_scores(ds, n_comp=8):
    F = chan_mean_matrix(ds)
    load = np.abs(PCA(n_components=min(n_comp, F.shape[1])).fit(F - F.mean(0)).components_).sum(0)
    return {gc: float(load[i]) for i, gc in enumerate(flat_channels(ds))}


def mi_scores(ds, seed=0):
    F, y = chan_mean_matrix(ds), ds.labels.numpy()
    mi = mutual_info_classif(F, y, random_state=seed)
    return {gc: float(mi[i]) for i, gc in enumerate(flat_channels(ds))}


def ours_order(ds, seed=0):
    torch.manual_seed(seed)
    data = {g: ds.data[g] for g in ds.groups}
    time_len = ds.data[ds.groups[0]].shape[1]
    model = BottleneckTemporalAutoencoder(ds.channels_per_group, time_len, latent_dim=LATENT,
                                          pool=1, latent_act=False)
    train_autoencoder(model, data, epochs=EPOCHS, lr=1e-3, batch_size=128, device=DEVICE)
    model = model.to("cpu")
    cfg = SelectionConfig(
        dimension_selection_method="variance", n_important_dimensions=LATENT,
        n_channels_to_select=KMAX, normalization_method="none",
        diversity_method="mmr", lambda_diversity=0.3,
        perturbation_method="percentile", perturbation_magnitudes=[50, 70, 90],
    )
    return run_selection(model, data, cfg).selected   # greedy MMR order (len KMAX)


def coverage(selected, roles):
    hit = sum(1 for cl in roles.clusters if cl & set(selected))
    return hit / len(roles.clusters)


def noise_frac(selected, roles):
    return len(set(selected) & roles.noise) / len(selected) if selected else float("nan")


def acc_curve(ds, order, ks):
    return [knn_acc(ds, order[:k]) for k in ks]


def rand_curve(ds, ks, seeds=25):
    flat = flat_channels(ds)
    out = []
    for k in ks:
        accs = [knn_acc(ds, [flat[i] for i in np.random.default_rng(s).choice(len(flat), k, replace=False)], seed=s)
                for s in range(seeds)]
        out.append(float(np.mean(accs)))
    return out


def main():
    torch.manual_seed(0)
    cfg = SyntheticConfig()
    ds, roles = make_grouped_synthetic(cfg, seed=0)
    lines = ["Controlled synthetic benchmark (factor-coverage)",
             f"device={DEVICE} epochs={EPOCHS} latent={LATENT}",
             f"{cfg.total_channels} channels = {cfg.n_factors} factor-clusters x "
             f"{cfg.cluster_size} + {cfg.n_noise} noise; {cfg.n_classes} classes "
             f"(need all {cfg.n_factors} factors)", ""]

    orders = {
        "ours (label-free CAE)": ours_order(ds),
        "variance": order_by(variance_scores(ds)),
        "pca": order_by(pca_scores(ds)),
        "mutual-info (SUPERVISED)": order_by(mi_scores(ds)),
    }
    full = knn_acc(ds, flat_channels(ds))

    # --- headline: accuracy vs K ---
    lines.append(f"Accuracy vs #channels kept  (full {cfg.total_channels}-channel ceiling = {full:.3f})")
    lines.append(f"{'method':26s} " + " ".join(f"K={k:<4d}" for k in KS))
    lines.append("-" * 82)
    for name, order in orders.items():
        row = acc_curve(ds, order, KS)
        lines.append(f"{name:26s} " + " ".join(f"{a:6.3f}" for a in row))
    lines.append(f"{'random-K':26s} " + " ".join(f"{a:6.3f}" for a in rand_curve(ds, KS)))
    lines.append("")

    # --- noise removal + factor coverage at a small budget ---
    kh = 8
    lines.append(f"At K={kh}:  factor coverage (higher=better)  |  noise fraction (lower=better)")
    lines.append(f"{'method':26s} {'coverage':>10s} {'noise_frac':>12s}")
    lines.append("-" * 50)
    for name, order in orders.items():
        lines.append(f"{name:26s} {coverage(order[:kh], roles):10.2f} {noise_frac(order[:kh], roles):12.2f}")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
