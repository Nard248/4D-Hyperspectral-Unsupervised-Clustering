"""Train the bottleneck CAE on the controlled synthetic benchmark (same locked config as
run_synthetic_recovery.py) and dump everything the CODASSCA poster figures need:
raw data sample, channel roles, correlation, latent code, per-dimension and accumulated
perturbation influence, and the MMR order.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/synthetic/dump_synthetic_attribution.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_synthetic_recovery import (SyntheticConfig, make_grouped_synthetic, BottleneckTemporalAutoencoder,
                                    train_autoencoder, run_selection, SelectionConfig, flat_channels, knn_acc,
                                    order_by, variance_scores, pca_scores, mi_scores, DEVICE, EPOCHS, LATENT, KMAX, KS)
from selection_core import select_important_dimensions, latent_statistics, perturbation_amount, measure_influence

OUT = Path("publications/codassca2026/poster/data/synthetic_attribution.npz")


def main():
    torch.manual_seed(0)
    cfg = SyntheticConfig()
    ds, roles = make_grouped_synthetic(cfg, seed=0)
    flat = flat_channels(ds)
    data = {g: ds.data[g] for g in ds.groups}
    time_len = ds.data[ds.groups[0]].shape[1]
    torch.manual_seed(0)
    model = BottleneckTemporalAutoencoder(ds.channels_per_group, time_len, latent_dim=LATENT, pool=1, latent_act=False)
    train_autoencoder(model, data, epochs=EPOCHS, lr=1e-3, batch_size=128, device=DEVICE)
    model = model.to("cpu")
    sel_cfg = SelectionConfig(dimension_selection_method="variance", n_important_dimensions=LATENT,
                              n_channels_to_select=KMAX, normalization_method="none", diversity_method="mmr",
                              lambda_diversity=0.3, perturbation_method="percentile", perturbation_magnitudes=[50, 70, 90])
    res = run_selection(model, data, sel_cfg)
    with torch.no_grad():
        latent = model.encode(data); base = model.decode(latent)
    important = select_important_dimensions(latent, "variance", LATENT)
    latent_dims = tuple(latent.shape[1:])
    stats = latent_statistics(latent.reshape(latent.shape[0], -1))
    per_dim = np.zeros((LATENT, len(flat))); dim_scores = np.zeros(LATENT)
    example_pert = {}
    for score, coord in important:
        j = coord[0]; dim_scores[j] = score
        inf = {g: np.zeros(ds.channels_per_group[g]) for g in ds.groups}
        for mag in [50, 70, 90]:
            for sign in (-1, 1):
                amt = perturbation_amount(coord, latent_dims, mag, sign, stats, latent, "percentile")
                pert = latent.clone(); pert[(slice(None),) + coord] += amt
                contrib = measure_influence(model.decode, model.groups, pert, base, score)
                for g in contrib: inf[g] += contrib[g] * 0.5
                if mag == 90 and sign == 1:
                    with torch.no_grad():
                        rec = model.decode(pert)
                    example_pert[j] = np.concatenate([rec[g][0].numpy() for g in ds.groups], axis=1)  # (T, 64)
        per_dim[j] = np.concatenate([inf[g] for g in ds.groups])
    accumulated = np.concatenate([res.influence[g] for g in ds.groups])
    idx = {gc: i for i, gc in enumerate(flat)}
    order = [idx[gc] for gc in res.selected]
    X = np.concatenate([ds.data[g].numpy() for g in ds.groups], axis=2)          # (N, T, 64)
    factor_of = np.array([roles.factor_of.get(gc, -1) for gc in flat])
    corr = np.corrcoef(X.reshape(-1, X.shape[2]).T)
    base_rec0 = np.concatenate([base[g][0].numpy() for g in ds.groups], axis=1)
    accs = {name: [knn_acc(ds, o[:k]) for k in KS] for name, o in
            {"ours": res.selected, "variance": order_by(variance_scores(ds)), "pca": order_by(pca_scores(ds)),
             "mi": order_by(mi_scores(ds))}.items()}
    print("this-run accuracy vs K", KS, json.dumps(accs, indent=1))
    print("MMR order (flat idx):", order, "roles:", [int(factor_of[i]) for i in order])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, X0=X[0], X1=X[1], y=ds.labels.numpy(), factor_states=roles.factor_states,
                        factor_of=factor_of, chan_var=X.var(axis=(0, 1)), corr=corr,
                        latent=latent.reshape(latent.shape[0], -1).numpy(), dim_scores=dim_scores,
                        per_dim=per_dim, accumulated=accumulated, order=np.array(order),
                        base_rec0=base_rec0, pert_rec0=np.stack([example_pert[j] for j in range(LATENT)]),
                        ks=np.array(KS), acc_ours=np.array(accs["ours"]), acc_var=np.array(accs["variance"]),
                        acc_pca=np.array(accs["pca"]), acc_mi=np.array(accs["mi"]),
                        groups=np.array([str(g) for g in ds.groups]), channels_per_group=np.array([ds.channels_per_group[g] for g in ds.groups]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
