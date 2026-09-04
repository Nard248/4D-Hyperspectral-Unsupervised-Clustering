"""Dump per-subject electrode selections + per-electrode scores for the CODASSCA poster
scalp-map figure. Same protocol as run_bci_loso.py (within-subject, session 1 -> 2,
CSP+LDA macro-F1). Writes publications/codassca2026/poster/data/eeg_selections.json.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/eeg/dump_bci_selections.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bci_loso import (flat_channels, csp_lda_f1, bandpower_matrix, KS, KMAX,
                          DEVICE, EPOCHS, LATENT, DECIMATE, SUBSAMPLE)
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from channel_select.adapters.bci_eeg import load_bci_iv_2a, CANONICAL_CHANNELS, REGION_GROUPS
from channel_select.models.temporal_bottleneck import BottleneckTemporalAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

OUT = Path("publications/codassca2026/poster/data/eeg_selections.json")


def ours_full(ds, tr):
    torch.manual_seed(0)
    sub = tr if len(tr) <= SUBSAMPLE else list(np.random.default_rng(0).choice(tr, SUBSAMPLE, replace=False))
    sel_data = {g: ds.data[g][sub][:, ::DECIMATE, :] for g in ds.groups}
    time_len = sel_data[ds.groups[0]].shape[1]
    model = BottleneckTemporalAutoencoder(ds.channels_per_group, time_len,
                                          latent_dim=LATENT, pool=1, latent_act=False)
    train_autoencoder(model, sel_data, epochs=EPOCHS, lr=1e-3, batch_size=128, device=DEVICE)
    model = model.to("cpu")
    cfg = SelectionConfig(
        dimension_selection_method="variance", n_important_dimensions=LATENT,
        n_channels_to_select=KMAX, normalization_method="none",
        diversity_method="mmr", lambda_diversity=0.4,
        perturbation_method="percentile", perturbation_magnitudes=[50, 70, 90],
    )
    return run_selection(model, sel_data, cfg)


def main():
    ds = load_bci_iv_2a()
    subj_ids, sess = ds.subject_ids.numpy(), ds.session_ids.numpy()
    flat = flat_channels(ds)
    name_of = {}
    for region, idxs in REGION_GROUPS.items():
        for c, i in enumerate(idxs):
            name_of[(region, c)] = CANONICAL_CHANNELS[i]
    names = [name_of[gc] for gc in flat]
    res = {"channels": CANONICAL_CHANNELS, "groups": REGION_GROUPS, "flat_names": names,
           "ks": KS, "subjects": {}}
    for s in sorted(set(subj_ids.tolist())):
        tr = np.where((subj_ids == s) & (sess == 0))[0].tolist()
        te = np.where((subj_ids == s) & (sess == 1))[0].tolist()
        sel = ours_full(ds, tr)
        ours_scores = [float(sel.influence[g][c]) for (g, c) in flat]
        var_scores = [float(ds.data[g].numpy()[tr][:, :, c].var()) for (g, c) in flat]
        F = bandpower_matrix(ds, tr)
        mi_scores = mutual_info_classif(F, ds.labels[tr].numpy(), random_state=0).tolist()
        pca_scores = np.abs(PCA(n_components=8).fit(F - F.mean(0)).components_).sum(0).tolist()
        order = lambda sc: [names[i] for i in np.argsort(-np.asarray(sc))]
        orders = {"ours": [name_of[gc] for gc in sel.selected],
                  "pca": order(pca_scores), "variance": order(var_scores), "mutual_info": order(mi_scores)}
        idx = {n: i for i, n in enumerate(names)}
        f1 = {"full": csp_lda_f1(ds, tr, te, flat)}
        for m, o in orders.items():
            f1[m] = {str(k): csp_lda_f1(ds, tr, te, [flat[idx[n]] for n in o[:k]]) for k in KS}
        f1["random"] = {str(k): float(np.mean([csp_lda_f1(ds, tr, te, [flat[i] for i in np.random.default_rng(sd).choice(len(flat), k, False)]) for sd in range(5)])) for k in KS}
        res["subjects"][str(s)] = {"orders": orders, "f1": f1,
                                   "scores": {"ours": ours_scores, "variance": var_scores,
                                              "mutual_info": mi_scores, "pca": pca_scores}}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"subject {s} done | K=6 ours={f1['ours']['6']:.3f} pca={f1['pca']['6']:.3f} "
              f"var={f1['variance']['6']:.3f} mi={f1['mutual_info']['6']:.3f} rand={f1['random']['6']:.3f} "
              f"full={f1['full']:.3f} | ours top6={orders['ours'][:6]} pca top6={orders['pca'][:6]}", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
