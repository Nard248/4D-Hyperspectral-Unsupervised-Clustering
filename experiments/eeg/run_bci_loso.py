"""BCI-IV-2a motor-imagery: unsupervised channel selection, leave-one-subject-out.

Third verification domain (neural). The identical label-free engine + MMR is applied to
EEG by swapping in the Conv1D encoder; groups are scalp regions (montage metadata, not
labels). Downstream decoding uses CSP + LDA (the standard motor-imagery decoder) on the
SELECTED channels, so the ceiling is realistic and channel selection is testable.

Per held-out subject: train the bottleneck CAE unsupervised on the other subjects' trials,
select K electrodes, fit CSP+LDA on those electrodes (train), score macro-F1 on the
held-out subject. Compare vs variance / supervised MI / random selection.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/eeg/run_bci_loso.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from channel_select.adapters.bci_eeg import load_bci_iv_2a
from channel_select.models.temporal_bottleneck import BottleneckTemporalAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 20
LATENT = 8
DECIMATE = 3          # 1001 -> 334 for the AE; Nyquist ~41 Hz preserves the 8-30 Hz band
SUBSAMPLE = 1500
KS = [3, 4, 6, 8, 10]
KMAX = max(KS)
REPORT = Path("publications/generalization/reports/eeg_bci_loso.txt")


def flat_channels(ds):
    return [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]


def sel_epochs(ds, idx, selected):
    """Selected channels' band-passed epochs -> (n_trials, K, time) for CSP."""
    return np.stack([ds.data[g].numpy()[idx][:, :, c] for (g, c) in selected], axis=1)


def csp_lda_f1(ds, tr, te, selected):
    if not selected:
        return float("nan")
    Xtr, Xte = sel_epochs(ds, tr, selected), sel_epochs(ds, te, selected)
    ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
    n_comp = min(6, len(selected))
    clf = make_pipeline(
        CSP(n_components=n_comp, reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )
    clf.fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average="macro")


def bandpower_matrix(ds, idx):
    flat = flat_channels(ds)
    return np.stack([np.log(ds.data[g].numpy()[idx][:, :, c].var(axis=1) + 1e-8) for (g, c) in flat], axis=1)


def variance_order(ds, tr):
    scored = {gc: float(ds.data[gc[0]].numpy()[tr][:, :, gc[1]].var()) for gc in flat_channels(ds)}
    return [gc for gc, _ in sorted(scored.items(), key=lambda x: -x[1])]


def mi_order(ds, tr):
    F, flat = bandpower_matrix(ds, tr), flat_channels(ds)
    mi = mutual_info_classif(F, ds.labels[tr].numpy(), random_state=0)
    return [gc for gc, _ in sorted(zip(flat, mi), key=lambda x: -x[1])]


def ours_order(ds, tr):
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
    return run_selection(model, sel_data, cfg).selected


def pca_order(ds, tr):
    from sklearn.decomposition import PCA
    F = bandpower_matrix(ds, tr)
    load = np.abs(PCA(n_components=8).fit(F - F.mean(0)).components_).sum(0)
    return [gc for gc, _ in sorted(zip(flat_channels(ds), load), key=lambda x: -x[1])]


def main():
    ds = load_bci_iv_2a()
    subj_ids, sess = ds.subject_ids.numpy(), ds.session_ids.numpy()
    subjects = sorted(set(subj_ids.tolist()))
    flat = flat_channels(ds)
    methods = ["ours", "pca", "variance", "mutual_info", "random"]
    res = {m: {k: [] for k in KS} for m in methods}
    ceil = []

    for s in subjects:
        # within-subject: session 0 -> train (+ unsupervised selection), session 1 -> test
        tr = np.where((subj_ids == s) & (sess == 0))[0].tolist()
        te = np.where((subj_ids == s) & (sess == 1))[0].tolist()
        orders = {"ours": ours_order(ds, tr), "pca": pca_order(ds, tr),
                  "variance": variance_order(ds, tr), "mutual_info": mi_order(ds, tr)}
        ceil.append(csp_lda_f1(ds, tr, te, flat))
        for k in KS:
            for m in ("ours", "pca", "variance", "mutual_info"):
                res[m][k].append(csp_lda_f1(ds, tr, te, orders[m][:k]))
            rs = [csp_lda_f1(ds, tr, te, [flat[i] for i in np.random.default_rng(sd).choice(len(flat), k, False)])
                  for sd in range(5)]
            res["random"][k].append(float(np.mean(rs)))
        print(f"subject {s} done | K=6: ours={res['ours'][6][-1]:.3f} var={res['variance'][6][-1]:.3f} "
              f"MI={res['mutual_info'][6][-1]:.3f} rand={res['random'][6][-1]:.3f} | ceil={ceil[-1]:.3f}", flush=True)

    lines = ["BCI-IV-2a motor imagery | within-subject (session 1 -> 2) | CSP+LDA macro-F1",
             f"full 22-electrode ceiling = {np.mean(ceil):.3f} +/- {np.std(ceil):.3f}", "",
             f"{'method':14s} " + " ".join(f"K={k:<5d}" for k in KS), "-" * 60]
    for m in methods:
        lines.append(f"{m:14s} " + " ".join(f"{np.mean(res[m][k]):5.3f}" for k in KS))
    lines += ["", "std across subjects (stability; lower = more consistent):"]
    for m in ("ours", "variance"):
        lines.append(f"{m:14s} " + " ".join(f"{np.std(res[m][k]):5.3f}" for k in KS))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
