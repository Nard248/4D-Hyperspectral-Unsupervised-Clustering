"""Re-evaluate CAE reconstruction with an HONEST metric + the activation/scheduler fixes.

Two bugs were found: (1) the per-band-averaged R2 metric is dominated by the ~40 near-constant
off-peak bands (even a perfect MLP scores ~0.2), and (2) train_with_masking's LR scheduler
(patience=5) collapses the LR early. Here we use POOLED (variance-weighted) R2 -- SS_res/SS_tot summed
over all bands so the signal bands dominate -- and a high scheduler patience so the LR does not
collapse, and the relu/gelu activation fix. An MLP control proves the metric.

Run:  python reports/cae_recon_metric.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import torch
import torch.nn as nn

from cae_data_search import build_simple
from classification_experiment import build_dataset as clean_build, cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config
from sweep_common import oracle_cols


def pooled_r2(inp, rec):
    """Variance-weighted R2: 1 - sum (x-r)^2 / sum (x - per-band-mean)^2, pooled across all bands."""
    ssr = sst = 0.0
    for ex in inp:
        X = inp[ex].reshape(-1, inp[ex].shape[-1]); R = rec[ex].reshape(-1, rec[ex].shape[-1])
        ssr += float(((X - R) ** 2).sum())
        sst += float(((X - X.mean(0, keepdims=True)) ** 2).sum())
    return 1 - ssr / (sst + 1e-12)


def signal_corr(inp, rec, topk=15):
    cs = []
    for ex in inp:
        X = inp[ex].reshape(-1, inp[ex].shape[-1]); R = rec[ex].reshape(-1, rec[ex].shape[-1])
        for b in np.argsort(X.var(0))[::-1][:topk]:
            if X[:, b].std() > 1e-9 and R[:, b].std() > 1e-9:
                cs.append(np.corrcoef(X[:, b], R[:, b])[0, 1])
    return float(np.nanmean(cs)) if cs else float("nan")


def fit(spectra, hid, out, epochs=400, sched=10000):
    kw = dict(sample_name="m", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none", use_diversity_constraint=True,
              training_epochs=epochs, training_scheduler_patience=sched,
              model_dropout_rate=0.0, model_sparsity_weight=0.0,
              model_hidden_activation=hid, model_output_activation=out,
              device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    return a


def cae_metrics(a):
    model, data = a._model, a._dataset.get_all_data()
    with torch.no_grad():
        inp = {ex: data[ex].unsqueeze(0) for ex in model.excitation_wavelengths}
        rec = model.decode(model.encode(inp))
    inp_np = {ex: inp[ex][0].numpy() for ex in inp}
    rec_np = {ex: rec[ex][0].numpy() for ex in rec}
    return pooled_r2(inp_np, rec_np), signal_corr(inp_np, rec_np)


def mlp_control(spectra):
    X, _ = feature_matrix(spectra)
    sc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xt = torch.tensor(sc.astype(np.float32))
    d = Xt.shape[1]
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 8), nn.ReLU(),
                      nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, d))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(800):
        opt.zero_grad(); loss = ((m(Xt) - Xt) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        R = m(Xt).numpy()
    ssr = float(((sc - R) ** 2).sum()); sst = float(((sc - sc.mean(0)) ** 2).sum())
    return 1 - ssr / sst


def main():
    rank1, _ = build_simple(1, 1)
    clean, _, yc, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    print("=" * 90)
    print("CAE RECONSTRUCTION, re-evaluated: POOLED (variance-weighted) R2 + activation/LR fixes")
    print("=" * 90)
    print(f"MLP control pooled-R2: rank1={mlp_control(rank1):.3f} clean={mlp_control(clean):.3f}  "
          f"(near 1.0 => metric is sane and data is fittable)\n")
    print(f"{'scene':<9}{'hidden/out':<18}{'pooledR2':>10}{'signalCorr':>12}{'selF1':>8}{'oracle':>8}")
    for sname, spectra, y in [("rank1", rank1, None), ("clean", clean, np.asarray(yc))]:
        if y is not None:
            X, colmap = feature_matrix(spectra)
            orc = knn_macro_f1(X[:, oracle_cols(X, y, 12)], y, 1)
        else:
            orc = float("nan")
        for hid, out in [("sigmoid", "sigmoid"), ("gelu", "sigmoid"), ("gelu", "identity")]:
            a = fit(spectra, hid, out)
            pr2, sc = cae_metrics(a)
            if y is not None:
                X, colmap = feature_matrix(spectra)
                cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
                f1 = knn_macro_f1(X[:, cols], y, 1) if cols else float("nan")
            else:
                f1 = float("nan")
            print(f"{sname:<9}{hid + '/' + out:<18}{pr2:>10.3f}{sc:>12.3f}{f1:>8.3f}{orc:>8.3f}")
    print("-" * 90)
    print("pooledR2 -> 1 and signalCorr -> 1 mean the CAE DOES reconstruct (fixes worked).")


if __name__ == "__main__":
    main()
