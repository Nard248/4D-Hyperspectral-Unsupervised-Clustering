"""Debug the autoencoder: does fixing the activations (3 stacked hidden sigmoids -> relu/gelu) make
the SAME CAE architecture actually reconstruct? Keeps the idea (spatial CAE + band-collapse +
per-excitation branches + perturbation); only the activation functions change. Trains with
regularization off (the real test of representational/trainability capacity) and measures
reconstruction R2 (vs per-band-mean) + selection F1, on rank-1, clean, and realistic scenes. Saves
input-vs-recon images for the published-sigmoid vs the fixed activation.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import realistic_benchmark as rb
from cae_data_search import build_simple, recon_r2
from classification_experiment import build_dataset as clean_build, cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config
from sweep_common import oracle_cols

OUT = pathlib.Path(__file__).parent / "exp_records" / "recon_audit"
OUT.mkdir(parents=True, exist_ok=True)
ACTS = [("sigmoid", "sigmoid"), ("relu", "sigmoid"), ("relu", "identity"), ("gelu", "identity")]


def fit(spectra, hid, out, epochs=200):
    kw = dict(sample_name="dbg", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none", use_diversity_constraint=True,
              training_epochs=epochs, model_dropout_rate=0.0, model_sparsity_weight=0.0,
              model_hidden_activation=hid, model_output_activation=out,
              device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    return a


def save_recon(tag, model, data):
    ex = list(model.excitation_wavelengths)[0]
    with torch.no_grad():
        inp = {e: data[e].unsqueeze(0) for e in model.excitation_wavelengths}
        rec = model.decode(model.encode(inp))
    X, R = inp[ex][0].numpy(), rec[ex][0].numpy()
    b = int(np.argmax(X.reshape(-1, X.shape[2]).var(0)))
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a_, img, t in zip(ax, [X[:, :, b], R[:, :, b], R[:, :, b] - X[:, :, b]], ["input", "recon", "residual"]):
        im = a_.imshow(img, cmap="viridis"); a_.set_title(f"{t}", fontsize=9); a_.axis("off")
        plt.colorbar(im, ax=a_, fraction=0.046)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(OUT / f"dbg_{tag}.png", dpi=90); plt.close(fig)


def main():
    rank1, _ = build_simple(1, 1)
    clean, _, yc, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    real, _, yr, _ = rb.build_dataset(1)
    scenes = [("rank1", rank1, None), ("clean", clean, np.asarray(yc)), ("realistic", real, np.asarray(yr))]
    print("=" * 78)
    print("AUTOENCODER DEBUG — activation fix (no-reg, 200 epochs); reconstruction R2 + selection F1")
    print("=" * 78)
    print(f"{'scene':<11}{'hidden':<8}{'output':<10}{'reconR2':>10}{'F1':>8}{'oracle':>8}")
    for sname, spectra, y in scenes:
        X, colmap = feature_matrix(spectra)
        orc = knn_macro_f1(X[:, oracle_cols(X, y, 12)], y, 1) if y is not None else float("nan")
        for hid, out in ACTS:
            a = fit(spectra, hid, out)
            r2 = recon_r2(a._model, a._dataset.get_all_data())
            if y is not None:
                cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
                f1 = knn_macro_f1(X[:, cols], y, 1) if cols else float("nan")
            else:
                f1 = float("nan")
            print(f"{sname:<11}{hid:<8}{out:<10}{r2:>+10.3f}{f1:>8.3f}{orc:>8.3f}")
            if sname == "clean" and (hid, out) in (("sigmoid", "sigmoid"), ("relu", "identity")):
                save_recon(f"clean_{hid}_{out}", a._model, a._dataset.get_all_data())
    print("-" * 78)
    print("If relu/gelu lifts reconR2 from <0 to >0, the 3 stacked hidden sigmoids were the bug.")


if __name__ == "__main__":
    main()
