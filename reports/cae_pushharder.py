"""Push the fixed CAE + perturbation toward the best known methods (~0.48 = pca_load/oracle level).
Two levers, both keeping the AE+perturbation idea:

A. Band-preserving spatial CAE (no collapse) trained once; then SWEEP the perturbation/selection knobs
   cheaply on the same trained latent (n_important dims, dimension-selection method, perturbation
   method) — selection is where the gains are, since reconstruction fidelity is anti-correlated with
   selection quality.
B. The per-pixel collapse-bottleneck conv AE (already ~0.50 on clean) over clean+realistic x seeds.

Run:  python reports/cae_pushharder.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np

import selection_core
from classification_experiment import build_dataset as clean_build, cols_for_bands, feature_matrix, knn_macro_f1
import realistic_benchmark as rb
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config
from spectral_select.architectures.conv_spectral_ae import _ConvSpectralBase
from spectral_select.architectures.deep_cae import DeepSpectralCAE
from sweep_common import oracle_cols, topn_diverse


def train_bandpreserving(spectra, epochs=600, k=24):
    kw = dict(sample_name="ph", n_bands_to_select=12, perturbation_method="percentile",
              normalization_method="none", use_diversity_constraint=True, training_epochs=epochs,
              training_scheduler_patience=10000, model_dropout_rate=0.0, model_sparsity_weight=0.0,
              model_k1=k, model_k3=k, autoencoder_architecture=DeepSpectralCAE,
              device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    return a


def influence_cols(a, colmap, n, n_imp, dim_method, pert_method, mags):
    model = a._model
    bl, br = a._baseline_latent, a._baseline_reconstruction
    important = selection_core.select_important_dimensions(bl, dim_method, n_imp)
    cpg = {ex: model.emission_bands[ex] for ex in model.excitation_wavelengths}
    infl = selection_core.accumulate_influence(model.decode, list(model.excitation_wavelengths), cpg,
                                               bl, br, important, magnitudes=mags,
                                               directions=["bidirectional"], perturbation_method=pert_method)
    first = {}
    for j, (ex, em) in enumerate(colmap):
        first.setdefault(ex, j)
    score = np.array([infl[ex][j - first[ex]] if ex in infl else 0.0 for j, (ex, em) in enumerate(colmap)])
    return topn_diverse(score, colmap, n)


class _CollapseConvAE(_ConvSpectralBase):           # per-pixel conv AE with band-collapse bottleneck
    name = "collapse-bottleneck conv-AE"

    def _make_module(self, d):
        from spectral_select.architectures.conv_spectral_ae import _ConvAE
        ex = sorted({e for e, _ in self.colmap}); n_ex = len(ex); n_band = d // n_ex
        return _ConvAE(n_ex, n_band, latent=self.latent_dim, width=self.width)

    def _loss(self, model, Xt, gen):
        return ((model.decode(model.encode(Xt)) - Xt) ** 2).mean()


def main():
    clean, _, yc, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    yc = np.asarray(yc); Xc, colmap = feature_matrix(clean)
    orc = knn_macro_f1(Xc[:, oracle_cols(Xc, yc, 12)], yc, 1)
    print("=" * 86)
    print(f"PUSH HARDER — target ~{orc:.3f} (oracle) / 0.48 (pca_load). clean 3-dye, seed 1")
    print("=" * 86)

    print("\n[A] band-preserving CAE (no collapse), 600 epochs — sweep selection knobs:")
    a = train_bandpreserving(clean, epochs=600, k=24)
    best = (-1, None)
    print(f"  {'dimsel':<10}{'pert':<14}{'n_imp':>6}{'selF1':>8}")
    for dim_method in ("variance", "activation", "pca"):
        for pert, mags in (("percentile", [10, 20, 30]), ("standard_deviation", [50, 100, 200])):
            for n_imp in (15, 40, 80):
                cols = influence_cols(a, colmap, 12, n_imp, dim_method, pert, mags)
                f1 = knn_macro_f1(Xc[:, cols], yc, 1) if cols else 0.0
                if f1 > best[0]:
                    best = (f1, (dim_method, pert, n_imp))
                print(f"  {dim_method:<10}{pert:<14}{n_imp:>6}{f1:>8.3f}")
    print(f"  -> best band-preserving CAE selF1 = {best[0]:.3f} at {best[1]}  (oracle {orc:.3f})")

    print("\n[B] per-pixel collapse-bottleneck conv-AE over clean+realistic x 3 seeds:")
    for regime in ("clean", "realistic"):
        f1s = []
        for s in (1, 2, 3):
            if regime == "clean":
                sp, _, y, _ = clean_build(s, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
            else:
                sp, _, y, _ = rb.build_dataset(s)
            y = np.asarray(y); X, cm = feature_matrix(sp)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                m = _CollapseConvAE(latent_dim=8, epochs=300, seed=s).fit(sp)
            cols = cols_for_bands(cm, m.select(12))
            f1s.append(knn_macro_f1(X[:, cols], y, s))
        print(f"  {regime:<10} selF1 = {np.mean(f1s):.3f} +- {np.std(f1s):.3f}")


if __name__ == "__main__":
    main()
