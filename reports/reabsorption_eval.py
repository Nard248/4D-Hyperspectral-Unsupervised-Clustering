"""FAIR evaluation of band selection on a genuinely nonlinear regime (renderer reabsorption).

The point (raised explicitly): on nonlinear data the *metric itself* decides the winner. A single
linear oracle (f_classif) or a single classifier (KNN) bakes in an inductive bias and can fail to
credit a selection that captured nonlinear (band-shape / ratio) information. So we evaluate every
selected subset with a PANEL of classifiers spanning biases, cross-validated, and report:

  * linear  = LogisticRegression macro-F1            (what a linear model/PCA-aligned eval can use)
  * best-NL = max(KNN, RandomForest, MLP) macro-F1   (what an expressive model can use)
  * gap     = best-NL - linear                       (information that is ONLY nonlinearly accessible)

"Good selection" for the nonlinear case = retains high *best-NL* F1 (information-retention under an
expressive model). The gap ATTRIBUTES the advantage: a method whose subset has a large gap found
nonlinearly-informative bands that a linear method/eval would undervalue. Sanity check: on the
reabsorption scene the all-bands gap must be > 0 (nonlinear info really exists) — otherwise the regime
is not a fair nonlinear test and any "win" is an artefact.

Run:  python reports/reabsorption_eval.py
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from realistic_benchmark import band_is_discriminative, build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

BUDGET = 12


def _clfs(seed):
    return {
        "linear:logreg": LogisticRegression(max_iter=300),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "rf": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed),
    }


def panel_scores(X, y, cols, seed, folds=4):
    """CV macro-F1 of each classifier on the standardized selected subset. Returns dict name->mean."""
    if not len(cols):
        return {k: 0.0 for k in _clfs(seed)}
    Xs = X[:, cols]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {}
    for name, clf in _clfs(seed).items():
        pipe = make_pipeline(StandardScaler(), clf)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            out[name] = float(np.mean(cross_val_score(pipe, Xs, y, cv=cv, scoring="f1_macro")))
    return out


def summarize(p):
    lin = p["linear:logreg"]
    nl = max(p["knn"], p["rf"], p["mlp"])
    return lin, nl, nl - lin


def _ae_select(spectra, colmap, seed):
    kw = dict(backbone="conv", act="gelu", mask_ratio=0.5, latent_dim=8, depth=3, width=64, epochs=300)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
    return cols_for_bands(colmap, m.select(BUDGET))


def selectors(X, colmap, spectra, y, seed):
    rng = np.random.default_rng(seed)
    MI = mutual_info_classif(X, y, random_state=seed)
    return {
        "all-bands": list(range(X.shape[1])),
        "random": list(rng.choice(X.shape[1], BUDGET, replace=False)),
        "variance": topn_diverse(X.var(0), colmap, BUDGET),
        "pca_load[k6]": mz.pca_load(X, colmap, BUDGET, seed, rng, spectra, k=6),
        "AE+perturb (conv,mask)": _ae_select(spectra, colmap, seed),
        "oracle mutual_info": topn_diverse(MI, colmap, BUDGET),
    }


def main():
    seeds = [1, 2]
    print("=" * 96)
    print("FAIR eval on reabsorption (nonlinear) vs LINEAR control — CV macro-F1 panel, 12-band budget")
    print("headline = best-NL (max of knn/rf/mlp);  gap = best-NL - linear(logreg)  [nonlinear-only info]")
    print("=" * 96)
    for tag, reab in [("LINEAR (reabsorption off)", False), ("REABSORB (nonlinear, s=2.5)", True)]:
        agg = {}
        disc_frac = {}
        for seed in seeds:
            spectra, gt, y, acq = build_dataset(seed, size=48, reabsorption=reab)
            X, colmap = feature_matrix(spectra)
            disc = band_is_discriminative(colmap)
            for name, cols in selectors(X, colmap, spectra, y, seed).items():
                agg.setdefault(name, []).append(summarize(panel_scores(X, y, cols, seed)))
                disc_frac.setdefault(name, []).append(100 * float(disc[cols].mean()) if len(cols) else 0.0)
        print(f"\n--- {tag} ---")
        print(f"{'method':<26}{'linear':>9}{'best-NL':>9}{'gap':>8}{'%disc':>8}")
        pca_nl = np.mean([nl for _, nl, _ in agg['pca_load[k6]']])
        for name, vals in agg.items():
            lin = np.mean([a for a, _, _ in vals]); nl = np.mean([b for _, b, _ in vals])
            gap = np.mean([g for _, _, g in vals]); df = np.mean(disc_frac[name])
            mark = "  <-- beats pca (NL)" if name.startswith("AE") and nl > pca_nl else ""
            print(f"{name:<26}{lin:>9.3f}{nl:>9.3f}{gap:>8.3f}{df:>7.0f}%{mark}")
    print("\nRead: (1) sanity — all-bands gap>0 under REABSORB proves nonlinear info exists; (2) AE wins")
    print("if its best-NL >= pca AND its gap is larger (it captured the shape info pca/linear miss).")


if __name__ == "__main__":
    main()
