"""Five synthetic full-ME-HSI datasets at increasing MIXTURE+NOISE, high spectral resolution.

Pipeline (with a verification gate printed at every stage):
  1. Build 5 levels (pristine -> severe): more nuisance spectral mixture + more photon/read/scatter noise.
     High resolution: emission 420-700 nm @ 2 nm = 141 bands x 4 excitations = 564 features.
  2. ROIs: because we author the scene, the ground-truth concentration fields give a purity margin per
     pixel; the clearest 50% form the ROI (clean labels, like an analyst drawing regions on clear areas).
  3. Full-data classification (KNN + RF + MLP + logreg, CV) on ALL 564 bands -> the ceiling per level.
  4. Select k=24 bands with PCA (pca_load) and the AE (conv & mlp) + supervised mutual_info reference.
  5. Re-classify on the selected bands; report the ACCURACY DIFFERENCE vs full data, and AE - PCA.

Run:  python reports/mixnoise_experiment.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from realistic_benchmark import DISC, build_dataset
from spectraforge.scenegen import random_field
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

BUDGET = 24
SIZE = 64
EM_STEP = 2
# 5 levels: (nuisance_amp = spectral mixture, photon_scale & read_sigma & scatter = noise)
LEVELS = {
    "L1-pristine": dict(nuisance_amp=0.2, turbidity_amp=0.1, rayleigh=0.10, raman=0.10, photon_scale=100000, read_sigma=0.001),
    "L2-low":      dict(nuisance_amp=0.6, turbidity_amp=0.3, rayleigh=0.20, raman=0.20, photon_scale=20000,  read_sigma=0.003),
    "L3-moderate": dict(nuisance_amp=1.2, turbidity_amp=0.6, rayleigh=0.35, raman=0.30, photon_scale=2000,   read_sigma=0.005),
    "L4-high":     dict(nuisance_amp=2.0, turbidity_amp=1.0, rayleigh=0.50, raman=0.40, photon_scale=600,    read_sigma=0.008),
    "L5-severe":   dict(nuisance_amp=3.0, turbidity_amp=1.4, rayleigh=0.70, raman=0.50, photon_scale=200,    read_sigma=0.012),
}
_AE_CONV = dict(backbone="conv", act="gelu", mask_ratio=0.5, latent_dim=8, depth=3, width=64, epochs=250)
_AE_MLP = dict(backbone="mlp", act="gelu", mask_ratio=0.5, latent_dim=6, depth=3, width=128, epochs=300)


def roi_mask(seed, size, frac=0.5):
    fields = np.stack([random_field(size, size, seed * 13 + 31 * k) for k in range(len(DISC))])
    purity = (np.sort(fields, 0)[-1] - np.sort(fields, 0)[-2]).ravel()
    return purity >= np.quantile(purity, 1 - frac)


def _clfs(seed):
    return {"knn": KNeighborsClassifier(7),
            "rf": RandomForestClassifier(150, random_state=seed, n_jobs=-1),
            "mlp": MLPClassifier(hidden_layer_sizes=(48,), max_iter=250, random_state=seed),
            "logreg": LogisticRegression(max_iter=300)}


def classify(X, y, cols, seed, folds=4):
    """CV macro-F1 + balanced-acc on standardized selected bands; return per-clf and the best-NL/knn."""
    Xs = X[:, cols]
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    f1, ba = {}, {}
    for n, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            yh = cross_val_predict(make_pipeline(StandardScaler(), c), Xs, y, cv=cv)
        f1[n] = f1_score(y, yh, average="macro"); ba[n] = balanced_accuracy_score(y, yh)
    bestnl = max(f1["knn"], f1["rf"], f1["mlp"])
    return dict(knn=f1["knn"], rf=f1["rf"], mlp=f1["mlp"], logreg=f1["logreg"],
                bestNL=bestnl, knn_ba=ba["knn"], bestNL_ba=max(ba["knn"], ba["rf"], ba["mlp"]))


def ae_cols(spectra, colmap, seed, kw):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
    return cols_for_bands(colmap, m.select(BUDGET))


def main():
    smoke = "--smoke" in sys.argv
    levels = (["L1-pristine", "L5-severe"] if smoke else list(LEVELS))
    seeds = ([1] if smoke else [1, 2, 3])
    print("=" * 100)
    print(f"5-LEVEL MIX/NOISE EXPERIMENT — high-res 564-band ME-HSI, ROI pixels, budget={BUDGET}/{4*141}")
    print("=" * 100)
    summary = []
    for lvl in levels:
        params = LEVELS[lvl]
        full, selp, sela_c, sela_m, selmi = [], [], [], [], []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE)
            Xr, yr = X[roi], y[roi]
            # ---- per-step verification ----
            assert X.shape[1] == 4 * 141, X.shape
            assert np.isfinite(X).all()
            assert set(np.unique(yr)) == {0, 1, 2}
            full.append(classify(Xr, yr, list(range(X.shape[1])), seed))
            rng = np.random.default_rng(seed)
            selp.append(classify(Xr, yr, mz.pca_load(X, colmap, BUDGET, seed, rng, sp, k=6), seed))
            sela_c.append(classify(Xr, yr, ae_cols(sp, colmap, seed, _AE_CONV), seed))
            sela_m.append(classify(Xr, yr, ae_cols(sp, colmap, seed, _AE_MLP), seed))
            mi = topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET)
            selmi.append(classify(Xr, yr, mi, seed))

        def avg(rows, k):
            return float(np.mean([r[k] for r in rows]))

        f_nl, p_nl = avg(full, "bestNL"), avg(selp, "bestNL")
        ac_nl, am_nl = avg(sela_c, "bestNL"), avg(sela_m, "bestNL")
        ae_nl = max(ac_nl, am_nl)
        mi_nl = avg(selmi, "bestNL")
        print(f"\n### {lvl}  (nuis={params['nuisance_amp']}, photon={params['photon_scale']}, "
              f"read={params['read_sigma']})  | {len(seeds)} seeds, ROI px")
        print(f"  VERIFY: 564 bands, 3 classes, finite — OK")
        print(f"  full-data(564)  bestNL={f_nl:.3f}  knn={avg(full,'knn'):.3f}  logreg={avg(full,'logreg'):.3f}")
        print(f"  PCA  (k={BUDGET})  bestNL={p_nl:.3f}  knn={avg(selp,'knn'):.3f}   Δ(full-sel)={f_nl-p_nl:+.3f}")
        print(f"  AE-conv(k={BUDGET}) bestNL={ac_nl:.3f}  knn={avg(sela_c,'knn'):.3f}   Δ(full-sel)={f_nl-ac_nl:+.3f}")
        print(f"  AE-mlp (k={BUDGET}) bestNL={am_nl:.3f}  knn={avg(sela_m,'knn'):.3f}   Δ(full-sel)={f_nl-am_nl:+.3f}")
        print(f"  mutual_info* k={BUDGET} bestNL={mi_nl:.3f}  (supervised reference)")
        print(f"  >>> AE(best) - PCA = {ae_nl - p_nl:+.3f}   |   retention: PCA {p_nl/f_nl:.0%}, AE {ae_nl/f_nl:.0%}")
        summary.append((lvl, f_nl, p_nl, ae_nl, mi_nl, ae_nl - p_nl))

    print("\n" + "=" * 100)
    print(f"{'level':<14}{'full564':>9}{'PCA':>8}{'AE':>8}{'mutInfo*':>10}{'AE-PCA':>9}{'PCAret':>8}{'AEret':>7}")
    for lvl, f_nl, p_nl, ae_nl, mi_nl, d in summary:
        print(f"{lvl:<14}{f_nl:>9.3f}{p_nl:>8.3f}{ae_nl:>8.3f}{mi_nl:>10.3f}{d:>+9.3f}{p_nl/f_nl:>8.0%}{ae_nl/f_nl:>7.0%}")
    print("\nΔ(full-sel) = accuracy lost by selecting 24/564 bands; AE-PCA = which selector retains more.")


if __name__ == "__main__":
    main()
