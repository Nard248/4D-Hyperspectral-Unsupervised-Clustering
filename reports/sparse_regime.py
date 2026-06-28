"""Round-3 M-sparse — make band selection GENUINELY matter: a sparse-nonlinear-discriminative regime.
Two NARROW fluorophore peaks (few signal bands) with FRET (class = XOR co-localization), buried under
bright broad nuisances + heavy fixed-pattern clutter (most bands = noise). Here full-data overfits the
noise bands, random misses the few signal bands, and a method that FINDS the few (nonlinear) signal
bands should win decisively. Tests whether the supervised AE wins big (and where selection beats full).

Run:  python reports/sparse_regime.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_interaction_scene

import method_zoo as mz
from classification_experiment import feature_matrix
from mixnoise_experiment import BUDGET, SIZE, _clfs, roi_mask
from realistic_benchmark import add_cube_clutter
from supervised_ae import supae_influence, train_supae
from sweep_common import topn_diverse

PER_CLASS = 30
REPEATS = 4
EXC = [405.0, 470.0, 488.0, 510.0, 530.0]
D1 = Fluorophore("D1", ex_peak_nm=470, ex_fwhm_nm=28, em_peak_nm=520, em_fwhm_nm=10, extinction=0.7, quantum_yield=0.6)
D2 = Fluorophore("D2", ex_peak_nm=510, ex_fwhm_nm=28, em_peak_nm=590, em_fwhm_nm=10, extinction=0.7, quantum_yield=0.6)
NADH = Fluorophore("NADH", ex_peak_nm=405, ex_fwhm_nm=60, em_peak_nm=460, em_fwhm_nm=90, extinction=1.2, quantum_yield=0.9, em_skew=0.5)
LIPO = Fluorophore("lipofuscin", ex_peak_nm=488, ex_fwhm_nm=80, em_peak_nm=640, em_fwhm_nm=110, extinction=1.2, quantum_yield=0.9, em_skew=0.4)
LIB = {"D1": D1, "D2": D2, "NADH": NADH, "lipofuscin": LIPO}


def build_sparse(seed, size=SIZE, clutter_amp=2.5, clutter_modes=30, nuisance_amp=2.5):
    acq = AcquisitionConfig(excitations=EXC, em_min=420, em_max=700, em_step=5)
    donor, acceptor = Material("D1", {"D1": 1.0}), Material("D2", {"D2": 1.0})
    nuis = [Material("NADH", {"NADH": 1.0}), Material("lipofuscin", {"lipofuscin": 1.0})]
    scene, labels, scatter = make_interaction_scene(donor, acceptor, nuis, size, size, seed,
                                                    disc_amp=1.0, nuisance_amp=nuisance_amp, turbidity_amp=1.0)
    artifacts = ArtifactConfig(rayleigh_strength=0.6, raman_strength=0.5, second_order=True,
                               photon_scale=500, read_sigma=0.01)
    spectra, gt = render(scene, LIB, acq, artifacts=artifacts, physics=PhysicsConfig(psf_sigma_px=1.0),
                         seed=seed, scatter_field=scatter, fret_pairs=[("D1", "D2", 6.0)])
    if clutter_amp > 0:
        add_cube_clutter(spectra, size, seed, n_modes=clutter_modes, amp=clutter_amp)
    return spectra, labels.ravel()


def evalc(X, y, cols, L, T, seed):
    sc = StandardScaler().fit(X[L][:, cols])
    Xtr, Xte = sc.transform(X[L][:, cols]), sc.transform(X[T][:, cols])
    from sklearn.metrics import f1_score
    fs = []
    for n, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.fit(Xtr, y[L]); fs.append(f1_score(y[T], c.predict(Xte), average="macro"))
    return max(fs)


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    methods = ["random", "full", "variance", "pca_load", "mutInfo*", "RFimp*", "supAE*"]
    agg = {m: [] for m in methods}
    rnd_all = []
    for seed in seeds:
        sp, y = build_sparse(seed)
        X, colmap = feature_matrix(sp); nb = X.shape[1]
        roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
        Xstd = StandardScaler().fit_transform(Xr)
        for r in range(REPEATS):
            rng = np.random.default_rng(seed * 100 + r)
            L = np.concatenate([rng.choice(np.where(yr == c)[0], PER_CLASS, replace=False) for c in np.unique(yr)])
            T = np.setdiff1d(np.arange(len(yr)), L)
            MI = np.nan_to_num(mutual_info_classif(Xr[L], yr[L], random_state=seed))
            rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                rf.fit(Xstd[L], yr[L])
            model, Xt = train_supae(Xstd, L, yr[L], seed * 10 + r, nc=len(np.unique(yr)),
                                    epochs=(80 if smoke else 400))
            infl = supae_influence(model, Xt, seed * 10 + r)
            sels = {"full": list(range(nb)), "variance": topn_diverse(Xr.var(0), colmap, BUDGET),
                    "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, sp, k=6),
                    "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                    "RFimp*": topn_diverse(rf.feature_importances_, colmap, BUDGET),
                    "supAE*": topn_diverse(infl, colmap, BUDGET),
                    "random": list(rng.choice(nb, BUDGET, replace=False))}
            for m, cols in sels.items():
                agg[m].append(evalc(Xr, yr, cols, L, T, seed))
            for j in range(8):
                rj = np.random.default_rng(seed * 1000 + r * 8 + j)
                rnd_all.append(evalc(Xr, yr, list(rj.choice(nb, BUDGET, replace=False)), L, T, seed))
    rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
    full = np.mean(agg["full"])
    print("=" * 80)
    print(f"M-sparse — narrow dyes + FRET + clutter, {len(seeds)} seeds, few-shot {PER_CLASS}/class best-NL F1")
    print(f"random µ={rnd.mean():.3f} 95th={r95:.3f} | full-data={full:.3f}")
    print("=" * 80)
    for m in methods:
        v = np.mean(agg[m])
        flags = []
        if m not in ("random", "full"):
            if v > r95: flags.append("beats random")
            if v > full: flags.append("BEATS FULL")
        print(f"  {m:<10}{v:>8.3f}  {', '.join(flags)}")
    print("\nWin = supAE* beats random AND full (selection matters + supervised AE finds the sparse signal).")


if __name__ == "__main__":
    main()
