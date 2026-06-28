"""Round-2 H12 — the actionable decision map: WHEN is band selection worth it, and which kind?
Sweep label budget × clutter strength; per cell report (supervised mutInfo − full) and (blind PCA −
random). Tells you, for a given (labels, clutter): use full data, blind selection, or supervised selection.

Run:  python reports/phase_map.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, NBANDS, SIZE, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse

BASE = dict(nuisance_amp=1.0, turbidity_amp=0.5, rayleigh=0.3, raman=0.25, photon_scale=2000, read_sigma=0.005)
LABELS = [10, 25, 50, 100]
CLUTTER = [0.0, 1.5, 3.0, 4.5]
MODES = 32
N_RANDOM = 8


def fewshot(X, y, cols, seed, per_class, repeats=3):
    out = []
    for r in range(repeats):
        rng = np.random.default_rng(seed * 100 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(X[tr][:, cols])
        Xtr, Xte = sc.transform(X[tr][:, cols]), sc.transform(X[te][:, cols])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Xtr, y[tr]); fs.append(f1_score(y[te], c.predict(Xte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out))


def main():
    smoke = "--smoke" in sys.argv
    labels = ([25, 100] if smoke else LABELS)
    clutter = ([0.0, 4.5] if smoke else CLUTTER)
    seeds = ([1] if smoke else [1, 2])
    sup_full = {}   # (clutter,label) -> mutInfo - full
    blind_rand = {}
    for camp in clutter:
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, clutter_modes=MODES, clutter_amp=camp, **BASE)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            MI = topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET)
            pca = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            for pc in labels:
                full = fewshot(Xr, yr, list(range(NBANDS)), seed, pc)
                sup = fewshot(Xr, yr, MI, seed, pc)
                blind = fewshot(Xr, yr, pca, seed, pc)
                rnd = np.mean([fewshot(Xr, yr, list(np.random.default_rng(seed * 99 + j).choice(NBANDS, BUDGET, replace=False)), seed, pc)
                               for j in range(N_RANDOM)])
                sup_full.setdefault((camp, pc), []).append(sup - full)
                blind_rand.setdefault((camp, pc), []).append(blind - rnd)
    print("=" * 92)
    print("H12 PHASE MAP — rows=clutter, cols=labels/class.  Cell A: supervised mutInfo − full-data")
    print("(positive => supervised selection beats full).   Cell B: blind PCA − random.")
    print("=" * 92)
    for title, grid in [("A) mutInfo − full", sup_full), ("B) PCA − random", blind_rand)]:
        print(f"\n{title}")
        print("  clutter\\labels " + "".join(f"{pc:>9}" for pc in labels))
        for camp in clutter:
            cells = "".join(f"{np.mean(grid[(camp, pc)]):>+9.3f}" for pc in labels)
            print(f"  {camp:>12.1f} {cells}")
    print("\nRead: supervised selection wins (A>0) at high clutter + low labels; blind ~ random (B~0)")
    print("whenever clutter>0. The decision map for real use.")


if __name__ == "__main__":
    main()
