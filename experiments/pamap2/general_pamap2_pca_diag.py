"""PAMAP2 LOSO: PCA-loading baseline (unsupervised, marginal) to complete the boundary
chart of the CODASSCA poster. Same protocol as general_pamap2_baseline_diag.py
(KNN-5 macro-F1, mean+std features, subjects 1-8 leave-one-subject-out).

Run: .venv/bin/python experiments/pamap2/general_pamap2_pca_diag.py
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from general_pamap2_baseline_diag import (load_pamap2_monster, channel_features, knn_f1, cols_for,
                                          DATA_DIR, SUBJECTS, KS)

def main():
    ds = load_pamap2_monster(DATA_DIR)
    res = {K: [] for K in KS}
    for holdout in SUBJECTS:
        tr, te = ds.loso_split(holdout)
        Xtr, owner, pairs = channel_features(ds, tr)
        Xte, _, _ = channel_features(ds, te)
        ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
        Z = StandardScaler().fit_transform(Xtr)
        load = np.abs(PCA(n_components=8).fit(Z).components_).sum(0)
        pca_by_ch = {p: float(load[cols_for(owner, {p})].sum()) for p in pairs}
        rank = sorted(pairs, key=lambda p: -pca_by_ch[p])
        for K in KS:
            res[K].append(knn_f1(Xtr, ytr, Xte, yte, cols_for(owner, set(rank[:K]))))
        print(f"subject {holdout} done", flush=True)
    lines = ["PAMAP2 LOSO PCA-loading baseline | KNN-5 macro-F1"]
    for K in KS:
        a = np.array(res[K]); lines.append(f"K={K}\tpca \t{a.mean():.4f} +/- {a.std():.4f}")
    out = Path("publications/generalization/reports/pamap2_pca_diag.txt")
    out.write_text("\n".join(lines)); print("\n".join(lines)); print("wrote", out)

if __name__ == "__main__":
    main()
