"""Round-2 sweep: densify the round-1 winner (PCA-loadings) across component count x preprocessing,
add preprocessing to variance, and a consensus (rank-fusion) method. Agent-discovered winners are
appended to CONFIGS as they are integrated. Writes reports/exp_records/round2_sweep.csv.

Run:  python reports/round2_sweep.py
"""
from __future__ import annotations

import csv
import functools
import pathlib

import numpy as np

import method_zoo as mz
import sweep_common as sc

OUT = pathlib.Path(__file__).parent / "exp_records"


def variance_prep(X, colmap, n, seed, rng, spectra=None, prep="none"):
    return sc.topn_diverse(mz._prep(X, colmap, prep).var(0), colmap, n)


def consensus(X, colmap, n, seed, rng, spectra=None, k=8):
    """Rank-fusion: z-scored sum of (variance, PCA-loadings on raw, PCA-loadings on shape-normalized,
    per-excitation derivative variance) — combines magnitude, subspace and shape cues."""
    from sklearn.decomposition import PCA

    def z(s):
        s = np.asarray(s, float)
        return (s - s.mean()) / (s.std() + 1e-9)
    raw = mz._std(X)
    l2 = mz._std(mz._prep(X, colmap, "l2"))
    s_var = X.var(0)
    s_pca = np.sum(np.abs(PCA(n_components=k, random_state=seed).fit(raw).components_), axis=0)
    s_pcal2 = np.sum(np.abs(PCA(n_components=k, random_state=seed).fit(l2).components_), axis=0)
    s_der = mz._prep(X, colmap, "deriv").var(0)
    score = z(s_var) + z(s_pca) + z(s_pcal2) + z(s_der)
    return sc.topn_diverse(score, colmap, n)


CONFIGS = []
for prep in ("none", "l2", "snv", "deriv"):
    for k in (5, 6, 7, 8, 9, 10, 12):
        CONFIGS.append((f"pca_load[k={k},prep={prep}]", functools.partial(mz.pca_load, k=k, prep=prep)))
for prep in ("none", "l2", "snv", "deriv"):
    CONFIGS.append((f"variance[prep={prep}]", functools.partial(variance_prep, prep=prep)))
for k in (6, 8, 10):
    CONFIGS.append((f"consensus[k={k}]", functools.partial(consensus, k=k)))

# --- agent-discovered winners get appended here (round 3) ---
AGENT_CONFIGS = []
CONFIGS += AGENT_CONFIGS


def main():
    cache = sc.build_cache(sc.REGIMES, (1, 2, 3))
    oracle = float(np.mean([cache[(g, s)][4] for g in sc.REGIMES for s in (1, 2, 3)]))
    print(f"[round2] {len(CONFIGS)} configs | oracle mean F1 = {oracle:.3f}")
    rows = []
    for label, fn in CONFIGS:
        r = sc.score_method(fn, cache)
        rows.append([label, r["MEAN_F1"], r["VS_ORACLE"], *[r[g] for g in sc.REGIMES]])
        print(f"  {label:<28} mean={r['MEAN_F1']:.3f} vsOracle={r['VS_ORACLE']:.2f}  "
              f"({'/'.join(f'{r[g]:.2f}' for g in sc.REGIMES)})")
    rows.sort(key=lambda x: -x[1])
    with open(OUT / "round2_sweep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "mean_f1", "vs_oracle", *sc.REGIMES])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
    print(f"\n==== ROUND-2 LEADERBOARD (oracle {oracle:.3f}) ====")
    for i, r in enumerate(rows[:15], 1):
        print(f"{i:>3}  {r[0]:<28}{r[1]:>8.3f}{r[2]:>8.2f}")


if __name__ == "__main__":
    main()
