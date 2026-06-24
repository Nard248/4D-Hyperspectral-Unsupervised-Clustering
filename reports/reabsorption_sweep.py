"""Decisive test: does the AE's advantage over pca_load GROW as the regime becomes more nonlinear?

Sweep the reabsorption (self-absorption) strength s. For each s, evaluate selections with the FAIR
panel (reabsorption_eval.panel_scores): report the all-bands ceiling (linear vs best-NL = how nonlinear
the regime is), then pca_load vs AE on best-NL and on the nonlinear-only gap. If the AE margin
(AE_bestNL - pca_bestNL) and the AE gap rise with s while all-bands best-NL stays up (info relocated,
not destroyed), that is direct evidence the AE exceeds the linear baseline by more the more nonlinear
the data is.

Run:  python reports/reabsorption_sweep.py
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from realistic_benchmark import build_dataset
from reabsorption_eval import _ae_select, panel_scores, summarize
from sweep_common import topn_diverse

BUDGET = 12
STRENGTHS = [0.0, 1.5, 3.0, 5.0, 8.0]
SEEDS = [1, 2]


def lean_selectors(X, colmap, spectra, y, seed):
    rng = np.random.default_rng(seed)
    MI = mutual_info_classif(X, y, random_state=seed)
    return {
        "all-bands": list(range(X.shape[1])),
        "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, spectra, k=6),
        "AE": _ae_select(spectra, colmap, seed),
        "oracle_MI": topn_diverse(MI, colmap, BUDGET),
    }


def main():
    print("=" * 92)
    print("REABSORPTION-STRENGTH SWEEP — fair panel (CV macro-F1). Does AE beat pca_load by MORE as the")
    print("regime gets more nonlinear?  margin = AE.bestNL - pca.bestNL ;  nl% = all-bands gap (regime)")
    print("=" * 92)
    print(f"{'s':>5} | {'allbands lin/NL':>16} | {'pca lin/NL/gap':>20} | {'AE lin/NL/gap':>20} | {'margin':>7}")
    rows = []
    for s in STRENGTHS:
        acc = {m: [] for m in ["all-bands", "pca_load", "AE", "oracle_MI"]}
        for seed in SEEDS:
            spectra, gt, y, acq = build_dataset(seed, size=48, reabsorption=(s > 0),
                                                reabsorption_strength=max(s, 1e-6))
            X, colmap = feature_matrix(spectra)
            for name, cols in lean_selectors(X, colmap, spectra, y, seed).items():
                acc[name].append(summarize(panel_scores(X, y, cols, seed)))

        def m(name):  # -> (lin, nl, gap) averaged over seeds
            a = acc[name]
            return (np.mean([x[0] for x in a]), np.mean([x[1] for x in a]), np.mean([x[2] for x in a]))

        ab, pca, ae, ora = m("all-bands"), m("pca_load"), m("AE"), m("oracle_MI")
        margin = ae[1] - pca[1]
        rows.append((s, ab, pca, ae, ora, margin))
        print(f"{s:>5.1f} | {ab[0]:>6.3f}/{ab[1]:<5.3f}  gap{ab[2]:+.3f} | "
              f"{pca[0]:.3f}/{pca[1]:.3f}/{pca[2]:+.3f} | {ae[0]:.3f}/{ae[1]:.3f}/{ae[2]:+.3f} | {margin:>+7.3f}")
    print("-" * 92)
    print("Interpretation: rising 'margin' with s (while all-bands NL stays up) = AE exceeds pca_load by")
    print("more as nonlinearity grows. If margin plateaus/falls and all-bands NL collapses, strong")
    print("reabsorption is DESTROYING info (not a fair test past that point).")
    best = max(rows, key=lambda r: r[5])
    print(f"\nMax AE margin over pca_load: {best[5]:+.3f} at s={best[0]} "
          f"(AE best-NL {best[3][1]:.3f} vs pca {best[2][1]:.3f}); AE nonlinear gap {best[3][2]:+.3f}.")


if __name__ == "__main__":
    main()
