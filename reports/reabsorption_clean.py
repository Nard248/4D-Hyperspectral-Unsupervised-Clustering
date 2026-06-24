"""Cleaner nonlinear-DISCRIMINATIVE regime: boost the discriminative dyes' extinction so *their*
concentration (not the nuisances') dominates the reabsorption reshaping. With more seeds, test whether
the AE+perturbation beats pca_load by a clear, significant margin under the fair panel metric.

Reports mean +/- std over seeds of best-NL for pca_load vs AE, the margin, and the nonlinear-only gap.

Run:  python reports/reabsorption_clean.py
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

import method_zoo as mz
from classification_experiment import feature_matrix
from realistic_benchmark import build_dataset
from reabsorption_eval import _ae_select, panel_scores, summarize
from sweep_common import topn_diverse

BUDGET = 12
SEEDS = [1, 2, 3, 4]
STRENGTHS = [2.5, 4.0]
DISC_EXT = 2.0   # discriminative dyes self-absorb strongly -> discriminative info -> band shape


def main():
    print("=" * 90)
    print(f"CLEAN nonlinear-discriminative regime (disc_extinction={DISC_EXT}), {len(SEEDS)} seeds")
    print("fair panel CV macro-F1; best-NL mean+/-std; margin = AE - pca_load; gap = nonlinear-only info")
    print("=" * 90)
    print(f"{'s':>5} | {'allbands NL':>11} | {'pca best-NL':>16} | {'AE best-NL':>16} | {'margin':>8} | {'AEgap':>6}")
    for s in STRENGTHS:
        rec = {m: [] for m in ["all-bands", "pca_load", "AE", "oracle_MI"]}
        for seed in SEEDS:
            spectra, gt, y, acq = build_dataset(seed, size=48, reabsorption=True,
                                                reabsorption_strength=s, disc_extinction=DISC_EXT)
            X, colmap = feature_matrix(spectra)
            rng = np.random.default_rng(seed)
            MI = mutual_info_classif(X, y, random_state=seed)
            sels = {
                "all-bands": list(range(X.shape[1])),
                "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, spectra, k=6),
                "AE": _ae_select(spectra, colmap, seed),
                "oracle_MI": topn_diverse(MI, colmap, BUDGET),
            }
            for name, cols in sels.items():
                rec[name].append(summarize(panel_scores(X, y, cols, seed)))

        def stat(name):
            nl = np.array([x[1] for x in rec[name]]); gap = np.array([x[2] for x in rec[name]])
            return nl.mean(), nl.std(), gap.mean()

        ab_nl = np.mean([x[1] for x in rec["all-bands"]])
        ab_gap = np.mean([x[2] for x in rec["all-bands"]])
        p_nl, p_sd, p_gap = stat("pca_load")
        a_nl, a_sd, a_gap = stat("AE")
        margin = a_nl - p_nl
        sig = "  SIGNIF" if margin > (p_sd + a_sd) else ""
        print(f"{s:>5.1f} | {ab_nl:.3f} g{ab_gap:+.3f} | {p_nl:.3f}+/-{p_sd:.3f} | "
              f"{a_nl:.3f}+/-{a_sd:.3f} | {margin:>+8.3f} | {a_gap:>+6.3f}{sig}")
    print("-" * 90)
    print("SIGNIF flag: margin exceeds the summed per-method std (a rough significance bar). 'by far")
    print("more' would need margin >> std AND a large positive AE gap vs a ~0 pca gap.")


if __name__ == "__main__":
    main()
