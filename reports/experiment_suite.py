"""Extended, recorded experiment suite for the spectral band-selection candidates (doc 08).

Four studies, each writing a CSV to reports/exp_records/ and printing a mean±std table:

  1. masking-rescue       — does adding the *masking objective* to the over-capacity deep net (C5 ->
                            C5b) rescue selection? (+ a mask-ratio sweep)
  2. confound phase map   — sweep the confound strength from 0 (clean) up; trace where variance-ranking
                            collapses and the learned advantage opens up; record corr(variance, F).
  3. recon-vs-selection   — every candidate's reconstruction R vs downstream F1 on realistic data, and
                            the across-candidate corr(R, F1) (the "fidelity != selection" inversion).
  4. budget robustness    — F1 vs band budget {6,12,18,24} for the main methods.

Run:  python reports/experiment_suite.py            (all studies; ~20-30 min on CPU)
      python reports/experiment_suite.py rescue     (one study by name)
"""
from __future__ import annotations

import contextlib
import csv
import os
import pathlib
import sys

import numpy as np
from sklearn.feature_selection import f_classif

import realistic_benchmark as rb
from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectral_select.architectures import (
    ConvSpectralAE, DeepMaskedSpectralAE, DeepSpectralAE, MaskedConvSpectralAE, MaskedSpectralAE,
    SpectralAE,
)

OUT = pathlib.Path(__file__).parent / "exp_records"
OUT.mkdir(exist_ok=True)
BUDGET = 12

CANDS = {
    "C2 spectral": SpectralAE, "C3 masked": MaskedSpectralAE, "C5 deep": DeepSpectralAE,
    "C5b deep-masked": DeepMaskedSpectralAE, "C6 conv": ConvSpectralAE, "C7 masked-conv": MaskedConvSpectralAE,
}


def _fit(Cls, spectra, seed, **kw):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return Cls(seed=seed, **kw).fit(spectra)


def _ms(vals):
    return f"{np.mean(vals):.3f}+-{np.std(vals):.3f}"


def _write_csv(name, header, rows):
    path = OUT / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> wrote {path.relative_to(OUT.parent)}")


def _prep(seed, **confound):
    spectra, gt, y, acq = rb.build_dataset(seed, **confound)
    X, colmap = feature_matrix(spectra)
    disc = rb.band_is_discriminative(colmap)
    F, _ = f_classif(X, y); F = np.nan_to_num(F)
    return spectra, X, colmap, disc, F, y


# ==================================================================================================
def study_candidates(seeds, extra=None):
    """Studies 1 (main) + 3: every candidate's F1 / disc% / recon R / infl-corr on realistic data."""
    print("\n[study] candidates on realistic data (F1 / disc% / recon R / infl-corr)")
    cand = dict(CANDS); cand.update(extra or {})
    per = {k: {"f1": [], "disc": [], "R": [], "corr": []} for k in
           ["variance-ranking", "discriminability-oracle", "all bands", "random", *cand]}
    rows = []
    for seed in seeds:
        spectra, X, colmap, disc, F, y = _prep(seed)
        var = X.var(0)
        def base(name, cols):
            per[name]["f1"].append(knn_macro_f1(X[:, cols], y, seed))
            per[name]["disc"].append(float(disc[cols].mean()))
            rows.append([name, seed, per[name]["f1"][-1], per[name]["disc"][-1], "", ""])
        base("all bands", list(range(X.shape[1])))
        base("variance-ranking", list(np.argsort(var)[::-1][:BUDGET]))
        base("discriminability-oracle", list(np.argsort(F)[::-1][:BUDGET]))
        base("random", list(np.random.default_rng(seed).choice(X.shape[1], BUDGET, replace=False)))
        for name, Cls in cand.items():
            m = _fit(Cls, spectra, seed)
            cols = cols_for_bands(colmap, m.select(BUDGET))
            f1 = knn_macro_f1(X[:, cols], y, seed); dfrac = float(disc[cols].mean())
            R = m.reconstruction_r(); corr = m.influence_signal_corr()
            for key, v in zip(("f1", "disc", "R", "corr"), (f1, dfrac, R, corr)):
                per[name][key].append(v)
            rows.append([name, seed, f1, dfrac, R, corr])
    _write_csv("study_candidates.csv", ["method", "seed", "f1", "disc_frac", "recon_R", "infl_corr"], rows)

    print(f"  {'method':<24}{'F1':>14}{'disc%':>8}{'recon R':>16}{'infl-corr':>16}")
    learned_R, learned_F1 = [], []
    for name in per:
        d = per[name]
        R = _ms(d["R"]) if d["R"] else "    -"
        corr = _ms(d["corr"]) if d["corr"] else "    -"
        print(f"  {name:<24}{_ms(d['f1']):>14}{100*np.mean(d['disc']):>7.0f}%{R:>16}{corr:>16}")
        if d["R"]:
            learned_R.append(np.mean(d["R"])); learned_F1.append(np.mean(d["f1"]))
    if len(learned_R) > 2:
        c = float(np.corrcoef(learned_R, learned_F1)[0, 1])
        print(f"  >> across-candidate corr(reconstruction R, F1) = {c:+.3f}  "
              f"({'fidelity does NOT buy selection' if c < 0.3 else 'fidelity tracks selection'})")


# ==================================================================================================
def study_mask_ratio(seeds):
    """Study 1b: mask-ratio sweep for the deep-masked (C5b) and masked-conv (C7) nets."""
    print("\n[study] mask-ratio sweep (C5b deep-masked, C7 masked-conv) on realistic data")
    rows = []
    print(f"  {'model':<16}{'mask':>6}{'F1':>14}{'disc%':>8}{'infl-corr':>16}")
    for label, Cls in [("C5b deep-masked", DeepMaskedSpectralAE), ("C7 masked-conv", MaskedConvSpectralAE)]:
        for mr in (0.3, 0.5, 0.7):
            f1s, dfs, corrs = [], [], []
            for seed in seeds:
                spectra, X, colmap, disc, F, y = _prep(seed)
                m = _fit(Cls, spectra, seed, mask_ratio=mr)
                cols = cols_for_bands(colmap, m.select(BUDGET))
                f1s.append(knn_macro_f1(X[:, cols], y, seed)); dfs.append(float(disc[cols].mean()))
                corrs.append(m.influence_signal_corr())
                rows.append([label, mr, seed, f1s[-1], dfs[-1], corrs[-1]])
            print(f"  {label:<16}{mr:>6.1f}{_ms(f1s):>14}{100*np.mean(dfs):>7.0f}%{_ms(corrs):>16}")
    _write_csv("study_mask_ratio.csv", ["model", "mask_ratio", "seed", "f1", "disc_frac", "infl_corr"], rows)


# ==================================================================================================
def study_phase(seeds):
    """Study 2: sweep the confound level (0=clean -> strong); fast methods only."""
    print("\n[study] confound phase diagram (F1 vs confound level; corr(var,F) = decoupling)")
    levels = [0.0, 0.25, 0.5, 1.0, 2.0]
    methods = ["all bands", "discriminability-oracle", "variance-ranking", "C2 spectral", "C3 masked", "random"]
    rows = []
    header = f"  {'level':>6}{'corr(var,F)':>12}" + "".join(f"{m.split()[0]:>10}" for m in methods)
    print(header)
    for L in levels:
        agg = {m: [] for m in methods}; vf = []
        for seed in seeds:
            spectra, X, colmap, disc, F, y = _prep(seed, nuisance_amp=2.0 * L, turbidity_amp=1.0 * L)
            var = X.var(0); vf.append(float(np.corrcoef(var, F)[0, 1]))
            agg["all bands"].append(knn_macro_f1(X, y, seed))
            agg["discriminability-oracle"].append(knn_macro_f1(X[:, np.argsort(F)[::-1][:BUDGET]], y, seed))
            agg["variance-ranking"].append(knn_macro_f1(X[:, np.argsort(var)[::-1][:BUDGET]], y, seed))
            agg["random"].append(knn_macro_f1(
                X[:, np.random.default_rng(seed).choice(X.shape[1], BUDGET, replace=False)], y, seed))
            for name, Cls in [("C2 spectral", SpectralAE), ("C3 masked", MaskedSpectralAE)]:
                m = _fit(Cls, spectra, seed)
                agg[name].append(knn_macro_f1(X[:, cols_for_bands(colmap, m.select(BUDGET))], y, seed))
            for m_ in methods:
                rows.append([L, seed, m_, agg[m_][-1], vf[-1] if m_ == methods[0] else ""])
        print(f"  {L:>6.2f}{np.mean(vf):>12.2f}" + "".join(f"{np.mean(agg[m]):>10.3f}" for m in methods))
    _write_csv("study_phase.csv", ["confound_level", "seed", "method", "f1", "corr_var_F"], rows)


# ==================================================================================================
def study_budget(seeds):
    """Study 4: F1 vs band budget for the main methods on realistic data."""
    print("\n[study] band-budget robustness (F1 vs n_bands)")
    budgets = [6, 12, 18, 24]
    methods = ["discriminability-oracle", "variance-ranking", "C2 spectral", "C3 masked", "all bands"]
    rows = []
    print(f"  {'budget':>7}" + "".join(f"{m.split()[0]:>12}" for m in methods))
    for B in budgets:
        agg = {m: [] for m in methods}
        for seed in seeds:
            spectra, X, colmap, disc, F, y = _prep(seed)
            var = X.var(0)
            agg["all bands"].append(knn_macro_f1(X, y, seed))
            agg["discriminability-oracle"].append(knn_macro_f1(X[:, np.argsort(F)[::-1][:B]], y, seed))
            agg["variance-ranking"].append(knn_macro_f1(X[:, np.argsort(var)[::-1][:B]], y, seed))
            for name, Cls in [("C2 spectral", SpectralAE), ("C3 masked", MaskedSpectralAE)]:
                m = _fit(Cls, spectra, seed)
                agg[name].append(knn_macro_f1(X[:, cols_for_bands(colmap, m.select(B))], y, seed))
            for m_ in methods:
                rows.append([B, seed, m_, agg[m_][-1]])
        print(f"  {B:>7}" + "".join(f"{np.mean(agg[m]):>12.3f}" for m in methods))
    _write_csv("study_budget.csv", ["budget", "seed", "method", "f1"], rows)


STUDIES = {"rescue": lambda: (study_candidates([1, 2, 3]), study_mask_ratio([1, 2, 3])),
           "phase": lambda: study_phase([1, 2, 3, 4, 5]),
           "budget": lambda: study_budget([1, 2, 3, 4, 5])}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 92)
    print("EXTENDED EXPERIMENT SUITE — recorded to reports/exp_records/")
    print("=" * 92)
    if which in ("all", "rescue"):
        study_candidates([1, 2, 3])
        study_mask_ratio([1, 2, 3])
    if which in ("all", "phase"):
        study_phase([1, 2, 3, 4, 5])
    if which in ("all", "budget"):
        study_budget([1, 2, 3, 4, 5])
    print("\n" + "=" * 92)
    print("done. CSVs in reports/exp_records/ ; see docs/spectraforge/08-extended-experiments.md")


if __name__ == "__main__":
    main()
