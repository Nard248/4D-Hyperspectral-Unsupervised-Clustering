"""Sweep per-pixel parallel-branch conv setups (band-kernel, depth, merge, band-collapse on/off) and
compare to references (plain spectral MLP-AE, PCA-loadings). Reports reconstruction corr + selection
KNN-F1 on clean and realistic. Answers: does the CAE's parallel-branch design work per-pixel (no
spatial conv)? Is the band-collapse the killer? Which conv setup is best?

Run:  python reports/conv_arch_search.py
"""
from __future__ import annotations

import contextlib
import functools
import os

import numpy as np

import method_zoo as mz
import sweep_common as sc
from classification_experiment import cols_for_bands, knn_macro_f1
from conv_arch_zoo import ParallelBranchSpectralAE
from spectral_select.architectures import SpectralAE


def configs():
    c = {}
    c["pca_load[k6] (ref)"] = ("fn", functools.partial(mz.pca_load, k=6))
    c["C2 spectral-MLP (ref)"] = ("model", SpectralAE)
    for kb in (3, 5, 9):
        for depth in (1, 2, 3):
            c[f"branch kb{kb} d{depth} concat"] = (
                "model", functools.partial(ParallelBranchSpectralAE, kb=kb, depth=depth,
                                           collapse=False, merge="concat"))
    c["branch kb5 d2 MEAN-merge"] = ("model", functools.partial(ParallelBranchSpectralAE, kb=5, depth=2,
                                                                collapse=False, merge="mean"))
    c["branch kb5 d2 +COLLAPSE concat"] = ("model", functools.partial(ParallelBranchSpectralAE, kb=5, depth=2,
                                                                       collapse=True, merge="concat"))
    c["branch kb5 d2 +COLLAPSE mean"] = ("model", functools.partial(ParallelBranchSpectralAE, kb=5, depth=2,
                                                                     collapse=True, merge="mean"))
    return c


def main():
    seed = 1
    regs = {r: sc.make_regime(r, seed) for r in ("clean", "realistic")}
    orc = {r: knn_macro_f1(X[:, sc.oracle_cols(X, y, 12)], y, seed) for r, (X, cm, y, sp) in regs.items()}
    print("=" * 96)
    print("CONV-ARCHITECTURE SEARCH (per-pixel parallel branches; no spatial conv)")
    print(f"oracle F1: clean={orc['clean']:.3f} realistic={orc['realistic']:.3f}  "
          f"(spatial CAE ref: reconR~0, F1~0.33)")
    print("=" * 96)
    print(f"{'config':<32}{'clean reconR':>13}{'clean F1':>10}{'real reconR':>13}{'real F1':>10}")
    for name, (kind, factory) in configs().items():
        out = {}
        for r, (X, colmap, y, spectra) in regs.items():
            if kind == "fn":
                cols = list(factory(X, colmap, 12, seed, np.random.default_rng(seed), spectra))
                rr = float("nan")
            else:
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    m = factory(seed=seed).fit(spectra)
                cols = cols_for_bands(colmap, m.select(12))
                rr = m.reconstruction_r()
            out[r] = (rr, knn_macro_f1(X[:, cols], y, seed) if cols else float("nan"))
        print(f"{name:<32}{out['clean'][0]:>+13.3f}{out['clean'][1]:>10.3f}"
              f"{out['realistic'][0]:>+13.3f}{out['realistic'][1]:>10.3f}")
    print("-" * 96)
    print("reconR = mean per-band corr(input,recon) (>0 => learns structure). Compare COLLAPSE vs not.")


if __name__ == "__main__":
    main()
