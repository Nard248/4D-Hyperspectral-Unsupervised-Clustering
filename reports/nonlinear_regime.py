"""Can the AE+perturbation EXCEED the linear baselines by a wide margin where it physically should —
on NONLINEAR data?

The realistic regime is near-saturated because its structure is essentially *linear*: pca_load (a
linear basis) already reaches ~97% of the oracle. An autoencoder's real advantage is *nonlinear*
manifolds. Fluorescence supplies exactly such a regime:

  * inner-filter effect (IFE) / reabsorption: bright (nuisance) species attenuate excitation and
    emission, so the measured signal is a *multiplicative, concentration-coupled* (nonlinear) function
    of the per-pixel composition -- it couples the discriminative bands to the bright nuisances;
  * concentration quenching / detector saturation: a saturating nonlinearity on intensity.

We render the realistic scene, then apply this nonlinearity to the cube, and compare blind selectors.
Honest ceilings: f_classif (LINEAR ANOVA) and mutual_info_classif (NONLINEAR) top-k oracles -- the gap
between them measures how much *nonlinear* discriminative information exists for the AE to exploit.

Hypothesis: on the nonlinear scene, pca_load drops (linear) while the AE+perturbation holds, so the AE
exceeds pca_load by a margin that grows with the nonlinearity.

Run:  python reports/nonlinear_regime.py
"""
from __future__ import annotations

import contextlib
import functools
import os

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from realistic_benchmark import band_is_discriminative, build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

BUDGET = 12


def nonlinearize(spectra, k_ife=2.5, sat=0.6):
    """Apply per-pixel inner-filter attenuation (coupled to total brightness, i.e. the bright
    nuisances) + a saturating quench, in place on each excitation cube. Returns the modified object."""
    # absorbance proxy: per-pixel total brightness across all excitations/bands, normalised
    tot = None
    cubes = []
    for ex in spectra.excitation_wavelengths:
        c = spectra.get_excitation(ex).cube
        cubes.append((ex, c))
        s = c.reshape(c.shape[0] * c.shape[1], -1).sum(1)
        tot = s if tot is None else tot + s
    A = (tot / (np.median(tot) + 1e-9)).reshape(cubes[0][1].shape[:2])  # (H,W) absorbance
    atten = np.exp(-k_ife * A)[..., None]                               # multiplicative IFE coupling
    for ex, c in cubes:
        m = c * atten                       # inner-filter: bright pixels suppress ALL their bands
        m = m / (1.0 + m / sat)             # saturating quench (nonlinear in intensity)
        spectra.get_excitation(ex).cube = m.astype(c.dtype)
    return spectra


def _ae_select(spectra, colmap, seed):
    kw = dict(backbone="conv", act="relu", mask_ratio=0.0, latent_dim=8, depth=3, width=64, epochs=300)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
    return cols_for_bands(colmap, m.select(BUDGET))


def _eval(spectra, y, seed):
    X, colmap = feature_matrix(spectra)
    disc = band_is_discriminative(colmap)
    rng = np.random.default_rng(seed)
    F = np.nan_to_num(f_classif(X, y)[0])
    MI = mutual_info_classif(X, y, random_state=seed)
    sel = {
        "variance": topn_diverse(X.var(0), colmap, BUDGET),
        "pca_load[k6]": mz.pca_load(X, colmap, BUDGET, seed, rng, spectra, k=6),
        "AE+perturb (conv)": _ae_select(spectra, colmap, seed),
        "oracle f_classif (lin)": topn_diverse(F, colmap, BUDGET),
        "oracle mutual_info (nl)": topn_diverse(MI, colmap, BUDGET),
    }
    return {name: (knn_macro_f1(X[:, c], y, seed), 100 * float(disc[c].mean()) if len(c) else 0.0)
            for name, c in sel.items()}


def main():
    seeds = [1, 2, 3]
    print("=" * 88)
    print("NONLINEAR regime (inner-filter + saturation) vs the same scene LINEAR — KNN macro-F1, 12 bands")
    print("=" * 88)
    for tag, nl in [("LINEAR (control)", False), ("NONLINEAR (IFE+sat)", True)]:
        agg = {}
        for seed in seeds:
            spectra, gt, y, acq = build_dataset(seed)
            if nl:
                spectra = nonlinearize(spectra)
            for name, (f1, frac) in _eval(spectra, y, seed).items():
                agg.setdefault(name, []).append((f1, frac))
        print(f"\n--- {tag} ---")
        print(f"{'method':<26}{'KNN macro-F1':>14}{'% disc-window':>16}")
        pca = np.mean([f for f, _ in agg["pca_load[k6]"]])
        for name, vals in agg.items():
            f1 = np.mean([f for f, _ in vals]); frac = np.mean([d for _, d in vals])
            beat = "  <-- beats pca" if (name.startswith("AE") and f1 > pca) else ""
            print(f"{name:<26}{f1:>14.3f}{frac:>15.0f}%{beat}")
    print("\nRead: if NONLINEAR widens (AE - pca_load) and (mutual_info - f_classif) oracle gap vs LINEAR,")
    print("the AE is exploiting nonlinear discriminative structure that the linear baseline cannot.")


if __name__ == "__main__":
    main()
