"""Improvement experiments (expert-review follow-up).

Reuses the EXACT paper harness (build_dataset, roi_mask, feature_matrix, _clfs, fewshot, topn_diverse,
BUDGET=24, PER_CLASS=30) so every number here is directly comparable to beat_random.py / Figure 5.

Three probes:

E3  --clutter-blind : Does any CLUTTER-STRUCTURE-AWARE *blind* selector beat the random-95th-percentile
                      under the L2-L5 clutter regimes, where plain PCA/AE collapse to random? The clutter
                      is provably LOW-RANK by construction (add_cube_clutter = n_modes spatial fields x
                      per-band gains), so removing the dominant low-rank subspace BEFORE ranking is the
                      untested escape. If it works, the headline ("no blind method beats random under
                      clutter") needs the qualifier "no variance-ranking blind method"; if it doesn't,
                      the negative result is hardened. Hardened random: N_RANDOM_HARD draws -> empirical
                      p-value, not a 12-sample 95th percentile.

E1  --sup-select    : The principled supervised neural selector the program never built: an end-to-end
                      DIFFERENTIABLE k-subset selector (concrete/Gumbel selection layer + classifier
                      head, trained on the few labels to maximize class separability directly). Unlike
                      supAE (post-hoc occlusion) it optimizes the SELECTION through the gradient, so it
                      can lock onto JOINTLY-discriminative bands (XOR/FRET) that marginal mutInfo misses.
                      Compared to mutInfo*, RFimp*, supAE* on clutter (L4) and the sparse XOR+FRET regime.

E4  --harden        : Statistical hardening of the beat_random headline: many random draws, report the
                      empirical p-value that PCA/AE/mutInfo beat random (vs the 2-seed/12-draw original).

Run:  python reports/improve_experiments.py [--clutter-blind] [--sup-select] [--harden] [--smoke]
      (no flag = run all)
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from beat_random import PER_CLASS, REPEATS, fewshot, fit_ae
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _clfs, roi_mask
from realistic_benchmark import add_cube_clutter, build_dataset
from sparse_regime import build_sparse
from supervised_ae import supae_influence, train_supae
from sweep_common import topn_diverse

N_RANDOM_HARD = 60          # many random draws -> empirical null, not a 12-sample 95th pct


# ============================================================================================
# Clutter-structure-aware BLIND selectors (E3). All take (X, colmap, k, seed) -> column list.
# Key idea: the clutter is low-rank; strip the dominant low-rank subspace, THEN rank.
# ============================================================================================
def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def lowrank_residual(X, r):
    """Remove the top-r PCA directions (the loud clutter/nuisance subspace) -> residual."""
    Xs = _std(X)
    p = PCA(n_components=r, random_state=0).fit(Xs)
    return Xs - p.inverse_transform(p.transform(Xs))


def denoised_pca_load(X, colmap, k, seed, r=12):
    """pca_load on the residual after removing the top-r loud (clutter) directions."""
    Xr = lowrank_residual(X, r)
    p = PCA(n_components=min(8, Xr.shape[1]), random_state=seed).fit(Xr)
    return topn_diverse(np.sum(np.abs(p.components_), axis=0), colmap, k)


def residual_clusterF(X, colmap, k, seed, r=12, nclust=6, pca_embed=8):
    """Strip the loud low-rank clutter, BLINDLY cluster the residual, rank bands by between-cluster
    F-ratio on the *cluster* labels (an unsupervised discriminability proxy that is NOT variance)."""
    Xr = lowrank_residual(X, r)
    Z = PCA(n_components=min(pca_embed, Xr.shape[1]), random_state=seed).fit_transform(Xr)
    lab = KMeans(nclust, random_state=seed, n_init=4).fit_predict(Z)
    if len(np.unique(lab)) < 2:
        return topn_diverse(Xr.var(0), colmap, k)
    Fv, _ = f_classif(Xr, lab)
    return topn_diverse(np.nan_to_num(Fv), colmap, k)


def residual_variance(X, colmap, k, seed, r=12):
    """Plain variance ranking on the clutter-stripped residual (control: does stripping alone help?)."""
    return topn_diverse(lowrank_residual(X, r).var(0), colmap, k)


CLUTTER_BLIND = {
    "pca_load(plain)":   lambda X, cm, k, s: mz.pca_load(X, cm, k, s, np.random.default_rng(s), None, k=6),
    "resid-var(r12)":    lambda X, cm, k, s: residual_variance(X, cm, k, s, r=12),
    "denoised-pca(r12)": lambda X, cm, k, s: denoised_pca_load(X, cm, k, s, r=12),
    "resid-clusterF(r8)":  lambda X, cm, k, s: residual_clusterF(X, cm, k, s, r=8),
    "resid-clusterF(r16)": lambda X, cm, k, s: residual_clusterF(X, cm, k, s, r=16),
    "resid-clusterF(r24)": lambda X, cm, k, s: residual_clusterF(X, cm, k, s, r=24),
}


def run_clutter_blind(smoke):
    levels = (["L2-low", "L4-high"] if smoke else ["L2-low", "L3-moderate", "L4-high", "L5-severe"])
    seeds = ([1] if smoke else [1, 2, 3])
    print("=" * 110)
    print(f"E3  CLUTTER-STRUCTURE-AWARE BLIND SELECTION  (k={BUDGET}, few-shot {PER_CLASS}/class best-NL F1)")
    print(f"    Does stripping the low-rank clutter let a BLIND selector beat random? ({len(seeds)} seeds, "
          f"{N_RANDOM_HARD} random draws/seed)")
    print("=" * 110)
    names = list(CLUTTER_BLIND)
    header = f"{'level':<13}{'rand mu':>9}{'rand95':>8}" + "".join(f"{n[:13]:>15}" for n in names)
    print(header)
    for lvl in levels:
        params = LEVELS[lvl]
        rnd_v = []
        vals = {n: [] for n in names}
        pvals = {n: [] for n in names}
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            # hardened random null
            seed_rnd = []
            for j in range(N_RANDOM_HARD):
                rng = np.random.default_rng(seed * 1000 + j)
                seed_rnd.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
            rnd_v.extend(seed_rnd)
            seed_rnd = np.array(seed_rnd)
            for n in names:
                cols = CLUTTER_BLIND[n](X, colmap, BUDGET, seed)
                v = fewshot(Xr, yr, cols, seed)
                vals[n].append(v)
                pvals[n].append(float((seed_rnd >= v).mean()))   # empirical p that random matches/beats
        rnd = np.array(rnd_v); r95 = np.percentile(rnd, 95)
        row = f"{lvl:<13}{rnd.mean():>9.3f}{r95:>8.3f}"
        for n in names:
            m = np.mean(vals[n]); p = np.mean(pvals[n])
            flag = "*" if m > r95 else " "
            row += f"{m:>11.3f}{flag}p{p:<2.2f}"
        print(row)
    print("-" * 110)
    print("* = mean exceeds random 95th pct; pXX = mean empirical p-value (frac of random draws >= method).")
    print("If resid-clusterF beats random where pca_load(plain) does not -> the negative result needs the")
    print("qualifier 'no VARIANCE-RANKING blind method'; clutter's low-rank structure IS exploitable blind.")


# ============================================================================================
# Supervised end-to-end DIFFERENTIABLE k-subset selector (E1).
# ============================================================================================
class ConcreteSelector(nn.Module):
    """k Gumbel-softmax selection rows over d bands -> select k features -> classifier head.
    An optional decoder reconstructs the full input (semi-supervised regularizer on all pixels).
    Trained: CE on the few labels (drives SELECTION toward jointly-discriminative bands) + aux recon."""

    def __init__(self, d, k, nc, hidden=128, recon=True):
        super().__init__()
        self.d, self.k = d, k
        self.logits = nn.Parameter(torch.randn(k, d) * 0.01)
        self.clf = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, nc))
        self.recon = recon
        if recon:
            self.dec = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, d))

    def gates(self, temp, gen):
        u = torch.rand(self.k, self.d, generator=gen).clamp_(1e-20, 1.0)
        g = -torch.log(-torch.log(u))
        return torch.softmax((self.logits + g) / temp, dim=1)         # (k, d) soft one-hots

    def forward(self, x, temp, gen):
        sel = self.gates(temp, gen)
        feat = x @ sel.t()                                            # (B, k)
        out = {"logit": self.clf(feat)}
        if self.recon:
            out["recon"] = self.dec(feat)
        return out


def train_concrete_selector(Xstd, L, yL, k, nc, seed, epochs=400, recon_w=0.3, bs=256, smoke=False):
    gen = torch.Generator().manual_seed(int(seed))
    Xt = torch.tensor(Xstd, dtype=torch.float32)
    XL = torch.tensor(Xstd[L], dtype=torch.float32)
    yt = torch.tensor(yL, dtype=torch.long)
    d = Xt.shape[1]
    model = ConcreteSelector(d, k, nc, recon=recon_w > 0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = Xt.shape[0]
    epochs = 80 if smoke else epochs
    for ep in range(epochs):
        temp = 5.0 * (0.05 / 5.0) ** (ep / max(1, epochs - 1))        # anneal 5 -> 0.05
        # supervised CE on labeled subset (the loss that shapes the SELECTION)
        outL = model(XL, temp, gen)
        loss = F.cross_entropy(outL["logit"], yt)
        if model.recon:                                              # aux recon on a random pixel batch
            idx = torch.randperm(n, generator=gen)[:bs]
            outU = model(Xt[idx], temp, gen)
            loss = loss + recon_w * F.mse_loss(outU["recon"], Xt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rows = model.logits.argmax(1).tolist()
        rank = model.logits.max(0).values.numpy()
    # dedup to k distinct bands, pad by next-highest logit
    chosen = []
    for j in rows:
        if j not in chosen:
            chosen.append(int(j))
    for j in np.argsort(rank)[::-1]:
        if len(chosen) >= k:
            break
        if int(j) not in chosen:
            chosen.append(int(j))
    return chosen[:k]


def _eval_subset(Xr, yr, cols, L, T, seed):
    sc = StandardScaler().fit(Xr[L][:, cols])
    Xtr, Xte = sc.transform(Xr[L][:, cols]), sc.transform(Xr[T][:, cols])
    fs = []
    for _, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.fit(Xtr, yr[L]); fs.append(f1_score(yr[T], c.predict(Xte), average="macro"))
    return max(fs)


def run_sup_select(smoke):
    seeds = ([1] if smoke else [1, 2, 3])
    budgets = ([30] if smoke else [30, 100])           # realistic few-shot AND generous-label budget
    regimes = (["clutter(L4)"] if smoke else ["clutter(L4)", "sparse(XOR+FRET)"])
    methods = ["random", "full", "mutInfo*", "RFimp*", "supAE*", "concrete-sup*"]
    print("=" * 100)
    print(f"E1  SUPERVISED DIFFERENTIABLE k-SUBSET SELECTION  ({len(seeds)} seeds, k={BUDGET}, best-NL F1)")
    print("   marginal mutInfo is per-band (blind to XOR/FRET joint structure); concrete-sup is end-to-end")
    print("=" * 100)
    for regime in regimes:
        # build datasets once per seed (reused across label budgets)
        data = []
        for seed in seeds:
            if regime.startswith("clutter"):
                sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **LEVELS["L4-high"])
            else:
                sp, y = build_sparse(seed)
            X, colmap = feature_matrix(sp); nb = X.shape[1]
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            Xstd = StandardScaler().fit_transform(Xr)
            data.append((seed, Xr, yr, Xstd, colmap, nb))
        for per_class in budgets:
            agg = {m: [] for m in methods}
            rnd_all = []
            for seed, Xr, yr, Xstd, colmap, nb in data:
                nc = len(np.unique(yr))
                for r in range(REPEATS):
                    rng = np.random.default_rng(seed * 100 + r)
                    L = np.concatenate([rng.choice(np.where(yr == c)[0], per_class, replace=False) for c in np.unique(yr)])
                    T = np.setdiff1d(np.arange(len(yr)), L)
                    MI = np.nan_to_num(mutual_info_classif(Xr[L], yr[L], random_state=seed))
                    rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
                    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                        rf.fit(Xstd[L], yr[L])
                    model, Xt = train_supae(Xstd, L, yr[L], seed * 10 + r, nc=nc, epochs=(80 if smoke else 400))
                    infl = supae_influence(model, Xt, seed * 10 + r)
                    conc = train_concrete_selector(Xstd, L, yr[L], BUDGET, nc, seed * 10 + r, smoke=smoke)
                    sels = {"full": list(range(nb)),
                            "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                            "RFimp*": topn_diverse(rf.feature_importances_, colmap, BUDGET),
                            "supAE*": topn_diverse(infl, colmap, BUDGET),
                            "concrete-sup*": conc,
                            "random": list(rng.choice(nb, BUDGET, replace=False))}
                    for m, cols in sels.items():
                        agg[m].append(_eval_subset(Xr, yr, cols, L, T, seed))
                    for j in range(6):
                        rj = np.random.default_rng(seed * 7000 + r * 6 + j)
                        rnd_all.append(_eval_subset(Xr, yr, list(rj.choice(nb, BUDGET, replace=False)), L, T, seed))
            rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95); full = np.mean(agg["full"])
            print(f"\n### {regime}  [{per_class} labels/class]   random mu={rnd.mean():.3f} 95th={r95:.3f} | full={full:.3f}")
            mi = np.mean(agg["mutInfo*"])
            for m in methods:
                v = np.mean(agg[m])
                tag = []
                if m not in ("random", "full"):
                    if v > r95: tag.append("beats random")
                    if v > full: tag.append("BEATS FULL")
                if m in ("supAE*", "concrete-sup*"):
                    tag.append(f"vs mutInfo {v-mi:+.3f}")
                print(f"  {m:<14}{v:>8.3f}  {', '.join(tag)}")
    print("\nWin = concrete-sup* >= best supervised baseline, esp. the XOR+FRET joint regime where marginal")
    print("mutInfo is blind; and the label-budget axis shows whether the differentiable selector needs labels.")


# ============================================================================================
def run_harden(smoke):
    """E4: empirical p-value that PCA/AE/mutInfo beat random, with many random draws."""
    from sklearn.feature_selection import mutual_info_classif as _mi
    levels = (["L1-pristine", "L4-high"] if smoke else ["L1-pristine", "L2-low", "L3-moderate", "L4-high", "L5-severe"])
    seeds = ([1] if smoke else [1, 2])
    print("=" * 96)
    print(f"E4  HARDENED beat-random  ({len(seeds)} seeds, {N_RANDOM_HARD} random draws/seed -> empirical p)")
    print("=" * 96)
    print(f"{'level':<13}{'rand mu':>9}{'PCA':>8}{'p(PCA)':>9}{'AE':>8}{'p(AE)':>8}{'mutInfo*':>10}{'p(MI)':>8}")
    for lvl in levels:
        params = LEVELS[lvl]
        rows = {k: [] for k in ("pca", "ae", "mi", "pp", "pa", "pm")}
        rnd_mu = []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            ae = fit_ae(sp, seed)
            MI = np.nan_to_num(_mi(Xr, yr, random_state=seed))
            pca = fewshot(Xr, yr, mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6), seed)
            aev = fewshot(Xr, yr, cols_for_bands(colmap, ae.select(BUDGET)), seed)
            miv = fewshot(Xr, yr, topn_diverse(MI, colmap, BUDGET), seed)
            rdraw = []
            for j in range(N_RANDOM_HARD):
                rng = np.random.default_rng(seed * 1000 + j)
                rdraw.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
            rdraw = np.array(rdraw); rnd_mu.append(rdraw.mean())
            rows["pca"].append(pca); rows["ae"].append(aev); rows["mi"].append(miv)
            rows["pp"].append((rdraw >= pca).mean()); rows["pa"].append((rdraw >= aev).mean())
            rows["pm"].append((rdraw >= miv).mean())
        print(f"{lvl:<13}{np.mean(rnd_mu):>9.3f}{np.mean(rows['pca']):>8.3f}{np.mean(rows['pp']):>9.2f}"
              f"{np.mean(rows['ae']):>8.3f}{np.mean(rows['pa']):>8.2f}{np.mean(rows['mi']):>10.3f}{np.mean(rows['pm']):>8.2f}")
    print("-" * 96)
    print("p = empirical fraction of random draws matching/beating the method (small p = genuinely beats random).")


# ============================================================================================
# E6  TYPE-(b) regime: a LOW-VARIANCE but HIGH-FISHER discriminative band (Prop 3 boundary test).
# Dim, spectrally-DISTINCT, low-noise marker dyes (small amplitude => low total variance, but a clean
# consistent class shift => high Fisher) buried under bright nuisances + clutter. The program's
# "extractable <=> variance-prominent" claim predicts no method helps; Prop 3 predicts SUPERVISED
# selection strictly beats blind + random here (and that the negative result was partly a benchmark
# parameter choice). Cheap: no AE training; pca_load/variance represent the blind family (AE~=PCA).
# ============================================================================================
def build_typeb(seed, size=SIZE, ext=0.06, nuisance_amp=2.5, clutter_amp=2.5, clutter_modes=30,
                photon_scale=40000):
    """Spectrally-DISTINCT, non-overlapping marker dyes + bright nuisances + clutter, LOW photon/read
    noise. ``ext`` sets marker brightness => class-SNR: ext~0.06 gives class-SNR<1 (markers too dim to
    extract even with labels); ext~0.25 gives class-SNR>1 (extractable) while heavy clutter keeps marker
    TOTAL variance below the clutter bands (the variance!=extractability decoupling, Prop-C / synthesis 6.2)."""
    from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
    from spectraforge.fluorophore import Fluorophore
    from spectraforge.forward import render
    from spectraforge.scenegen import make_confounded_scene
    acq = AcquisitionConfig(excitations=[405.0, 470.0, 488.0, 506.0], em_min=420, em_max=700, em_step=EM_STEP)
    disc = {
        "M1": Fluorophore("M1", ex_peak_nm=470, ex_fwhm_nm=20, em_peak_nm=470, em_fwhm_nm=18, extinction=ext, quantum_yield=0.3),
        "M2": Fluorophore("M2", ex_peak_nm=488, ex_fwhm_nm=20, em_peak_nm=540, em_fwhm_nm=18, extinction=ext, quantum_yield=0.3),
        "M3": Fluorophore("M3", ex_peak_nm=506, ex_fwhm_nm=20, em_peak_nm=610, em_fwhm_nm=18, extinction=ext, quantum_yield=0.3),
    }
    from realistic_benchmark import NUIS
    lib = {**disc, **NUIS}
    disc_mats = [Material(n, {n: 1.0}) for n in disc]
    nuis_mats = [Material(n, {n: 1.0}) for n in NUIS]
    scene, labels, scatter = make_confounded_scene(disc_mats, nuis_mats, size, size, seed,
                                                   disc_amp=1.0, nuisance_amp=nuisance_amp, turbidity_amp=0.6)
    artifacts = ArtifactConfig(rayleigh_strength=0.25, raman_strength=0.2, second_order=True,
                               photon_scale=photon_scale, read_sigma=0.0015)   # LOW noise -> markers stay clean
    spectra, gt = render(scene, lib, acq, artifacts=artifacts, physics=PhysicsConfig(psf_sigma_px=1.0),
                         seed=seed, scatter_field=scatter)
    add_cube_clutter(spectra, size, seed, n_modes=clutter_modes, amp=clutter_amp)
    return spectra, labels.ravel()


TYPEB_CONFIGS = {
    "dim (class-SNR<1)":       dict(ext=0.06, nuisance_amp=2.5, clutter_amp=2.5, clutter_modes=30),
    "bright-but-buried(SNR>1)": dict(ext=0.30, nuisance_amp=3.0, clutter_amp=5.0, clutter_modes=30),
}


def run_typeb(smoke):
    seeds = ([1] if smoke else [1, 2, 3])
    per_class = 30
    configs = (list(TYPEB_CONFIGS.items())[:1] if smoke else list(TYPEB_CONFIGS.items()))
    methods = ["random", "full", "variance", "pca_load", "mutInfo*", "RFimp*", "oracle*"]
    print("=" * 100)
    print(f"E6  TYPE-(b) variance!=extractability counterexample  ({len(seeds)} seeds, {per_class} lab/class, k={BUDGET})")
    print("   Constructive Prop-C test (synthesis 6.2): a marker with class-SNR>1 (extractable) but TOTAL")
    print("   variance BELOW the clutter bands. oracle/supervised should find it; variance/pca/random miss it.")
    print("=" * 100)
    for cfgname, cfg in configs:
        data, fisher_rank, classsnr = [], [], []
        for seed in seeds:
            sp, y = build_typeb(seed, **cfg)
            X, colmap = feature_matrix(sp); nb = X.shape[1]
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            var = Xr.var(0); Fv, _ = f_classif(Xr, yr); Fv = np.nan_to_num(Fv)
            best_F = int(np.argmax(Fv))
            fisher_rank.append(int((var > var[best_F]).sum()))
            classsnr.append(float(Fv[best_F]))   # ANOVA F of the best marker band (~class-SNR proxy)
            oracle_cols = topn_diverse(Fv, colmap, BUDGET)
            data.append((seed, Xr, yr, X, colmap, nb, var, oracle_cols))
        agg = {m: [] for m in methods}; rnd_all = []
        for seed, Xr, yr, X, colmap, nb, var, oracle_cols in data:
            Xstd = StandardScaler().fit_transform(Xr)
            for r in range(REPEATS):
                rng = np.random.default_rng(seed * 100 + r)
                L = np.concatenate([rng.choice(np.where(yr == c)[0], per_class, replace=False) for c in np.unique(yr)])
                T = np.setdiff1d(np.arange(len(yr)), L)
                MI = np.nan_to_num(mutual_info_classif(Xr[L], yr[L], random_state=seed))
                rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    rf.fit(Xstd[L], yr[L])
                sels = {"full": list(range(nb)),
                        "variance": topn_diverse(var, colmap, BUDGET),
                        "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, None, k=6),
                        "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                        "RFimp*": topn_diverse(rf.feature_importances_, colmap, BUDGET),
                        "oracle*": oracle_cols,
                        "random": list(rng.choice(nb, BUDGET, replace=False))}
                for m, cols in sels.items():
                    agg[m].append(_eval_subset(Xr, yr, cols, L, T, seed))
                for j in range(8):
                    rj = np.random.default_rng(seed * 7000 + r * 8 + j)
                    rnd_all.append(_eval_subset(Xr, yr, list(rj.choice(nb, BUDGET, replace=False)), L, T, seed))
        rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95); full = np.mean(agg["full"])
        print(f"\n### {cfgname}  (ext={cfg['ext']}, clutter={cfg['clutter_amp']}x{cfg['clutter_modes']})")
        print(f"    best marker variance-rank ~{np.mean(fisher_rank):.0f}/{data[0][5]} (buried); best-band F~{np.mean(classsnr):.0f}")
        print(f"    random mu={rnd.mean():.3f} 95th={r95:.3f} | full={full:.3f}")
        for m in methods:
            v = np.mean(agg[m]); tag = []
            if m not in ("random", "full"):
                if v > r95: tag.append("beats random")
                if v > full: tag.append("BEATS FULL")
            print(f"      {m:<10}{v:>8.3f}  {', '.join(tag)}")
    print("\nIf in bright-but-buried, oracle/mutInfo/RFimp BEAT random while variance/pca do NOT, the marker is")
    print("extractable-with-labels yet variance-buried => 'extractable<=>variance-prominent' is decoupled (Prop C).")


def run_mileak(smoke):
    """P0 validation: the headline claim 'only supervised mutInfo beats random under clutter' uses
    mutInfo computed on the FULL ROI labels (incl. the few-shot TEST pixels) -> test-label leakage
    (beat_random.py:67). Recompute it the LEAKY way (full ROI) and the CORRECT way (train-subset only,
    the same labels the classifier sees) across all 5 levels, and report beats-random for each.
    If the 'fixed' mutInfo no longer beats random under clutter, the one positive headline is an artifact."""
    levels = (["L1-pristine", "L4-high"] if smoke else list(LEVELS))
    seeds = ([1] if smoke else [1, 2, 3])
    print("=" * 104)
    print(f"P0  mutInfo* TEST-LABEL LEAK CHECK  ({len(seeds)} seeds, {N_RANDOM_HARD} random draws/seed, k={BUDGET})")
    print("   MI-leak = mutual_info on FULL ROI labels (as in beat_random.py:67); MI-fix = on few-shot TRAIN subset only")
    print("=" * 104)
    print(f"{'level':<13}{'rand mu':>9}{'rand95':>8}{'MI-leak':>9}{'p':>6}{'MI-fix':>9}{'p':>6}{'PCA(blind)':>12}{'p':>6}")
    for lvl in levels:
        params = LEVELS[lvl]
        leak_v, fix_v, pca_v = [], [], []
        leak_p, fix_p, pca_p = [], [], []
        rnd_all = []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            MI_leak = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))   # FULL ROI (leak)
            pca_cols = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            seed_rnd = []
            for j in range(N_RANDOM_HARD):
                rng = np.random.default_rng(seed * 1000 + j)
                seed_rnd.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
            seed_rnd = np.array(seed_rnd); rnd_all.extend(seed_rnd.tolist())
            # MI-fix: recompute MI per few-shot split on the TRAIN labels only, average (REPEATS handled in fewshot;
            # here we approximate by a single labeled-train draw of PER_CLASS/class, matching the classifier budget)
            rng = np.random.default_rng(seed * 13)
            L = np.concatenate([rng.choice(np.where(yr == c)[0], PER_CLASS, replace=False) for c in np.unique(yr)])
            MI_fix = np.nan_to_num(mutual_info_classif(Xr[L], yr[L], random_state=seed))
            v_leak = fewshot(Xr, yr, topn_diverse(MI_leak, colmap, BUDGET), seed)
            v_fix = fewshot(Xr, yr, topn_diverse(MI_fix, colmap, BUDGET), seed)
            v_pca = fewshot(Xr, yr, pca_cols, seed)
            leak_v.append(v_leak); fix_v.append(v_fix); pca_v.append(v_pca)
            leak_p.append(float((seed_rnd >= v_leak).mean()))
            fix_p.append(float((seed_rnd >= v_fix).mean()))
            pca_p.append(float((seed_rnd >= v_pca).mean()))
        rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
        def fmt(vs, ps):
            m = np.mean(vs); p = np.mean(ps)
            return f"{m:>9.3f}{('*' if m > r95 else ' ')}p{p:<4.2f}"
        print(f"{lvl:<13}{rnd.mean():>9.3f}{r95:>8.3f}{fmt(leak_v, leak_p)}{fmt(fix_v, fix_p)}{fmt(pca_v, pca_p)}")
    print("-" * 104)
    print("* = beats random 95th pct. If MI-leak beats random under clutter but MI-fix does NOT, the headline")
    print("'only supervised selection beats random under clutter' is an artifact of test-label leakage.")


# ============================================================================================
# E7  SMOOTH (realistic, calibratable) clutter vs WHITE (adversarial) clutter (physics-review #4).
# Real fixed-pattern clutter (illumination roll-off, detector PRNU, etalon fringing) is spectrally
# SMOOTH and flat-field-calibratable. The benchmark's add_cube_clutter uses i.i.d. per-band gains
# (spectrally WHITE) -> un-regressable by a spectral AE. Test whether blind selection beats random
# under SMOOTH clutter where it does not under WHITE clutter.
# ============================================================================================
def add_smooth_clutter(spectra, size, seed, n_modes=30, amp=3.3, n_freq=3):
    """Like add_cube_clutter but each mode's per-band gain is a SMOOTH low-frequency function of the
    emission band index (sum of a few low cosines) instead of i.i.d. noise on a random subset -> the
    spectrally-smooth, low-rank, flat-field-calibratable clutter real instruments actually produce."""
    from spectraforge.scenegen import random_field
    rng = np.random.default_rng(seed * 7919 + 5)
    exs = list(spectra.excitation_wavelengths)
    nem = spectra.get_excitation(exs[0]).cube.shape[-1]
    sig = float(np.mean([spectra.get_excitation(e).cube.std() for e in exs]))
    bb = np.linspace(0, 1, nem)
    for k in range(n_modes):
        field = random_field(size, size, seed * 101 + 991 * k + 3)
        field = field - field.mean()
        for e in exs:
            gain = np.zeros(nem)
            for _ in range(n_freq):
                f = rng.uniform(0.5, 3.0); ph = rng.uniform(0, 2 * np.pi); a = rng.normal(0, 1)
                gain += a * np.cos(2 * np.pi * f * bb + ph)        # smooth low-frequency spectral gain
            gain /= (np.linalg.norm(gain) / np.sqrt(nem) + 1e-9)   # match white-gain energy
            exd = spectra.get_excitation(e)
            exd.cube = exd.cube + amp * sig * field[:, :, None] * gain[None, None, :]
    return spectra


def build_clutter_variant(seed, kind, amp=3.3, modes=30):
    """L4-style realistic dataset (nuisances+scatter+noise) with WHITE or SMOOTH fixed-pattern clutter."""
    params = dict(LEVELS["L4-high"]); params.pop("clutter_amp"); params.pop("clutter_modes")
    sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, clutter_amp=0.0, clutter_modes=0, **params)
    if kind == "white":
        add_cube_clutter(sp, SIZE, seed, n_modes=modes, amp=amp)
    elif kind == "smooth":
        add_smooth_clutter(sp, SIZE, seed, n_modes=modes, amp=amp)
    elif kind == "none":
        pass
    return sp, y


def run_smoothclutter(smoke):
    seeds = ([1] if smoke else [1, 2, 3])
    kinds = ["none", "white", "smooth"]
    methods = ["pca_load", "resid-clusterF", "variance"]
    print("=" * 104)
    print(f"E7  SMOOTH vs WHITE clutter  ({len(seeds)} seeds, k={BUDGET}, few-shot {PER_CLASS}/class, {N_RANDOM_HARD} rand draws)")
    print("   Real fixed-pattern clutter is spectrally SMOOTH/calibratable; the benchmark's is WHITE/adversarial.")
    print("=" * 104)
    print(f"{'clutter':<10}{'rand mu':>9}{'rand95':>8}{'full':>8}{'pca_load':>11}{'p':>6}{'resid-clustF':>14}{'p':>6}{'variance':>10}{'p':>6}")
    for kind in kinds:
        vals = {m: [] for m in methods}; ps = {m: [] for m in methods}
        rnd_all, full_all = [], []
        for seed in seeds:
            sp, y = build_clutter_variant(seed, kind)
            X, colmap = feature_matrix(sp); nb = X.shape[1]
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            seed_rnd = np.array([fewshot(Xr, yr, list(np.random.default_rng(seed * 1000 + j).choice(nb, BUDGET, replace=False)), seed)
                                 for j in range(N_RANDOM_HARD)])
            rnd_all.extend(seed_rnd.tolist())
            full_all.append(fewshot(Xr, yr, list(range(nb)), seed))
            sels = {"pca_load": mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6),
                    "resid-clusterF": residual_clusterF(X, colmap, BUDGET, seed, r=24),
                    "variance": topn_diverse(Xr.var(0), colmap, BUDGET)}
            for m, cols in sels.items():
                v = fewshot(Xr, yr, cols, seed)
                vals[m].append(v); ps[m].append(float((seed_rnd >= v).mean()))
        rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
        def cell(m):
            mv = np.mean(vals[m]); return f"{mv:>11.3f}{('*' if mv > r95 else ' ')}p{np.mean(ps[m]):<4.2f}"
        print(f"{kind:<10}{rnd.mean():>9.3f}{r95:>8.3f}{np.mean(full_all):>8.3f}"
              f"{cell('pca_load')}{cell('resid-clusterF')}{cell('variance')}")
    print("-" * 104)
    print("* = beats random 95th pct. If blind beats random under SMOOTH clutter but not WHITE, the headline")
    print("'no blind method beats random under clutter' is specific to adversarial white clutter (physics #4).")


def main():
    smoke = "--smoke" in sys.argv
    flags = [a for a in sys.argv[1:] if a != "--smoke"]
    run_all = not flags
    if run_all or "--mi-leak" in flags:
        run_mileak(smoke)
    if run_all or "--clutter-blind" in flags:
        run_clutter_blind(smoke)
    if run_all or "--sup-select" in flags:
        run_sup_select(smoke)
    if run_all or "--typeb" in flags:
        run_typeb(smoke)
    if run_all or "--smooth-clutter" in flags:
        run_smoothclutter(smoke)
    if run_all or "--harden" in flags:
        run_harden(smoke)


if __name__ == "__main__":
    main()
