"""Optimization swarm: many FlexSpectralAE configs (activation x masking x bottleneck x depth) scored
on clean+realistic, tracked live, targeting selF1 > pca_load. Stage 1 = broad; pass 'stage2' to refine
the top configs with more epochs/depth/seeds. Writes reports/exp_records/swarm.csv.

Run:  python reports/swarm_run.py          (stage 1, broad)
      python reports/swarm_run.py stage2    (refine winners)
"""
from __future__ import annotations

import csv
import functools
import itertools
import pathlib
import sys
import time

import numpy as np

import method_zoo as mz
import sweep_common as sc
from classification_experiment import cols_for_bands
from swarm_zoo import FlexSpectralAE

OUT = pathlib.Path(__file__).parent / "exp_records"
REGIMES = ["clean", "realistic"]


def model_fn(**kw):
    def f(X, colmap, n, seed, rng, spectra):
        import contextlib, os
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
        return cols_for_bands(colmap, m.select(n))
    return f


def stage1_configs():
    cfgs = {}
    for act, mask, latent, depth in itertools.product(
            ["relu", "gelu", "silu", "leaky", "elu"], [0.0, 0.5], [6, 12], [2, 4]):
        cfgs[f"flex a={act} m={mask} L={latent} d={depth}"] = dict(
            act=act, mask_ratio=mask, latent_dim=latent, depth=depth, epochs=400, width=128)
    return cfgs


def stage2_configs():
    # The MLP plateaus ~0.45 on realistic; the CONV backbone (band-locality) is the hope. Conv-focused
    # swarm at practical epochs. masked + small latent (the winning recipe) emphasized.
    cfgs = {}
    # deeper conv (d=3 beat d=2 on realistic), no-mask, more epochs/width — final push on realistic
    for act, latent, depth, width in itertools.product(
            ["relu", "gelu"], [8, 12], [3, 4, 5], [64, 96]):
        cfgs[f"conv a={act} L={latent} d={depth} w={width}"] = dict(
            backbone="conv", act=act, mask_ratio=0.0, latent_dim=latent, depth=depth, width=width,
            epochs=500, weight_decay=1e-5, scheduler="cosine")
    return cfgs


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    seeds = (1, 2)
    cache = sc.build_cache(REGIMES, seeds)
    pca = sc.score_method(functools.partial(mz.pca_load, k=6), cache, regimes=REGIMES, seeds=seeds)
    oracle = float(np.mean([cache[(g, s)][4] for g in REGIMES for s in seeds]))
    cfgs = stage2_configs() if stage == "stage2" else stage1_configs()
    print(f"[swarm {stage}] {len(cfgs)} configs | seeds={seeds} | "
          f"TARGET pca_load realistic={pca['realistic']:.3f} clean={pca['clean']:.3f} | oracle={oracle:.3f}")
    print(f"{'config':<34}{'clean':>8}{'realistic':>11}{'mean':>8}{'beats pca?':>11}")
    rows = []
    best = {"clean": (0, ""), "realistic": (0, ""), "mean": (0, "")}
    for name, kw in cfgs.items():
        t = time.time()
        r = sc.score_method(model_fn(**kw), cache, regimes=REGIMES, seeds=seeds)
        beat = "YES" if r["realistic"] > pca["realistic"] else ""
        rows.append([name, r["clean"], r["realistic"], r["MEAN_F1"], time.time() - t])
        for k, v in (("clean", r["clean"]), ("realistic", r["realistic"]), ("mean", r["MEAN_F1"])):
            if v > best[k][0]:
                best[k] = (v, name)
        print(f"{name:<34}{r['clean']:>8.3f}{r['realistic']:>11.3f}{r['MEAN_F1']:>8.3f}{beat:>11}")
    rows.sort(key=lambda x: -x[2])
    with open(OUT / f"swarm_{stage}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "clean", "realistic", "mean_f1", "sec"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" if isinstance(v, float) else v for v in r[1:]])
    print(f"\nBEST realistic: {best['realistic'][1]} = {best['realistic'][0]:.3f} "
          f"(pca_load {pca['realistic']:.3f}); BEST mean: {best['mean'][1]} = {best['mean'][0]:.3f}")
    print(f"TOP-5 by mean:")
    for r in rows[:5]:
        print(f"  {r[0]:<34} clean={r[1]:.3f} realistic={r[2]:.3f}")


if __name__ == "__main__":
    main()
