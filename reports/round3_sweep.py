"""Round 3: validate the agent winners on the FULL benchmark (agents only ran 2 regimes x 2 seeds)
and on a GENERALIZATION battery of held-out synthetic regimes (regime_zoo) — does the winner hold
when the data-generation assumptions change? Writes reports/exp_records/round3_{full,general}.csv.

Run:  python reports/round3_sweep.py
"""
from __future__ import annotations

import csv
import functools
import importlib
import pathlib

import numpy as np

import method_zoo as mz
import regime_zoo as rz
import sweep_common as sc
from classification_experiment import knn_macro_f1

OUT = pathlib.Path(__file__).parent / "exp_records"


def _load(modname):
    try:
        return importlib.import_module(modname).select
    except Exception as e:
        print(f"  [import failed] {modname}: {e}")
        return None


METHODS = {}
for nm, mod in [("robust_disc", "am_robust_discriminability"), ("barlow", "am_barlow_twins"),
                ("derivative_pca", "am_derivative_pca")]:
    fn = _load(mod)
    if fn:
        METHODS[nm] = fn
METHODS["pca_load[k6]"] = functools.partial(mz.pca_load, k=6)
METHODS["ae_masked(C3)"] = mz.ae_masked
METHODS["variance"] = mz.variance
# fast methods get the full generalization battery; slow torch ones (barlow, AE) skip it
FAST = {"robust_disc", "derivative_pca", "pca_load[k6]", "variance"}


def part_full():
    cache = sc.build_cache(sc.REGIMES, (1, 2, 3))
    oracle = float(np.mean([cache[(g, s)][4] for g in sc.REGIMES for s in (1, 2, 3)]))
    print(f"\n==== FULL BENCHMARK (4 regimes x 3 seeds; oracle {oracle:.3f}) ====")
    rows = []
    for nm, fn in METHODS.items():
        r = sc.score_method(fn, cache)
        rows.append([nm, r["MEAN_F1"], r["VS_ORACLE"], *[r[g] for g in sc.REGIMES]])
        print(f"  {nm:<18} mean={r['MEAN_F1']:.3f} vsOracle={r['VS_ORACLE']:.2f}  "
              f"({'/'.join(f'{r[g]:.2f}' for g in sc.REGIMES)})")
    rows.sort(key=lambda x: -x[1])
    with open(OUT / "round3_full.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "mean_f1", "vs_oracle", *sc.REGIMES])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
    return oracle


def part_general():
    variants = list(rz.VARIANTS.items())
    seeds = (1, 2)
    print(f"\n==== GENERALIZATION ({len(variants)} held-out regimes x {len(seeds)} seeds) ====")
    # build all variant datasets once
    cache = {}
    for vname, kw in variants:
        for s in seeds:
            X, colmap, y, spectra = rz.build_variant(s, **kw)
            ocl = sc.oracle_cols(X, y, 12)
            cache[(vname, s)] = (X, colmap, y, spectra, knn_macro_f1(X[:, ocl], y, s))
    gmethods = {nm: fn for nm, fn in METHODS.items() if nm in FAST}
    header = f"  {'regime':<16}{'oracle':>8}" + "".join(f"{nm[:10]:>11}" for nm in gmethods)
    print(header)
    rows = []
    agg = {nm: [] for nm in gmethods}
    oragg = []
    for vname, kw in variants:
        line = {}
        orc = np.mean([cache[(vname, s)][4] for s in seeds]); oragg.append(orc)
        for nm, fn in gmethods.items():
            f1s = []
            for s in seeds:
                X, colmap, y, spectra, _ = cache[(vname, s)]
                try:
                    cols = list(fn(X, colmap, 12, s, np.random.default_rng(s), spectra))
                    f1s.append(knn_macro_f1(X[:, cols], y, s))
                except Exception:
                    f1s.append(float("nan"))
            line[nm] = float(np.nanmean(f1s)); agg[nm].append(line[nm])
            rows.append([vname, nm, line[nm], float(orc)])
        print(f"  {vname:<16}{orc:>8.3f}" + "".join(f"{line[nm]:>11.3f}" for nm in gmethods))
    print(f"  {'MEAN':<16}{np.mean(oragg):>8.3f}" + "".join(f"{np.mean(agg[nm]):>11.3f}" for nm in gmethods))
    with open(OUT / "round3_general.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["regime", "method", "f1", "oracle"]); w.writerows(rows)


def main():
    print(f"methods: {list(METHODS)}")
    part_full()
    part_general()
    print(f"\n-> wrote {OUT/'round3_full.csv'} and {OUT/'round3_general.csv'}")


if __name__ == "__main__":
    main()
