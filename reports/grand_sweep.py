"""Grand sweep: evaluate every method x hyperparameter config across data regimes x seeds and rank
them by mean KNN macro-F1 (and F1/oracle). Writes reports/exp_records/grand_sweep.csv.

Run:  python reports/grand_sweep.py           (full)
      python reports/grand_sweep.py quick     (1 seed, clean+realistic, classical only)
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

import method_zoo as mz
import sweep_common as sc

OUT = pathlib.Path(__file__).parent / "exp_records"
OUT.mkdir(exist_ok=True)


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    regimes = ["clean", "realistic"] if quick else sc.REGIMES
    seeds = (1,) if quick else (1, 2, 3)
    print(f"[grand_sweep] regimes={regimes} seeds={seeds} quick={quick}")
    print("building datasets ...")
    cache = sc.build_cache(regimes, seeds)
    oracle = float(np.mean([cache[(g, s)][4] for g in regimes for s in seeds]))

    configs = list(mz.expand())
    if quick:
        configs = [(lab, fn) for lab, fn in configs if not lab.startswith("ae_")]
    print(f"evaluating {len(configs)} configs | oracle mean F1 = {oracle:.3f}")

    rows = []
    for label, fn in configs:
        t = time.time()
        r = sc.score_method(fn, cache, regimes=regimes, seeds=seeds)
        dt = time.time() - t
        rows.append([label, r["MEAN_F1"], r["VS_ORACLE"], *[r[g] for g in regimes], dt])
        print(f"  {label:<30} mean={r['MEAN_F1']:.3f} vsOracle={r['VS_ORACLE']:.2f}  [{dt:.1f}s]")

    rows.sort(key=lambda x: (-x[1] if not np.isnan(x[1]) else 1e9))
    with open(OUT / "grand_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_f1", "vs_oracle", *regimes, "sec"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" if isinstance(v, float) else v for v in r[1:]])

    print(f"\n==== LEADERBOARD (oracle ceiling mean F1 = {oracle:.3f}) ====")
    print(f"{'rank':>4}  {'method':<30}{'mean F1':>9}{'vsOracle':>9}   per-regime")
    for i, r in enumerate(rows[:25], 1):
        pr = "/".join(f"{v:.2f}" for v in r[3:3 + len(regimes)])
        print(f"{i:>4}  {r[0]:<30}{r[1]:>9.3f}{r[2]:>9.2f}   {pr}")
    print(f"-> wrote {OUT / 'grand_sweep.csv'}")


if __name__ == "__main__":
    main()
