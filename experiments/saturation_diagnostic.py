"""Dataset suitability probe: does a dataset have *selection headroom*?

A channel selector can only beat random when a minority of channels carry the
discriminative signal. This probe quantifies that BEFORE we invest in the full
unsupervised pipeline: it ranks channels by a supervised score, then compares the
downstream accuracy of the best-K channels against random-K and the full set.

  headroom(K) = acc(best-K, supervised) - acc(random-K)

- headroom ~ 0  -> saturated: any subset works, selection cannot beat random (PAMAP2).
- headroom >> 0 -> informative concentrated in few channels: selection should win.

Domain-agnostic: pass a per-(trial, channel) feature function.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/saturation_diagnostic.py
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score


def flat_channels(ds):
    return [(g, c) for g in ds.groups for c in range(ds.channels_per_group[g])]


def meanstd_feats(ds, idx, selected):
    out = []
    for (g, c) in selected:
        arr = ds.data[g].numpy()[idx][:, :, c]
        out.append(arr.mean(axis=1)); out.append(arr.std(axis=1))
    return np.stack(out, axis=1)


def _acc(ds, tr, te, selected, feat_fn):
    Xtr, Xte = feat_fn(ds, tr, selected), feat_fn(ds, te, selected)
    ytr, yte = ds.labels[tr].numpy(), ds.labels[te].numpy()
    sc = StandardScaler().fit(Xtr)
    clf = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
    return float(accuracy_score(yte, clf.predict(sc.transform(Xte))))


def headroom(ds, ks=(3, 5, 10), feat_fn=meanstd_feats, seed=0, rand_seeds=15):
    """Return per-K dict: {best_supervised, random_mean, full, headroom}."""
    flat = flat_channels(ds)
    y = ds.labels.numpy()
    idx = np.arange(ds.n_windows)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    # supervised per-channel ranking (MI of each channel's mean feature with the label)
    Fmean = np.stack([ds.data[g].numpy()[tr][:, :, c].mean(axis=1) for (g, c) in flat], axis=1)
    mi = mutual_info_classif(Fmean, y[tr], random_state=seed)
    ranked = [flat[i] for i in np.argsort(-mi)]
    full = _acc(ds, tr, te, flat, feat_fn)
    out = {}
    for k in ks:
        best = _acc(ds, tr, te, ranked[:k], feat_fn)
        rand = np.mean([_acc(ds, tr, te, [flat[i] for i in np.random.default_rng(s).choice(len(flat), k, False)], feat_fn)
                        for s in range(rand_seeds)])
        out[k] = {"best_supervised": best, "random_mean": float(rand), "full": full,
                  "headroom": best - float(rand)}
    return out


def report(name, ds, ks=(3, 5, 10)):
    n_ch = sum(ds.channels_per_group.values())
    print(f"\n=== {name}: {n_ch} channels, {ds.n_windows} windows, "
          f"{len(set(ds.labels.tolist()))} classes ===")
    h = headroom(ds, ks=ks)
    print(f"{'K':>4} {'best(sup)':>10} {'random':>8} {'full':>7} {'HEADROOM':>9}")
    for k, r in h.items():
        print(f"{k:>4} {r['best_supervised']:>10.3f} {r['random_mean']:>8.3f} "
              f"{r['full']:>7.3f} {r['headroom']:>9.3f}")
    verdict = "SATURATED (selection can't beat random)" if max(r["headroom"] for r in h.values()) < 0.05 \
        else "HAS HEADROOM (selection should help)"
    print(f"verdict: {verdict}")
    return h


if __name__ == "__main__":
    from pathlib import Path
    from channel_select.adapters.pamap2_monster import load_pamap2_monster
    ds = load_pamap2_monster(Path("Data/Raw/PAMAP2_MONSTER"))
    report("PAMAP2 (27 ch)", ds)
