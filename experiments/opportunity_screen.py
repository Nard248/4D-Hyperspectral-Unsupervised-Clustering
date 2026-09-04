"""Screen a high-channel HAR dataset (Opportunity via MONSTER, 113 channels): does our
UNSUPERVISED selection track supervised MI / PCA and beat random? Within-distribution
split (the protocol that reveals selection headroom).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from channel_select.data import GroupedChannelDataset
from channel_select.models.temporal_bottleneck import BottleneckTemporalAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
R = "monster-monash/Opportunity"
X = np.nan_to_num(np.load(hf_hub_download(R, "Opportunity_X.npy", repo_type="dataset")).astype(np.float32))
y = np.load(hf_hub_download(R, "Opportunity_y.npy", repo_type="dataset")).astype(int)
N, C, T = X.shape
print(f"Opportunity {X.shape} | classes={len(set(y.tolist()))}", flush=True)

NG = 8
bnd = np.linspace(0, C, NG + 1).astype(int)
data = {f"g{i}": torch.tensor(X[:, bnd[i]:bnd[i+1], :].transpose(0, 2, 1), dtype=torch.float32) for i in range(NG)}
ds = GroupedChannelDataset(data, axis_type="temporal1d", labels=torch.tensor(y, dtype=torch.long))
def g2glob(gc): i = int(gc[0][1:]); return bnd[i] + gc[1]

Xmean = X.mean(axis=2); Xstd = X.std(axis=2)
F = np.concatenate([Xmean, Xstd], axis=1)  # (N, 2C) mean+std per channel
def feat(sel_glob, rows): return np.concatenate([Xmean[rows][:, sel_glob], Xstd[rows][:, sel_glob]], axis=1)

tr, te = train_test_split(np.arange(N), test_size=0.3, random_state=0, stratify=y)
def acc(sel_glob):
    if len(sel_glob) == 0: return float("nan")
    sc = StandardScaler().fit(feat(sel_glob, tr))
    clf = KNeighborsClassifier(5).fit(sc.transform(feat(sel_glob, tr)), y[tr])
    return accuracy_score(y[te], clf.predict(sc.transform(feat(sel_glob, te))))

# baselines (global channel scores on train)
mi = mutual_info_classif(Xmean[tr], y[tr], random_state=0)
var = X[tr].var(axis=(0, 2))
pca = np.abs(PCA(n_components=8).fit(Xmean[tr] - Xmean[tr].mean(0)).components_).sum(0)
order = lambda s: list(np.argsort(-s))
orders = {"MI(sup)": order(mi), "PCA": order(pca), "variance": order(var)}

# ours: unsupervised bottleneck CAE
torch.manual_seed(0)
sub = np.random.default_rng(0).choice(tr, size=min(4000, len(tr)), replace=False)
sel_data = {g: ds.data[g][sub] for g in ds.groups}
m = BottleneckTemporalAutoencoder(ds.channels_per_group, T, latent_dim=10, pool=1, latent_act=False)
train_autoencoder(m, sel_data, epochs=30, lr=1e-3, batch_size=128, device=DEVICE); m = m.to("cpu")
cfg = SelectionConfig(dimension_selection_method="variance", n_important_dimensions=10,
                      n_channels_to_select=30, normalization_method="none", diversity_method="mmr",
                      lambda_diversity=0.4, perturbation_method="percentile", perturbation_magnitudes=[50,70,90])
ours_glob = [g2glob(gc) for gc in run_selection(m, sel_data, cfg).selected]

KS = [5, 10, 15, 20, 30]
print(f"full {C}-channel ceiling = {acc(list(range(C))):.3f}\n", flush=True)
print(f"{'method':10} " + " ".join(f"K={k:<4}" for k in KS))
print("ours       " + " ".join(f"{acc(ours_glob[:k]):.3f}" for k in KS), flush=True)
for name, o in orders.items():
    print(f"{name:10} " + " ".join(f"{acc(o[:k]):.3f}" for k in KS), flush=True)
rnd = [np.mean([acc(list(np.random.default_rng(z).choice(C, k, False))) for z in range(10)]) for k in KS]
print("random     " + " ".join(f"{r:.3f}" for r in rnd), flush=True)
