"""Tennessee Eastman Process (52 heterogeneous sensors, 21 fault classes): does our
UNSUPERVISED CAE track supervised MI / beat PCA & random? KNN accuracy on selected sensors."""
import warnings; warnings.filterwarnings("ignore")
import urllib.request, zipfile, io, numpy as np, pandas as pd, torch
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
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
    "https://industrial-makarov.obs.ru-moscow-1.hc.sbercloud.ru/small_tep.zip", timeout=120).read()))
df = pd.read_csv(z.open("dataset.csv")); lab = pd.read_csv(z.open("labels.csv"))
feat = [c for c in df.columns if c.startswith(("xmeas_", "xmv_"))]
C = len(feat)
X = df[feat].to_numpy(np.float32); y_s = lab["labels"].to_numpy(int); runs = df["run_id"].to_numpy()
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
# window per run (W=20, stride 10); label = majority
W, S = 20, 10
Xw, yw = [], []
for r in np.unique(runs):
    idx = np.where(runs == r)[0]
    for s in range(0, len(idx) - W + 1, S):
        w = idx[s:s+W]; Xw.append(X[w]); yw.append(np.bincount(y_s[w]).argmax())
Xw = np.stack(Xw); yw = np.array(yw)                     # (n, W, 52)
print(f"TEP {Xw.shape} windows, classes={len(set(yw.tolist()))}", flush=True)
mean = Xw.mean(1); std = Xw.std(1)
tr, te = train_test_split(np.arange(len(yw)), test_size=0.4, random_state=0, stratify=yw)
def acc(cols):
    if len(cols) == 0: return float("nan")
    F = lambda r: np.concatenate([mean[r][:, cols], std[r][:, cols]], 1)
    sc = StandardScaler().fit(F(tr)); clf = KNeighborsClassifier(7).fit(sc.transform(F(tr)), yw[tr])
    return accuracy_score(yw[te], clf.predict(sc.transform(F(te))))

mi = mutual_info_classif(mean[tr], yw[tr], random_state=0); var = Xw[tr].reshape(len(tr), -1, C).var((0, 1))
pca = np.abs(PCA(n_components=10).fit(mean[tr]).components_).sum(0)
orders = {"MI(sup)": np.argsort(-mi), "PCA": np.argsort(-pca), "variance": np.argsort(-var)}

# ours: 3 functional groups (continuous meas / composition / manipulated)
gb = [(0, 22), (22, 41), (41, 52)]
sub = np.random.default_rng(0).choice(tr, size=min(4000, len(tr)), replace=False)
data = {f"g{i}": torch.tensor(Xw[sub][:, :, a:b], dtype=torch.float32) for i, (a, b) in enumerate(gb)}
ds = GroupedChannelDataset(data, axis_type="temporal1d", labels=torch.tensor(yw[sub], dtype=torch.long))
g2glob = lambda gc: gb[int(gc[0][1:])][0] + gc[1]
torch.manual_seed(0)
m = BottleneckTemporalAutoencoder(ds.channels_per_group, W, latent_dim=8, pool=1, latent_act=False)
train_autoencoder(m, {g: ds.data[g] for g in ds.groups}, epochs=30, lr=1e-3, batch_size=128, device=DEVICE)
m = m.to("cpu")
cfg = SelectionConfig(dimension_selection_method="variance", n_important_dimensions=8,
                      n_channels_to_select=15, normalization_method="none", diversity_method="mmr",
                      lambda_diversity=0.4, perturbation_method="percentile", perturbation_magnitudes=[50,70,90])
ours = [g2glob(gc) for gc in run_selection(m, {g: ds.data[g] for g in ds.groups}, cfg).selected]

KS = [3, 5, 8, 12, 15]
print(f"full {C}-sensor ceiling = {acc(list(range(C))):.3f}\n", flush=True)
print(f"{'method':10} " + " ".join(f"K={k:<4}" for k in KS))
print("ours       " + " ".join(f"{acc(ours[:k]):.3f}" for k in KS), flush=True)
for nm, o in orders.items():
    print(f"{nm:10} " + " ".join(f"{acc(list(o[:k])):.3f}" for k in KS), flush=True)
rnd = [np.mean([acc(list(np.random.default_rng(zz).choice(C, k, False))) for zz in range(10)]) for k in KS]
print("random     " + " ".join(f"{r:.3f}" for r in rnd), flush=True)
