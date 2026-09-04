"""Standard HSI band-selection screen (Indian Pines, 200 bands): does our UNSUPERVISED
Conv2D-bottleneck CAE track supervised MI / PCA and beat random? Pixel classification."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from huggingface_hub import hf_hub_download
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from channel_select.data import GroupedChannelDataset
from channel_select.models.spatial_bottleneck import SpatialBottleneckAutoencoder
from channel_select.models.training import train_autoencoder
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
REPO = "danaroth/indian_pines"
cube = loadmat(hf_hub_download(REPO, "Indian_pines_corrected.mat", repo_type="dataset"))["indian_pines_corrected"].astype(np.float32)
gt = loadmat(hf_hub_download(REPO, "Indian_pines_gt.mat", repo_type="dataset"))["indian_pines_gt"].astype(int)
H, W, B = cube.shape
cube = (cube - cube.reshape(-1, B).mean(0)) / (cube.reshape(-1, B).std(0) + 1e-8)
print(f"Indian Pines {cube.shape}, classes={len(set(gt[gt>0].tolist()))}", flush=True)

k = 7; pad = k // 2
cubep = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
ys, xs = np.where(gt > 0)
y = gt[ys, xs]
spectra = cube[ys, xs]                                   # (n, B) per-pixel for baselines/downstream
tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=0, stratify=y)

def acc(cols):
    if len(cols) == 0: return float("nan")
    sc = StandardScaler().fit(spectra[tr][:, cols])
    clf = KNeighborsClassifier(5).fit(sc.transform(spectra[tr][:, cols]), y[tr])
    return accuracy_score(y[te], clf.predict(sc.transform(spectra[te][:, cols])))

# baselines (train pixels)
mi = mutual_info_classif(spectra[tr], y[tr], random_state=0)
var = spectra[tr].var(0)
pca = np.abs(PCA(n_components=10).fit(spectra[tr]).components_).sum(0)
orders = {"MI(sup)": np.argsort(-mi), "PCA": np.argsort(-pca), "variance": np.argsort(-var)}

# ours: unsupervised Conv2D-bottleneck CAE on patches of a train subsample
sub = np.random.default_rng(0).choice(tr, size=min(3000, len(tr)), replace=False)
patches = np.stack([cubep[ys[i]:ys[i]+k, xs[i]:xs[i]+k, :] for i in sub])   # (n,k,k,B)
NG = 5; bnd = np.linspace(0, B, NG + 1).astype(int)
data = {f"g{i}": torch.tensor(patches[:, :, :, bnd[i]:bnd[i+1]], dtype=torch.float32) for i in range(NG)}
ds = GroupedChannelDataset(data, axis_type="spatial2d", labels=torch.tensor(y[sub], dtype=torch.long))
g2glob = lambda gc: bnd[int(gc[0][1:])] + gc[1]
torch.manual_seed(0)
m = SpatialBottleneckAutoencoder(ds.channels_per_group, (k, k), latent_dim=10, latent_act=False)
train_autoencoder(m, {g: ds.data[g] for g in ds.groups}, epochs=30, lr=1e-3, batch_size=128, device=DEVICE)
m = m.to("cpu")
cfg = SelectionConfig(dimension_selection_method="variance", n_important_dimensions=10,
                      n_channels_to_select=30, normalization_method="none", diversity_method="mmr",
                      lambda_diversity=0.4, perturbation_method="percentile", perturbation_magnitudes=[50,70,90])
ours = [g2glob(gc) for gc in run_selection(m, {g: ds.data[g] for g in ds.groups}, cfg).selected]

KS = [5, 10, 15, 20, 30]
print(f"full {B}-band ceiling = {acc(list(range(B))):.3f}\n", flush=True)
print(f"{'method':10} " + " ".join(f"K={kk:<4}" for kk in KS))
print("ours       " + " ".join(f"{acc(ours[:kk]):.3f}" for kk in KS), flush=True)
for nm, o in orders.items():
    print(f"{nm:10} " + " ".join(f"{acc(list(o[:kk])):.3f}" for kk in KS), flush=True)
rnd = [np.mean([acc(list(np.random.default_rng(z).choice(B, kk, False))) for z in range(10)]) for kk in KS]
print("random     " + " ".join(f"{r:.3f}" for r in rnd), flush=True)
