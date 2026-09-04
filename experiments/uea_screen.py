"""Screen UEA multivariate-TS datasets: ours vs PCA/variance/MI/random, KNN accuracy."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, torch
from aeon.datasets import load_classification
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

def screen(name, ng=8, latent=8, KS=(3,5,8,12)):
    Xtr,ytr = load_classification(name, split="train"); Xte,yte = load_classification(name, split="test")
    X = np.concatenate([Xtr,Xte]).astype(np.float32); yy = np.concatenate([ytr,yte])
    cl = {c:i for i,c in enumerate(sorted(set(yy)))}; y = np.array([cl[c] for c in yy])
    X = np.nan_to_num(X); n,C,T = X.shape
    X = (X - X.mean((0,2),keepdims=True))/(X.std((0,2),keepdims=True)+1e-8)
    mean = X.mean(2); std = X.std(2)
    tr,te = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=y)
    print(f"\n=== {name}: {C} channels, {T} time, {n} cases, {len(cl)} classes ===", flush=True)
    def acc(cols):
        if not len(cols): return float('nan')
        F=lambda r: np.concatenate([mean[r][:,cols],std[r][:,cols]],1)
        sc=StandardScaler().fit(F(tr)); clf=KNeighborsClassifier(5).fit(sc.transform(F(tr)),y[tr])
        return accuracy_score(y[te], clf.predict(sc.transform(F(te))))
    mi=mutual_info_classif(mean[tr],y[tr],random_state=0); var=X[tr].reshape(len(tr),C,-1).var((0,2))
    pca=np.abs(PCA(n_components=min(10,C)).fit(mean[tr]-mean[tr].mean(0)).components_).sum(0)
    orders={"MI(sup)":np.argsort(-mi),"PCA":np.argsort(-pca),"variance":np.argsort(-var)}
    # ours
    dec=max(1,T//300); Xa=np.ascontiguousarray(X[:,:,::dec].transpose(0,2,1))  # (n,Ta,C)
    bnd=np.linspace(0,C,ng+1).astype(int)
    sub=np.random.default_rng(0).choice(tr,size=min(3000,len(tr)),replace=False)
    data={f"g{i}":torch.tensor(Xa[sub][:,:,bnd[i]:bnd[i+1]],dtype=torch.float32) for i in range(ng)}
    ds=GroupedChannelDataset(data,axis_type="temporal1d",labels=torch.tensor(y[sub],dtype=torch.long))
    g2g=lambda gc: bnd[int(gc[0][1:])]+gc[1]
    torch.manual_seed(0)
    m=BottleneckTemporalAutoencoder(ds.channels_per_group,Xa.shape[1],latent_dim=latent,pool=1,latent_act=False)
    train_autoencoder(m,{g:ds.data[g] for g in ds.groups},epochs=30,lr=1e-3,batch_size=128,device=DEVICE); m=m.to("cpu")
    cfg=SelectionConfig(dimension_selection_method="variance",n_important_dimensions=latent,n_channels_to_select=max(KS),
        normalization_method="none",diversity_method="mmr",lambda_diversity=0.4,perturbation_method="percentile",perturbation_magnitudes=[50,70,90])
    ours=[g2g(gc) for gc in run_selection(m,{g:ds.data[g] for g in ds.groups},cfg).selected]
    print(f"full {C}-ch = {acc(list(range(C))):.3f} | " + " ".join(f"K{k}" for k in KS), flush=True)
    print("ours     "+" ".join(f"{acc(ours[:k]):.3f}" for k in KS), flush=True)
    for nm,o in orders.items(): print(f"{nm:8} "+" ".join(f"{acc(list(o[:k])):.3f}" for k in KS), flush=True)
    print("random   "+" ".join(f"{np.mean([acc(list(np.random.default_rng(z).choice(C,k,False))) for z in range(8)]):.3f}" for k in KS), flush=True)

for nm in ["FaceDetection", "MotorImagery"]:
    try: screen(nm)
    except Exception as e: print(f"{nm} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
