"""Load the processed ME-HSI cubes (Lichens, Collagen) and save small arrays for the poster's
dataset-explainer figures: downsampled band images per excitation, per-class mean EEMs,
class masks. Run from the repo root."""
import pickle, json
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path("publications/codassca2026/poster/data")
SETS = {
    "lichens": ("Data/processed/Lichens Dataset 1", 3,
                {(255, 0, 0): 1, (0, 0, 255): 2, (0, 200, 0): 3, (255, 165, 0): 4}),
    "collagen": ("Data/processed/Collagen Pepsin", 1,
                 {(255, 0, 0): 1, (0, 0, 255): 2, (0, 200, 0): 3}),
}
for name, (folder, ds_step, colors) in SETS.items():
    folder = Path(folder)
    with open(folder / "spectra_masked.pkl", "rb") as f:
        d = pickle.load(f)
    exs = [float(e) for e in d["excitation_wavelengths"]]
    mask_rgb = np.array(Image.open(folder / "class_mask.png").convert("RGB"))
    cls = np.zeros(mask_rgb.shape[:2], dtype=np.int16)
    for rgb, cid in colors.items():
        cls[np.all(mask_rgb == np.array(rgb), axis=-1)] = cid
    arrays = {"excitations": np.array(exs), "class_mask": cls, "valid_mask": np.asarray(d["mask"]),
              "ds_step": np.array(ds_step)}
    all_em = sorted({float(w) for e in exs for w in d["data"][str(e)]["wavelengths"]})
    arrays["emissions_union"] = np.array(all_em)
    # per-class mean EEM on the union emission grid (NaN where an excitation lacks that emission)
    n_cls = len(colors)
    eem = np.full((n_cls + 1, len(exs), len(all_em)), np.nan)
    for i, e in enumerate(exs):
        cube = d["data"][str(e)]["cube"]; wl = [float(w) for w in d["data"][str(e)]["wavelengths"]]
        arrays[f"cube_{int(e)}"] = cube[::ds_step, ::ds_step, :].astype(np.float32)
        arrays[f"wl_{int(e)}"] = np.array(wl)
        for cid in range(1, n_cls + 1):
            m = cls == cid
            if m.sum() == 0: continue
            spec = cube[m].mean(axis=0)
            for j, w in enumerate(wl):
                eem[cid, i, all_em.index(w)] = spec[j]
        valid = np.asarray(d["mask"]) > 0
        spec = cube[valid].mean(axis=0)
        for j, w in enumerate(wl):
            eem[0, i, all_em.index(w)] = spec[j]
    arrays["class_eem"] = eem
    total = sum(len(d["data"][str(e)]["wavelengths"]) for e in exs)
    print(name, "excitations", exs, "total bands", total, "H,W", cls.shape,
          "class px", {c: int((cls == c).sum()) for c in range(1, n_cls + 1)},
          "exposure", d.get("exposure_times"))
    np.savez_compressed(OUT / f"hsi_{name}.npz", **arrays)
    print("wrote", OUT / f"hsi_{name}.npz")
