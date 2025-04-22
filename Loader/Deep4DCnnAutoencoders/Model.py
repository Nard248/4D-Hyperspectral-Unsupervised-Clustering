# hyperspectral_4d_pipeline.py  — v2
"""One‑stop unsupervised clustering of 4‑D excitation–emission hyperspectral cubes
================================================================================

Usage in a Jupyter notebook
---------------------------
```python
from hyperspectral_4d_pipeline import HyperspectralClusterer

cube = my_custom_reader("sample.pkl")          # (H,W,N_em,N_ex)
model = HyperspectralClusterer(pixelwise=True,  # or False for patch mode
                               n_clusters=12,
                               patch=12,
                               stride=6,
                               d_spec=64,
                               z_dim=128)
model.fit(cube, epochs=300)
labels = model.predict(cube)                    # (H,W) integer map
model.save("minerva_pixwise.h5")               # optional
```
The only function you still have to supply is **your own data reader**; the
pipeline begins once you feed a `float32` 4‑D array `(H,W,N_em,N_ex)`.

Key design points
-----------------
* **pixelwise vs patch mode** is toggled by the `pixelwise` flag.
* All hyper‑parameters (`patch`, `stride`, `d_spec`, `z_dim` …) are kwargs.
* Internally the class builds three sub‑nets:
  1. **EEMEncoder φ**  – per‑pixel spectral CNN (em × ex → d).
  2. **SpatialEncoder ψ** – either fully‑convolutional (pixelwise) or
     patch‑wise trunk.
  3. **StudentCluster**  – DEC soft clustering head.
* Training follows the two‑stage DEC schedule: CAE pre‑train ➔ joint fine‑tune.
* Everything is pure `tf.keras`; no external MATLAB is needed.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks, optimizers
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
#                         LOW‑LEVEL BUILDING BLOCKS
# ---------------------------------------------------------------------------

def build_eem_encoder(d_spec: int, n_em: int, n_ex: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_em, n_ex, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(d_spec, activation=None)(x)
    return models.Model(inp, out, name="EEMEncoder")

class StudentCluster(layers.Layer):
    def __init__(self, n_clusters: int, alpha: float = 1., **kw):
        super().__init__(**kw)
        self.n_clusters, self.alpha = n_clusters, alpha
        self.clusters = None

    def build(self, in_shape):
        # before (buggy):
        # self.clusters = self.add_weight("centroids", shape=(self.n_clusters, in_shape[-1]),
        #                                 initializer="glorot_uniform", trainable=True)

        # after (explicit keywords):
        self.clusters = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, in_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
    def call(self, z):
        q = 1. / (1. + tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - self.clusters), -1)/self.alpha)
        q **= (self.alpha + 1.)/2.
        q /= tf.reduce_sum(q, -1, keepdims=True)
        return q

def target_distribution(q):
    w = q**2 / tf.reduce_sum(q, 0)
    return tf.transpose(tf.transpose(w) / tf.reduce_sum(w, 1))

# ---------------------------------------------------------------------------
#                            MAIN USER API CLASS
# ---------------------------------------------------------------------------
class HyperspectralClusterer:
    """Unsupervised DEC clustering on 4‑D excitation–emission cubes.

    Parameters
    ----------
    pixelwise : bool
        *True* → keep full H×W resolution (no patching, stride‑1 convolutions).
        *False* → use patch encoder with `patch` / `stride`.
    n_clusters : int
    patch, stride : int
        Only used when `pixelwise=False`.
    d_spec : int
        Spectral embedding dimension.
    z_dim : int
        Spatial latent dimension before clustering.
    n_em, n_ex : int
        Spectral sizes.  If *None* we infer them at runtime from the first cube.
    """

    def __init__(self,
                 pixelwise: bool = True,
                 n_clusters: int = 10,
                 patch: int = 10,
                 stride: int = 5,
                 d_spec: int = 32,
                 z_dim:  int = 128,
                 n_em: Optional[int] = None,
                 n_ex: Optional[int] = None,
                 lr: float = 1e-3):
        self.pixelwise, self.patch, self.stride = pixelwise, patch, stride
        self.n_clusters, self.d_spec, self.z_dim = n_clusters, d_spec, z_dim
        self.n_em, self.n_ex = n_em, n_ex
        self.lr = lr
        self._built = False

    # -------------------------- public API -----------------------------
    def fit(self, cube: np.ndarray, epochs: int = 200):
        cube = cube.astype('float32')
        if self.n_em is None:
            self.n_em, self.n_ex = cube.shape[2:4]
        self._build_models()

        # Spectral pass --------------------------------------------------
        flat = cube.reshape(-1, self.n_em, self.n_ex, 1)
        spec_feat = self.eem_enc.predict(flat, batch_size=1024, verbose=0)
        spec_map  = spec_feat.reshape(cube.shape[0], cube.shape[1], self.d_spec)

        if self.pixelwise:
            # fully‑conv route — training on whole image may be VRAM heavy; crop
            lat_map = self.spatial_enc.predict(spec_map[np.newaxis,...], verbose=0)[0]
            z_flat  = lat_map.reshape(-1, self.z_dim)
            self._init_centroids(z_flat)
            self._train_pixelwise(spec_map, epochs)
        else:
            patches, idx = self._make_patches(spec_map)
            z_init = self.spatial_enc.predict(patches, batch_size=256, verbose=0)
            self._init_centroids(z_init)
            self._train_patchwise(patches, epochs)
            self._patch_idx = idx  # store for predict()

        self._built = True
        return self

    def predict(self, cube: np.ndarray) -> np.ndarray:
        assert self._built, "Call fit() before predict()"
        cube = cube.astype('float32')
        flat = cube.reshape(-1, self.n_em, self.n_ex, 1)
        spec_feat = self.eem_enc.predict(flat, batch_size=1024, verbose=0)
        spec_map  = spec_feat.reshape(cube.shape[0], cube.shape[1], self.d_spec)

        if self.pixelwise:
            lat_map = self.spatial_enc.predict(spec_map[np.newaxis,...], verbose=0)[0]
            q = self.cluster_head.predict(lat_map.reshape(-1, self.z_dim), batch_size=1024, verbose=0)
            labels = q.argmax(-1).reshape(cube.shape[0], cube.shape[1])
            return labels
        else:
            patches, idx = self._make_patches(spec_map)
            q = self.cluster_head.predict(self.spatial_enc.predict(patches, verbose=0), verbose=0)
            lab = q.argmax(-1)
            return self._reconstruct_label_map(lab, idx, cube.shape[:2])

    def save(self, path: str | Path):
        self.model.save(path)

    def load(self, path: str | Path):
        self.model = tf.keras.models.load_model(path, custom_objects={'StudentCluster': StudentCluster})
        self.eem_enc, self.spatial_enc = self.model.get_layer('EEMEncoder'), self.model.get_layer('SpatialEncoder')
        self.cluster_head = self.model.get_layer('student_cluster')
        self._built = True
        return self

    # ------------------------ internal helpers -------------------------
    def _build_models(self):
        # φ
        self.eem_enc = build_eem_encoder(self.d_spec, self.n_em, self.n_ex)

        # ψ – choose architecture
        if self.pixelwise:
            self.spatial_enc = self._build_fullconv_spatial()
        else:
            self.spatial_enc = self._build_patch_spatial()

        self.cluster_head = StudentCluster(self.n_clusters, name='student_cluster')

        # dummy forward to build variables --------------------------------
        if self.pixelwise:
            dummy = tf.zeros((1, 32, 32, self.d_spec))
            _ = self.spatial_enc(dummy)
        else:
            dummy = tf.zeros((1, self.patch, self.patch, self.d_spec))
            _ = self.spatial_enc(dummy)

        self.model = None  # not used – we train parts manually in pixel mode

    def _build_fullconv_spatial(self):
        inp = layers.Input(shape=(None, None, self.d_spec))
        x = inp
        for i in range(4):
            f = 64 * 2**i
            x = layers.Conv2D(f, 3, padding="same", dilation_rate=2**i, activation="relu")(x)
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        out = layers.Conv2D(self.z_dim, 1, padding="same", activation=None)(x)
        return models.Model(inp, out, name="SpatialEncoder")

    def _build_patch_spatial(self):
        inp = layers.Input(shape=(self.patch, self.patch, self.d_spec))
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        out = layers.Dense(self.z_dim, activation=None)(x)
        return models.Model(inp, out, name="SpatialEncoder")

    # ---------------- patch utilities -------------------------
    def _make_patches(self, feat_map: np.ndarray):
        H, W, C = feat_map.shape
        patches, idx = [], []
        for y in range(0, H - self.patch + 1, self.stride):
            for x in range(0, W - self.patch + 1, self.stride):
                patches.append(feat_map[y:y+self.patch, x:x+self.patch, :])
                idx.append((y, x))
        return np.stack(patches).astype('float32'), np.asarray(idx, dtype=np.int32)

    def _reconstruct_label_map(self, labels: np.ndarray, idx: np.ndarray, canvas: Tuple[int,int]):
        H, W = canvas
        out = -np.ones((H, W), np.int32); cnt = np.zeros_like(out)
        for lab, (y,x) in zip(labels, idx):
            out[y:y+self.patch, x:x+self.patch] += lab + 1
            cnt[y:y+self.patch, x:x+self.patch] += 1
        return out // np.maximum(cnt, 1)

    # -------------- clustering initialisation & training ---------------
    def _init_centroids(self, z: np.ndarray):
        """Run K‑means and copy centroids into the StudentCluster layer."""
        if not self.cluster_head.built:
            self.cluster_head.build((None, z.shape[-1]))  # Make sure weights are ready

        km = KMeans(self.n_clusters, n_init=20).fit(z)

        # Debug: check K-means centroids
        print(f"Initialized centroids: {km.cluster_centers_[:5]}")  # Display first 5 centroids

        self.cluster_head.set_weights([km.cluster_centers_])

    def _train_pixelwise(self, spec_map: np.ndarray, epochs: int):
        opt = optimizers.Adam(self.lr)
        mse = losses.MeanSquaredError()

        # ---- 1. one forward pass for K‑means init ------------------------
        latent_map = self.spatial_enc(spec_map[np.newaxis, ...])[0]  # Tensor
        z_flat = tf.reshape(latent_map, (-1, self.z_dim)).numpy()  # NumPy
        self._init_centroids(z_flat)  # <-- initialize centroids after first forward pass

        # ---- 2. DEC training loop ---------------------------------------
        for ep in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
                latent_map = self.spatial_enc(spec_map[np.newaxis, ...])[0]  # (1,H,W,z)
                z = tf.reshape(latent_map, (-1, self.z_dim))  # (H·W, z)
                q = self.cluster_head(z)  # (H·W, K)

                p = target_distribution(q)  # (H·W, K)
                kl = tf.keras.losses.kld(p, q)
                kl = tf.reduce_mean(kl)  # Mean KL

                # Debugging outputs
                if ep % 20 == 0:
                    print(f"ep={ep:03d}, q_mean={tf.reduce_mean(q):.4f}, p_mean={tf.reduce_mean(p):.4f}, KL={kl:.4f}")
                    # Check if the gradients are being computed
                    print("Gradients:", tape.gradient(kl, self.spatial_enc.trainable_weights))
                    print("Gradients:", tape.gradient(kl, self.cluster_head.trainable_weights))

                rec = 0.0  # Placeholder for reconstruction loss (if needed)
                loss = kl + rec

            grads = tape.gradient(loss, self.spatial_enc.trainable_weights + self.cluster_head.trainable_weights)
            opt.apply_gradients(zip(grads, self.spatial_enc.trainable_weights + self.cluster_head.trainable_weights))

            if ep % 20 == 0:
                print(f"ep={ep:03d}, KL={kl:.4f}")

    def _train_patchwise(self, patches: np.ndarray, epochs: int):
        # build mini DEC model for patches
        inp = layers.Input(shape=(self.patch, self.patch, self.d_spec))
        z   = self.spatial_enc(inp)
        q   = self.cluster_head(z)
        dec = models.Model(inp, q)
        dec.compile(optimizers.Adam(self.lr), loss='kld')

        # calculate p inside custom callback --------------------------------
        class DECUpdater(callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                q = dec.predict(patches, batch_size=256, verbose=0)
                p = target_distribution(q)
                dec.fit(patches, p, epochs=1, batch_size=256, verbose=0)
        dec.fit(patches, np.zeros((len(patches), self.n_clusters)),
                epochs=epochs, batch_size=256, callbacks=[DECUpdater()], verbose=0)

# ---------------------------------------------------------------------------
#                                END OF FILE
# ---------------------------------------------------------------------------
