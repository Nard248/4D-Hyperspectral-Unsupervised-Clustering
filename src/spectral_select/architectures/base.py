"""Anti-cheat autoencoder ladder — a common interface + shared helpers (see
``docs/spectraforge/03-architecture-plan.md`` and ``04-training-runbook.md``).

Every candidate (C0..C4) is a :class:`BandSelectorModel`:

    fit(spectra) -> self
    select(n_bands) -> [(excitation_nm, emission_nm), ...]
    reconstruction_r() -> float      # mean per-band corr(input, recon); >0 == learned structure
    influence_signal_corr() -> float # corr(per-band influence, per-band signal variance); CAE ~ -0.8

All candidates reuse :mod:`selection_core` for the perturbation -> influence math, so only the
*model + training objective* differ between candidates. The two diagnostics operationalize the
synthetic gate from doc 03/04 (reconstruction R > 0 and influence-vs-variance corr > 0).

Two implementation families:
  * :class:`SpectralSelector` — per-pixel spectral autoencoders (C2/C3/C4). Each pixel's
    concatenated multi-excitation spectrum is one training sample; a single feature "group" of
    D = sum(bands) channels is handed to ``selection_core``.
  * :class:`AnalyzerBackedSelector` — the production *spatial* CAE pipeline (C0/C1), driven through
    the real :class:`spectral_select.Analyzer` via the ``autoencoder_architecture`` seam.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

import selection_core


# --------------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------------
def feature_matrix(spectra):
    """``(n_pixels, D)`` feature matrix over every excitation x emission band, plus a
    ``column -> (excitation_nm, emission_nm)`` map. Mirrors
    ``reports/classification_experiment.feature_matrix`` so KNN columns line up exactly."""
    cols, colmap = [], []
    for ex in spectra.excitation_wavelengths:
        exd = spectra.get_excitation(ex)
        cube = exd.cube.reshape(-1, exd.cube.shape[-1])
        for b, em in enumerate(exd.emission_wavelengths):
            cols.append(cube[:, b])
            colmap.append((float(ex), float(em)))
    return np.column_stack(cols).astype(np.float64), colmap


def diverse_topk(band_influence, colmap, n, *, dedup_nm=10.0, ex_tol=1.0):
    """Pick ``n`` bands by descending influence, skipping any within ``dedup_nm`` of an
    already-picked band at the same excitation (the simple diversity rule from
    ``reports/cae_vs_spectral_ae.py``)."""
    chosen: list[int] = []
    for j in np.argsort(band_influence)[::-1]:
        ex, em = colmap[int(j)]
        if all(not (abs(ex - colmap[c][0]) < ex_tol and abs(em - colmap[c][1]) < dedup_nm)
               for c in chosen):
            chosen.append(int(j))
        if len(chosen) == n:
            break
    return [colmap[c] for c in chosen]


def safe_corr(a, b) -> float:
    """Pearson r that returns 0.0 (not NaN) for a constant input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------------------------------
# the common interface
# --------------------------------------------------------------------------------------------------
class BandSelectorModel(ABC):
    """Common interface for every candidate architecture in the ladder."""

    name = "base"

    @abstractmethod
    def fit(self, spectra) -> "BandSelectorModel":
        ...

    @abstractmethod
    def select(self, n_bands: int) -> list[tuple[float, float]]:
        ...

    @abstractmethod
    def reconstruction_r(self) -> float:
        ...

    @abstractmethod
    def influence_signal_corr(self) -> float:
        ...


# --------------------------------------------------------------------------------------------------
# per-pixel spectral family (C2/C3/C4)
# --------------------------------------------------------------------------------------------------
class SpectralSelector(BandSelectorModel):
    """Per-pixel spectral autoencoder + the shared perturbation-influence selection.

    Subclasses supply the torch module (``_make_module``) and the training objective (``_loss``).
    The perturbation/selection/diagnostics are identical across subclasses so the only scientific
    variable is the architecture + objective.
    """

    name = "spectral-base"
    latent_dim = 8
    epochs = 300
    lr = 1e-3

    # opt-in training tricks (defaults reproduce the original full-batch Adam loop exactly)
    batch_size = None        # None -> full batch
    weight_decay = 0.0       # >0 -> AdamW
    scheduler = None         # None | "cosine"
    patience = None          # None -> no early stopping; else stop after N epochs w/o improvement
    grad_clip = None         # None | float (max grad norm)

    def __init__(self, *, latent_dim=None, epochs=None, lr=None, seed=0, device="cpu",
                 n_important=None, batch_size=None, weight_decay=None, scheduler=None,
                 patience=None, grad_clip=None):
        if latent_dim is not None:
            self.latent_dim = latent_dim
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr
        if batch_size is not None:
            self.batch_size = batch_size
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if scheduler is not None:
            self.scheduler = scheduler
        if patience is not None:
            self.patience = patience
        if grad_clip is not None:
            self.grad_clip = grad_clip
        self.seed = int(seed)
        self.device = device
        self.n_important = n_important  # None -> perturb every latent dim

    # ---- subclass hooks -------------------------------------------------------------------------
    def _make_module(self, d: int) -> torch.nn.Module:
        raise NotImplementedError

    def _loss(self, model: torch.nn.Module, Xt: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def _encode_baseline(self, model: torch.nn.Module, Xt: torch.Tensor) -> torch.Tensor:
        """Latent used as the perturbation baseline (the mean, for a VAE)."""
        return model.encode(Xt)

    # ---- pipeline -------------------------------------------------------------------------------
    def fit(self, spectra) -> "SpectralSelector":
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.seed)
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 1)

        X, self.colmap = feature_matrix(spectra)
        self._signal_var = X.var(axis=0)                       # raw per-band signal variance
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X).astype(np.float32), device=self.device)
        self.d = Xt.shape[1]

        model = self._make_module(self.d).to(self.device)
        self._train(model, Xt, gen)
        model.eval()

        self.model = model
        with torch.no_grad():
            self._latent = self._encode_baseline(model, Xt)
            self._recon = model.decode(self._latent)
            self._Xt = Xt
        self._infl_cache = None
        return self

    def _train(self, model, Xt, gen):
        """Train ``model`` on ``Xt``. With all training knobs at their defaults this is the original
        full-batch Adam loop (byte-identical for C2/C3/C4); set ``batch_size``/``weight_decay``/
        ``scheduler``/``patience`` to enable minibatch SGD + AdamW + cosine LR + early stopping."""
        default = (self.batch_size is None and self.weight_decay == 0.0
                   and self.scheduler is None and self.patience is None and self.grad_clip is None)
        if default:
            opt = torch.optim.Adam(model.parameters(), self.lr)
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = self._loss(model, Xt, gen)
                loss.backward()
                opt.step()
            return

        opt = (torch.optim.AdamW(model.parameters(), self.lr, weight_decay=self.weight_decay)
               if self.weight_decay > 0 else torch.optim.Adam(model.parameters(), self.lr))
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
                 if self.scheduler == "cosine" else None)
        n = Xt.shape[0]
        bs = self.batch_size or n
        best, best_state, bad = float("inf"), None, 0
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen, device=Xt.device)
            running = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = self._loss(model, Xt[idx], gen)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                running += float(loss.item()) * len(idx)
            if sched is not None:
                sched.step()
            if self.patience is not None:
                epoch_loss = running / n
                if epoch_loss < best - 1e-5:
                    best, bad = epoch_loss, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)

    def _band_influence(self) -> np.ndarray:
        if self._infl_cache is not None:
            return self._infl_cache
        latent = self._latent
        n_imp = self.n_important or latent.shape[1]
        important = selection_core.select_important_dimensions(latent, "variance", n_imp)

        def decode_fn(perturbed):
            with torch.no_grad():
                return {0: self.model.decode(perturbed)}

        baseline = {0: self._recon}
        infl = selection_core.accumulate_influence(
            decode_fn, [0], {0: self.d}, latent, baseline, important,
            magnitudes=[100.0], directions=["bidirectional"],
            perturbation_method="standard_deviation",
        )
        out = selection_core.normalize_influence(infl, {0: self._Xt}, "none")[0]
        self._infl_cache = out
        return out

    def select(self, n_bands: int) -> list[tuple[float, float]]:
        return diverse_topk(self._band_influence(), self.colmap, n_bands)

    def reconstruction_r(self) -> float:
        x = self._Xt.cpu().numpy()
        r = self._recon.cpu().numpy()
        return float(np.nanmean([safe_corr(x[:, b], r[:, b]) for b in range(self.d)]))

    def influence_signal_corr(self) -> float:
        return safe_corr(self._band_influence(), self._signal_var)


# --------------------------------------------------------------------------------------------------
# spatial CAE family (C0/C1) — driven through the real Analyzer
# --------------------------------------------------------------------------------------------------
class AnalyzerBackedSelector(BandSelectorModel):
    """Run the production spatial-CAE pipeline via :class:`spectral_select.Analyzer`.

    ``architecture`` is forwarded to ``Config.autoencoder_architecture`` — ``"standard"`` for C0
    (the published CAE, untouched) or a custom ``nn.Module`` class for C1. Selection is the real
    published path (patch baseline -> selection_core -> diversity). The two diagnostics are computed
    on the *full* normalized cube through the same fitted model.
    """

    name = "analyzer-base"
    architecture = "standard"

    def __init__(self, *, n_bands=12, training_epochs=30, n_important=18, seed=0, device="cpu",
                 perturbation_method="percentile", normalization="none"):
        self.n_bands = int(n_bands)
        self.training_epochs = int(training_epochs)
        self.n_important = int(n_important)
        self.seed = int(seed)
        self.device = device
        self.perturbation_method = perturbation_method
        self.normalization = normalization

    def fit(self, spectra) -> "AnalyzerBackedSelector":
        import contextlib
        import os
        import pathlib
        import tempfile

        from spectral_select import Analyzer, Config

        cfg = Config(
            sample_name="candidate",
            n_important_dimensions=self.n_important,
            n_bands_to_select=self.n_bands,
            perturbation_method=self.perturbation_method,
            normalization_method=self.normalization,
            use_diversity_constraint=True,
            training_epochs=self.training_epochs,
            device=self.device,
            random_seed=self.seed,
            autoencoder_architecture=self.architecture,
            output_dir=pathlib.Path(tempfile.mkdtemp()),
        )
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            self.analyzer = Analyzer(cfg)
            self.analyzer.fit(spectra)
        self._spectra = spectra
        self._diag = None
        return self

    def _raw_signal_var(self):
        """Per-(excitation, band) RAW signal variance over pixels — the same 'signal variance'
        the per-pixel spectral family correlates against, so the corr metric is comparable."""
        model = self.analyzer._model
        out = {}
        for ex in model.excitation_wavelengths:
            cube = self._spectra.get_excitation(ex).cube
            out[ex] = cube.reshape(-1, cube.shape[-1]).astype(np.float64).var(axis=0)
        return out

    def select(self, n_bands: int) -> list[tuple[float, float]]:
        return [(float(b.excitation_nm), float(b.emission_nm))
                for b in self.analyzer.get_wavelengths()]

    @staticmethod
    def _tile_batch(cube_hwb: np.ndarray, t: int) -> np.ndarray:
        """Split an ``(H, W, bands)`` cube into a batch of ``t x t`` tiles (drops the remainder).
        Needed so the diagnostic latent has batch > 1 (``select_important_dimensions`` takes a
        variance over the batch — a single sample would give NaN)."""
        h, w, _ = cube_hwb.shape
        tiles = [cube_hwb[i * t:(i + 1) * t, j * t:(j + 1) * t, :]
                 for i in range(h // t) for j in range(w // t)]
        return np.stack(tiles)

    def _diagnostics(self):
        if self._diag is not None:
            return self._diag
        a = self.analyzer
        model = a._model
        data = a._dataset.get_all_data()                       # {ex: (H, W, bands) normalized}
        h, w = next(iter(data.values())).shape[:2]
        t = min(16, h, w)
        if (h // t) * (w // t) < 2:                            # guarantee batch >= 2
            t = max(2, min(h, w) // 2)
        input_dict = {ex: torch.tensor(self._tile_batch(d.cpu().numpy(), t).astype(np.float32)).to(a._device)
                      for ex, d in data.items()}
        with torch.no_grad():
            latent = model.encode(input_dict)
            recon = model.decode(latent)

        # Prefer the Analyzer's REAL selection-driving influence (computed over the patch baseline —
        # this is the quantity doc 02 found anti-correlated with signal). Fall back to a tiled
        # recompute only if it is unavailable.
        infl_real = getattr(a, "_influence_matrix", None)
        if not (isinstance(infl_real, dict) and all(ex in infl_real for ex in model.excitation_wavelengths)):
            important = selection_core.select_important_dimensions(
                latent, a._config.dimension_selection_method, a._config.n_important_dimensions)
            cpg = {ex: model.emission_bands[ex] for ex in model.excitation_wavelengths}
            infl_real = selection_core.accumulate_influence(
                model.decode, list(model.excitation_wavelengths), cpg, latent, recon, important,
                magnitudes=a._config.perturbation_magnitudes,
                directions=a._config.perturbation_directions,
                perturbation_method=a._config.perturbation_method,
            )

        raw_var = self._raw_signal_var()
        rs: list[float] = []
        infl_all: list[np.ndarray] = []
        var_all: list[np.ndarray] = []
        for ex in model.excitation_wavelengths:
            x = input_dict[ex].cpu().numpy().reshape(-1, input_dict[ex].shape[-1])
            r = recon[ex].cpu().numpy().reshape(-1, recon[ex].shape[-1])
            for b in range(x.shape[1]):
                rs.append(safe_corr(x[:, b], r[:, b]))
            infl_all.append(np.asarray(infl_real[ex], dtype=float))
            var_all.append(raw_var[ex])                          # RAW signal variance (comparable to C2..C4)
        self._diag = (
            float(np.nanmean(rs)),
            safe_corr(np.concatenate(infl_all), np.concatenate(var_all)),
        )
        return self._diag

    def reconstruction_r(self) -> float:
        return self._diagnostics()[0]

    def influence_signal_corr(self) -> float:
        return self._diagnostics()[1]
