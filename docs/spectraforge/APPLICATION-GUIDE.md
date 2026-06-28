# SpectraForge Synthetic Data Generation — Application Guide

A complete, step-by-step guide to how the SpectraForge generator builds synthetic multi-excitation
hyperspectral imaging (ME-HSI) cubes: the physics, the parameters, the API, and recipes for every
regime. Everything here is reproducible from the project venv.

---

## 1. Mental model

A SpectraForge dataset is a stack of **excitation–emission images**. For each chosen **excitation
wavelength** λ_ex, you get an image cube of shape `(H, W, n_emission_bands)`. Stacking all excitations
gives, per pixel, a full **excitation–emission matrix (EEM)** flattened to a feature vector of length
`n_excitations × n_emission_bands`. A per-pixel **class label** is assigned from the dominant
discriminative fluorophore.

The generator is **physically additive and linear by default** — `render(A+B) = render(A)+render(B)` —
and becomes nonlinear only when you switch on inner-filter, reabsorption, or FRET. Realism (and
difficulty) is added through nuisances, scatter, noise, and fixed-pattern clutter.

Pipeline: **fluorophores → scene (concentration maps + labels) → forward render (per excitation) →
optional physics → artifacts (scatter + noise) → optional clutter → feature matrix.**

---

## 2. The fluorophore model (`spectraforge/fluorophore.py`)

Each fluorophore is a frozen dataclass with Gaussian bands (dilute-regime model). Its dilute contribution
to a pixel is:

```
contribution(λ_ex, λ_em) = extinction · quantum_yield · excitation(λ_ex) · emission(λ_em) · concentration
```

Parameters:

| field | meaning |
|-------|---------|
| `ex_peak_nm`, `ex_fwhm_nm` | excitation (≈ absorption) Gaussian peak and full-width-half-max |
| `em_peak_nm`, `em_fwhm_nm` | emission Gaussian peak and FWHM |
| `quantum_yield` (Φ) | radiative efficiency (Vavilov: ≈ excitation-independent) |
| `extinction` (ε) | relative absorption strength |
| `em_skew` | 0 = symmetric; >0 = red-tailed emission (real asymmetry) |
| `vibronic` | >0 adds a Franck–Condon shoulder ~1300 cm⁻¹ to the red (characteristic shape) |

- `excitation(wl)` returns a peak-normalised Gaussian (relative absorption probability).
- `emission(wl)` returns an **area-normalised** (split-)Gaussian; with `em_skew`/`vibronic` it becomes a
  red-tailed shape plus an optional vibronic shoulder — encoding **Kasha** (shape independent of
  excitation) and **Franck–Condon** (vibronic progression).

Example:

```python
from spectraforge.fluorophore import Fluorophore
D1 = Fluorophore("D1", ex_peak_nm=470, ex_fwhm_nm=35, em_peak_nm=515, em_fwhm_nm=45,
                 extinction=0.5, quantum_yield=0.45, em_skew=0.4)
```

---

## 3. The scene (`spectraforge/scenegen.py`, `scene.py`)

A **Scene** holds a per-fluorophore concentration map `(H, W)`. Helpers:

- `random_field(H, W, seed, blur=2)` — a smooth random concentration field in [0,1] (organic gradients).
- `make_confounded_scene(disc_materials, nuisance_materials, H, W, seed, *, disc_amp, nuisance_amp,
  turbidity_amp)` — **the workhorse.** Discriminative materials get smooth fields whose per-pixel
  **argmax defines the class label**; nuisance materials get independent fields (bright, class-irrelevant).
  Returns `(scene, labels, scatter_field)`. This is what decouples **variance from informativeness**.
- `make_interaction_scene(donor, acceptor, nuisances, H, W, seed, ...)` — class = **XOR** of two dyes'
  median-binarised co-localisation (each dye's marginal is class-independent; the information is purely
  in the joint). Used with FRET for nonlinear regimes.

---

## 4. The forward render (`spectraforge/forward.py::render`)

For each excitation λ_ex:

1. Compute a scale = lamp · exposure · power for that excitation.
2. For each fluorophore: `amp = extinction · quantum_yield · excitation(λ_ex)`; accumulate
   `concentration · amp · emission(λ_em)` into the cube; accumulate excitation absorbance for inner-filter.
3. If `physics` is set: apply primary inner-filter and/or reabsorption (Section 6).
4. If `artifacts` is set: add Rayleigh + Raman scatter (scaled by the turbidity field) and photon + read
   noise (Section 5).
5. If `fret_pairs` is set: apply donor→acceptor transfer before summing per-fluorophore contributions.

Result: a `SpectraData` object (`.excitation_wavelengths`, `.get_excitation(ex).cube`,
`.emission_wavelengths`) plus a `GroundTruth` (concentration maps, clean cubes, per-fluorophore spectra).

---

## 5. Artifacts: scatter and noise (`spectraforge/artifacts.py`)

`ArtifactConfig(rayleigh_strength, raman_strength, second_order, photon_scale, read_sigma)`:

- **Rayleigh** elastic scatter at λ_em = λ_ex (and 2λ_ex if `second_order`), scaled per pixel by the
  turbidity field — high-variance, zero chemical information.
- **Raman** inelastic scatter at a fixed energy offset.
- **Photon (shot) noise** via `photon_scale` (higher = less noise); **read noise** via Gaussian
  `read_sigma`.

---

## 6. Optional nonlinear physics (`spectraforge/physics.py`)

`PhysicsConfig(psf_sigma_px, inner_filter, inner_filter_strength, reabsorption, reabsorption_strength,
autofluorescence, ...)`:

- **`psf_sigma_px`** — optical point-spread blur (per band).
- **`inner_filter`** — primary Beer–Lambert: signal × `exp(-strength · A_ex)` (per-pixel scalar; first
  nonlinearity).
- **`reabsorption`** — secondary inner-filter / self-absorption: emission × `exp(-strength · A_em(λ))`
  where `A_em(λ)=Σ_k ε_k c_k · absorption_k(λ)`. Because absorption overlaps the blue edge of emission,
  this **reshapes the band** (suppresses blue edge, apparent red-shift) as a function of concentration —
  a nonlinearity that relocates information into band *shape*.
- **`apply_fret(contribs, absorbed, conc, library, em, pairs)`** — for each `(donor, acceptor, k)`,
  transfer fraction `E = k·c_A/(1+k·c_A)` of donor energy: donor quenched by `(1-E)`, acceptor sensitised
  by `E · absorbed_donor · Φ_A`. Encodes dye co-localisation in the band ratio (saturating, product).

---

## 7. Fixed-pattern clutter (`reports/realistic_benchmark.py::add_cube_clutter`)

The dominant real-instrument confound. `add_cube_clutter(spectra, size, seed, n_modes, amp)` injects
`n_modes` independent **class-irrelevant** smooth spatial fields, each imprinted on a random band subset
with a random per-band gain (amplitude `amp · cube_std`). Clutter is high-variance but carries no class
information — which is exactly why unsupervised, variance/reconstruction-based selection collapses to
chance under it. (Modelled as the analogue of illumination/detector fixed-pattern noise.)

---

## 8. The one-call API: `build_dataset` (`reports/realistic_benchmark.py`)

```python
from realistic_benchmark import build_dataset
spectra, ground_truth, labels, acquisition = build_dataset(
    seed=1, size=64, em_step=2,                 # 64x64 scene, emission 420-700 nm @ 2 nm = 141 bands
    nuisance_amp=1.2, turbidity_amp=0.6,        # nuisance brightness, turbidity
    rayleigh=0.35, raman=0.30,                  # scatter strengths
    photon_scale=2000, read_sigma=0.005,        # noise
    reabsorption=False, reabsorption_strength=2.5,  # nonlinear self-absorption
    disc_extinction=None,                        # override discriminative dye brightness
    clutter_modes=0, clutter_amp=0.0)            # fixed-pattern clutter
```

| parameter | default | effect |
|-----------|---------|--------|
| `seed` | — | reproducibility (scene + noise) |
| `size` | 64 | scene H=W |
| `em_step` | 5 | emission sampling (nm); 2 → 141 bands, 5 → 57 bands (× 4 excitations) |
| `nuisance_amp` | 2.0 | brightness of class-irrelevant nuisance dyes |
| `turbidity_amp` | 1.0 | spatially-varying scatter strength |
| `rayleigh`, `raman` | 0.5, 0.4 | scatter ridge strengths |
| `photon_scale` | 600 | shot-noise (higher = cleaner) |
| `read_sigma` | 0.005 | Gaussian read noise |
| `reabsorption[_strength]` | False / 2.5 | nonlinear band reshaping |
| `disc_extinction` | None | override discriminative dye extinction (signal strength) |
| `clutter_modes`, `clutter_amp` | 0, 0.0 | fixed-pattern clutter (count, amplitude) |

Fixed in the call: excitations {405, 470, 488, 506} nm, emission 420–700 nm. To get the feature matrix:

```python
from classification_experiment import feature_matrix
X, colmap = feature_matrix(spectra)   # X: (n_pixels, n_features); colmap[j] = (excitation_nm, emission_nm)
y = labels.ravel()
```

---

## 9. Regime recipes

| regime | call |
|--------|------|
| **clean** | `build_dataset(seed, nuisance_amp=0.4, turbidity_amp=0.1, rayleigh=0.1, raman=0.1, photon_scale=50000, read_sigma=0.002)` |
| **realistic** | `build_dataset(seed)` (defaults) |
| **clutter (L4)** | `build_dataset(seed, nuisance_amp=2.0, photon_scale=600, read_sigma=0.009, clutter_modes=36, clutter_amp=3.3)` |
| **reabsorption (nonlinear)** | `build_dataset(seed, reabsorption=True, reabsorption_strength=4.0)` |
| **FRET / XOR (nonlinear)** | `reports/fret_regime.py::build_fret_dataset(seed, k=5.0)` |
| **spatial texture** | `reports/spatial_regime.py::build_spatial(seed)` |

---

## 10. Reproducing the paper figures

```
python reports/paper_figures.py    # re-runs the simulations, writes paper/figures/*.png + params.json
```

`paper/figures/params.json` records the exact configuration/parameters behind every figure. For the
headline result numbers see `beat_random.py`, `compression_curve.py`, `spatial_regime.py`, and the full
campaign log `docs/spectraforge/OVERNIGHT-LOG.md`.

---

## 11. Honest-use checklist

When evaluating any selector on SpectraForge data, **always include two controls**:
- **`random`** band selection — if a method cannot beat the 95th percentile of random subsets, it adds
  nothing.
- **`full`** spectrum — the accuracy ceiling a selector is trying to approach.

For spatial regimes, use **spatial-block cross-validation** (train/test on disjoint image blocks) to
avoid spatial-autocorrelation leakage. These controls are the difference between an honest result and a
self-deception (the central lesson of this project).
