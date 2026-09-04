# SpectraForge: synthetic ME-HSI data with ground truth

## Purpose and status

SpectraForge renders multi-excitation hyperspectral cubes from **chemistry you define**: fluorophores with excitation and emission spectra, materials that mix them, a scene you paint the materials onto, and an instrument description (excitations, emission grid, lamp, exposure, power). The output is a `spectral_select.SpectraData` that loads exactly like real data, plus a `GroundTruth` sidecar that records where each fluorophore is and which emission bands carry its signal. Its reason for existing is validation: it is the oracle used to ask whether an unsupervised band selector recovers the bands that genuinely carry information.

Status in September 2026:

* The engine (`src/spectraforge/*.py`) and the painter GUI (`src/spectraforge/gui/`) are complete and covered by about 70 tests.
* The **research claim was withdrawn.** An earlier headline ("the selector recovers realistic spectra with recovery 1.00") was an artifact of a saturated ground-truth mask; a uniformly random selector matched the autoencoder. Commits `9304f7a` and `48bf4e4` record the retraction. The harness now ships a chance baseline (`make_random_selector`), a tight `peak_recovery` metric and a `mask_coverage` diagnostic, and the official verdict is **inconclusive**: on the tight metric the autoencoder hits about 0 of the true emission peaks versus about 0.33 for random, but emission peaks need not be the discriminative target. No paper claim is made from SpectraForge today.
* A follow-up investigation reframed the question as classification (KNN macro-F1 on labelled synthetic scenes) and found that the spatial convolutional autoencoder minimises MSE by predicting per-band means on this sparse data, so its perturbation influence is noise. A per-pixel spectral MLP autoencoder fixes it in preliminary runs. The next phase (`docs/spectraforge/03-architecture-plan.md`, `04-training-runbook.md`) is designed for a training machine but **not built**.

Read `docs/spectraforge/README.md` and `02-investigation-and-findings.md` before quoting any SpectraForge number.

## Files

```
src/spectraforge/
├── fluorophore.py    Fluorophore: Gaussian excitation / emission profiles
├── measured.py       MeasuredFluorophore + from_fpbase_payload: real spectra by interpolation
├── material.py       Material: {fluorophore name: concentration}
├── scene.py          Scene: H x W canvas, paint_rect / paint_circle / paint_polygon / paint_map, resolve()
├── scenegen.py       random_field, random_scene, make_labeled_scene: procedural scenes with per-pixel variance
├── acquisition.py    AcquisitionConfig: excitations, emission grid, lamp / exposure / power per excitation
├── physics.py        PhysicsConfig + apply_physics: inner filter, autofluorescence, PSF blur (all default off)
├── artifacts.py      ArtifactConfig, add_scatter_lines (Rayleigh + second order), add_noise (Poisson + read)
├── forward.py        render(): the forward model -> (SpectraData, GroundTruth)
├── groundtruth.py    GroundTruth: concentration maps, clean cubes, informative_bands(), save()
├── library.py        load_builtin_library(): 12 parametric fluorophores from data/fluorophores.json
├── validation.py     validate_selection(): precision / recall / F1 / per-fluorophore / peak_recovery / mask_coverage
├── sweep.py          run_validation_sweep, aggregate_metrics, make_random_selector, make_analyzer_selector
├── demo.py           build_demo() + the spectraforge-demo console script
├── data/fluorophores.json
└── gui/              the Forge: app.py (ForgeWindow), state.py (ForgeState), layer.py, project.py (save/load),
                      render_ops.py, workers.py, panels/{library,material,canvas,layers,acquire_render}_panel.py,
                      widgets/spectrum_plot.py
tests/spectraforge/   ~70 tests, GUI tests offscreen
reports/              validation scripts and artifacts (spectraforge_validation_report.py, fpbase_validation.py,
                      classification_experiment.py, cae_vs_spectral_ae.py, fpbase_spectra/)
docs/spectraforge/    01-system-overview, 02-investigation-and-findings, 03-architecture-plan, 04-training-runbook
docs/superpowers/specs/2026-06-19-spectraforge-design.md, -gui-design.md; plans/2026-06-19-spectraforge-{engine,gui}.md;
docs/superpowers/spectraforge-roadmap.md
```

## The forward model

For each excitation `λex` (`forward.py:11-73`):

```
scale       = lamp(λex) · exposure(λex) · power(λex)
cube        = Σ_k  c_k(x, y) · ε_k · Φ_k · exc_k(λex) · em_k(λem grid)      # per fluorophore k
absorbance  = Σ_k  ε_k · exc_k(λex) · c_k(x, y)                             # per pixel, for the inner filter
cube       *= scale
cube        = apply_physics(cube, ...)          # optional: inner filter, autofluorescence, PSF
clean_cubes[λex] = cube.copy()                  # ground truth snapshot, before artifacts
add_scatter_lines(cube, λex, ...)               # Rayleigh line at λex, second order at 2 λex
cube        = add_noise(cube, ...)              # Poisson(photon_scale) then Gaussian read noise
```

Closed form: `F(x, y, λex, λem) = lamp·exposure·power · Σ_k c_k(x,y)·ε_k·Φ_k·exc_k(λex)·em_k(λem)`.

* `Fluorophore.excitation(wl)` is a Gaussian **peak-normalised to 1**; `emission(wl)` is a Gaussian **area-normalised to unit sum on the query grid** (`σ = FWHM / 2.3548`). Because the normalisation is over the grid, changing `em_step` rescales absolute intensities.
* Physics (`physics.py`), all off by default so that `render(A + B) == render(A) + render(B)` holds: inner filter `exp(-strength · absorbance)` (plain Beer-Lambert transmission, no emission re-absorption), spatially uniform autofluorescence floor, Gaussian PSF blur in the spatial axes.
* Artifacts (`artifacts.py`): Rayleigh Gaussian at `λex` with `rayleigh_strength · scale`, second-order line at `2 λex` when inside the grid, then `Poisson(clip(cube) · photon_scale) / photon_scale + N(0, read_sigma)`. A single seeded `np.random.default_rng(seed)` makes renders reproducible.

## Data classes and API

| Class / function | Signature | Notes |
|---|---|---|
| `Fluorophore` | `(name, ex_peak_nm, ex_fwhm_nm, em_peak_nm, em_fwhm_nm, quantum_yield=0.5, extinction=1.0)` | frozen dataclass; `excitation(wl)`, `emission(wl)` |
| `MeasuredFluorophore` | `(name, ex_wavelengths, ex_values, em_wavelengths, em_values, quantum_yield=0.5, extinction=1.0)` | `np.interp`, zero outside support; drop-in for `Fluorophore` (duck typed) |
| `from_fpbase_payload(payload, quantum_yield=None, extinction=1.0)` | | accepts FPbase API payloads tagged by `subtype` or `state`; no network |
| `Material` | `(name, recipe: dict[str, float])` | `fluorophores()` |
| `Scene` | `(height, width)` | `paint_rect(material, r0, r1, c0, c1, amount=1.0)`, `paint_circle(material, cy, cx, radius, amount)`, `paint_polygon(material, vertices, amount)`, `paint_map(material, amount_map)`, `resolve() -> {fluorophore: (H, W)}`, `__add__` |
| `AcquisitionConfig` | `(excitations, em_min, em_max, em_step, lamp={}, exposure={}, power={})` | `emission_grid()`; per-excitation lookups default to 1.0 |
| `ArtifactConfig` | `(rayleigh_strength=0.0, rayleigh_fwhm=10.0, second_order=True, photon_scale=0.0, read_sigma=0.0)` | 0 disables |
| `PhysicsConfig` | `(psf_sigma_px=0.0, inner_filter=False, inner_filter_strength=1.0, autofluorescence=0.0, autofluor_peak_nm=480.0, autofluor_fwhm_nm=150.0)` | |
| `render` | `(scene, library, acquisition, artifacts=None, physics=None, seed=None, sample_name="synthetic") -> (SpectraData, GroundTruth)` | `library` is any `{name: fluorophore-like}` mapping |
| `GroundTruth` | fields `concentration_maps`, `clean_cubes`, `emission_grid`, `excitations`, `materials={}`, `per_fluorophore_spectra`, `seed` | `save(out_dir)` writes `groundtruth.npz` (`conc__<name>`, `clean__<ex>`, `emission_grid`) + `groundtruth.json`; `informative_bands(threshold=0.01) -> {ex: bool mask}`; `informative_bands_per_fluorophore(threshold)` |
| `load_builtin_library()` | `-> dict[str, Fluorophore]` | tryptophan, collagen, elastin, NADH, FAD, DAPI, EGFP, fluorescein, rhodamine, TexasRed, mCherry, Cy5; literature peak values, **not** FPbase downloads |
| `validate_selection` | `(ground_truth, selected, tol_nm=10.0, threshold=0.01) -> dict` | `selected` = `WavelengthBand`s or `(ex, em)` tuples; excitation must match within 1 nm |
| `run_validation_sweep` | `(scene_factory, library, acquisition, selector, seeds, artifacts=None, physics=None, tol_nm=10.0) -> list[dict]` | `selector(spectra) -> selection`; `aggregate_metrics(results)` gives mean/std |
| `make_random_selector(k=12, seed=0)` / `make_analyzer_selector(config)` | | the chance baseline and the `spectral_select.Analyzer` wrapper |
| `random_field(h, w, seed, blur=2)`, `random_scene(materials, h, w, seed)`, `make_labeled_scene(materials, h, w, seed) -> (Scene, labels)` | | procedural scenes; `labels` is the per-pixel argmax material, used by the classification benchmark |

`validate_selection` keys: `precision`, `recall`, `f1`, `hits`, `n_selected`, `n_informative`, `per_fluorophore`, `fluorophores_recovered` (broad-mask metrics, which saturate), `peak_hits`, `peak_recovery` (did a selected band land within `tol_nm` of each fluorophore's true emission peak at its best excitation: the metric that discriminates), `mask_coverage` (fraction of the grid flagged informative; if it is above 0.8 the broad metrics are meaningless).

Rule encoded by the July 2026 retraction: **never read precision or recovery without a random baseline and `mask_coverage` next to it.**

## Running it

```bash
spectraforge-demo -o my_dataset      # 64x64, 3 excitations (340/450/488), 4 fluorophores, seed 42
                                     # writes spectra_unmasked.pkl + groundtruth.npz + groundtruth.json
spectraforge-gui                     # the Forge (needs PyQt6: pip install -e ".[gui]")
```

Programmatically (also in section 11 of the walkthrough notebook):

```python
from spectraforge import AcquisitionConfig, ArtifactConfig, Material, Scene, load_builtin_library, render
lib = load_builtin_library()
scene = Scene(64, 64)
scene.paint_rect(Material("tissue", {"collagen": 1.0, "NADH": 0.3}), 5, 50, 5, 50)
acq = AcquisitionConfig(excitations=[340, 450, 488], em_min=360, em_max=700, em_step=5)
spectra, gt = render(scene, lib, acq, artifacts=ArtifactConfig(rayleigh_strength=0.15, photon_scale=400), seed=42)
spectra.to_pickle("synthetic.pkl")        # loads in spectral-select-gui step 1 and in Analyzer
gt.informative_bands()                    # {ex: bool mask over the emission grid}
```

A validation sweep with the chance baseline:

```python
from spectraforge.sweep import run_validation_sweep, aggregate_metrics, make_random_selector, make_analyzer_selector
from spectraforge.scenegen import random_scene
factory = lambda seed: random_scene([Material("a", {"collagen": 1}), Material("b", {"FAD": 1})], 64, 64, seed)
rand = aggregate_metrics(run_validation_sweep(factory, lib, acq, make_random_selector(k=12), seeds=range(5)))
ours = aggregate_metrics(run_validation_sweep(factory, lib, acq, make_analyzer_selector(config), seeds=range(5)))
```

The scripts in `reports/` (`spectraforge_validation_report.py`, `fpbase_validation.py`, `classification_experiment.py`, `cae_vs_spectral_ae.py`) are the runnable record of the investigation; `docs/spectraforge/04-training-runbook.md` lists their commands.

## The Forge (GUI)

`ForgeWindow` (`gui/app.py`) is a fixed 1500 x 950 workbench around a `ForgeState` (default 64 x 64 canvas; excitations 340 / 450 / 488; emission 360 to 700 step 5; Rayleigh 0.15 / FWHM 12; photon scale 400; read sigma 0.005).

| Panel | What it does |
|---|---|
| Library (left tab) | list of fluorophores with a spectrum plot; a form to add a Gaussian fluorophore (`define_fluorophore`) |
| Materials (left tab) | compose a `Material` from fluorophore / concentration rows; previews the mixed emission at the first excitation |
| Canvas (centre) | paint the **active layer's** amount map with brush (max), eraser (zero), rect and circle (add); brush radius and value; false-colour composite of visible layers |
| Layers (right) | one layer per material; visibility checkboxes; reorder; remove; selection sets the active layer |
| Acquire / Render / Export (bottom) | excitation list; **Render** (`RenderWorker` thread); **Export pkl + ground truth** (`export_dataset`); **Validate selection vs ground truth** (`ValidateWorker`, runs a CPU `Analyzer` with 12 bands / 30 epochs and prints `peak_recovery` first and `mask_coverage` always); slice preview with excitation and band sliders |

Projects save as a single **`.npz`** (`gui/project.py`): metadata as one JSON string in `_meta`, each layer's amount map as `_layer{i}`. File menu: New / Open / Save.

Threading: both workers catch exceptions and emit `failed`; `validate_state` renders into a local and never writes `state.last_render` (fix for a data race, commit `50a424f`), and the panel refuses a second click while a validation is running.

## Tests

`tests/spectraforge/`: `test_fluorophore`, `test_material`, `test_scene`, `test_acquisition`, `test_artifacts`, `test_physics` (physics-off preserves linearity; inner filter breaks it), `test_forward` (shape, peak location, amplitude proportional to concentration, **linearity invariant**, exposure / power recorded), `test_groundtruth`, `test_library`, `test_measured` (both FPbase payload formats), `test_validation` (including the end-to-end contract test whose docstring says recovery is a research question, not a CI assertion), `test_sweep`, `test_binding` (render -> pickle round trip -> `Analyzer.fit` runs), `test_demo`, and `gui/` (layer, state, project round trip, render ops, workers, widgets, every panel offscreen, the concurrent-validate guard). GUI tests use `QT_QPA_PLATFORM=offscreen` and skip without PyQt6.

## Known issues and gotchas

1. **Emission normalisation depends on the grid** (`fluorophore.py:38`): datasets rendered with different `em_step` are not amplitude-comparable.
2. **`GroundTruth.materials` is never populated** by `render`; `groundtruth.json`'s `materials` is always empty. There is no `GroundTruth.load()`: unpack the `.npz` by hand (`conc__<name>`, `clean__<ex>`).
3. **`sample_name` and `metadata` are dropped** by `SpectraData.to_pickle`, so `render(..., sample_name=...)` does not survive export.
4. **Project files are `.npz`, not `.forge`** as the older docs say. Saving a project that contains a `MeasuredFluorophore` crashes (`asdict` + `json.dumps` on numpy arrays; the loader rebuilds `Fluorophore(**f)` unconditionally). The Forge is Gaussian-only in practice.
5. **The GUI cannot use `PhysicsConfig`** (`ForgeState` has no `physics` field; `render_state` passes none) and exposes only the excitation list, not the emission grid, seed or artifact parameters. A typo in the excitation box is silently ignored.
6. **`RenderWorker` still holds a live reference to the mutable state** while rendering (painting during a render is a race); only the Validate path was hardened.
7. The GUI imports `DEFAULT_CLASS_COLORS` and `ImageCanvas` from `mehsi_preprocessor`, so it breaks if that package moves.
8. Magic numbers: 1 nm excitation match tolerance and `threshold=0.01` in `validate_selection` (the source of mask saturation); `random_field` upsamples 4x with periodic (`np.roll`) blur; `SpectrumPlot` covers 250 to 750 nm only.
9. `add_scatter_lines` has a dead `reflectance` parameter (always ones); the inner filter does not implement the roadmap's `1 - 10^(-εcl)` plus self-absorption formula.
10. `spectraforge-batch` (roadmap increment E) was never built; run sweeps from Python or `reports/`.
11. `__init__` does not export `validate_selection`, the sweep helpers or `scenegen`; import them from the submodules.
12. Test counts quoted in the docs (436 / 453 / 460) are approximate.
