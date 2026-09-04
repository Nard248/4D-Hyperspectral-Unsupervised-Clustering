# Environment setup

This guide takes a fresh machine to a working `spectral-select` checkout: Python environment, the Java stack that reads the instrument's `.im3` files, the GUIs, optional GPU support, the datasets, and a verification checklist. Follow it top to bottom once; afterwards `git pull` and `pip install -e ".[all]"` keep you current.

Time budget: about 30 minutes plus downloads (Fiji is 350 MB, PyTorch 200 MB to 2.5 GB depending on the GPU build, datasets up to a few GB).

## 0. What gets installed and why

| Component | Why we need it | Required? |
|---|---|---|
| Git | clone and update the repository | yes |
| Python 3.11 or 3.12 | the package requires `>= 3.11`; 3.11 is what the maintainer runs | yes |
| `spectral-select` package (editable) | the code in `src/`: `spectral_select`, `selection_core`, `channel_select`, `mehsi_preprocessor`, `spectraforge` | yes |
| `[dev]` extra: pytest, JupyterLab, nbval | run the tests and the example notebooks | yes |
| `[gui]` extra: PyQt6 | the preprocessing wizard (`spectral-select-gui`) and the SpectraForge painter (`spectraforge-gui`) | for GUI users |
| `[im3]` extra: PyImageJ | reading raw `.im3` cubes through Fiji's Bio-Formats reader | for anyone touching raw data |
| JDK 17 (Temurin) | PyImageJ runs a Java virtual machine in-process | with `[im3]` |
| Apache Maven 3.9 | PyImageJ resolves and downloads Fiji through Maven on first use | with `[im3]` |
| tkinter | the desktop ME-HSI viewer; `import spectral_select` needs it even if you never open the viewer | yes (bundled on macOS/Windows, a package on Linux) |
| CUDA-enabled PyTorch | autoencoder training on NVIDIA GPUs; Apple Silicon uses the built-in MPS backend | optional |
| Fiji desktop application | handy for opening exported TIFF stacks by hand; the code does not need it | optional |
| `moabb`, `mne`, `huggingface_hub`, `aeon` | only for the EEG and benchmark-screening experiments under `experiments/` | optional |

The `.im3` format is PerkinElmer/CRi Nuance. There is no pure-Python reader in the repository: `HyperspectralDataLoader` calls `imagej.init("sc.fiji:fiji")`, which asks Maven to fetch Fiji into `~/.jgo` and `~/.m2` and boots a JVM. That is the only reason Java and Maven appear here.

## 1. Platform prerequisites

### macOS (Apple Silicon or Intel)

```bash
# Homebrew: https://brew.sh
brew install git python@3.11 maven
brew install --cask temurin@17

# Make Java 17 the default for your shell (add to ~/.zshrc)
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
export PATH="$JAVA_HOME/bin:$PATH"
```

Check:

```bash
python3.11 --version   # Python 3.11.x
java -version          # openjdk version "17.0.x" ... Temurin
mvn -v                 # Apache Maven 3.9.x, Java version: 17.0.x
```

Notes: Homebrew's `maven` formula pulls in a newer OpenJDK as a dependency. That is fine, as long as `JAVA_HOME` points at 17 when you run Python. Xcode command-line tools are installed automatically by Homebrew. tkinter ships with Homebrew Python (`brew install python-tk@3.11` if `import tkinter` fails).

### Windows 10 / 11

1. **Python 3.11** from [python.org](https://www.python.org/downloads/windows/). Tick **Add python.exe to PATH** and keep the **tcl/tk** option selected.
2. **Git for Windows** from [git-scm.com](https://git-scm.com/download/win).
3. **Temurin 17 JDK** (MSI) from [adoptium.net](https://adoptium.net/temurin/releases/?version=17). In the installer enable **Set JAVA_HOME variable** and **Add to PATH**.
4. **Apache Maven**: download the binary zip from [maven.apache.org](https://maven.apache.org/download.cgi), extract to `C:\maven`, and add `C:\maven\bin` to the user `PATH` (Settings > System > About > Advanced system settings > Environment Variables).

Or with a package manager (run in an elevated PowerShell):

```powershell
winget install --id Git.Git
winget install --id Python.Python.3.11
winget install --id EclipseAdoptium.Temurin.17.JDK
choco install maven          # Chocolatey; or extract the zip as above
```

Open a **new** terminal after changing environment variables, then check:

```powershell
python --version
java -version
mvn -v
echo $env:JAVA_HOME
```

Use PowerShell or Command Prompt for everything below; the venv activation line differs, the rest is identical.

### Linux (Ubuntu 22.04 / 24.04, Debian)

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv python3.11-dev python3-tk \
                    openjdk-17-jdk maven \
                    libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0 libdbus-1-3
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64     # add to ~/.bashrc
```

`python3-tk` is mandatory: the package imports tkinter at import time. The `libxcb*` and `libgl*` packages are what PyQt6 needs to show a window. On a headless server (no display) set `QT_QPA_PLATFORM=offscreen` for the GUI tests and skip the GUIs.

Other distributions: use the equivalent packages (`java-17-openjdk-devel maven python3.11-tkinter` on Fedora).

### Alternative for every platform: conda / mamba

If you already use conda, this is the most reliable way to get Java, Maven and PyImageJ that agree with each other:

```bash
mamba create -n hsi -c conda-forge python=3.11 pyimagej openjdk=17 maven
mamba activate hsi
```

Then continue from step 3 inside that environment (skip the venv creation; `pip install -e ".[dev,gui]"` is enough because PyImageJ came from conda).

## 2. Clone and create the environment

```bash
git clone https://github.com/narekmeloyan/spectral-select.git
cd spectral-select
```

Create and activate a virtual environment named `.venv` (git-ignored; PyCharm and VS Code pick it up automatically):

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Windows Command Prompt
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt now starts with `(.venv)`. Upgrade the installer tooling once:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install the package

Everything in one go (development tools, both GUIs, the `.im3` reader):

```bash
pip install -e ".[all]"
```

Or pick what you need:

| Command | Gets you |
|---|---|
| `pip install -e .` | the libraries only (`spectral_select`, `selection_core`, `channel_select`, `spectraforge`, the headless parts of `mehsi_preprocessor`) |
| `pip install -e ".[dev]"` | plus pytest, JupyterLab, nbval |
| `pip install -e ".[gui]"` | plus PyQt6 for `spectral-select-gui` and `spectraforge-gui` |
| `pip install -e ".[im3]"` | plus PyImageJ (needs the JDK and Maven from step 1) |
| `pip install -e ".[all]"` | all of the above |

The install pulls PyTorch. On Linux and Windows `pip` installs the CPU build by default; see step 5 to install a CUDA build **before** running the command above, or reinstall torch afterwards. On Apple Silicon the default wheel already contains the MPS backend.

Optional extras for the experiment scripts (not needed for the hyperspectral pipeline):

```bash
pip install ipywidgets            # interactive sliders and the ROIWidget in notebooks
pip install moabb mne             # experiments/eeg (BCI IV-2a via MOABB)
pip install huggingface_hub aeon  # experiments/*_screen.py benchmark screens
```

## 4. Reading `.im3` files: PyImageJ and Fiji

With `[im3]` installed, the first `.im3` read of your life takes a few minutes: PyImageJ asks Maven to resolve the `sc.fiji:fiji` artifact and downloads roughly 350 MB into `~/.jgo/sc.fiji/fiji/` and `~/.m2/repository/`. Every later start takes two to five seconds. Trigger the download now, while you are watching:

```bash
python - <<'EOF'
import imagej
ij = imagej.init("sc.fiji:fiji", mode="headless")
print("ImageJ", ij.getVersion())          # e.g. 2.17.0/1.54p
EOF
```

Expected: a wall of `[INFO]` Maven lines the first time, then the version string. If you see `Failed to create a JVM`, check `java -version` and `JAVA_HOME`; if you see `mvn: command not found` or `jgo` errors, Maven is not on the PATH.

Rules worth knowing before you write any code:

* **One `imagej.init()` per Python process.** A second call returns a broken gateway (it reports `Inactive`). The example notebook shows how to start ImageJ once and hand the gateway to `HyperspectralDataLoader`; `SpectraData.from_raw` starts its own, so use it from a fresh process (a script), not after another init.
* **Use `mode="headless"`** in scripts and notebooks. The interactive mode wants a GUI event loop and conflicts with Qt and Jupyter.
* **No Fiji desktop install is needed.** The old installation guide said the code would auto-detect `/Applications/Fiji.app`; it does not, and never did. If you *want* to use a local Fiji instead of the Maven download (for example, offline), pass its folder to `imagej.init("/Applications/Fiji")` (the folder that contains `jars/`; on recent Fiji builds that is `/Applications/Fiji`, not `Fiji.app`). Bio-Formats 7+ is required for the Nuance reader; the bundled Fiji has it.
* **Corporate proxies**: Maven reads `~/.m2/settings.xml` for proxy settings.
* Reading the raw loader from Python must not require PyQt6. Since this guide was written, `mehsi_preprocessor` imports its Qt app lazily; if you are on an older checkout and `from mehsi_preprocessor.io.hyperspectral_loader import HyperspectralDataLoader` fails with `No module named PyQt6`, update or install the `[gui]` extra.

Verify with a real cube (needs the data from step 6):

```bash
python - <<'EOF'
import imagej, numpy as np
ij = imagej.init("sc.fiji:fiji", mode="headless")
img = ij.io().open("Data/Raw/Lichens_2/365.im3")
cube = np.asarray(ij.py.from_java(img).values)
print(cube.shape, cube.dtype)             # (256, 348, 31) uint16
EOF
```

## 5. GPU support (optional)

Training the autoencoder on the full Lichens dataset takes hours on CPU and minutes on a GPU. Everything else in the repository is fine on CPU.

### NVIDIA (Linux, Windows)

Install the matching PyTorch build **inside the venv**, then reinstall the package so nothing downgrades it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu121 / cu118 to match your driver
pip install -e ".[all]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Set `Config(device="cuda")` (the default) in `spectral_select`. The `channel_select` experiment scripts hard-code `mps`-or-`cpu`; edit their `DEVICE` line to `"cuda"` if you run them on an NVIDIA machine.

### Apple Silicon (M-series)

Nothing to install. `python -c "import torch; print(torch.backends.mps.is_available())"` should print `True`. Use `Config(device="mps")`. The `channel_select` experiments need this environment variable because one operator is not implemented on MPS:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The GUI's training step always uses the CPU.

## 6. Get the data

`Data/` is git-ignored and sits at the repository root. The layout, file formats and the catalogue of every dataset are in [DATA_GUIDE.md](DATA_GUIDE.md); the short version:

```
Data/
├── Raw/<Sample>/          .im3 cubes + metadata.xlsx + TLS Scans/average_power.xlsx
└── processed/<Sample>/    .pkl datasets, masks, ROI annotations
```

Two sources:

1. **Lab storage** (the canonical copies, about 7 GB in total). Ask Narek for the shared folder and copy `Raw/` and `processed/` as they are; the folder names are load-bearing because scripts reference them (for example `Data/processed/Lichens Dataset 1/`).
2. **Public release** of the Lichens dataset on Zenodo: <https://doi.org/10.5281/zenodo.18640119> (`HSI.zip`, 1.8 GB, plus `Metadata_HSI.csv`, `Metadata_Classes.csv`, `Mask_Manual.png`, an RGB rendering and a sample-prep photo). Its file naming differs from the lab layout; unpack it into `Data/Raw/Lichens_Zenodo/` and rename the cubes to `<excitation>.im3` if you want the loader to read them directly.

Start with **`Lichens_2`** (50 MB raw, 256 x 348 pixels, 8 excitations): every example in the onboarding material uses it. Put the data somewhere else by setting `SPECTRAL_SELECT_DATA=/path/to/Data` for the notebook, but the experiment scripts expect `Data/` in the repository.

## 7. Verify the installation

Run these from the repository root with the venv active. Each line says what a good result looks like.

```bash
# 1. The package imports and reports its version
python -c "import spectral_select, spectraforge, channel_select, selection_core; print(spectral_select.__version__)"
# -> 0.1.0

# 2. Console scripts are on the PATH
spectraforge-demo -o /tmp/forge_demo       # writes spectra_unmasked.pkl + groundtruth.npz/json
spectral-select-gui --help 2>/dev/null; echo "gui entry point: $?"   # a window opens (close it); needs [gui]

# 3. The test suite (about one minute; GUI tests are skipped without PyQt6)
QT_QPA_PLATFORM=offscreen pytest -q --no-cov
# -> "... passed" with no failures

# 4. Jupyter sees the environment
python -m ipykernel install --user --name spectral-select --display-name "Python (spectral-select)"
jupyter lab examples/03_hsi_data_walkthrough.ipynb
```

The walkthrough notebook is the end-to-end check: it opens a raw `.im3` through ImageJ, loads a processed pickle, plots slices and spectra, runs the preprocessing functions and a three-epoch selection. It also runs without any data (it synthesises a cube), so it doubles as an environment smoke test.

## 8. Editor notes

* **PyCharm**: open the repository folder; choose `.venv/bin/python` as the interpreter (Settings > Project > Python Interpreter > Add > Existing). No need to mark `src/` as a sources root: the editable install makes the packages importable. Enable "Emulate terminal in output console" for scripts that print progress bars.
* **VS Code**: install the Python and Jupyter extensions, select the `.venv` interpreter from the status bar. Add `"python.testing.pytestEnabled": true` to use the test explorer.
* **Jupyter**: register the kernel as shown in step 7 so notebooks in other folders can select it.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `mvn: command not found`, or `jgo` raises during `imagej.init` | Maven is not on the PATH of the shell that launched Python. Install it (step 1) and open a new terminal. Conda users: `mamba install -c conda-forge maven`. |
| `RuntimeError: Failed to create a JVM with the requested environment` | No JDK visible, wrong `JAVA_HOME`, or (with a local path) the folder is not a Fiji installation. Run `java -version`; for local installs pass the folder that contains `jars/`. |
| Second `imagej.init()` prints a Java stack trace and the version ends in `Inactive` | ImageJ was already started in this process. Reuse the first gateway (see the notebook, section 4) or restart the kernel. |
| `HyperspectralDataLoader.load_data()` returns nothing and only warns | PyImageJ failed to import or to start; the loader silently falls back to a stub that cannot read `.im3`. Fix the Java stack, then retry. |
| `ValueError: could not convert string to float: '310 1500 SPF'` | The loader expects files named `<excitation>.im3`. Drop Data uses `<ex> <exposure> SPF.im3` plus `Background.im3`; read those with PyImageJ directly (notebook section 4b) or the archived Drop Data scripts. |
| `ModuleNotFoundError: No module named 'tkinter'` on `import spectral_select` | Linux: `sudo apt install python3-tk` (Fedora: `python3-tkinter`), then recreate the venv. |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | Missing Qt system libraries on Linux (step 1), or no display. For tests and scripts set `QT_QPA_PLATFORM=offscreen`. |
| `ImportError: PyQt6` when launching a GUI | Install the extra: `pip install -e ".[gui]"`. |
| PyTorch fails to install or wheels do not match | Install torch first with the index URL for your CUDA version (step 5), then `pip install -e ".[all]"`. |
| `NotImplementedError: ... MPS` in a `channel_select` experiment | `export PYTORCH_ENABLE_MPS_FALLBACK=1` before running. |
| A "new" training run finishes in seconds | `Config` reuses `model_output/<sample_name>/model.pth` when `model_path` is not set. Delete that file or set `model_path` explicitly. |
| Scripts cannot find `Data/...` on Linux but work on macOS | Linux file systems are case sensitive; one archived script uses lowercase `data/`. Use the exact folder names from `DATA_GUIDE.md`. |
| `MemoryError` loading a pickle | The legacy pickle dialect stores every cube twice (cut and uncut). Use the export dialect (`spectra_*.pkl`), or `del data.metadata["raw_data"]` after loading. |
| Jupyter kernel dies while starting ImageJ | Usually a 32-bit or mismatched JDK. Use Temurin 17 (64-bit) and make sure `JAVA_HOME` points at it. |

## 10. Updating and uninstalling

```bash
git pull
pip install -e ".[all]"        # picks up new dependencies and console scripts

# remove everything except the data
deactivate
rm -rf .venv                   # Windows: rmdir /s /q .venv
rm -rf ~/.jgo ~/.m2            # optional: the Fiji / Maven caches (350 MB + 400 MB)
```
