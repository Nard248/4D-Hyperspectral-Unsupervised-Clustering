# Installation Guide

> **Superseded.** The maintained, platform-by-platform guide is [`onboarding/SETUP.md`](onboarding/SETUP.md). It covers the Java + Maven stack that PyImageJ needs for `.im3` files, the `[gui]` / `[im3]` / `[all]` extras, GPU setup, where to get the data and a verification checklist. This page is kept for reference; where the two disagree, `onboarding/SETUP.md` is right.

This guide walks you through installing Spectral-Select on your computer.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.11 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 2 GB for software + space for your data

### Recommended for Large Datasets
- **RAM**: 32 GB or more
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **Storage**: SSD for faster data loading

## Step-by-Step Installation

### Step 1: Install Python

If you don't have Python installed:

**Windows:**
1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Click "Install Now"

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### Step 2: Download Spectral-Select

**Option A: Clone with Git (recommended)**
```bash
git clone https://github.com/narekmeloyan/spectral-select.git
cd spectral-select
```

**Option B: Download ZIP**
1. Go to the GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open a terminal in the extracted folder

### Step 3: Create a Virtual Environment

A virtual environment keeps this project's packages separate from other Python projects.

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your command prompt.

### Step 4: Install Dependencies

```bash
# Install the package in development mode with all dependencies
pip install -e ".[dev]"
```

This installs:
- Core dependencies (PyTorch, NumPy, scikit-learn, etc.)
- Development tools (pytest, jupyter)
- All optional dependencies

### Step 5: Verify Installation

Run the following to verify everything works:

```bash
python -c "from spectral_select import Analyzer, Config; print('Installation successful!')"
```

You should see: `Installation successful!`

### Step 6: Reading raw `.im3` files (optional)

Raw Nuance `.im3` cubes are read through Fiji's Bio-Formats plugin via **PyImageJ**, which needs a JDK (Temurin 17)
and Apache Maven on the PATH. Install them, then `pip install -e ".[im3]"`. The first `.im3` read downloads Fiji
(about 350 MB) into `~/.jgo` and `~/.m2`; a desktop Fiji installation is neither required nor auto-detected.
Details and troubleshooting: [`onboarding/SETUP.md`](onboarding/SETUP.md), section 4.

## GPU Support (Optional but Recommended)

GPU acceleration makes training 10-50x faster.

### NVIDIA GPU (CUDA)

1. **Check your GPU**: Open Device Manager (Windows) or run `lspci | grep -i nvidia` (Linux)

2. **Install CUDA Toolkit**:
   - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Download CUDA 11.8 or 12.x
   - Run the installer

3. **Install cuDNN**:
   - Create an NVIDIA Developer account
   - Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Follow installation instructions

4. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Verify GPU support**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
   ```

### Apple Silicon (M1/M2/M3)

Apple Silicon Macs can use the MPS (Metal Performance Shaders) backend:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

Use `device="mps"` in your configuration.

## Running Jupyter Notebooks

The example notebooks are the easiest way to learn the software.

### Start Jupyter

```bash
# Make sure your virtual environment is activated
jupyter notebook
```

This opens a browser window. Navigate to `examples/` and open the notebooks.

### Jupyter Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | Shift + Enter |
| Run cell, stay in cell | Ctrl + Enter |
| Insert cell below | B |
| Insert cell above | A |
| Delete cell | D, D (press D twice) |
| Save notebook | Ctrl + S |

## Project Structure

After installation, your directory should look like this:

```
spectral-select/
├── spectral_select/     # Core library
│   ├── models/          # Autoencoder, dataset, training
│   ├── analyzer.py      # Main analysis engine
│   ├── config.py        # Configuration class
│   ├── types.py         # Data types
│   ├── validation.py    # Validation tools
│   └── visualizer.py    # Visualization utilities
├── mehsi_preprocessor/  # Optional preprocessing GUI
│   └── io/              # Raw data loading
├── experiments/         # Paper reproduction scripts
├── examples/            # Tutorial notebooks
├── tests/               # Test suite
├── docs/                # Documentation
├── pyproject.toml       # Package configuration
└── README.md            # Main readme
```

## Common Installation Issues

### "Python not found"

**Windows**: Python wasn't added to PATH. Either:
- Reinstall Python and check "Add to PATH"
- Or use the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe`

### "pip not found"

Try using `python -m pip` instead of `pip`:
```bash
python -m pip install -e ".[dev]"
```

### "Permission denied" errors

**Windows**: Run Command Prompt as Administrator

**macOS/Linux**: Don't use `sudo` with pip in a virtual environment. Make sure your venv is activated.

### PyTorch installation fails

Try installing PyTorch separately first:
```bash
pip install torch torchvision
```

Then install the rest:
```bash
pip install -e ".[dev]"
```

### "No module named 'spectral_select'"

Make sure you:
1. Activated your virtual environment
2. Ran `pip install -e ".[dev]"` in the project directory
3. Are running Python from the same terminal where you installed

## Updating

To update to the latest version:

```bash
cd spectral-select
git pull
pip install -e ".[dev]"
```

## Uninstalling

```bash
# Deactivate virtual environment
deactivate

# Remove the project directory
# Windows: rmdir /s /q spectral-select
# macOS/Linux: rm -rf spectral-select
```

## Next Steps

After installation:

1. **Test with example data**: Open `examples/01_quickstart.ipynb`
2. **Prepare your data**: See [DATA_PROCESSING.md](guides/DATA_PROCESSING.md)
3. **Learn the configuration options**: See [CONFIGURATION.md](CONFIGURATION.md)

## Getting Help

If you encounter issues:

1. Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
2. Search existing GitHub issues
3. Open a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce
