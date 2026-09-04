"""
Notebook smoke tests for spectral_select example notebooks.

These tests verify that:
1. Notebooks are valid JSON (parseable)
2. Imports used in notebooks work
3. Notebooks can be executed (with nbval, when not skipped)

Run with:
    pytest tests/test_notebooks.py -v           # All tests
    pytest -m "not slow" tests/test_notebooks.py  # Skip slow tests
    pytest --nbval-lax examples/                # Full notebook execution
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.notebook
class TestNotebookParseable:
    """Test that notebooks are valid JSON and have expected structure."""

    def test_quickstart_is_valid_json(self) -> None:
        """Verify 01_quickstart.ipynb is valid Jupyter notebook JSON."""
        nb_path = NOTEBOOKS_DIR / "01_quickstart.ipynb"
        assert nb_path.exists(), f"Notebook not found: {nb_path}"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Verify notebook structure
        assert "cells" in nb, "Notebook missing 'cells' key"
        assert len(nb["cells"]) > 0, "Notebook has no cells"
        assert "metadata" in nb, "Notebook missing 'metadata' key"
        assert "nbformat" in nb, "Notebook missing 'nbformat' key"

    def test_validation_is_valid_json(self) -> None:
        """Verify 02_validation.ipynb is valid Jupyter notebook JSON."""
        nb_path = NOTEBOOKS_DIR / "02_validation.ipynb"
        assert nb_path.exists(), f"Notebook not found: {nb_path}"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Verify notebook structure
        assert "cells" in nb, "Notebook missing 'cells' key"
        assert len(nb["cells"]) > 0, "Notebook has no cells"
        assert "metadata" in nb, "Notebook missing 'metadata' key"
        assert "nbformat" in nb, "Notebook missing 'nbformat' key"

    def test_walkthrough_is_valid_json(self) -> None:
        """Verify 03_hsi_data_walkthrough.ipynb is valid and guarded for missing data."""
        nb_path = NOTEBOOKS_DIR / "03_hsi_data_walkthrough.ipynb"
        assert nb_path.exists(), f"Notebook not found: {nb_path}"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 20, f"Expected at least 20 code cells, found {len(code_cells)}"
        source = "\n".join("".join(c.get("source", [])) for c in code_cells)
        # The walkthrough must run without data and without PyImageJ (guarded sections).
        assert "HAVE_IMAGEJ" in source and "HAVE_PROCESSED" in source
        assert "build_demo" in source, "synthetic fallback missing"

    def test_quickstart_has_code_cells(self) -> None:
        """Verify quickstart notebook contains code cells."""
        nb_path = NOTEBOOKS_DIR / "01_quickstart.ipynb"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 5, f"Expected at least 5 code cells, found {len(code_cells)}"

    def test_validation_has_code_cells(self) -> None:
        """Verify validation notebook contains code cells."""
        nb_path = NOTEBOOKS_DIR / "02_validation.ipynb"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 5, f"Expected at least 5 code cells, found {len(code_cells)}"


@pytest.mark.notebook
class TestNotebookImports:
    """Test that imports used in example notebooks work."""

    def test_quickstart_imports(self) -> None:
        """Test that imports from quickstart notebook work."""
        from spectral_select import Analyzer, Config, SpectraData, Visualizer

        # Verify classes are importable and not None
        assert Analyzer is not None
        assert Config is not None
        assert SpectraData is not None
        assert Visualizer is not None

    def test_validation_imports(self) -> None:
        """Test that imports from validation notebook work."""
        import numpy as np

        from spectral_select import Validator, Visualizer, load_ground_truth_from_png

        # Verify classes/functions are importable
        assert Validator is not None
        assert Visualizer is not None
        assert load_ground_truth_from_png is not None
        assert np is not None

    def test_all_public_api_importable(self) -> None:
        """Test that all public API exports are importable."""
        from spectral_select import (
            Analyzer,
            Config,
            GroundTruth,
            SpectraData,
            Validator,
            ValidationMetrics,
            Visualizer,
            WavelengthBand,
            WavelengthResult,
            load_ground_truth_from_png,
        )

        # All should be non-None
        exports = [
            Analyzer,
            Config,
            GroundTruth,
            SpectraData,
            Validator,
            ValidationMetrics,
            Visualizer,
            WavelengthBand,
            WavelengthResult,
            load_ground_truth_from_png,
        ]
        for export in exports:
            assert export is not None


@pytest.mark.slow
@pytest.mark.notebook
class TestNotebookCellExtraction:
    """Test extraction of specific cells from notebooks for validation."""

    def test_quickstart_first_import_cell(self) -> None:
        """Extract and verify the first import cell from quickstart."""
        nb_path = NOTEBOOKS_DIR / "01_quickstart.ipynb"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Find first code cell
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        first_code = code_cells[0]

        # Should be the import statement
        source = "".join(first_code.get("source", []))
        assert "from spectral_select import" in source
        assert "Analyzer" in source
        assert "Config" in source

    def test_validation_first_import_cell(self) -> None:
        """Extract and verify the first import cell from validation."""
        nb_path = NOTEBOOKS_DIR / "02_validation.ipynb"

        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Find first code cell
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        first_code = code_cells[0]

        # Should be the import statement
        source = "".join(first_code.get("source", []))
        assert "from spectral_select import" in source
        assert "Validator" in source


@pytest.mark.notebook
class TestNotebookMetadata:
    """Test notebook metadata and structure."""

    def test_notebooks_have_kernel_spec(self) -> None:
        """Verify notebooks have kernel specification."""
        for nb_name in ["01_quickstart.ipynb", "02_validation.ipynb", "03_hsi_data_walkthrough.ipynb"]:
            nb_path = NOTEBOOKS_DIR / nb_name

            with open(nb_path, encoding="utf-8") as f:
                nb = json.load(f)

            metadata = nb.get("metadata", {})
            # Kernel spec is optional but recommended
            if "kernelspec" in metadata:
                ks = metadata["kernelspec"]
                assert "name" in ks or "display_name" in ks

    def test_notebooks_are_nbformat_4(self) -> None:
        """Verify notebooks use nbformat version 4."""
        for nb_name in ["01_quickstart.ipynb", "02_validation.ipynb", "03_hsi_data_walkthrough.ipynb"]:
            nb_path = NOTEBOOKS_DIR / nb_name

            with open(nb_path, encoding="utf-8") as f:
                nb = json.load(f)

            assert nb.get("nbformat") == 4, f"{nb_name} is not nbformat 4"
