"""MEHSI preprocessor: the .im3 reader, pure preprocessing functions and the PyQt6 wizard.

Importing this package must not require PyQt6. ``mehsi_preprocessor.io`` and
``mehsi_preprocessor.processing`` are used headlessly (Jupyter, scripts, and the
``spectral_select.DataLoader``), so the Qt application is imported lazily.
"""


def main() -> None:
    """Launch the preprocessing wizard (same as the ``spectral-select-gui`` command)."""
    from mehsi_preprocessor.app import main as _main

    _main()


__all__ = ["main"]
