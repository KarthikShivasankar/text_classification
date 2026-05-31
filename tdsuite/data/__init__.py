"""Data processing and loading utilities for technical debt classification."""

from .data_splitter import split_data


def __getattr__(name):
    """Lazy-load torch-dependent dataset classes only when requested.

    ``TDDataset``/``TDProcessor``/``BinaryTDProcessor`` live in ``dataset.py``,
    which imports torch. Loading them lazily keeps ``import tdsuite.data`` (and
    the ``tdsuite-split-data`` CLI) usable on the default torch-free install.
    """
    if name in ("TDDataset", "TDProcessor", "BinaryTDProcessor"):
        from . import dataset

        globals()[name] = getattr(dataset, name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TDDataset", "TDProcessor", "BinaryTDProcessor", "split_data"]
