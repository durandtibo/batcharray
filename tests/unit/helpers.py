r"""Contain helpers shared across the unit tests."""

from __future__ import annotations

__all__ = ["SORT_KINDS", "MyArray", "MyMaskedArray"]

import numpy as np

SORT_KINDS = ["quicksort", "mergesort", "heapsort", "stable"]


class MyArray(np.ndarray):
    r"""Minimal ``numpy.ndarray`` subclass to test subclass handling."""


class MyMaskedArray(np.ma.MaskedArray):
    r"""Minimal ``numpy.ma.MaskedArray`` subclass to test subclass
    handling."""
