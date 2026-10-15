r"""Contain helpers shared across the unit tests."""

from __future__ import annotations

__all__ = ["MyArray", "MyMaskedArray"]

import numpy as np


class MyArray(np.ndarray):
    r"""Minimal ``numpy.ndarray`` subclass to test subclass handling."""


class MyMaskedArray(np.ma.MaskedArray):
    r"""Minimal ``numpy.ma.MaskedArray`` subclass to test subclass
    handling."""
