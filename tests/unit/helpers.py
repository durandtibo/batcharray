r"""Contain helpers shared across the unit tests."""

from __future__ import annotations

__all__ = ["SORT_KINDS", "FixedPermutationGenerator", "MyArray", "MyMaskedArray"]

import numpy as np

SORT_KINDS = ["quicksort", "mergesort", "heapsort", "stable"]


class MyArray(np.ndarray):
    r"""Minimal ``numpy.ndarray`` subclass to test subclass handling."""


class MyMaskedArray(np.ma.MaskedArray):
    r"""Minimal ``numpy.ma.MaskedArray`` subclass to test subclass
    handling."""


class FixedPermutationGenerator(np.random.Generator):
    r"""Random number generator that always returns the same permutation.

    Args:
        permutation: The permutation returned by ``permutation``.
    """

    def __init__(self, permutation: np.ndarray) -> None:
        super().__init__(np.random.PCG64(0))
        self._permutation = permutation

    def permutation(self, x: int | np.ndarray, axis: int = 0) -> np.ndarray:  # noqa: ARG002
        return self._permutation
