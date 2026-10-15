r"""Contain types used in the remaining of the package."""

from __future__ import annotations

__all__ = ["SORT_KINDS", "ArrayT", "SortKind"]

from typing import Literal, TypeVar, get_args

import numpy as np

SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
SORT_KINDS: tuple[SortKind, ...] = get_args(SortKind)

ArrayT = TypeVar("ArrayT", bound=np.ndarray)
