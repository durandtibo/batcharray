r"""Contain types used in the remaining of the package."""

from __future__ import annotations

__all__ = ["NestedArray", "SortKind", "T"]

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np

SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]

NestedArray: TypeAlias = np.ndarray | Mapping[Any, "NestedArray"] | Sequence["NestedArray"]
r"""An array or a nested structure (dictionary, list, tuple and their
subclasses) of arrays."""

T = TypeVar("T", bound=NestedArray)
r"""A nested array type: a function of type ``T -> T`` returns the same
structure as its input."""
