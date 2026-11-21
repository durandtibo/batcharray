r"""Contain utilities shared by the ``array`` and ``nested``
packages."""

from __future__ import annotations

__all__ = [
    "NestedArray",
    "SortKind",
    "T",
    "apply_to_arrays",
    "combine_arrays",
    "get_namespace",
    "get_size",
    "split_arrays",
]

from batcharray.utils.namespace import get_namespace
from batcharray.utils.structure import apply_to_arrays, combine_arrays, get_size, split_arrays
from batcharray.utils.types import NestedArray, SortKind, T
