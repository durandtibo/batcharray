r"""Contain types used in the remaining of the package."""

from __future__ import annotations

__all__ = ["SortKind"]

from typing import Literal

SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
