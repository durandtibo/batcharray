r"""Contain input validation functions for arrays."""

from __future__ import annotations

__all__ = [
    "validate_min_ndim",
    "validate_split_sections",
]

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_min_ndim(array: np.ndarray, min_ndim: int = 2) -> None:
    r"""Check that the array has at least the given number of dimensions.

    Args:
        array: The array to check.
        min_ndim: The minimum number of dimensions.

    Raises:
        ValueError: if the array has fewer than ``min_ndim`` dimensions.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.validation import validate_min_ndim
        >>> validate_min_ndim(np.ones((2, 3)))

        ```
    """
    if array.ndim < min_ndim:
        msg = (
            f"array must have at least {min_ndim} dimensions (batch and sequence), "
            f"but got {array.ndim}"
        )
        raise ValueError(msg)


def validate_split_sections(size: int, split_size_or_sections: int | Sequence[int]) -> None:
    r"""Check that the split size or sections are valid for an axis.

    Args:
        size: The size of the axis to split.
        split_size_or_sections: The size of a single chunk (a Python
            or NumPy integer, not a boolean), or the sizes of each
            chunk.

    Raises:
        ValueError: if the split size is a boolean or is not positive,
            or if the sections are negative or do not sum to ``size``.

    Example:
        ```pycon
        >>> from batcharray.validation import validate_split_sections
        >>> validate_split_sections(5, 2)
        >>> validate_split_sections(5, [2, 3])

        ```
    """
    if isinstance(split_size_or_sections, Integral):
        if isinstance(split_size_or_sections, bool) or split_size_or_sections <= 0:
            msg = f"split size must be a positive integer, but got {split_size_or_sections}"
            raise ValueError(msg)
        return
    sections = np.asarray(split_size_or_sections, dtype=int)
    if (sections < 0).any() or sections.sum() != size:
        msg = (
            f"split sections {sections.tolist()} must be non-negative and sum "
            f"to the axis size ({size})"
        )
        raise ValueError(msg)
