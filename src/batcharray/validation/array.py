r"""Contain input validation functions for arrays."""

from __future__ import annotations

__all__ = [
    "validate_indices",
    "validate_mask",
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


def validate_indices(indices: np.ndarray, *, ndim: int, length: int | None = None) -> None:
    r"""Check that the indices are an integer array with the given rank.

    A boolean array is rejected because NumPy reads it as the integers
    0 and 1. The values of the indices (bounds, negative values, float
    dtype) are checked by NumPy.

    Args:
        indices: The indices to check.
        ndim: The expected number of dimensions of the indices.
        length: The expected size of the first axis of the indices.
            ``None`` means that the size is not checked.

    Raises:
        TypeError: if the indices are not a ``numpy.ndarray``.
        ValueError: if the indices are a boolean array, if they do not
            have ``ndim`` dimensions, or if the size of their first
            axis is not ``length``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.validation import validate_indices
        >>> validate_indices(np.array([2, 0, 1]), ndim=1)
        >>> validate_indices(np.array([2, 0, 1]), ndim=1, length=3)

        ```
    """
    if not isinstance(indices, np.ndarray):
        msg = f"indices must be a numpy.ndarray, but got an object of type {type(indices).__name__}"
        raise TypeError(msg)
    if indices.dtype == np.bool_:
        msg = "indices must be an integer array, but got a boolean array"
        raise ValueError(msg)
    if indices.ndim != ndim:
        msg = f"indices must be a {ndim}-D array, but got {indices.ndim} dimension(s)"
        raise ValueError(msg)
    if length is not None and indices.shape[0] != length:
        msg = (
            f"indices must have {length} elements along the first axis, but got {indices.shape[0]}"
        )
        raise ValueError(msg)


def validate_mask(mask: np.ndarray) -> None:
    r"""Check that the mask is a 1-D boolean array.

    An integer array would be read as indices by NumPy, and a 2-D mask
    would index several axes.

    Args:
        mask: The mask to check.

    Raises:
        TypeError: if the mask is not a ``numpy.ndarray``.
        ValueError: if the mask is not a boolean array or is not 1-D.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.validation import validate_mask
        >>> validate_mask(np.array([True, False, True]))

        ```
    """
    if not isinstance(mask, np.ndarray):
        msg = f"mask must be a numpy.ndarray, but got an object of type {type(mask).__name__}"
        raise TypeError(msg)
    if mask.dtype != np.bool_:
        msg = f"mask must be a boolean array, but got dtype {mask.dtype}"
        raise ValueError(msg)
    if mask.ndim != 1:
        msg = f"mask must be a 1-D array, but got {mask.ndim} dimension(s)"
        raise ValueError(msg)
