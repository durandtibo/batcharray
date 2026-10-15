r"""Contain reusable input validation functions."""

from __future__ import annotations

__all__ = [
    "check_first_array",
    "check_flat_dict_of_arrays",
    "check_min_ndim",
    "check_same_keys",
    "check_split_sections",
]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola.iterator import dfs_iterate

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


def check_first_array(data: Any) -> np.ndarray:
    r"""Check that the first item of the data is an array and return it.

    Args:
        data: The data to check. It can be an array or a nested
            structure of arrays.

    Returns:
        The first array of the data, in depth-first order.

    Raises:
        ValueError: if the data is empty or its first item is not an array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils.validation import check_first_array
        >>> check_first_array({"a": np.ones((2, 3)), "b": np.zeros(2)}).shape
        (2, 3)

        ```
    """
    value = next(dfs_iterate(data), None)
    if not isinstance(value, np.ndarray):
        msg = "Expected data with at least one array, but the first item is not an array"
        raise ValueError(msg)  # noqa: TRY004
    return value


def check_flat_dict_of_arrays(data: Any) -> None:
    r"""Check that the data is a flat dictionary of arrays.

    Args:
        data: The data to check.

    Raises:
        TypeError: if the data is not a dictionary or if some of its
            values are not arrays.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils.validation import check_flat_dict_of_arrays
        >>> check_flat_dict_of_arrays({"a": np.ones((2, 3))})

        ```
    """
    if not isinstance(data, dict) or not all(isinstance(v, np.ndarray) for v in data.values()):
        msg = "Expected a flat dictionary of arrays (nested structures are not supported)"
        raise TypeError(msg)


def check_min_ndim(array: np.ndarray, min_ndim: int = 2) -> None:
    r"""Check that the array has at least the given number of dimensions.

    Args:
        array: The array to check.
        min_ndim: The minimum number of dimensions.

    Raises:
        ValueError: if the array has fewer than ``min_ndim`` dimensions.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils.validation import check_min_ndim
        >>> check_min_ndim(np.ones((2, 3)))

        ```
    """
    if array.ndim < min_ndim:
        msg = (
            f"array must have at least {min_ndim} dimensions (batch and sequence), "
            f"but got {array.ndim}"
        )
        raise ValueError(msg)


def check_same_keys(data: Sequence[dict[Hashable, Any]]) -> None:
    r"""Check that all the dictionaries have the same keys.

    Args:
        data: The dictionaries to check.

    Raises:
        ValueError: if some dictionaries have different keys.

    Example:
        ```pycon
        >>> from batcharray.utils.validation import check_same_keys
        >>> check_same_keys([{"a": 1, "b": 2}, {"b": 3, "a": 4}])

        ```
    """
    if not data:
        return
    keys = data[0].keys()
    for i, item in enumerate(data[1:], start=1):
        if item.keys() != keys:
            msg = (
                f"All dictionaries must have the same keys, but item 0 has {sorted(keys, key=str)} "
                f"and item {i} has {sorted(item.keys(), key=str)}"
            )
            raise ValueError(msg)


def check_split_sections(size: int, split_size_or_sections: int | Sequence[int]) -> None:
    r"""Check that the split size or sections are valid for an axis.

    Args:
        size: The size of the axis to split.
        split_size_or_sections: The size of a single chunk, or the
            sizes of each chunk.

    Raises:
        ValueError: if the split size is not positive, or if the
            sections are negative or do not sum to ``size``.

    Example:
        ```pycon
        >>> from batcharray.utils.validation import check_split_sections
        >>> check_split_sections(5, 2)
        >>> check_split_sections(5, [2, 3])

        ```
    """
    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            msg = f"split size must be positive, but got {split_size_or_sections}"
            raise ValueError(msg)
        return
    sections = np.asarray(split_size_or_sections, dtype=int)
    if (sections < 0).any() or sections.sum() != size:
        msg = (
            f"split sections {list(split_size_or_sections)} must be non-negative and sum "
            f"to the axis size ({size})"
        )
        raise ValueError(msg)
