r"""Contain input validation functions for nested data."""

from __future__ import annotations

__all__ = [
    "validate_first_array",
    "validate_same_batch_size",
    "validate_same_keys",
    "validate_same_seq_size",
]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola.iterator import dfs_iterate

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


def validate_first_array(data: Any) -> np.ndarray:
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
        >>> from batcharray.validation import validate_first_array
        >>> validate_first_array({"a": np.ones((2, 3)), "b": np.zeros(2)}).shape
        (2, 3)

        ```
    """
    value = next(dfs_iterate(data), None)
    if not isinstance(value, np.ndarray):
        msg = "Expected data with at least one array, but the first item is not an array"
        raise ValueError(msg)  # noqa: TRY004
    return value


def validate_same_keys(data: Sequence[dict[Hashable, Any]]) -> None:
    r"""Check that all the dictionaries have the same keys.

    Args:
        data: The dictionaries to check.

    Raises:
        ValueError: if some dictionaries have different keys.

    Example:
        ```pycon
        >>> from batcharray.validation import validate_same_keys
        >>> validate_same_keys([{"a": 1, "b": 2}, {"b": 3, "a": 4}])

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


def validate_same_batch_size(data: Any) -> None:
    r"""Check that all the arrays have the same batch size.

    This check is opt-in: the functions of ``batcharray.nested`` do
    not call it, because some valid calls involve arrays with different
    batch sizes (e.g. ``sum_along_batch``).

    Note:
        This function assumes the batch axis is the first axis.

    Args:
        data: The data to check. It can be an array or a nested
            structure of arrays.

    Raises:
        TypeError: if an item of the data is not an array.
        ValueError: if some arrays have different batch sizes or if
            some arrays are 0-d.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.validation import validate_same_batch_size
        >>> validate_same_batch_size({"a": np.ones((2, 3)), "b": np.zeros(2)})

        ```
    """
    _validate_same_size(data, axis=0, name="batch")


def validate_same_seq_size(data: Any) -> None:
    r"""Check that all the arrays have the same sequence size.

    This check is opt-in: the functions of ``batcharray.nested`` do
    not call it.

    Note:
        This function assumes the sequence axis is the second axis.

    Args:
        data: The data to check. It can be an array or a nested
            structure of arrays.

    Raises:
        TypeError: if an item of the data is not an array.
        ValueError: if some arrays have different sequence sizes or if
            some arrays have fewer than 2 dimensions.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.validation import validate_same_seq_size
        >>> validate_same_seq_size({"a": np.ones((2, 3)), "b": np.zeros((4, 3))})

        ```
    """
    _validate_same_size(data, axis=1, name="sequence")


def _validate_same_size(data: Any, axis: int, name: str) -> None:
    r"""Check that all the arrays have the same size along an axis."""
    sizes: dict[int, int] = {}
    for i, array in enumerate(dfs_iterate(data)):
        if not isinstance(array, np.ndarray):
            msg = f"Expected an array, but item {i} has type {type(array).__qualname__}"
            raise TypeError(msg)
        if array.ndim <= axis:
            msg = f"Cannot get the {name} size of item {i}: it has {array.ndim} dimension(s)"
            raise ValueError(msg)
        sizes.setdefault(array.shape[axis], i)
    if len(sizes) > 1:
        msg = f"All the arrays must have the same {name} size, but got {sorted(sizes)}"
        raise ValueError(msg)
