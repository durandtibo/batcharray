r"""Contain input validation functions for nested data."""

from __future__ import annotations

__all__ = [
    "validate_first_array",
    "validate_same_keys",
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
