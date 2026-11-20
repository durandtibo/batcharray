r"""Contain input validation functions for nested data."""

from __future__ import annotations

__all__ = ["validate_same_keys"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


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
