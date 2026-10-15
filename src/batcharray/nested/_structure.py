r"""Contain private helpers to split or join nested structures of
arrays."""

from __future__ import annotations

__all__ = ["concatenate_structures", "split_structure"]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola.recursive import recursive_apply

from batcharray.validation import validate_same_keys

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _Pieces:
    r"""Wrap the pieces of an array so they are treated as a leaf."""

    __slots__ = ("items",)

    def __init__(self, items: Sequence[np.ndarray]) -> None:
        self.items = list(items)


def split_structure(data: Any, fn: Callable[[np.ndarray], Sequence[np.ndarray]]) -> list[Any]:
    r"""Split a nested structure of arrays into a list of structures.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays.
        fn: The function used to split each array into pieces.

    Returns:
        A list of structures with the same structure as the input.
            The ``i``-th structure contains the ``i``-th piece of each
            array.

    Raises:
        TypeError: if an item of the data is not an array.
        ValueError: if the arrays are not split into the same number
            of pieces.
    """

    def _split(array: Any) -> _Pieces:
        if not isinstance(array, np.ndarray):
            msg = f"Expected an array, but got an object of type {type(array).__qualname__}"
            raise TypeError(msg)
        return _Pieces(fn(array))

    pieces = recursive_apply(data, _split)
    sizes: set[int] = set()
    recursive_apply(pieces, lambda p: sizes.add(len(p.items)))
    if len(sizes) > 1:
        msg = (
            f"All the arrays must be split into the same number of pieces, but got {sorted(sizes)}"
        )
        raise ValueError(msg)
    num_pieces = sizes.pop() if sizes else 0
    return [recursive_apply(pieces, lambda p, i=i: p.items[i]) for i in range(num_pieces)]


def concatenate_structures(items: Sequence[Any], fn: Callable[[Sequence[np.ndarray]], Any]) -> Any:
    r"""Concatenate nested structures of arrays with the same structure.

    Args:
        items: The structures to concatenate. They must have the same
            structure (same dictionary keys, same sequence lengths).
        fn: The function used to concatenate a sequence of arrays.

    Returns:
        The concatenated structure.

    Raises:
        TypeError: if the items have different types or contain
            something that is not an array.
        ValueError: if the dictionaries have different keys or the
            sequences have different lengths.
    """
    first = items[0]
    if isinstance(first, np.ndarray):
        if not all(isinstance(item, np.ndarray) for item in items):
            msg = "Expected all the items to be arrays"
            raise TypeError(msg)
        return fn(items)
    if isinstance(first, dict):
        if not all(isinstance(item, dict) for item in items):
            msg = "Expected all the items to be dictionaries"
            raise TypeError(msg)
        validate_same_keys(items)
        return type(first)(
            {key: concatenate_structures([d[key] for d in items], fn) for key in first}
        )
    if isinstance(first, (list, tuple)):
        if not all(isinstance(item, type(first)) for item in items):
            msg = f"Expected all the items to be of type {type(first).__qualname__}"
            raise TypeError(msg)
        if any(len(item) != len(first) for item in items):
            msg = "All the sequences must have the same length"
            raise ValueError(msg)
        values = [
            concatenate_structures([item[i] for item in items], fn) for i in range(len(first))
        ]
        return type(first)(values)
    msg = f"Expected an array or a nested structure of arrays, but got {type(first).__qualname__}"
    raise TypeError(msg)
