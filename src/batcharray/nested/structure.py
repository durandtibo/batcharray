r"""Contain helpers to split or join nested structures of arrays."""

from __future__ import annotations

__all__ = ["concatenate_structures", "split_structure"]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola.recursive import recursive_apply

from batcharray.constants import BATCH_AXIS
from batcharray.validation import (
    validate_same_batch_size,
    validate_same_keys,
    validate_same_seq_size,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _Pieces:
    r"""Wrap the pieces of an array so they are treated as a leaf."""

    __slots__ = ("items",)

    def __init__(self, items: Sequence[np.ndarray]) -> None:
        self.items = list(items)


def split_structure(
    data: Any, fn: Callable[[np.ndarray], Sequence[np.ndarray]], *, axis: int
) -> list[Any]:
    r"""Split a nested structure of arrays into a list of structures.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays.
        fn: The function used to split each array into pieces.
        axis: The axis along which ``fn`` splits the arrays. All the
            arrays must have the same size along this axis, otherwise
            the pieces of different arrays are not aligned.

    Returns:
        A list of structures with the same structure as the input.
            The ``i``-th structure contains the ``i``-th piece of each
            array.

    Raises:
        TypeError: if an item of the data is not an array.
        ValueError: if the arrays have different sizes along the axis.
    """
    if axis == BATCH_AXIS:
        validate_same_batch_size(data)
    else:
        validate_same_seq_size(data)

    pieces = recursive_apply(data, lambda array: _Pieces(fn(array)))
    sizes: set[int] = set()
    recursive_apply(pieces, lambda p: sizes.add(len(p.items)))
    num_pieces = sizes.pop() if sizes else 0
    return [recursive_apply(pieces, lambda p, i=i: p.items[i]) for i in range(num_pieces)]


def concatenate_structures(items: Sequence[Any], fn: Callable[[Sequence[np.ndarray]], Any]) -> Any:
    r"""Concatenate nested structures of arrays with the same layout.

    The containers of the output are the ones of the first item: a
    list and a tuple can be combined (the output is of the type of the
    first item), as can a dictionary and an ordered dictionary.

    Args:
        items: The structures to concatenate. They must have the same
            layout (same dictionary keys, same sequence lengths).
        fn: The function used to concatenate a sequence of arrays.

    Returns:
        The concatenated structure.

    Raises:
        TypeError: if the items contain something that is not an
            array, or if an item is an array and another is a
            container.
        ValueError: if no item is given, if a mapping is combined with
            a sequence, if the dictionaries have different keys or if
            the sequences have different lengths.
    """
    if len(items) == 0:
        msg = "Cannot concatenate an empty sequence of structures"
        raise ValueError(msg)
    first = items[0]
    if isinstance(first, np.ndarray):
        if not all(isinstance(item, np.ndarray) for item in items):
            msg = "Expected all the items to be arrays"
            raise TypeError(msg)
        return fn(items)
    if isinstance(first, dict):
        return _concatenate_dicts(items, fn)
    if isinstance(first, (list, tuple)):
        return _concatenate_sequences(items, fn)
    msg = f"Expected an array or a nested structure of arrays, but got {type(first).__qualname__}"
    raise TypeError(msg)


def _concatenate_dicts(items: Sequence[Any], fn: Callable[[Sequence[np.ndarray]], Any]) -> Any:
    r"""Concatenate dictionaries of nested structures of arrays."""
    if any(isinstance(item, (list, tuple)) for item in items):
        msg = "Cannot concatenate a dictionary with a sequence"
        raise ValueError(msg)
    if not all(isinstance(item, dict) for item in items):
        msg = "Expected all the items to be dictionaries"
        raise TypeError(msg)
    validate_same_keys(items)
    return type(items[0])(
        {key: concatenate_structures([d[key] for d in items], fn) for key in items[0]}
    )


def _concatenate_sequences(items: Sequence[Any], fn: Callable[[Sequence[np.ndarray]], Any]) -> Any:
    r"""Concatenate lists or tuples of nested structures of arrays."""
    if any(isinstance(item, dict) for item in items):
        msg = "Cannot concatenate a dictionary with a sequence"
        raise ValueError(msg)
    if not all(isinstance(item, (list, tuple)) for item in items):
        msg = "Expected all the items to be lists or tuples"
        raise TypeError(msg)
    first = items[0]
    if any(len(item) != len(first) for item in items):
        msg = "All the sequences must have the same length"
        raise ValueError(msg)
    values = [concatenate_structures([item[i] for item in items], fn) for i in range(len(first))]
    return type(first)(values)
