r"""Contain the helpers that traverse nested structures of arrays.

Every function of ``batcharray.nested`` goes through one of the three
helpers ``apply_to_arrays``, ``split_arrays`` or ``combine_arrays``, so
the rules on the leaves and on the containers are implemented once.
"""

from __future__ import annotations

__all__ = ["apply_to_arrays", "combine_arrays", "get_size", "split_arrays"]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola.iterator import dfs_iterate
from coola.recursive import recursive_apply, recursive_zip_apply

from batcharray.constants import BATCH_AXIS, SEQ_AXIS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcharray.utils.types import NestedArray, T


class _Pieces:
    r"""Wrap the pieces of an array so they are treated as a leaf."""

    __slots__ = ("items",)

    def __init__(self, items: Sequence[np.ndarray]) -> None:
        self.items = list(items)


def apply_to_arrays(data: T, fn: Callable[[np.ndarray], Any]) -> T:
    r"""Apply a function to every array of a nested structure.

    The output has the same structure as the input: the containers
    (``dict``, ``list``, ``tuple`` and their subclasses) are kept and
    every array is replaced by the output of the function. This is also
    the tool to apply a function that the package does not wrap, like
    a pointwise function.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays.
        fn: The function applied to each array.

    Returns:
        The data with every array replaced by the output of ``fn``.

    Raises:
        TypeError: if a leaf of the data is not an array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils import apply_to_arrays
        >>> apply_to_arrays({"a": np.array([1.0, 4.0]), "b": [np.array([9.0])]}, np.sqrt)
        {'a': array([1., 2.]), 'b': [array([3.])]}

        ```
    """

    def _apply(leaf: Any) -> Any:
        _validate_array(leaf)
        return fn(leaf)

    return recursive_apply(data, _apply)


def split_arrays(
    data: T, fn: Callable[[np.ndarray], Sequence[np.ndarray]], *, axis: int
) -> list[T]:
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
        TypeError: if a leaf of the data is not an array.
        ValueError: if the data has no array, or if the arrays have
            different sizes along the axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils import split_arrays
        >>> split_arrays(
        ...     {"a": np.arange(4), "b": [np.arange(4) + 10]},
        ...     lambda array: np.array_split(array, 2),
        ...     axis=0,
        ... )
        [{'a': array([0, 1]), 'b': [array([10, 11])]},
         {'a': array([2, 3]), 'b': [array([12, 13])]}]

        ```
    """
    num_pieces = 0

    def _split(array: np.ndarray) -> _Pieces:
        nonlocal num_pieces
        pieces = _Pieces(fn(array))
        num_pieces = len(pieces.items)
        return pieces

    get_size(data, axis=axis)
    pieces = recursive_apply(data, _split)
    return [recursive_apply(pieces, lambda p, i=i: p.items[i]) for i in range(num_pieces)]


def combine_arrays(items: Sequence[T], fn: Callable[[Sequence[np.ndarray]], Any]) -> T:
    r"""Combine nested structures of arrays that have the same layout.

    The structures are traversed in parallel, and ``fn`` receives the
    arrays found at the same position in every structure. The
    containers of the output are the ones of the first item: a list and
    a tuple can be combined (the output has the type of the first item),
    as can a dictionary and an ordered dictionary.

    Args:
        items: The structures to combine. They must have the same
            layout (same dictionary keys, same sequence lengths).
        fn: The function used to combine the arrays found at the same
            position.

    Returns:
        The combined structure.

    Raises:
        TypeError: if a leaf of the items is not an array.
        ValueError: if no item is given, if the items have different
            layouts (dictionary keys, sequence lengths) or if a mapping
            is combined with a sequence.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils import combine_arrays
        >>> combine_arrays(
        ...     [{"a": np.array([1])}, {"a": np.array([2])}],
        ...     np.concatenate,
        ... )
        {'a': array([1, 2])}

        ```
    """

    def _combine(leaves: list[Any]) -> Any:
        for leaf in leaves:
            _validate_array(leaf)
        return fn(leaves)

    return recursive_zip_apply(items, _combine)


def get_size(data: NestedArray, axis: int) -> int:
    r"""Return the size of the axis shared by all the arrays.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays.
        axis: The axis. ``BATCH_AXIS`` is the batch axis and
            ``SEQ_AXIS`` is the sequence axis.

    Returns:
        The size of the axis.

    Raises:
        TypeError: if a leaf of the data is not an array.
        ValueError: if the data has no array, if an array has fewer
            dimensions than needed, or if the arrays have different
            sizes along the axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.constants import BATCH_AXIS
        >>> from batcharray.utils import get_size
        >>> get_size({"a": np.ones((2, 3)), "b": np.zeros(2)}, axis=BATCH_AXIS)
        2

        ```
    """
    name = {BATCH_AXIS: "batch", SEQ_AXIS: "sequence"}.get(axis, f"axis {axis}")
    sizes: dict[int, int] = {}
    for i, leaf in enumerate(dfs_iterate(data)):
        _validate_array(leaf)
        if leaf.ndim <= axis:
            msg = f"Cannot get the {name} size of item {i}: it has {leaf.ndim} dimension(s)"
            raise ValueError(msg)
        sizes.setdefault(leaf.shape[axis], i)
    if not sizes:
        msg = "Expected data with at least one array, but got none"
        raise ValueError(msg)
    if len(sizes) > 1:
        msg = f"All the arrays must have the same {name} size, but got {sorted(sizes)}"
        raise ValueError(msg)
    return next(iter(sizes))


def _validate_array(leaf: Any) -> None:
    r"""Check that a leaf is an array.

    Args:
        leaf: The leaf to check.

    Raises:
        TypeError: if the leaf is not an array.
    """
    if not isinstance(leaf, np.ndarray):
        msg = f"Expected an array, but got an object of type {type(leaf).__qualname__}"
        raise TypeError(msg)
