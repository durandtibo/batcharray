r"""Contain some array joining functions for nested data."""

from __future__ import annotations

__all__ = [
    "concatenate_along_batch",
    "concatenate_along_seq",
    "repeat_along_batch",
    "repeat_along_seq",
    "tile_along_seq",
]

from functools import partial
from typing import TYPE_CHECKING, Any

from coola.recursive import recursive_apply

from batcharray import array as ba
from batcharray.nested._structure import concatenate_structures

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike


def concatenate_along_batch(
    data: Sequence[Any],
    *,
    dtype: DTypeLike | None = None,
) -> Any:
    r"""Concatenate the given arrays in the batch axis.

    All arrays must either have the same data type and shape (except
    in the concatenating axis) or be empty.

    Note:
        This function assumes the batch axis is the first
            axis of the arrays. All the arrays should have the
            same batch size.

    Args:
        data: The input data to concatenate. It can be arrays or nested
            structures of arrays (dictionaries, lists, tuples). All the
            items must have the same structure (same dictionary keys
            and same sequence lengths).
        dtype: If provided, the destination array will have this
            data type.

    Returns:
        The concatenated arrays along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import concatenate_along_batch
        >>> data = [
        ...     {
        ...         "a": np.array([[0, 1, 2], [4, 5, 6]]),
        ...         "b": np.array([[10, 11, 12], [13, 14, 15]]),
        ...     },
        ...     {"a": np.array([[7, 8, 9]]), "b": np.array([[17, 18, 19]])},
        ... ]
        >>> out = concatenate_along_batch(data)
        >>> out
        {'a': array([[0, 1, 2], [4, 5, 6], [7, 8, 9]]),
         'b': array([[10, 11, 12], [13, 14, 15], [17, 18, 19]])}

        ```
    """
    if not data:
        return {}
    return concatenate_structures(
        data, lambda arrays: ba.concatenate_along_batch(list(arrays), dtype=dtype)
    )


def concatenate_along_seq(
    data: Sequence[Any],
    *,
    dtype: DTypeLike | None = None,
) -> Any:
    r"""Concatenate the given arrays in the sequence axis.

    All arrays must either have the same data type and shape (except
    in the concatenating axis) or be empty.

    Note:
        This function assumes the sequence axis is the second
            axis of the arrays. All the arrays should have the
            same sequence size.

    Args:
        data: The input data to concatenate. It can be arrays or nested
            structures of arrays (dictionaries, lists, tuples). All the
            items must have the same structure (same dictionary keys
            and same sequence lengths).
        dtype: If provided, the destination array will have this
            data type.

    Returns:
        The concatenated arrays along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import concatenate_along_seq
        >>> data = [
        ...     {
        ...         "a": np.array([[0, 1, 2], [4, 5, 6]]),
        ...         "b": np.array([[10, 11, 12], [13, 14, 15]]),
        ...     },
        ...     {"a": np.array([[7], [8]]), "b": np.array([[17], [18]])},
        ... ]
        >>> out = concatenate_along_seq(data)
        >>> out
        {'a': array([[0, 1, 2, 7], [4, 5, 6, 8]]),
         'b': array([[10, 11, 12, 17], [13, 14, 15, 18]])}

        ```
    """
    if not data:
        return {}
    return concatenate_structures(
        data, lambda arrays: ba.concatenate_along_seq(list(arrays), dtype=dtype)
    )


def tile_along_seq(data: Any, reps: int) -> Any:
    r"""Repeat all the arrays along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis of the arrays. All the arrays should have the
            same sequence size.

    Args:
        data: The input data. Each item must be an array.
        reps: The number of repetitions data along the
            sequence axis.

    Returns:
        The arrays repeated along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import tile_along_seq
        >>> data = {
        ...     "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": np.array([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = tile_along_seq(data, 2)
        >>> out
        {'a': array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
         'b': array([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]])}

        ```
    """
    return recursive_apply(data, partial(ba.tile_along_seq, reps=reps))


def repeat_along_batch(data: Any, repeats: int) -> Any:
    r"""Repeat each element of all the arrays along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis of the arrays.

    Args:
        data: The input data. Each item must be an array.
        repeats: The number of repetitions of each element.

    Returns:
        The repeated arrays. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import repeat_along_batch
        >>> data = {"a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), "b": np.array([2, 1, 0, -1, -2])}
        >>> out = repeat_along_batch(data, 2)
        >>> out
        {'a': array([[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]]), 'b': array([ 2,  2,  1,  1,  0,  0, -1, -1, -2, -2])}

        ```
    """
    return recursive_apply(data, partial(ba.repeat_along_batch, repeats=repeats))


def repeat_along_seq(data: Any, repeats: int) -> Any:
    r"""Repeat each element of all the arrays along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis of the arrays.

    Args:
        data: The input data. Each item must be an array.
        repeats: The number of repetitions of each element.

    Returns:
        The repeated arrays. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import repeat_along_seq
        >>> data = {"a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), "b": np.array([[4, 3, 2, 1, 0], [0, 1, 2, 3, 4]])}
        >>> out = repeat_along_seq(data, 2)
        >>> out
        {'a': array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]]), 'b': array([[4, 4, 3, 3, 2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])}

        ```
    """
    return recursive_apply(data, partial(ba.repeat_along_seq, repeats=repeats))
