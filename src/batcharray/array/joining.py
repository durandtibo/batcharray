r"""Contain some joining functions for arrays."""

from __future__ import annotations

__all__ = [
    "concatenate_along_batch",
    "concatenate_along_seq",
    "repeat_along_batch",
    "repeat_along_seq",
    "tile_along_seq",
]


from typing import TYPE_CHECKING

import numpy as np

from batcharray import computation as cmpt
from batcharray.constants import BATCH_AXIS, SEQ_AXIS
from batcharray.validation import validate_min_ndim

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def concatenate_along_batch(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    dtype: DTypeLike | None = None,
) -> np.ndarray:
    r"""Concatenate the given arrays in the batch axis.

    All arrays must either have the same data type and shape (except
    in the concatenating axis) or be empty.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        arrays: The arrays to concatenate.
        dtype: If provided, the destination array will have this
            data type.

    Returns:
        The concatenated arrays along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import concatenate_along_batch
        >>> arrays = [
        ...     np.array([[0, 1, 2], [4, 5, 6]]),
        ...     np.array([[10, 11, 12], [13, 14, 15]]),
        ... ]
        >>> out = concatenate_along_batch(arrays)
        >>> out
        array([[ 0,  1,  2],
               [ 4,  5,  6],
               [10, 11, 12],
               [13, 14, 15]])

        ```
    """
    return cmpt.concatenate(arrays, axis=BATCH_AXIS, dtype=dtype)


def concatenate_along_seq(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    dtype: DTypeLike | None = None,
) -> np.ndarray:
    r"""Concatenate the given arrays in the sequence axis.

    All arrays must either have the same data type and shape (except
    in the concatenating axis) or be empty.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        arrays: The arrays to concatenate.
        dtype: If provided, the destination array will have this
            data type.

    Returns:
        The concatenated arrays along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import concatenate_along_seq
        >>> arrays = [
        ...     np.array([[0, 1, 2], [4, 5, 6]]),
        ...     np.array([[10, 11], [12, 13]]),
        ... ]
        >>> out = concatenate_along_seq(arrays)
        >>> out
        array([[ 0,  1,  2, 10, 11],
               [ 4,  5,  6, 12, 13]])

        ```
    """
    return cmpt.concatenate(arrays, axis=SEQ_AXIS, dtype=dtype)


def tile_along_seq(array: np.ndarray, reps: int) -> np.ndarray:
    r"""Construct an array by repeating the input array along the
    sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        reps: The number of repetitions data along the
            sequence axis.

    Returns:
        A new array with the data repeated along the sequence
            axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import tile_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = tile_along_seq(array, 2)
        >>> out
        array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]])

        ```
    """
    validate_min_ndim(array)
    repeats = [1] * array.ndim
    repeats[1] = reps
    return np.tile(array, repeats)


def repeat_along_batch(array: np.ndarray, repeats: int) -> np.ndarray:
    r"""Repeat each element along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        repeats: The number of repetitions of each element.

    Returns:
        The repeated array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import repeat_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = repeat_along_batch(array, 2)
        >>> out
        array([[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]])

        ```
    """
    return np.repeat(array, repeats, axis=BATCH_AXIS)


def repeat_along_seq(array: np.ndarray, repeats: int) -> np.ndarray:
    r"""Repeat each element along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        repeats: The number of repetitions of each element.

    Returns:
        The repeated array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import repeat_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = repeat_along_seq(array, 2)
        >>> out
        array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]])

        ```
    """
    validate_min_ndim(array)
    return np.repeat(array, repeats, axis=SEQ_AXIS)
