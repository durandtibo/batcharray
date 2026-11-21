r"""Contain some functions to permute data in arrays."""

from __future__ import annotations

__all__ = [
    "flip_along_batch",
    "flip_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
]


import numpy as np

from batcharray.constants import BATCH_AXIS, SEQ_AXIS
from batcharray.validation import validate_indices, validate_min_ndim


def permute_along_batch(array: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    r"""Permute the array along the batch axis.

    Args:
        array: The array to split.
        permutation: The 1-D array containing the indices of the
            permutation. The shape should match the batch axis
            of the array.

    Returns:
        The array with permuted data along the batch axis.

    Raises:
        TypeError: if the permutation is not a ``numpy.ndarray``.
        ValueError: if the permutation is a boolean array, is not 1-D
            or its length does not match the batch axis of the array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import permute_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = permute_along_batch(array, np.array([2, 1, 3, 0, 4]))
        >>> out
        array([[4, 5],
               [2, 3],
               [6, 7],
               [0, 1],
               [8, 9]])

        ```
    """
    validate_indices(permutation, ndim=1, length=array.shape[BATCH_AXIS])
    return np.take(array, permutation, axis=BATCH_AXIS)


def permute_along_seq(array: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    r"""Permute the array along the sequence axis.

    Args:
        array: The array to split.
        permutation: The 1-D array containing the indices of the
            permutation. The shape should match the sequence axis
            of the array.

    Returns:
        The array with permuted data along the sequence axis.

    Raises:
        TypeError: if the permutation is not a ``numpy.ndarray``.
        ValueError: if the array has fewer than 2 dimensions, if the
            permutation is a boolean array, is not 1-D or its length
            does not match the sequence axis of the array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import permute_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = permute_along_seq(array, np.array([2, 1, 3, 0, 4]))
        >>> out
        array([[2, 1, 3, 0, 4],
               [7, 6, 8, 5, 9]])

        ```
    """
    validate_min_ndim(array)
    validate_indices(permutation, ndim=1, length=array.shape[SEQ_AXIS])
    return np.take(array, permutation, axis=SEQ_AXIS)


def shuffle_along_batch(
    array: np.ndarray, *, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    r"""Shuffle the array along the batch dimension.

    Args:
        array: The array to split.
        rng: A random number generator, a seed or ``None``. It is
            passed to ``numpy.random.default_rng``.

    Returns:
        The shuffled array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import shuffle_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = shuffle_along_batch(array)
        >>> out
        array([[...]])

        ```
    """
    rng = np.random.default_rng(rng)
    return permute_along_batch(
        array=array,
        permutation=rng.permutation(array.shape[BATCH_AXIS]),
    )


def shuffle_along_seq(
    array: np.ndarray, *, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    r"""Shuffle the array along the batch dimension.

    Args:
        array: The array to split.
        rng: A random number generator, a seed or ``None``. It is
            passed to ``numpy.random.default_rng``.

    Returns:
        The shuffled array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import shuffle_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = shuffle_along_seq(array)
        >>> out
        array([[...]])

        ```
    """
    rng = np.random.default_rng(rng)
    validate_min_ndim(array)
    return permute_along_seq(
        array=array,
        permutation=rng.permutation(array.shape[SEQ_AXIS]),
    )


def flip_along_batch(array: np.ndarray) -> np.ndarray:
    r"""Reverse the order of the elements along the batch axis.

    Args:
        array: The input array.

    Returns:
        The flipped array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import flip_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = flip_along_batch(array)
        >>> out
        array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]])

        ```
    """
    return np.flip(array, axis=BATCH_AXIS)


def flip_along_seq(array: np.ndarray) -> np.ndarray:
    r"""Reverse the order of the elements along the sequence axis.

    Args:
        array: The input array.

    Returns:
        The flipped array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import flip_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = flip_along_seq(array)
        >>> out
        array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]])

        ```
    """
    return np.flip(array, axis=SEQ_AXIS)
