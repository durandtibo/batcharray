r"""Contain some indexing functions for arrays."""

from __future__ import annotations

__all__ = [
    "gather_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "masked_select_along_batch",
    "masked_select_along_seq",
]

import numpy as np

from batcharray.constants import BATCH_AXIS, SEQ_AXIS
from batcharray.validation import validate_indices, validate_mask, validate_min_ndim


def index_select_along_batch(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    r"""Return a new array which indexes the input array along the batch
    axis using the entries in ``indices``.

    Note:
        This function assumes the batch axis is the first axis.

    Args:
        array: The input array.
        indices: The 1-D array containing the indices to take.

    Returns:
        The indexed array along the batch axis.

    Raises:
        TypeError: if the indices are not a ``numpy.ndarray``.
        ValueError: if the indices are not a 1-D array or are a
            boolean array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import index_select_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = index_select_along_batch(array, np.array([2, 4]))
        >>> out
        array([[4, 5],
               [8, 9]])
        >>> out = index_select_along_batch(array, np.array([4, 3, 2, 1, 0]))
        >>> out
        array([[8, 9],
               [6, 7],
               [4, 5],
               [2, 3],
               [0, 1]])

        ```
    """
    validate_indices(indices, ndim=1)
    return np.take(array, indices, axis=BATCH_AXIS)


def index_select_along_seq(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    r"""Return a new array which indexes the input array along the
    sequence axis using the entries in ``indices``.

    Note:
        This function assumes the sequence axis is the second axis.

    Args:
        array: The input array.
        indices: The 1-D array containing the indices to take. Use
            ``gather_along_seq`` to take different indices for each
            sample.

    Returns:
        The indexed array along the sequence axis.

    Raises:
        TypeError: if the indices are not a ``numpy.ndarray``.
        ValueError: if the indices are not a 1-D array or are a
            boolean array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import index_select_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = index_select_along_seq(array, np.array([2, 4]))
        >>> out
        array([[2, 4],
               [7, 9]])
        >>> out = index_select_along_seq(array, np.array([4, 3, 2, 1, 0]))
        >>> out
        array([[4, 3, 2, 1, 0],
               [9, 8, 7, 6, 5]])

        ```
    """
    validate_indices(indices, ndim=1)
    return np.take(array, indices, axis=SEQ_AXIS)


def masked_select_along_batch(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r"""Return a new array which indexes the input array along the batch
    axis according to the boolean mask ``mask``.

    Note:
        This function assumes the batch axis is the first axis.

    Args:
        array: The input array.
        mask: The 1-D boolean array containing the mask to index with.

    Returns:
        The indexed array along the batch axis.

    Raises:
        TypeError: if the mask is not a ``numpy.ndarray``.
        ValueError: if the mask is not a 1-D boolean array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import masked_select_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = masked_select_along_batch(array, np.array([False, False, True, False, True]))
        >>> out
        array([[4, 5], [8, 9]])

        ```
    """
    validate_mask(mask)
    return array[mask]


def masked_select_along_seq(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r"""Return a new array which indexes the input array along the
    sequence axis according to the boolean mask ``mask``.

    Note:
        This function assumes the sequence axis is the second axis.

    Args:
        array: The input array.
        mask: The 1-D boolean array containing the mask to index with.

    Returns:
        The indexed array along the sequence axis.

    Raises:
        TypeError: if the mask is not a ``numpy.ndarray``.
        ValueError: if the mask is not a 1-D boolean array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import masked_select_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = masked_select_along_seq(array, np.array([False, False, True, False, True]))
        >>> out
        array([[2, 4], [7, 9]])

        ```
    """
    validate_mask(mask)
    return array[:, mask]


def gather_along_seq(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    r"""Gather different elements of the sequence axis for each sample.

    The output is ``out[b, i] = array[b, indices[b, i]]``. It is
    ``numpy.take_along_axis`` with the indices reshaped over the
    feature axes. There is no ``gather_along_batch``: a per-sample
    gather along the batch axis has no meaning, and
    ``index_select_along_batch`` is the operation of that axis.

    Note:
        This function assumes the sequence axis is the second axis.
        A typical use is the last valid step of a padded batch:
        ``gather_along_seq(x, (lengths - 1)[:, None])``.

    Args:
        array: The input array, with at least 2 dimensions.
        indices: The 2-D integer array of shape ``(batch_size, n)``
            containing the indices to take. A negative index counts
            from the end.

    Returns:
        The gathered array of shape ``(batch_size, n, *array.shape[2:])``.

    Raises:
        TypeError: if the indices are not a ``numpy.ndarray``.
        ValueError: if the array has fewer than 2 dimensions, if the
            indices are not a 2-D array or are a boolean array, or if
            the batch size of the indices differs from the batch size
            of the array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import gather_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = gather_along_seq(array, np.array([[0, 4], [1, 3]]))
        >>> out
        array([[0, 4],
               [6, 8]])

        ```
    """
    validate_min_ndim(array)
    validate_indices(indices, ndim=2, length=array.shape[0])
    indices = indices.reshape(*indices.shape, *([1] * (array.ndim - 2)))
    return np.take_along_axis(array, indices, axis=SEQ_AXIS)
