r"""Contain some indexing functions for arrays."""

from __future__ import annotations

__all__ = [
    "chunk_along_batch",
    "chunk_along_seq",
    "select_along_batch",
    "select_along_seq",
]


import numpy as np

from batcharray.constants import BATCH_AXIS, SEQ_AXIS


def chunk_along_batch(array: np.ndarray, chunks: int) -> list[np.ndarray]:
    r"""Split the array into chunks along the batch axis.

    Each chunk is a view of the input array.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The array to split.
        chunks: Number of chunks to return.

    Returns:
        The array chunks.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.array import chunk_along_batch
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> outputs = chunk_along_batch(array, chunks=3)
    >>> outputs
    [array([[0, 1], [2, 3]]), array([[4, 5], [6, 7]]), array([[8, 9]])]

    ```
    """
    return np.array_split(array, indices_or_sections=chunks, axis=BATCH_AXIS)


def chunk_along_seq(array: np.ndarray, chunks: int) -> list[np.ndarray]:
    r"""Split the array into chunks along the sequence axis.

    Each chunk is a view of the input array.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The array to split.
        chunks: Number of chunks to return.

    Returns:
        The array chunks.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.array import chunk_along_seq
    >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    >>> outputs = chunk_along_seq(array, chunks=3)
    >>> outputs
    [array([[0, 1], [5, 6]]), array([[2, 3], [7, 8]]), array([[4], [9]])]

    ```
    """
    return np.array_split(array, indices_or_sections=chunks, axis=SEQ_AXIS)


def select_along_batch(array: np.ndarray, index: int) -> np.ndarray:
    r"""Slice the input array along the batch axis at the given index.

    This function returns a view of the original array with the batch axis removed.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        index: The index to select with.

    Returns:
        The sliced array along the batch axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.array import select_along_batch
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = select_along_batch(array, index=2)
    >>> out
    array([4, 5])

    ```
    """
    return array[index]


def select_along_seq(array: np.ndarray, index: int) -> np.ndarray:
    r"""Slice the input array along the sequence axis at the given index.

    This function returns a view of the original array with the sequence axis removed.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        index: The index to select with.

    Returns:
        The sliced array along the sequence axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.array import select_along_seq
    >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    >>> out = select_along_seq(array, index=2)
    >>> out
    array([2, 7])

    ```
    """
    return array[:, index]
