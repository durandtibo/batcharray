r"""Contain some array indexing functions for nested data."""

from __future__ import annotations

__all__ = [
    "gather_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "masked_select_along_batch",
    "masked_select_along_seq",
]


from functools import partial
from typing import TYPE_CHECKING

from batcharray import array
from batcharray.utils import apply_to_arrays

if TYPE_CHECKING:
    import numpy as np

    from batcharray.utils import T


def index_select_along_batch(data: T, indices: np.ndarray) -> T:
    r"""Return the arrays which index the arrays along the batch axis
    using the entries in ``indices``.

    Args:
        data: The input data. Each item must be an array.
        indices: The 1-D array containing the indices to take.

    Returns:
        The indexed arrays along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import index_select_along_batch
        >>> arrays = {
        ...     "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": np.array([4, 3, 2, 1, 0]),
        ... }
        >>> out = index_select_along_batch(arrays, np.array([2, 4]))
        >>> out
        {'a': array([[4, 5], [8, 9]]), 'b': array([2, 0])}
        >>> out = index_select_along_batch(arrays, np.array([4, 3, 2, 1, 0]))
        >>> out
        {'a': array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]), 'b': array([0, 1, 2, 3, 4])}

        ```
    """
    return apply_to_arrays(data, partial(array.index_select_along_batch, indices=indices))


def index_select_along_seq(data: T, indices: np.ndarray) -> T:
    r"""Return the arrays which index the arrays along the sequence axis
    using the entries in ``indices``.

    Args:
        data: The input data. Each item must be an array.
        indices: The 1-D array containing the indices to take.

    Returns:
        The indexed arrays along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import index_select_along_seq
        >>> arrays = {
        ...     "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": np.array([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = index_select_along_seq(arrays, np.array([2, 4]))
        >>> out
        {'a': array([[2, 4], [7, 9]]), 'b': array([[2, 0]])}
        >>> out = index_select_along_seq(arrays, np.array([4, 3, 2, 1, 0]))
        >>> out
        {'a': array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]), 'b': array([[0, 1, 2, 3, 4]])}

        ```
    """
    return apply_to_arrays(data, partial(array.index_select_along_seq, indices=indices))


def masked_select_along_batch(data: T, mask: np.ndarray) -> T:
    r"""Return the arrays which index the arrays along the batch axis
    according to the boolean mask ``mask``.

    Args:
        data: The input data. Each item must be an array.
        mask: The 1-D array containing the binary mask to index with.

    Returns:
        The indexed arrays along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import masked_select_along_batch
        >>> arrays = {
        ...     "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
        ...     "b": np.array([4, 3, 2, 1, 0]),
        ... }
        >>> out = masked_select_along_batch(arrays, np.array([False, False, True, False, True]))
        >>> out
        {'a': array([[4, 5], [8, 9]]), 'b': array([2, 0])}

        ```
    """
    return apply_to_arrays(data, partial(array.masked_select_along_batch, mask=mask))


def masked_select_along_seq(data: T, mask: np.ndarray) -> T:
    r"""Return the arrays which index the arrays along the sequence axis
    according to the boolean mask ``mask``.

    Args:
        data: The input data. Each item must be an array.
        mask: The 1-D array containing the binary mask to index with.

    Returns:
        The indexed arrays along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import masked_select_along_seq
        >>> arrays = {
        ...     "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": np.array([[4, 3, 2, 1, 0]]),
        ... }
        >>> out = masked_select_along_seq(arrays, np.array([False, False, True, False, True]))
        >>> out
        {'a': array([[2, 4], [7, 9]]), 'b': array([[2, 0]])}

        ```
    """
    return apply_to_arrays(data, partial(array.masked_select_along_seq, mask=mask))


def gather_along_seq(data: T, indices: np.ndarray) -> T:
    r"""Gather different elements of the sequence axis for each sample of
    all the arrays.

    The output of each array is ``out[b, i] = array[b, indices[b, i]]``.
    The indices are one array shared by all the arrays.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays (dictionaries, lists, tuples).
        indices: The 2-D integer array of shape ``(batch_size, n)``
            containing the indices to take.

    Returns:
        The gathered data. The output data has the same structure as
            the input data.

    Raises:
        TypeError: if a leaf of the data is not an array.
        ValueError: if an array has fewer than 2 dimensions, if the
            indices are not a 2-D array, or if the batch size of the
            indices differs from the batch size of an array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import gather_along_seq
        >>> data = {
        ...     "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        ...     "b": np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
        ... }
        >>> out = gather_along_seq(data, np.array([[0, 4], [1, 3]]))
        >>> out
        {'a': array([[0, 4], [6, 8]]), 'b': array([[4, 0], [8, 6]])}

        ```
    """
    return apply_to_arrays(data, partial(array.gather_along_seq, indices=indices))
