r"""Contain functions to get the batch size and the sequence length of
nested data."""

from __future__ import annotations

__all__ = ["batch_size", "seq_len"]

from typing import TYPE_CHECKING

from batcharray.constants import BATCH_AXIS, SEQ_AXIS
from batcharray.utils import get_size

if TYPE_CHECKING:
    from batcharray.utils import NestedArray


def batch_size(data: NestedArray) -> int:
    r"""Return the batch size shared by all the arrays.

    Note:
        This function assumes the batch axis is the first
            axis of the arrays.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays (dictionaries, lists, tuples).

    Returns:
        The batch size.

    Raises:
        TypeError: if a leaf of the data is not an array.
        ValueError: if the data has no array, if an array is a 0-d
            array, or if the arrays have different batch sizes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import batch_size
        >>> batch_size({"a": np.ones((2, 3)), "b": np.zeros(2)})
        2

        ```
    """
    return get_size(data, axis=BATCH_AXIS)


def seq_len(data: NestedArray) -> int:
    r"""Return the sequence length shared by all the arrays.

    Note:
        This function assumes the sequence axis is the second
            axis of the arrays.

    Args:
        data: The input data. It can be an array or a nested
            structure of arrays (dictionaries, lists, tuples).

    Returns:
        The sequence length.

    Raises:
        TypeError: if a leaf of the data is not an array.
        ValueError: if the data has no array, if an array has fewer
            than 2 dimensions, or if the arrays have different
            sequence lengths.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import seq_len
        >>> seq_len({"a": np.ones((2, 3)), "b": np.zeros((4, 3, 5))})
        3

        ```
    """
    return get_size(data, axis=SEQ_AXIS)
