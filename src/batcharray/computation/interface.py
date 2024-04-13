r"""Contain public functions."""

from __future__ import annotations

__all__ = ["concatenate"]

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from batcharray.computation.auto import AutoComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

T = TypeVar("T", bound=np.ndarray)


_comp_model = AutoComputationModel()


def concatenate(arrays: Sequence[T], axis: int | None = None, *, dtype: DTypeLike = None) -> T:
    r"""Concatenate a sequence of arrays along an existing axis.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis along which the arrays will be joined.
            If ``axis`` is None, arrays are flattened before use.
        dtype: If provided, the destination array will have this
            data type.

    Returns:
        The concatenated array.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import concatenate
    >>> arrays = [
    ...     np.array([[0, 1, 2], [4, 5, 6]]),
    ...     np.array([[10, 11, 12], [13, 14, 15]]),
    ... ]
    >>> out = concatenate(arrays, axis=0)
    >>> out
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [10, 11, 12],
           [13, 14, 15]])
    >>> out = concatenate(arrays, axis=1)
    >>> out
    array([[ 0,  1,  2, 10, 11, 12],
           [ 4,  5,  6, 13, 14, 15]])

    ```
    """
    return _comp_model.concatenate(arrays=arrays, axis=axis, dtype=dtype)
