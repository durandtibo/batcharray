r"""Contain public functions."""

from __future__ import annotations

__all__ = ["argmax", "argmin", "concatenate", "max", "mean", "median", "min"]

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from batcharray.computation.auto import AutoComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

T = TypeVar("T", bound=np.ndarray)


_comp_model = AutoComputationModel()


def argmax(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:
    r"""Return the array of indices of the maximum values along the given
    axis.

    Args:
        arr: The input array.
        axis: Axis along which the argmax are computed.
            The default (``None``) is to compute the argmax along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        The array of indices of the maximum values along the given
            axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import argmax
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = argmax(array, axis=0)
    >>> out
    array([4, 4])
    >>> out = argmax(array, axis=1)
    >>> out
    array([1, 1, 1, 1, 1])
    >>> out = argmax(array, axis=0, keepdims=True)
    >>> out
    array([[4, 4]])

    ```
    """
    return _comp_model.argmax(arr=arr, axis=axis, keepdims=keepdims)


def argmin(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:
    r"""Return the array of indices of the minimum values along the given
    axis.

    Args:
        arr: The input array.
        axis: Axis along which the argmin are computed.
            The default (``None``) is to compute the argmin along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        The array of indices of the minimum values along the given
            axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import argmin
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = argmin(array, axis=0)
    >>> out
    array([0, 0])
    >>> out = argmin(array, axis=1)
    >>> out
    array([0, 0, 0, 0, 0])
    >>> out = argmin(array, axis=0, keepdims=True)
    >>> out
    array([[0, 0]])

    ```
    """
    return _comp_model.argmin(arr=arr, axis=axis, keepdims=keepdims)


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


def max(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:  # noqa: A001
    r"""Return the maximum along the specified axis.

    Args:
        arr: The input array.
        axis: Axis along which the maximum values are computed.
            The default (``None``) is to compute the maximum along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        The maximum of the input array along the given axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import max
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = max(array, axis=0)
    >>> out
    array([8, 9])
    >>> out = max(array, axis=1)
    >>> out
    array([1, 3, 5, 7, 9])
    >>> out = max(array, axis=0, keepdims=True)
    >>> out
    array([[8, 9]])

    ```
    """
    return _comp_model.max(arr=arr, axis=axis, keepdims=keepdims)


def mean(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:
    r"""Return the mean along the specified axis.

    Args:
        arr: The input array.
        axis: Axis along which the means are computed.
            The default (``None``) is to compute the mean along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        A new array holding the result. If the input contains integers
            or floats smaller than ``np.float64``, then the output
            data-type is ``np.float64``. Otherwise, the data-type of
            the output is the same as that of the input.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import mean
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = mean(array, axis=0)
    >>> out
    array([4., 5.])
    >>> out = mean(array, axis=1)
    >>> out
    array([0.5, 2.5, 4.5, 6.5, 8.5])
    >>> out = mean(array, axis=0, keepdims=True)
    >>> out
    array([[4., 5.]])

    ```
    """
    return _comp_model.mean(arr=arr, axis=axis, keepdims=keepdims)


def median(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:
    r"""Return the median along the specified axis.

    Args:
        arr: The input array.
        axis: Axis along which the medians are computed.
            The default (``None``) is to compute the median along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        A new array holding the result. If the input contains integers
            or floats smaller than ``np.float64``, then the output
            data-type is ``np.float64``. Otherwise, the data-type of
            the output is the same as that of the input.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import median
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = median(array, axis=0)
    >>> out
    array([4., 5.])
    >>> out = median(array, axis=1)
    >>> out
    array([0.5, 2.5, 4.5, 6.5, 8.5])
    >>> out = median(array, axis=0, keepdims=True)
    >>> out
    array([[4., 5.]])

    ```
    """
    return _comp_model.median(arr=arr, axis=axis, keepdims=keepdims)


def min(arr: T, axis: int | None = None, *, keepdims: bool = False) -> T:  # noqa: A001
    r"""Return the minimum along the specified axis.

    Args:
        arr: The input array.
        axis: Axis along which the minimum values are computed.
            The default (``None``) is to compute the minimum along
            a flattened version of the array.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the input array.

    Returns:
        The minimum of the input array along the given axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import min
    >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    >>> out = min(array, axis=0)
    >>> out
    array([0, 1])
    >>> out = min(array, axis=1)
    >>> out
    array([0, 2, 4, 6, 8])
    >>> out = min(array, axis=0, keepdims=True)
    >>> out
    array([[0, 1]])

    ```
    """
    return _comp_model.min(arr=arr, axis=axis, keepdims=keepdims)
