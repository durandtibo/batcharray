r"""Contain some reduction functions for arrays."""

from __future__ import annotations

__all__ = [
    "all_along_batch",
    "all_along_seq",
    "amax_along_batch",
    "amax_along_seq",
    "amin_along_batch",
    "amin_along_seq",
    "any_along_batch",
    "any_along_seq",
    "argmax_along_batch",
    "argmax_along_seq",
    "argmin_along_batch",
    "argmin_along_seq",
    "max_along_batch",
    "max_along_seq",
    "mean_along_batch",
    "mean_along_seq",
    "median_along_batch",
    "median_along_seq",
    "min_along_batch",
    "min_along_seq",
    "prod_along_batch",
    "prod_along_seq",
    "std_along_batch",
    "std_along_seq",
    "sum_along_batch",
    "sum_along_seq",
    "var_along_batch",
    "var_along_seq",
]


import numpy as np

from batcharray import computation as cmpt
from batcharray.constants import BATCH_AXIS, SEQ_AXIS


def amax_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the maximum of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The maximum of all elements along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import amax_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = amax_along_batch(array)
        >>> out
        array([8, 9])
        >>> out = amax_along_batch(array, keepdims=True)
        >>> out
        array([[8, 9]])

        ```
    """
    return cmpt.max(array, axis=BATCH_AXIS, keepdims=keepdims)


def amax_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the maximum of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The maximum of all elements along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import amax_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = amax_along_seq(array)
        >>> out
        array([4, 9])
        >>> out = amax_along_seq(array, keepdims=True)
        >>> out
        array([[4], [9]])

        ```
    """
    return cmpt.max(array, axis=SEQ_AXIS, keepdims=keepdims)


def amin_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the minimum of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The minimum of all elements along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import amin_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = amin_along_batch(array)
        >>> out
        array([0, 1])
        >>> out = amin_along_batch(array, keepdims=True)
        >>> out
        array([[0, 1]])

        ```
    """
    return cmpt.min(array, axis=BATCH_AXIS, keepdims=keepdims)


def amin_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the minimum of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The minimum of all elements along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import amin_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = amin_along_seq(array)
        >>> out
        array([0, 5])
        >>> out = amin_along_seq(array, keepdims=True)
        >>> out
        array([[0], [5]])

        ```
    """
    return cmpt.min(array, axis=SEQ_AXIS, keepdims=keepdims)


def argmax_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the indices of the maximum value of all elements along the
    batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import argmax_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = argmax_along_batch(array)
        >>> out
        array([4, 4])
        >>> out = argmax_along_batch(array, keepdims=True)
        >>> out
        array([[4, 4]])

        ```
    """
    return cmpt.argmax(array, axis=BATCH_AXIS, keepdims=keepdims)


def argmax_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the indices of the maximum value of all elements along the
    sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The indices of the maximum value of all elements along the
            sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import argmax_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = argmax_along_seq(array)
        >>> out
        array([4, 4])
        >>> out = argmax_along_seq(array, keepdims=True)
        >>> out
        array([[4], [4]])

        ```
    """
    return cmpt.argmax(array, axis=SEQ_AXIS, keepdims=keepdims)


def argmin_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the indices of the minimum value of all elements along the
    batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import argmin_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = argmin_along_batch(array)
        >>> out
        array([0, 0])
        >>> out = argmin_along_batch(array, keepdims=True)
        >>> out
        array([[0, 0]])

        ```
    """
    return cmpt.argmin(array, axis=BATCH_AXIS, keepdims=keepdims)


def argmin_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the indices of the minimum value of all elements along the
    sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The indices of the minimum value of all elements along the
            sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import argmin_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = argmin_along_seq(array)
        >>> out
        array([0, 0])
        >>> out = argmin_along_seq(array, keepdims=True)
        >>> out
        array([[0], [0]])

        ```
    """
    return cmpt.argmin(array, axis=SEQ_AXIS, keepdims=keepdims)


def max_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the maximum of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The maximum of the input array along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import max_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = max_along_batch(array)
        >>> out
        array([8, 9])
        >>> out = max_along_batch(array, keepdims=True)
        >>> out
        array([[8, 9]])

        ```
    """
    return cmpt.max(array, axis=BATCH_AXIS, keepdims=keepdims)


def max_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the maximum of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The maximum of the input array along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import max_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = max_along_seq(array)
        >>> out
        array([4, 9])
        >>> out = max_along_seq(array, keepdims=True)
        >>> out
        array([[4], [9]])

        ```
    """
    return cmpt.max(array, axis=SEQ_AXIS, keepdims=keepdims)


def mean_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the mean of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The mean of all elements along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import mean_along_batch
        >>> array = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        >>> out = mean_along_batch(array)
        >>> out
        array([4., 5.])
        >>> out = mean_along_batch(array, keepdims=True)
        >>> out
        array([[4., 5.]])

        ```
    """
    return np.mean(array, axis=BATCH_AXIS, keepdims=keepdims)


def mean_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the mean of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The mean of all elements along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import mean_along_seq
        >>> array = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
        >>> out = mean_along_seq(array)
        >>> out
        array([2., 7.])
        >>> out = mean_along_seq(array, keepdims=True)
        >>> out
        array([[2.], [7.]])

        ```
    """
    return np.mean(array, axis=SEQ_AXIS, keepdims=keepdims)


def median_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the median of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        A new array holding the result. If the input contains integers
            or floats smaller than ``np.float64``, then the output
            data-type is ``np.float64``. Otherwise, the data-type of
            the output is the same as that of the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import median_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = median_along_batch(array)
        >>> out
        array([4., 5.])
        >>> out = median_along_batch(array, keepdims=True)
        >>> out
        array([[4., 5.]])

        ```
    """
    return cmpt.median(array, axis=BATCH_AXIS, keepdims=keepdims)


def median_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the median of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        A new array holding the result. If the input contains integers
            or floats smaller than ``np.float64``, then the output
            data-type is ``np.float64``. Otherwise, the data-type of
            the output is the same as that of the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import median_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = median_along_seq(array)
        >>> out
        array([2., 7.])
        >>> out = median_along_seq(array, keepdims=True)
        >>> out
        array([[2.], [7.]])

        ```
    """
    return cmpt.median(array, axis=SEQ_AXIS, keepdims=keepdims)


def min_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the minimum of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The minimum of the input array along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import min_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = min_along_batch(array)
        >>> out
        array([0, 1])
        >>> out = min_along_batch(array, keepdims=True)
        >>> out
        array([[0, 1]])

        ```
    """
    return cmpt.min(array, axis=BATCH_AXIS, keepdims=keepdims)


def min_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the minimum of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The minimum of the input array along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import min_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = min_along_seq(array)
        >>> out
        array([0, 5])
        >>> out = min_along_seq(array, keepdims=True)
        >>> out
        array([[0], [5]])

        ```
    """
    return cmpt.min(array, axis=SEQ_AXIS, keepdims=keepdims)


def prod_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the product of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The product of all elements along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import prod_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = prod_along_batch(array)
        >>> out
        array([  0, 945])
        >>> out = prod_along_batch(array, keepdims=True)
        >>> out
        array([[  0, 945]])

        ```
    """
    return np.prod(array, axis=BATCH_AXIS, keepdims=keepdims)


def prod_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the product of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The product of all elements along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import prod_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = prod_along_seq(array)
        >>> out
        array([    0, 15120])
        >>> out = prod_along_seq(array, keepdims=True)
        >>> out
        array([[    0], [15120]])

        ```
    """
    return np.prod(array, axis=SEQ_AXIS, keepdims=keepdims)


def sum_along_batch(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the sum of all elements along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The sum of all elements along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import sum_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = sum_along_batch(array)
        >>> out
        array([20, 25])
        >>> out = sum_along_batch(array, keepdims=True)
        >>> out
        array([[20, 25]])

        ```
    """
    return np.sum(array, axis=BATCH_AXIS, keepdims=keepdims)


def sum_along_seq(array: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Return the sum of all elements along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The sum of all elements along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import sum_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = sum_along_seq(array)
        >>> out
        array([10, 35])
        >>> out = sum_along_seq(array, keepdims=True)
        >>> out
        array([[10], [35]])

        ```
    """
    return np.sum(array, axis=SEQ_AXIS, keepdims=keepdims)


def std_along_batch(array: np.ndarray, *, keepdims: bool = False, ddof: int = 0) -> np.ndarray:
    r"""Return the standard deviation along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.
        ddof: The delta degrees of freedom. The divisor is ``N - ddof``.

    Returns:
        The standard deviation along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import std_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = std_along_batch(array)
        >>> out
        array([2.82842712, 2.82842712])
        >>> out = std_along_batch(array, keepdims=True)
        >>> out
        array([[2.82842712, 2.82842712]])

        ```
    """
    return np.std(array, axis=BATCH_AXIS, keepdims=keepdims, ddof=ddof)


def std_along_seq(array: np.ndarray, *, keepdims: bool = False, ddof: int = 0) -> np.ndarray:
    r"""Return the standard deviation along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.
        ddof: The delta degrees of freedom. The divisor is ``N - ddof``.

    Returns:
        The standard deviation along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import std_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = std_along_seq(array)
        >>> out
        array([1.41421356, 1.41421356])
        >>> out = std_along_seq(array, keepdims=True)
        >>> out
        array([[1.41421356], [1.41421356]])

        ```
    """
    return np.std(array, axis=SEQ_AXIS, keepdims=keepdims, ddof=ddof)


def var_along_batch(array: np.ndarray, *, keepdims: bool = False, ddof: int = 0) -> np.ndarray:
    r"""Return the variance along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.
        ddof: The delta degrees of freedom. The divisor is ``N - ddof``.

    Returns:
        The variance along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import var_along_batch
        >>> array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        >>> out = var_along_batch(array)
        >>> out
        array([8., 8.])
        >>> out = var_along_batch(array, keepdims=True)
        >>> out
        array([[8., 8.]])

        ```
    """
    return np.var(array, axis=BATCH_AXIS, keepdims=keepdims, ddof=ddof)


def var_along_seq(array: np.ndarray, *, keepdims: bool = False, ddof: int = 0) -> np.ndarray:
    r"""Return the variance along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.
        ddof: The delta degrees of freedom. The divisor is ``N - ddof``.

    Returns:
        The variance along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import var_along_seq
        >>> array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        >>> out = var_along_seq(array)
        >>> out
        array([2., 2.])
        >>> out = var_along_seq(array, keepdims=True)
        >>> out
        array([[2.], [2.]])

        ```
    """
    return np.var(array, axis=SEQ_AXIS, keepdims=keepdims, ddof=ddof)


def any_along_batch(array: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    r"""Return the logical OR (whether any element is true) along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The logical OR (whether any element is true) along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import any_along_batch
        >>> array = np.array([[True, False], [True, True], [True, False]])
        >>> out = any_along_batch(array)
        >>> out
        array([ True,  True])
        >>> out = any_along_batch(array, keepdims=True)
        >>> out
        array([[ True,  True]])

        ```
    """
    return np.any(array, axis=BATCH_AXIS, keepdims=keepdims)


def any_along_seq(array: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    r"""Return the logical OR (whether any element is true) along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The logical OR (whether any element is true) along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import any_along_seq
        >>> array = np.array([[True, False, True], [True, True, True]])
        >>> out = any_along_seq(array)
        >>> out
        array([ True,  True])
        >>> out = any_along_seq(array, keepdims=True)
        >>> out
        array([[ True], [ True]])

        ```
    """
    return np.any(array, axis=SEQ_AXIS, keepdims=keepdims)


def all_along_batch(array: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    r"""Return the logical AND (whether all elements are true) along the batch axis.

    Note:
        This function assumes the batch axis is the first
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The logical AND (whether all elements are true) along the batch axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import all_along_batch
        >>> array = np.array([[True, False], [True, True], [True, False]])
        >>> out = all_along_batch(array)
        >>> out
        array([ True, False])
        >>> out = all_along_batch(array, keepdims=True)
        >>> out
        array([[ True, False]])

        ```
    """
    return np.all(array, axis=BATCH_AXIS, keepdims=keepdims)


def all_along_seq(array: np.ndarray, *, keepdims: bool = False) -> np.ndarray:
    r"""Return the logical AND (whether all elements are true) along the sequence axis.

    Note:
        This function assumes the sequence axis is the second
            axis.

    Args:
        array: The input array.
        keepdims: Whether the output array has dim retained or not.

    Returns:
        The logical AND (whether all elements are true) along the sequence axis.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.array import all_along_seq
        >>> array = np.array([[True, False, True], [True, True, True]])
        >>> out = all_along_seq(array)
        >>> out
        array([False,  True])
        >>> out = all_along_seq(array, keepdims=True)
        >>> out
        array([[False], [ True]])

        ```
    """
    return np.all(array, axis=SEQ_AXIS, keepdims=keepdims)
