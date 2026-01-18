r"""Contain some array point-wise functions for nested data."""

from __future__ import annotations

__all__ = [
    "abs",
    "clip",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
]

from functools import partial
from typing import Any

import numpy as np
from coola.recursive import recursive_apply


def abs(data: Any) -> Any:  # noqa: A001
    r"""Return new arrays with the absolute value of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The absolute value of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import abs
        >>> data = {
        ...     "a": np.array([[-4, -3], [-2, -1], [0, 1], [2, 3], [4, 5]]),
        ...     "b": np.array([2, 1, 0, -1, -2]),
        ... }
        >>> out = abs(data)
        >>> out
        {'a': array([[4, 3], [2, 1], [0, 1], [2, 3], [4, 5]]), 'b': array([2, 1, 0, 1, 2])}

        ```
    """
    return recursive_apply(data, np.abs)


def clip(data: Any, a_min: float | None = None, a_max: float | None = None) -> Any:
    r"""Clamp all elements in input into the range ``[min, max]``.

    Args:
        data: The input data. Each item must be an array.
        a_min: The lower-bound of the range to be clamped to.
        a_max: The upper-bound of the range to be clamped to.

    Returns:
        The clamp value of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import clip
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = clip(data, a_min=1, a_max=5)
        >>> out
        {'a': array([[1, 2], [3, 4], [5, 5], [5, 5], [5, 5]]), 'b': array([5, 4, 3, 2, 1])}

        ```
    """
    return recursive_apply(data, partial(np.clip, a_min=a_min, a_max=a_max))


def exp(data: Any) -> Any:
    r"""Return new arrays with the exponential of the elements.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The exponential of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import exp
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = exp(data)
        >>> out
        {'a': array([[2.71828183e+00, 7.38905610e+00],
               [2.00855369e+01, 5.45981500e+01],
               [1.48413159e+02, 4.03428793e+02],
               [1.09663316e+03, 2.98095799e+03],
               [8.10308393e+03, 2.20264658e+04]]), 'b': array([148.4131591 ,  54.5981500 ,  20.08553692,   7.3890561 ,   2.71828183])}

        ```
    """
    return recursive_apply(data, np.exp)


def exp2(data: Any) -> Any:
    r"""Return new arrays with the base two exponential of the elements.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The base two exponential of the elements. The output has the
            same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import exp2
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = exp2(data)
        >>> out
        {'a': array([[  2.,   4.],
               [  8.,  16.],
               [ 32.,  64.],
               [128., 256.],
               [512., 1024.]]), 'b': array([32., 16.,  8.,  4.,  2.])}

        ```
    """
    return recursive_apply(data, np.exp2)


def expm1(data: Any) -> Any:
    r"""Return new arrays with the exponential of the elements minus 1.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The exponential of the elements minus 1. The output has the
            same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import expm1
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = expm1(data)
        >>> out
        {'a': array([[1.71828183e+00, 6.38905610e+00],
               [1.90855369e+01, 5.35981500e+01],
               [1.47413159e+02, 4.02428793e+02],
               [1.09563316e+03, 2.97995799e+03],
               [8.10208393e+03, 2.20254658e+04]]), 'b': array([147.4131591 ,  53.5981500 ,  19.08553692,   6.3890561 ,   1.71828183])}

        ```
    """
    return recursive_apply(data, np.expm1)


def log(data: Any) -> Any:
    r"""Return new arrays with the natural logarithm of the elements.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The natural logarithm of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import log
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = log(data)
        >>> out
        {'a': array([[0.        , 0.69314718],
               [1.09861229, 1.38629436],
               [1.60943791, 1.79175947],
               [1.94591015, 2.07944154],
               [2.19722458, 2.30258509]]), 'b': array([1.60943791, 1.38629436, 1.09861229, 0.69314718, 0.        ])}

        ```
    """
    return recursive_apply(data, np.log)


def log2(data: Any) -> Any:
    r"""Return new arrays with the logarithm to the base 2 of the
    elements.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The logarithm to the base 2 of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import log2
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = log2(data)
        >>> out
        {'a': array([[0.        , 1.        ],
               [1.5849625 , 2.        ],
               [2.32192809, 2.5849625 ],
               [2.80735492, 3.        ],
               [3.169925  , 3.32192809]]), 'b': array([2.32192809, 2.        , 1.5849625 , 1.        , 0.        ])}

        ```
    """
    return recursive_apply(data, np.log2)


def log10(data: Any) -> Any:
    r"""Return new arrays with the logarithm to the base 10 of the
    elements.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The with the logarithm to the base 10 of the elements. The
            output has the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import log10
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = log10(data)
        >>> out
        {'a': array([[0.        , 0.30103   ],
               [0.47712125, 0.60205999],
               [0.69897   , 0.77815125],
               [0.84509804, 0.90308999],
               [0.95424251, 1.        ]]), 'b': array([0.69897   , 0.60205999, 0.47712125, 0.30103   , 0.        ])}

        ```
    """
    return recursive_apply(data, np.log10)


def log1p(data: Any) -> Any:
    r"""Return new arrays with the natural logarithm of ``(1 + input)``.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The natural logarithm of ``(1 + input)``. The output has the
            same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import log1p
        >>> data = {
        ...     "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        ...     "b": np.array([5, 4, 3, 2, 1]),
        ... }
        >>> out = log1p(data)
        >>> out
        {'a': array([[0.69314718, 1.09861229],
               [1.38629436, 1.60943791],
               [1.79175947, 1.94591015],
               [2.07944154, 2.19722458],
               [2.30258509, 2.39789527]]), 'b': array([1.79175947, 1.60943791, 1.38629436, 1.09861229, 0.69314718])}

        ```
    """
    return recursive_apply(data, np.log1p)
