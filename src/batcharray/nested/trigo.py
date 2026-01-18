r"""Contain some array trigonometric functions for nested data."""

from __future__ import annotations

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
]

from typing import Any

import numpy as np
from coola.recursive import recursive_apply


def arccos(data: Any) -> Any:
    r"""Return new arrays with the inverse cosine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The inverse cosine of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arccos
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arccos(data)
        >>> out
        {'a': array([[0., 0., 0.],
               [0., 0., 0.]]), 'b': array([1.57079633, 0.        ,        nan,        nan,        nan])}

        ```
    """
    return recursive_apply(data, np.arccos)


def arccosh(data: Any) -> Any:
    r"""Return new arrays with the inverse hyperbolic cosine of each
    element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The inverse hyperbolic cosine of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arccosh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arccosh(data)
        >>> out
        {'a': array([[0., 0., 0.],
               [0., 0., 0.]]), 'b': array([       nan, 0.        , 1.3169579 , 1.76274717, 2.06343707])}

        ```
    """
    return recursive_apply(data, np.arccosh)


def arcsin(data: Any) -> Any:
    r"""Return new arrays with the arcsine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The arcsine of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arcsin
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arcsin(data)
        >>> out
        {'a': array([[1.57079633, 1.57079633, 1.57079633],
               [1.57079633, 1.57079633, 1.57079633]]), 'b': array([0.        , 1.57079633,        nan,        nan,        nan])}

        ```
    """
    return recursive_apply(data, np.arcsin)


def arcsinh(data: Any) -> Any:
    r"""Return new arrays with the inverse hyperbolic sine of each
    element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The inverse hyperbolic sine of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arcsinh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arcsinh(data)
        >>> out
        {'a': array([[0.88137359, 0.88137359, 0.88137359],
               [0.88137359, 0.88137359, 0.88137359]]), 'b': array([0.        , 0.88137359, 1.44363548, 1.81844646, 2.09471255])}

        ```
    """
    return recursive_apply(data, np.arcsinh)


def arctan(data: Any) -> Any:
    r"""Return new arrays with the arctangent of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The arctangent of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arctan
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arctan(data)
        >>> out
        {'a': array([[0.78539816, 0.78539816, 0.78539816],
               [0.78539816, 0.78539816, 0.78539816]]), 'b': array([0.        , 0.78539816, 1.10714872, 1.24904577, 1.32581766])}

        ```
    """
    return recursive_apply(data, np.arctan)


def arctanh(data: Any) -> Any:
    r"""Return new arrays with the inverse hyperbolic tangent of each
    element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The inverse hyperbolic tangent of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import arctanh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = arctanh(data)
        >>> out
        {'a': array([[inf, inf, inf],
               [inf, inf, inf]]), 'b': array([0.        ,        inf,        nan,        nan,        nan])}

        ```
    """
    return recursive_apply(data, np.arctanh)


def cos(data: Any) -> Any:
    r"""Return new arrays with the cosine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The cosine of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import cos
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = cos(data)
        >>> out
        {'a': array([[0.54030231, 0.54030231, 0.54030231],
               [0.54030231, 0.54030231, 0.54030231]]), 'b': array([ 1.        ,  0.54030231, -0.41614684, -0.9899925 , -0.65364362])}

        ```
    """
    return recursive_apply(data, np.cos)


def cosh(data: Any) -> Any:
    r"""Return new arrays with the hyperbolic cosine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The hyperbolic cosine of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import cosh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = cosh(data)
        >>> out
        {'a': array([[1.54308063, 1.54308063, 1.54308063],
               [1.54308063, 1.54308063, 1.54308063]]), 'b': array([ 1.        ,  1.54308063,  3.76219569, 10.067662  , 27.30823284])}

        ```
    """
    return recursive_apply(data, np.cosh)


def sin(data: Any) -> Any:
    r"""Return new arrays with the sine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The sine of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import sin
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = sin(data)
        >>> out
        {'a': array([[0.84147098, 0.84147098, 0.84147098],
               [0.84147098, 0.84147098, 0.84147098]]), 'b': array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ])}

        ```
    """
    return recursive_apply(data, np.sin)


def sinh(data: Any) -> Any:
    r"""Return new arrays with the hyperbolic sine of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The hyperbolic sine of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import sinh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = sinh(data)
        >>> out
        {'a': array([[1.17520119, 1.17520119, 1.17520119],
               [1.17520119, 1.17520119, 1.17520119]]), 'b': array([ 0.        ,  1.17520119,  3.62686041, 10.01787493, 27.2899172 ])}

        ```
    """
    return recursive_apply(data, np.sinh)


def tan(data: Any) -> Any:
    r"""Return new arrays with the tangent of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The tangent of the elements. The output has the same
            structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import tan
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = tan(data)
        >>> out
        {'a': array([[1.55740772, 1.55740772, 1.55740772],
               [1.55740772, 1.55740772, 1.55740772]]), 'b': array([ 0.        ,  1.55740772, -2.18503986, -0.14254654,  1.15782128])}

        ```
    """
    return recursive_apply(data, np.tan)


def tanh(data: Any) -> Any:
    r"""Return new arrays with the hyperbolic tangent of each element.

    Args:
        data: The input data. Each item must be an array.

    Returns:
        The hyperbolic tangent of the elements. The output has
            the same structure as the input.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.nested import tanh
        >>> data = {"a": np.ones((2, 3)), "b": np.arange(5)}
        >>> out = tanh(data)
        >>> out
        {'a': array([[0.76159416, 0.76159416, 0.76159416],
               [0.76159416, 0.76159416, 0.76159416]]), 'b': array([0.        , 0.76159416, 0.96402758, 0.99505475, 0.9993293 ])}

        ```
    """
    return recursive_apply(data, np.tanh)
