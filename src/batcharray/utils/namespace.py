r"""Contain the helper to choose between ``numpy`` and ``numpy.ma``."""

from __future__ import annotations

__all__ = ["get_namespace"]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType


def get_namespace(array: np.ndarray) -> ModuleType:
    r"""Return the NumPy namespace that implements the operations of an
    array.

    A subclass of ``numpy.ma.MaskedArray`` is dispatched as a masked
    array. A subclass of ``numpy.ndarray`` is a plain array.

    Args:
        array: The input array.

    Returns:
        ``numpy.ma`` if the array is a masked array, otherwise
            ``numpy``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from batcharray.utils import get_namespace
        >>> get_namespace(np.ones((2, 3))).__name__
        'numpy'
        >>> get_namespace(np.ma.ones((2, 3))).__name__
        'numpy.ma'

        ```
    """
    return np.ma if isinstance(array, np.ma.MaskedArray) else np
