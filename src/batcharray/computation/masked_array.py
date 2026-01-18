r"""Contain the computation model for ``numpy.ma.MaskedArray``s."""

from __future__ import annotations

__all__ = ["MaskedArrayComputationModel"]


from typing import TYPE_CHECKING

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from batcharray.types import SortKind


class MaskedArrayComputationModel(BaseComputationModel[np.ma.MaskedArray]):  # noqa: PLW1641
    r"""Implement a computation model for ``numpy.ma.MaskedArray``s."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def argmax(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        r"""Return the array of indices of the maximum values along the
        given axis.

        Args:
            arr: The input masked array.
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

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]]),
            ... )
            >>> out = comp_model.argmax(array, axis=0)
            >>> out
            array([3, 4])
            >>> out = comp_model.argmax(array, axis=1)
            >>> out
            array([1, 1, 1, 1, 1])

            ```
        """
        return arr.argmax(axis=axis, keepdims=keepdims)

    def argmin(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        r"""Return the array of indices of the minimum values along the
        given axis.

        Args:
            arr: The input masked array.
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

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            ... )
            >>> out = comp_model.argmin(array, axis=0)
            >>> out
            array([1, 0])
            >>> out = comp_model.argmin(array, axis=1)
            >>> out
            array([1, 0, 0, 0, 0])

            ```
        """
        return arr.argmin(axis=axis, keepdims=keepdims)

    def argsort(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, kind: SortKind | None = None
    ) -> np.ma.MaskedArray:
        r"""Return the indices that sort an array along the given axis in
        ascending order by value.

        Args:
            arr: The input masked array.
            axis: Axis along which to sort. The default (``None``) is
                to sort along a flattened version of the array.
            kind: Sorting algorithm. The default is `quicksort`.
                Note that both `stable` and `mergesort` use timsort
                under the covers and, in general, the actual
                implementation will vary with datatype.
                The `mergesort` option is retained for backwards
                compatibility.

        Returns:
            The indices that sort the array along the given axis.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 9, 5], [7, 5, 8, 9, 0]]),
            ...     mask=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
            ... )
            >>> out = comp_model.argsort(array, axis=0)
            >>> out
            masked_array(
              data=[[0, 0, 0, 0, 2],
                    [1, 2, 1, 1, 0],
                    [2, 1, 2, 2, 1]],
              mask=False,
              fill_value=999999)

            ```
        """
        return np.ma.argsort(arr, axis=axis, kind=kind)

    def concatenate(
        self,
        arrays: Sequence[np.ma.MaskedArray],
        axis: int | None = None,
        *,
        dtype: DTypeLike = None,
    ) -> np.ma.MaskedArray:
        r"""Concatenate a sequence of masked arrays along an existing axis.

        Args:
            arrays: The masked arrays to concatenate.
            axis: The axis along which the arrays will be joined.
                If ``axis`` is None, arrays are flattened before use.
            dtype: If provided, the destination array will have this
                data type.

        Returns:
            The concatenated masked array.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> arrays = [
            ...     np.ma.masked_array(
            ...         data=np.array([[0, 1, 2], [4, 5, 6]]),
            ...         mask=np.array([[0, 0, 0], [0, 0, 1]]),
            ...     ),
            ...     np.ma.masked_array(
            ...         data=np.array([[10, 11, 12], [13, 14, 15]]),
            ...         mask=np.array([[0, 1, 0], [0, 0, 0]]),
            ...     ),
            ... ]
            >>> out = comp_model.concatenate(arrays, axis=0)
            >>> out
            masked_array(
              data=[[ 0,  1,  2],
                    [ 4,  5, --],
                    [10, --, 12],
                    [13, 14, 15]],
              mask=[[False, False, False],
                    [False, False,  True],
                    [False,  True, False],
                    [False, False, False]],
              fill_value=999999)

            ```
        """
        out = np.ma.concatenate(arrays, axis=axis)
        if dtype:
            out = np.ma.masked_array(data=out.data.astype(dtype), mask=out.mask)
        return out

    def max(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        r"""Return the maximum along the specified axis.

        Args:
            arr: The input masked array.
            axis: Axis along which the maximum values are computed.
                The default (``None``) is to compute the maximum along
                a flattened version of the array.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the input array.

        Returns:
            The maximum of the input array along the given axis.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]]),
            ... )
            >>> out = comp_model.max(array, axis=0)
            >>> out
            masked_array(data=[6, 9],
                         mask=[False, False],
                   fill_value=999999)
            >>> out = comp_model.max(array, axis=1)
            >>> out
            masked_array(data=[1, 3, 5, 7, 9],
                         mask=[False, False, False, False, False],
                   fill_value=999999)

            ```
        """
        return np.ma.max(arr, axis=axis, keepdims=keepdims)

    def mean(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        r"""Return the mean along the specified axis.

        Args:
            arr: The input masked array.
            axis: Axis along which the means are computed.
                The default (``None``) is to compute the mean along
                a flattened version of the array.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the input array.

        Returns:
            A new masked array holding the result. If the input contains
                integers or floats smaller than ``np.float64``, then the
                output data-type is ``np.float64``. Otherwise, the
                data-type of the output is the same as that of the input.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]]),
            ... )
            >>> out = comp_model.mean(array, axis=0)
            >>> out
            masked_array(data=[3.0, 5.0],
                         mask=[False, False],
                   fill_value=1e+20)

            ```
        """
        return np.ma.mean(arr, axis=axis, keepdims=keepdims)

    def median(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        r"""Return the median along the specified axis.

        Args:
            arr: The input masked array.
            axis: Axis along which the medians are computed.
                The default (``None``) is to compute the median along
                a flattened version of the array.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the input array.

        Returns:
            A new masked array holding the result. If the input contains
                integers or floats smaller than ``np.float64``, then the
                output data-type is ``np.float64``. Otherwise, the
                data-type of the output is the same as that of the input.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]]),
            ... )
            >>> out = comp_model.median(array, axis=0)
            >>> out
            masked_array(data=[3.0, 5.0],
                         mask=[False, False],
                   fill_value=1e+20)

            ```
        """
        return np.ma.median(arr, axis=axis, keepdims=keepdims)

    def min(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        r"""Return the minimum along the specified axis.

        Args:
            arr: The input masked array.
            axis: Axis along which the minimum values are computed.
                The default (``None``) is to compute the minimum along
                a flattened version of the array.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the input array.

        Returns:
            The minimum of the input array along the given axis.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            ...     mask=np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            ... )
            >>> out = comp_model.min(array, axis=0)
            >>> out
            masked_array(data=[2, 1],
                         mask=[False, False],
                   fill_value=999999)
            >>> out = comp_model.min(array, axis=1)
            >>> out
            masked_array(data=[1, 2, 4, 6, 8],
                         mask=[False, False, False, False, False],
                   fill_value=999999)

            ```
        """
        return np.ma.min(arr, axis=axis, keepdims=keepdims)

    def sort(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, kind: SortKind | None = None
    ) -> np.ma.MaskedArray:
        r"""Sort the elements of the input masked array along the given
        axis in ascending order by value.

        Args:
            arr: The input masked array.
            axis: Axis along which to sort. The default (``None``) is
                to sort along a flattened version of the array.
            kind: Sorting algorithm. The default is `quicksort`.
                Note that both `stable` and `mergesort` use timsort
                under the covers and, in general, the actual
                implementation will vary with datatype.
                The `mergesort` option is retained for backwards
                compatibility.

        Returns:
            A sorted copy of the input masked array.

        Example:
            ```pycon
            >>> import numpy as np
            >>> from batcharray.computation import MaskedArrayComputationModel
            >>> comp_model = MaskedArrayComputationModel()
            >>> array = np.ma.masked_array(
            ...     data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 9, 5], [7, 5, 8, 9, 0]]),
            ...     mask=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
            ... )
            >>> out = comp_model.sort(array, axis=0)
            >>> out
            masked_array(
              data=[[3, 5, 0, 2, 4],
                    [4, 5, 8, --, 5],
                    [7, 7, 8, --, --]],
              mask=[[False, False, False, False, False],
                    [False, False, False,  True, False],
                    [False, False, False,  True,  True]],
              fill_value=999999)

            ```
        """
        return np.ma.sort(arr, axis=axis, kind=kind)
