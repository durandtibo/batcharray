r"""Contain the computation model for ``numpy.ma.MaskedArray``s."""

from __future__ import annotations

__all__ = ["MaskedArrayComputationModel"]


from typing import TYPE_CHECKING

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy._typing import DTypeLike


class MaskedArrayComputationModel(BaseComputationModel[np.ma.MaskedArray]):
    r"""Implement a computation model for ``numpy.ma.MaskedArray``s."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def concatenate(
        self,
        arrays: Sequence[np.ma.MaskedArray],
        axis: int | None = None,
        *,
        dtype: DTypeLike = None,
    ) -> np.ma.MaskedArray:
        out = np.ma.concatenate(arrays, axis=axis)
        if dtype:
            out = np.ma.masked_array(data=out.data.astype(dtype), mask=out.mask)
        return out

    def median(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.median(arr, axis=axis, keepdims=keepdims)
