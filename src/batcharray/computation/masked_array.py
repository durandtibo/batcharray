r"""Contain the computation model for ``numpy.ma.MaskedArray``s."""

from __future__ import annotations

__all__ = ["MaskedArrayComputationModel"]


from typing import TYPE_CHECKING, cast

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from batcharray.types import SortKind


class MaskedArrayComputationModel(BaseComputationModel[np.ma.MaskedArray]):
    r"""Implement a computation model for ``numpy.ma.MaskedArray``s."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def argmax(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return cast("np.ndarray", np.ma.argmax(arr, axis=axis, keepdims=keepdims))

    def argmin(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return cast("np.ndarray", np.ma.argmin(arr, axis=axis, keepdims=keepdims))

    def argsort(
        self,
        arr: np.ma.MaskedArray,
        axis: int | None = None,
        *,
        kind: SortKind | None = None,
    ) -> np.ma.MaskedArray:
        return np.ma.argsort(arr, axis=axis, kind=kind)

    def concatenate(
        self,
        arrays: Sequence[np.ma.MaskedArray],
        axis: int | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> np.ma.MaskedArray:
        out = np.ma.concatenate(arrays, axis=axis)
        if dtype is not None:
            out = cast("np.ma.MaskedArray", out.astype(dtype))
        return out

    def max(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.max(arr, axis=axis, keepdims=keepdims)

    def mean(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.mean(arr, axis=axis, keepdims=keepdims)

    def median(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.median(arr, axis=axis, keepdims=keepdims)

    def min(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.min(arr, axis=axis, keepdims=keepdims)

    def sort(
        self,
        arr: np.ma.MaskedArray,
        axis: int | None = None,
        *,
        kind: SortKind | None = None,
    ) -> np.ma.MaskedArray:
        return cast("np.ma.MaskedArray", np.ma.sort(arr, axis=axis, kind=kind))

    def prod(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.prod(arr, axis=axis, keepdims=keepdims)

    def std(
        self,
        arr: np.ma.MaskedArray,
        axis: int | None = None,
        *,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> np.ma.MaskedArray:
        return np.ma.std(arr, axis=axis, keepdims=keepdims, ddof=ddof)

    def sum(
        self, arr: np.ma.MaskedArray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ma.MaskedArray:
        return np.ma.sum(arr, axis=axis, keepdims=keepdims)

    def var(
        self,
        arr: np.ma.MaskedArray,
        axis: int | None = None,
        *,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> np.ma.MaskedArray:
        return np.ma.var(arr, axis=axis, keepdims=keepdims, ddof=ddof)
