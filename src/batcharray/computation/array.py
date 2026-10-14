r"""Contain the computation model for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["ArrayComputationModel"]


from typing import TYPE_CHECKING, cast

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from batcharray.types import SortKind


class ArrayComputationModel(BaseComputationModel[np.ndarray]):
    r"""Implement a computation model for ``numpy.ndarray``s."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def argmax(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return cast("np.ndarray", np.argmax(arr, axis=axis, keepdims=keepdims))

    def argmin(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return cast("np.ndarray", np.argmin(arr, axis=axis, keepdims=keepdims))

    def argsort(
        self, arr: np.ndarray, axis: int | None = None, *, kind: SortKind | None = None
    ) -> np.ndarray:
        return np.argsort(arr, axis=axis, kind=kind)

    def concatenate(
        self,
        arrays: Sequence[np.ndarray],
        axis: int | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> np.ndarray:
        return np.concatenate(arrays, axis=axis, dtype=dtype)

    def max(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return np.max(arr, axis=axis, keepdims=keepdims)

    def mean(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return np.mean(arr, axis=axis, keepdims=keepdims)

    def median(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return np.median(arr, axis=axis, keepdims=keepdims)

    def min(
        self, arr: np.ndarray, axis: int | None = None, *, keepdims: bool = False
    ) -> np.ndarray:
        return np.min(arr, axis=axis, keepdims=keepdims)

    def sort(
        self, arr: np.ndarray, axis: int | None = None, *, kind: SortKind | None = None
    ) -> np.ndarray:
        return np.sort(arr, axis=axis, kind=kind)
