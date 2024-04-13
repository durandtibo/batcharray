r"""Contain the computation model for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["ArrayComputationModel"]


from typing import TYPE_CHECKING

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy._typing import DTypeLike


class ArrayComputationModel(BaseComputationModel[np.ndarray]):
    r"""Implement a computation model for ``numpy.ndarray``s."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def concatenate(
        self, arrays: Sequence[np.ndarray], axis: int | None = None, *, dtype: DTypeLike = None
    ) -> np.ndarray:
        return np.concatenate(arrays, axis=axis, dtype=dtype)
