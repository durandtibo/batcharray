r"""Contain the computation model for ``numpy.ma.MaskedArray``s."""

from __future__ import annotations

__all__ = ["MaskedArrayComputationModel"]


from typing import TYPE_CHECKING

import numpy as np

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy._typing import DTypeLike


class MaskedArrayComputationModel(BaseComputationModel[np.ndarray]):
    r"""Implement a computation model for ``numpy.ma.MaskedArray``s."""

    def concatenate(
        self, arrays: Sequence[np.ndarray], axis: int | None = None, *, dtype: DTypeLike = None
    ) -> np.ndarray:
        out = np.ma.concatenate(arrays, axis=axis)
        if dtype:
            out = np.ma.masked_array(data=out.data.astype(dtype), mask=out.mask)
        return out
