from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.computation import ArrayComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

###############
#     str     #
###############


def test_array_computation_model_repr() -> None:
    assert repr(ArrayComputationModel()).startswith("ArrayComputationModel(")


def test_array_computation_model_str() -> None:
    assert str(ArrayComputationModel()).startswith("ArrayComputationModel(")


#######################
#     concatenate     #
#######################


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        (np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])),
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11, 12]]),
            np.array([[13, 14, 15]]),
        ],
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.ones((0, 3), dtype=int),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_array_computation_model_concatenate_axis_0(arrays: Sequence[np.ndarray]) -> None:
    out = ArrayComputationModel().concatenate(arrays, axis=0)
    assert objects_are_equal(out, np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        (np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])),
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11], [13, 14]]),
            np.array([[12], [15]]),
        ],
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.ones((2, 0), dtype=int),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_array_computation_model_concatenate_axis_1(arrays: Sequence[np.ndarray]) -> None:
    out = ArrayComputationModel().concatenate(arrays, axis=1)
    assert objects_are_equal(out, np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]))


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        (np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])),
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11, 12]]),
            np.array([[13, 14, 15]]),
        ],
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.ones((0, 3), dtype=int),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_array_computation_model_concatenate_axis_none(arrays: Sequence[np.ndarray]) -> None:
    out = ArrayComputationModel().concatenate(arrays)
    assert objects_are_equal(out, np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]))


@pytest.mark.parametrize("dtype", [int, float])
def test_array_computation_model_concatenate_dtype(dtype: np.dtype) -> None:
    out = ArrayComputationModel().concatenate(
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
        axis=0,
        dtype=dtype,
    )
    assert objects_are_equal(
        out, np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]], dtype=dtype)
    )
