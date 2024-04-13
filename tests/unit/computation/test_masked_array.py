from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.computation import MaskedArrayComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_masked_array_computation_model_eq_true() -> None:
    assert MaskedArrayComputationModel() == MaskedArrayComputationModel()


def test_masked_array_computation_model_eq_false() -> None:
    assert MaskedArrayComputationModel() != "MaskedArrayComputationModel"


def test_masked_array_computation_model_repr() -> None:
    assert repr(MaskedArrayComputationModel()).startswith("MaskedArrayComputationModel(")


def test_masked_array_computation_model_str() -> None:
    assert str(MaskedArrayComputationModel()).startswith("MaskedArrayComputationModel(")


#######################
#     concatenate     #
#######################


@pytest.mark.parametrize(
    "arrays",
    [
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
        (
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ),
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12]]), mask=np.array([[False, False, True]])
            ),
            np.ma.masked_array(
                data=np.array([[13, 14, 15]]), mask=np.array([[False, False, False]])
            ),
        ],
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(data=np.ones((0, 3), dtype=int)),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
    ],
)
def test_masked_array_computation_model_concatenate_axis_0(arrays: Sequence[np.ndarray]) -> None:
    out = MaskedArrayComputationModel().concatenate(arrays, axis=0)
    assert objects_are_equal(
        out,
        np.ma.masked_array(
            data=np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
            mask=np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ]
            ),
        ),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
        (
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ),
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11], [13, 14]]),
                mask=np.array([[False, False], [False, False]]),
            ),
            np.ma.masked_array(data=np.array([[12], [15]]), mask=np.array([[True], [False]])),
        ],
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(data=np.ones((2, 0), dtype=int)),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
    ],
)
def test_masked_array_computation_model_concatenate_axis_1(arrays: Sequence[np.ndarray]) -> None:
    out = MaskedArrayComputationModel().concatenate(arrays, axis=1)
    assert objects_are_equal(
        out,
        np.ma.masked_array(
            data=np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
            mask=np.array(
                [
                    [False, False, False, False, False, True],
                    [False, True, False, False, False, False],
                ]
            ),
        ),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
        (
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ),
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12]]), mask=np.array([[False, False, True]])
            ),
            np.ma.masked_array(
                data=np.array([[13, 14, 15]]), mask=np.array([[False, False, False]])
            ),
        ],
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(data=np.ones((0, 3), dtype=int)),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
    ],
)
def test_masked_array_computation_model_concatenate_axis_none(arrays: Sequence[np.ndarray]) -> None:
    out = MaskedArrayComputationModel().concatenate(arrays)
    assert objects_are_equal(
        out,
        np.ma.masked_array(
            data=np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]),
            mask=np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ]
            ),
        ),
    )


@pytest.mark.parametrize("dtype", [int, float])
def test_masked_array_computation_model_concatenate_dtype(dtype: np.dtype) -> None:
    out = MaskedArrayComputationModel().concatenate(
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
        axis=0,
        dtype=dtype,
    )
    assert objects_are_equal(
        out,
        np.ma.masked_array(
            data=np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]], dtype=dtype),
            mask=np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ]
            ),
        ),
    )
