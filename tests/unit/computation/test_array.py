from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.computation import ArrayComputationModel

if TYPE_CHECKING:
    from collections.abc import Sequence

DTYPES = (np.float64, np.int64)


def test_array_computation_model_eq_true() -> None:
    assert ArrayComputationModel() == ArrayComputationModel()


def test_array_computation_model_eq_false() -> None:
    assert ArrayComputationModel() != "ArrayComputationModel"


def test_array_computation_model_repr() -> None:
    assert repr(ArrayComputationModel()).startswith("ArrayComputationModel(")


def test_array_computation_model_str() -> None:
    assert str(ArrayComputationModel()).startswith("ArrayComputationModel(")


##################
#     argmax     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmax_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmax(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0
        ),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmax_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmax(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1
        ),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmax_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmax(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.int64(9),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmax_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmax(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[4, 4]]),
    )


##################
#     argmin     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmin_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmin(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0
        ),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmin_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmin(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1
        ),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmin_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmin(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.int64(0),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_argmin_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().argmin(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[0, 0]]),
    )


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


###############
#     max     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_max_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().max(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0
        ),
        np.array([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_max_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().max(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1
        ),
        np.array([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_max_axis_none(dtype: np.dtype) -> None:
    assert (
        ArrayComputationModel().max(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype))
        == 9.0
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_max_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().max(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[8, 9]], dtype=dtype),
    )


################
#     mean     #
################


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_mean_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().mean(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0
        ),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_mean_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().mean(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1
        ),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_mean_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().mean(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_mean_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().mean(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[4.0, 5.0]]),
    )


##################
#     median     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_median_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().median(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0
        ),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_median_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().median(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1
        ),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_median_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().median(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_computation_model_median_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ArrayComputationModel().median(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[4.0, 5.0]]),
    )
