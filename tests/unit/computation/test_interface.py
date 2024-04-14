from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray import computation as cmpt

DTYPES = (np.float64, np.int64)

#######################
#     concatenate     #
#######################


def test_concatenate_array_axis_0() -> None:
    out = cmpt.concatenate(
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])], axis=0
    )
    assert objects_are_equal(out, np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


def test_concatenate_masked_array_axis_0() -> None:
    out = cmpt.concatenate(
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
    )
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


##################
#     mean     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.mean(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[4.0, 5.0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_masked_array(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.mean(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [[False, False], [False, False], [True, False], [False, False], [True, False]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([2.6666666666666665, 5.0]), mask=np.array([False, False])),
    )


##################
#     median     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_axis_0(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_axis_1(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_axis_none(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_keepdims_true(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.median(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[4.0, 5.0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_masked_array(dtype: np.dtype) -> None:
    assert objects_are_equal(
        cmpt.median(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [[False, False], [False, False], [True, False], [False, False], [True, False]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([2.0, 5.0]), mask=np.array([False, False])),
    )
