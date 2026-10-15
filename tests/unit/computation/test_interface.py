from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_allclose, objects_are_equal

from batcharray import computation as cmpt
from batcharray.types import SORT_KINDS

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from batcharray.types import SortKind

DTYPES = (np.float64, np.int64)


##################
#     argmax     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmax(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmax(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmax(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.int64(9),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmax(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[4, 4]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmax(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.array([3, 4]),
    )


##################
#     argmin     #
##################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmin(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmin(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmin(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.int64(0),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmin(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[0, 0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argmin(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [True, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [False, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.array([1, 0]),
    )


###################
#     argsort     #
###################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[1, 14, 4, 13, 6], [3, 0, 10, 5, 11], [2, 8, 7, 12, 9]], dtype=dtype),
            axis=0,
        ),
        np.array([[0, 1, 0, 1, 0], [2, 2, 2, 2, 2], [1, 0, 1, 0, 1]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_0_stable(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 4, 1, 3, 0]], dtype=dtype),
            axis=0,
            kind="stable",
        ),
        np.array([[0, 2, 0, 0, 2], [1, 0, 2, 2, 0], [2, 1, 1, 1, 1]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[1, 14, 4, 13, 6], [3, 0, 10, 5, 11], [2, 8, 7, 12, 9]], dtype=dtype),
            axis=1,
        ),
        np.array([[0, 2, 4, 3, 1], [1, 0, 3, 2, 4], [0, 2, 1, 4, 3]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_1_stable(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype),
            axis=1,
            kind="stable",
        ),
        np.array([[2, 3, 0, 4, 1], [0, 4, 1, 2, 3], [4, 1, 0, 2, 3]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[1, 14, 4, 13, 6], [3, 0, 10, 5, 11], [2, 8, 7, 12, 9]], dtype=dtype)
        ),
        np.array([6, 0, 10, 5, 2, 8, 4, 12, 11, 14, 7, 9, 13, 3, 1]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argsort_axis_none_stable(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype),
            kind="stable",
        ),
        np.array([2, 14, 3, 0, 4, 5, 1, 9, 11, 6, 7, 8, 10, 12, 13]),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_argsort_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 4, 1, 3, 0]]),
            axis=0,
            kind=kind,
        ),
        np.array([[0, 2, 0, 0, 2], [1, 0, 2, 2, 0], [2, 1, 1, 1, 1]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_auto_array_computation_model_argsort_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.argsort(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 4, 1, 3, 0]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.array([[0, 2, 0, 0, 2], [1, 0, 1, 2, 1], [2, 1, 2, 1, 0]]),
    )


#######################
#     concatenate     #
#######################


def test_concatenate_array_axis_0() -> None:
    out = cmpt.concatenate(
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        axis=0,
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


###############
#     max     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.max(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.max(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_axis_none(dtype: DTypeLike) -> None:
    assert cmpt.max(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)) == 9


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.max(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[8, 9]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.max(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([6, 9], dtype=dtype), mask=np.array([False, False])),
    )


################
#     mean     #
################


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.mean(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.mean(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[4.0, 5.0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.mean(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
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
def test_median_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.median(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(4.5),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.median(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[4.0, 5.0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.median(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([2.0, 5.0]), mask=np.array([False, False])),
    )


###############
#     min     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.min(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.min(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_axis_none(dtype: DTypeLike) -> None:
    assert cmpt.min(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)) == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.min(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            axis=0,
            keepdims=True,
        ),
        np.array([[0, 1]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.min(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                mask=np.array(
                    [
                        [True, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [False, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([2, 1], dtype=dtype), mask=np.array([False, False])),
    )


################
#     sort     #
################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.sort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype),
            axis=0,
        ),
        np.array([[3, 5, 0, 2, 0], [4, 5, 8, 8, 4], [8, 7, 8, 8, 5]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.sort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype),
            axis=1,
        ),
        np.array([[0, 2, 3, 4, 5], [4, 5, 7, 8, 8], [0, 5, 8, 8, 8]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.sort(np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype)),
        np.array([0, 0, 2, 3, 4, 4, 5, 5, 5, 7, 8, 8, 8, 8, 8], dtype=dtype),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_sort_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        cmpt.sort(
            np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]]),
            axis=0,
            kind=kind,
        ),
        np.array([[3, 5, 0, 2, 0], [4, 5, 8, 8, 4], [8, 7, 8, 8, 5]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_masked_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        cmpt.sort(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]], dtype=dtype),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(
            data=np.array([[3, 5, 0, 2, 0], [4, 5, 8, 8, 5], [8, 7, 8, 8, 4]], dtype=dtype),
            mask=np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, True, True, True],
                ]
            ),
        ),
    )


################
#     prod     #
################


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.prod(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([0, 945], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.prod(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([0, 15120], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.prod(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        dtype(0),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.prod(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[0, 945]], dtype=dtype),
    )


def test_prod_masked_array() -> None:
    assert objects_are_allclose(
        cmpt.prod(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, False], [False, True], [True, True]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([0, 15]), mask=np.array([False, False])),
    )


###############
#     std     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.std(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([2.8284271247461903, 2.8284271247461903]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.std(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([1.4142135623730951, 1.4142135623730951]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.std(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(2.8722813232690143),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.std(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[2.8284271247461903, 2.8284271247461903]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.std(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, ddof=1),
        np.array([3.1622776601683795, 3.1622776601683795]),
    )


def test_std_masked_array() -> None:
    assert objects_are_allclose(
        cmpt.std(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, False], [False, True], [True, True]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(
            data=np.array([2.23606797749979, 1.632993161855452]), mask=np.array([False, False])
        ),
    )


###############
#     sum     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.sum(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([20, 25], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.sum(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([10, 35], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.sum(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        dtype(45),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.sum(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[20, 25]], dtype=dtype),
    )


def test_sum_masked_array() -> None:
    assert objects_are_allclose(
        cmpt.sum(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, False], [False, True], [True, True]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([12, 9]), mask=np.array([False, False])),
    )


###############
#     var     #
###############


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_axis_0(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.var(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0),
        np.array([8.0, 8.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_axis_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.var(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), axis=1),
        np.array([2.0, 2.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_axis_none(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.var(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.float64(8.25),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.var(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, keepdims=True
        ),
        np.array([[8.0, 8.0]]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        cmpt.var(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), axis=0, ddof=1),
        np.array([10.0, 10.0]),
    )


def test_var_masked_array() -> None:
    assert objects_are_allclose(
        cmpt.var(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, False], [False, True], [True, True]]
                ),
            ),
            axis=0,
        ),
        np.ma.masked_array(data=np.array([5.0, 2.6666666666666665])),
    )
