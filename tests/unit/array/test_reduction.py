from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_allclose, objects_are_equal

from batcharray.array import (
    all_along_batch,
    all_along_seq,
    amax_along_batch,
    amax_along_seq,
    amin_along_batch,
    amin_along_seq,
    any_along_batch,
    any_along_seq,
    argmax_along_batch,
    argmax_along_seq,
    argmin_along_batch,
    argmin_along_seq,
    max_along_batch,
    max_along_seq,
    mean_along_batch,
    mean_along_seq,
    median_along_batch,
    median_along_seq,
    min_along_batch,
    min_along_seq,
    prod_along_batch,
    prod_along_seq,
    std_along_batch,
    std_along_seq,
    sum_along_batch,
    sum_along_seq,
    var_along_batch,
    var_along_seq,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

DTYPES = (np.float64, np.int64)
FLOATING_DTYPES = (np.float32, np.float64)


######################################
#     Tests for amax_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amax_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amax_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[8, 9]], dtype=dtype),
    )


def test_amax_along_batch_masked_array() -> None:
    assert objects_are_equal(
        amax_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([6, 9]), mask=np.array([[False, False]])),
    )


####################################
#     Tests for amax_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amax_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amax_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amax_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[4], [9]], dtype=dtype),
    )


def test_amax_along_seq_masked_array() -> None:
    assert objects_are_equal(
        amax_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([3, 9]), mask=np.array([[False], [False]])),
    )


######################################
#     Tests for amin_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amin_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amin_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[0, 1]], dtype=dtype),
    )


def test_amin_along_batch_masked_array() -> None:
    assert objects_are_equal(
        amin_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, True],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([0, 3]), mask=np.array([[False, False]])),
    )


####################################
#     Tests for amin_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amin_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_amin_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        amin_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[0], [5]], dtype=dtype),
    )


def test_amin_along_seq_masked_array() -> None:
    assert objects_are_equal(
        amin_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [True, False, True, False, False],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([1, 5]), mask=np.array([[False], [False]])),
    )


########################################
#     Tests for argmax_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmax_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmax_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[4, 4]]),
    )


def test_argmax_along_batch_masked_array() -> None:
    assert objects_are_equal(
        argmax_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.array([3, 4]),
    )


######################################
#     Tests for argmax_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmax_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([4, 4]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmax_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[4], [4]]),
    )


def test_argmax_along_seq_masked_array() -> None:
    assert objects_are_equal(
        argmax_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.array([3, 4]),
    )


########################################
#     Tests for argmin_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmin_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmin_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[0, 0]]),
    )


def test_argmin_along_batch_masked_array() -> None:
    assert objects_are_equal(
        argmin_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.array([0, 0]),
    )


######################################
#     Tests for argmin_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmin_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([0, 0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmin_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        argmin_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[0], [0]]),
    )


def test_argmin_along_seq_masked_array() -> None:
    assert objects_are_equal(
        argmin_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [True, False, True, False, False],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.array([1, 0]),
    )


#####################################
#     Tests for max_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        max_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([8, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        max_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[8, 9]], dtype=dtype),
    )


def test_max_along_batch_masked_array() -> None:
    assert objects_are_equal(
        max_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([6, 9]), mask=np.array([[False, False]])),
    )


###################################
#     Tests for max_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        max_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([4, 9], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        max_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[4], [9]], dtype=dtype),
    )


def test_max_along_seq_masked_array() -> None:
    assert objects_are_equal(
        max_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([3, 9]), mask=np.array([[False], [False]])),
    )


######################################
#     Tests for mean_along_batch     #
######################################


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        mean_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([4.0, 5.0], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        mean_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[4.0, 5.0]], dtype=dtype),
    )


def test_mean_along_batch_masked_array() -> None:
    assert objects_are_equal(
        mean_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [True, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([3.0, 5.0]), mask=np.array([[False, False]])),
    )


####################################
#     Tests for mean_along_seq     #
####################################


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        mean_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([2.0, 7.0], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_mean_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        mean_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[2.0], [7.0]], dtype=dtype),
    )


def test_mean_along_seq_masked_array() -> None:
    assert objects_are_allclose(
        mean_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(
            data=np.array([1.3333333333333333, 7.0]), mask=np.array([[False], [False]])
        ),
    )


########################################
#     Tests for median_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        median_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([4.0, 5.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        median_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[4.0, 5.0]]),
    )


def test_median_along_batch_masked_array() -> None:
    assert objects_are_equal(
        median_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([2.0, 5.0]), mask=np.array([[False, False]])),
    )


######################################
#     Tests for median_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        median_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([2.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        median_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[2.0], [7.0]]),
    )


def test_median_along_seq_masked_array() -> None:
    assert objects_are_equal(
        median_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([1.0, 7.0]), mask=np.array([[False], [False]])),
    )


#####################################
#     Tests for min_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        min_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([0, 1], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        min_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[0, 1]], dtype=dtype),
    )


def test_min_along_batch_masked_array() -> None:
    assert objects_are_equal(
        min_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([0, 1]), mask=np.array([[False, False]])),
    )


###################################
#     Tests for min_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        min_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([0, 5], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        min_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[0], [5]], dtype=dtype),
    )


def test_min_along_seq_masked_array() -> None:
    assert objects_are_equal(
        min_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [True, False, True, False, False],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([1, 5]), mask=np.array([[False], [False]])),
    )


######################################
#     Tests for prod_along_batch     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        prod_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([0, 945], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        prod_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[0, 945]], dtype=dtype),
    )


def test_prod_along_batch_masked_array() -> None:
    assert objects_are_equal(
        prod_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [True, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([96, 945]), mask=np.array([[False, False]])),
    )


####################################
#     Tests for prod_along_seq     #
####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        prod_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([0, 15120], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_prod_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        prod_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[0], [15120]], dtype=dtype),
    )


def test_prod_along_seq_masked_array() -> None:
    assert objects_are_equal(
        prod_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [True, False, False, True, False],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([8, 15120]), mask=np.array([[False], [False]])),
    )


#####################################
#     Tests for sum_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        sum_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([20, 25], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        sum_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
            keepdims=True,
        ),
        np.array([[20, 25]], dtype=dtype),
    )


def test_sum_along_batch_masked_array() -> None:
    assert objects_are_equal(
        sum_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [True, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([8, 25]), mask=np.array([[False, False]])),
    )


###################################
#     Tests for sum_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        sum_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([10, 35], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        sum_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[10], [35]], dtype=dtype),
    )


def test_sum_along_seq_masked_array() -> None:
    assert objects_are_equal(
        sum_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([4, 35]), mask=np.array([[False], [False]])),
    )


#############################################
#     Tests for axis equivalence            #
#############################################

AXIS_EQUIVALENCES = [
    ("amax", np.amax),
    ("amin", np.amin),
    ("argmax", np.argmax),
    ("argmin", np.argmin),
    ("max", np.max),
    ("mean", np.mean),
    ("median", np.median),
    ("min", np.min),
    ("prod", np.prod),
    ("sum", np.sum),
]


@pytest.mark.parametrize(("name", "reference"), AXIS_EQUIVALENCES)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_along_batch_and_seq_match_numpy_axis(
    name: str, reference: object, keepdims: bool, seed: int
) -> None:
    import batcharray.array as ba

    x = np.random.default_rng(seed).integers(1, 5, size=(4, 5, 3)).astype(np.float64)
    assert np.array_equal(
        getattr(ba, f"{name}_along_batch")(x, keepdims=keepdims),
        reference(x, axis=0, keepdims=keepdims),
    )
    assert np.array_equal(
        getattr(ba, f"{name}_along_seq")(x, keepdims=keepdims),
        reference(x, axis=1, keepdims=keepdims),
    )


#####################################
#     Tests for std_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([2.8284271247461903, 2.8284271247461903]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdims=True
        ),
        np.array([[2.8284271247461903, 2.8284271247461903]]),
    )


def test_std_along_batch_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        std_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_batch_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), ddof=1),
        np.array([3.1622776601683795, 3.1622776601683795]),
    )


def test_std_along_batch_masked_array() -> None:
    assert objects_are_allclose(
        std_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, True], [True, True], [True, True]]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([1.632993161855452, 1.0]), mask=np.array([False, False])),
    )


###################################
#     Tests for std_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([1.4142135623730951, 1.4142135623730951]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[1.4142135623730951], [1.4142135623730951]]),
    )


def test_std_along_seq_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        std_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_along_seq_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        std_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), ddof=1),
        np.array([1.5811388300841898, 1.5811388300841898]),
    )


def test_std_along_seq_masked_array() -> None:
    assert objects_are_allclose(
        std_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([1.4142135623730951, 0.0]), mask=np.array([False, False])),
    )


#####################################
#     Tests for var_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype)),
        np.array([8.0, 8.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_batch_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), keepdims=True
        ),
        np.array([[8.0, 8.0]]),
    )


def test_var_along_batch_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        var_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_batch_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), ddof=1),
        np.array([10.0, 10.0]),
    )


def test_var_along_batch_masked_array() -> None:
    assert objects_are_allclose(
        var_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [False, True], [True, True], [True, True]]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([2.6666666666666665, 1.0]), mask=np.array([False, False])),
    )


###################################
#     Tests for var_along_seq     #
###################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
        np.array([2.0, 2.0]),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_seq_keepdims_true(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), keepdims=True),
        np.array([[2.0], [2.0]]),
    )


def test_var_along_seq_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        var_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_along_seq_ddof_1(dtype: DTypeLike) -> None:
    assert objects_are_allclose(
        var_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), ddof=1),
        np.array([2.5, 2.5]),
    )


def test_var_along_seq_masked_array() -> None:
    assert objects_are_allclose(
        var_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]]
                ),
            )
        ),
        np.ma.masked_array(data=np.array([2.0, 0.0]), mask=np.array([False, False])),
    )


#####################################
#     Tests for any_along_batch     #
#####################################


def test_any_along_batch() -> None:
    assert objects_are_equal(
        any_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]])
        ),
        np.array([True, True]),
    )


def test_any_along_batch_keepdims_true() -> None:
    assert objects_are_equal(
        any_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]]),
            keepdims=True,
        ),
        np.array([[True, True]]),
    )


def test_any_along_batch_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        any_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]]),
            True,
        )


###################################
#     Tests for any_along_seq     #
###################################


def test_any_along_seq() -> None:
    assert objects_are_equal(
        any_along_seq(np.array([[True, False, True, True, True], [True, True, True, True, True]])),
        np.array([True, True]),
    )


def test_any_along_seq_keepdims_true() -> None:
    assert objects_are_equal(
        any_along_seq(
            np.array([[True, False, True, True, True], [True, True, True, True, True]]),
            keepdims=True,
        ),
        np.array([[True], [True]]),
    )


def test_any_along_seq_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        any_along_seq(
            np.array([[True, False, True, True, True], [True, True, True, True, True]]), True
        )


#####################################
#     Tests for all_along_batch     #
#####################################


def test_all_along_batch() -> None:
    assert objects_are_equal(
        all_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]])
        ),
        np.array([False, False]),
    )


def test_all_along_batch_keepdims_true() -> None:
    assert objects_are_equal(
        all_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]]),
            keepdims=True,
        ),
        np.array([[False, False]]),
    )


def test_all_along_batch_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        all_along_batch(
            np.array([[True, False], [True, True], [False, False], [True, True], [True, False]]),
            True,
        )


###################################
#     Tests for all_along_seq     #
###################################


def test_all_along_seq() -> None:
    assert objects_are_equal(
        all_along_seq(np.array([[True, False, True, True, True], [True, True, True, True, True]])),
        np.array([False, True]),
    )


def test_all_along_seq_keepdims_true() -> None:
    assert objects_are_equal(
        all_along_seq(
            np.array([[True, False, True, True, True], [True, True, True, True, True]]),
            keepdims=True,
        ),
        np.array([[False], [True]]),
    )


def test_all_along_seq_keepdims_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional"):
        all_along_seq(
            np.array([[True, False, True, True, True], [True, True, True, True, True]]), True
        )
