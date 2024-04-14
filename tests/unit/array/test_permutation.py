from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.array import permute_along_batch, permute_along_seq

INDEX_DTYPES = [np.int32, np.int64, np.uint32]

#########################################
#     Tests for permute_along_batch     #
#########################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        permute_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


def test_permute_along_batch_incorrect_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"permutation shape \(.*\) is not compatible with array shape \(.*\)",
    ):
        permute_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), np.array([4, 3, 2, 1, 0, 2, 0])
        )


def test_permute_along_batch_masked_array() -> None:
    assert objects_are_equal(
        permute_along_batch(
            np.ma.masked_array(
                data=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                mask=np.array(
                    [[False, False], [False, False], [True, False], [False, False], [True, False]]
                ),
            ),
            np.array([4, 3, 2, 1, 0]),
        ),
        np.ma.masked_array(
            data=np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            mask=np.array(
                [[True, False], [False, False], [True, False], [False, False], [False, False]]
            ),
        ),
    )


#######################################
#     Tests for permute_along_seq     #
#######################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_permute_along_seq(dtype: np.dtype) -> None:
    assert objects_are_equal(
        permute_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([4, 3, 2, 1, 0], dtype=dtype)
        ),
        np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


def test_permute_along_seq_incorrect_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"permutation shape \(.*\) is not compatible with array shape \(.*\)",
    ):
        permute_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([4, 3, 2, 1, 0, 2, 0])
        )


def test_permute_along_seq_masked_array() -> None:
    assert objects_are_equal(
        permute_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [[False, False, True, False, True], [False, False, False, False, False]]
                ),
            ),
            np.array([4, 3, 2, 1, 0]),
        ),
        np.ma.masked_array(
            data=np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            mask=np.array([[True, False, True, False, False], [False, False, False, False, False]]),
        ),
    )
