from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.nested import (
    index_select_along_batch,
    index_select_along_seq,
    take_along_batch,
    take_along_seq,
)
from tests.unit.array.test_indexing import INDEX_DTYPES

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


##############################################
#     Tests for index_select_along_batch     #
##############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5], [4], [3], [2], [1]]),
                    mask=np.array([[False], [False], [False], [True], [False]]),
                ),
            },
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            "b": np.ma.masked_array(
                data=np.array([[1], [2], [3], [4], [5]]),
                mask=np.array([[False], [True], [False], [False], [False]]),
            ),
        },
    )


############################################
#     Tests for index_select_along_seq     #
############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([4, 3, 2, 1, 0], dtype=dtype)
        ),
        np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5, 4, 3, 2, 1]]),
                    mask=np.array([[False, False, False, True, False]]),
                ),
            },
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            "b": np.ma.masked_array(
                data=np.array([[1, 2, 3, 4, 5]]),
                mask=np.array([[False, True, False, False, False]]),
            ),
        },
    )


######################################
#     Tests for take_along_batch     #
######################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_take_along_batch_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        take_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_take_along_batch_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        take_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5], [4], [3], [2], [1]]),
                    mask=np.array([[False], [False], [False], [True], [False]]),
                ),
            },
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
            "b": np.ma.masked_array(
                data=np.array([[1], [2], [3], [4], [5]]),
                mask=np.array([[False], [True], [False], [False], [False]]),
            ),
        },
    )


####################################
#     Tests for take_along_seq     #
####################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_take_along_seq_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        take_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([4, 3, 2, 1, 0], dtype=dtype)
        ),
        np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_take_along_seq_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        take_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5, 4, 3, 2, 1]]),
                    mask=np.array([[False, False, False, True, False]]),
                ),
            },
            np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        {
            "a": np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            "b": np.ma.masked_array(
                data=np.array([[1, 2, 3, 4, 5]]),
                mask=np.array([[False, True, False, False, False]]),
            ),
        },
    )
