from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.array import (
    gather_along_seq,
    index_select_along_batch,
    index_select_along_seq,
    masked_select_along_batch,
    masked_select_along_seq,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

INDEX_DTYPES = [np.int32, np.int64, np.uint32]


##############################################
#     Tests for index_select_along_batch     #
##############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_2(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            indices=np.array([2, 4], dtype=dtype),
        ),
        np.array([[4, 5], [8, 9]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_5(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            indices=np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_batch_7(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            indices=np.array([4, 3, 2, 1, 0, 2, 0], dtype=dtype),
        ),
        np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1], [4, 5], [0, 1]]),
    )


def test_index_select_along_batch_masked_array() -> None:
    assert objects_are_equal(
        index_select_along_batch(
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
            ),
            indices=np.array([2, 4]),
        ),
        np.ma.masked_array(
            data=np.array([[4, 5], [8, 9]]),
            mask=np.array([[True, False], [True, False]]),
        ),
    )


############################################
#     Tests for index_select_along_seq     #
############################################


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_2(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            indices=np.array([2, 4], dtype=dtype),
        ),
        np.array([[2, 4], [7, 9]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_5(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            indices=np.array([4, 3, 2, 1, 0], dtype=dtype),
        ),
        np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_index_select_along_seq_7(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        index_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            indices=np.array([4, 3, 2, 1, 0, 2, 0], dtype=dtype),
        ),
        np.array([[4, 3, 2, 1, 0, 2, 0], [9, 8, 7, 6, 5, 7, 5]]),
    )


def test_index_select_along_seq_masked_array_1d() -> None:
    assert objects_are_equal(
        index_select_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            ),
            indices=np.array([2, 4]),
        ),
        np.ma.masked_array(
            data=np.array([[2, 4], [7, 9]]),
            mask=np.array([[True, True], [False, False]]),
        ),
    )


###############################################
#     Tests for masked_select_along_batch     #
###############################################


def test_masked_select_along_batch() -> None:
    assert objects_are_equal(
        masked_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            mask=np.array([False, False, True, False, True]),
        ),
        np.array([[4, 5], [8, 9]]),
    )


def test_masked_select_along_batch_masked_array() -> None:
    assert objects_are_equal(
        masked_select_along_batch(
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
            ),
            mask=np.array([False, False, True, False, True]),
        ),
        np.ma.masked_array(
            np.array([[4, 5], [8, 9]]), mask=np.array([[True, False], [True, False]])
        ),
    )


#############################################
#     Tests for masked_select_along_seq     #
#############################################


def test_masked_select_along_seq() -> None:
    assert objects_are_equal(
        masked_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            mask=np.array([False, False, True, False, True]),
        ),
        np.array([[2, 4], [7, 9]]),
    )


def test_masked_select_along_seq_masked_array() -> None:
    assert objects_are_equal(
        masked_select_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, True, False, True],
                        [False, False, False, False, False],
                    ]
                ),
            ),
            mask=np.array([False, False, True, False, True]),
        ),
        np.ma.masked_array(
            data=np.array([[2, 4], [7, 9]]),
            mask=np.array([[True, True], [False, False]]),
        ),
    )


#############################################
#     Tests for index_select validation     #
#############################################


@pytest.mark.parametrize("func", [index_select_along_batch, index_select_along_seq])
@pytest.mark.parametrize("indices", [np.ones((2, 2), dtype=int), np.array(1)])
def test_index_select_indices_not_1d_raises(func: Callable, indices: np.ndarray) -> None:
    with pytest.raises(ValueError, match="indices must be a 1-D array"):
        func(np.ones((2, 3)), indices)


@pytest.mark.parametrize("func", [index_select_along_batch, index_select_along_seq])
def test_index_select_float_indices_raises_numpy_error(func: Callable) -> None:
    with pytest.raises(TypeError, match="Cannot cast array data"):
        func(np.ones((2, 3)), np.array([0.0, 1.0]))


@pytest.mark.parametrize("func", [index_select_along_batch, index_select_along_seq])
def test_index_select_does_not_modify_indices(func: Callable) -> None:
    indices = np.array([1, 0])
    func(np.ones((2, 3)), indices)
    assert objects_are_equal(indices, np.array([1, 0]))


##############################################
#     Tests for masked_select validation     #
##############################################


@pytest.mark.parametrize("func", [masked_select_along_batch, masked_select_along_seq])
@pytest.mark.parametrize(
    "mask", [np.array([0, 1, 2]), np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1], dtype=np.uint8)]
)
def test_masked_select_mask_not_bool_raises(func: Callable, mask: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mask must be a boolean array"):
        func(np.ones((3, 3)), mask)


def test_masked_select_along_batch_integer_mask_regression() -> None:
    # An integer mask was read as indices: rows 0, 1 and 2 of the array.
    with pytest.raises(ValueError, match="mask must be a boolean array"):
        masked_select_along_batch(np.arange(6).reshape(3, 2), np.array([0, 1, 2]))


######################################
#     Tests for gather_along_seq     #
######################################


@pytest.mark.parametrize("shape", [(3, 5), (3, 5, 2), (3, 5, 2, 4)])
def test_gather_along_seq_equal_stack_of_samples(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    array = rng.normal(size=shape)
    indices = rng.integers(0, 5, size=(3, 4))
    expected = np.stack([array[b, indices[b]] for b in range(3)])
    assert objects_are_equal(gather_along_seq(array, indices), expected)


@pytest.mark.parametrize("shape", [(3, 5), (3, 5, 2), (3, 5, 2, 4)])
def test_gather_along_seq_masked_array_equal_stack_of_samples(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=shape)
    array = np.ma.masked_array(data, mask=rng.random(size=shape) > 0.5)
    indices = rng.integers(0, 5, size=(3, 4))
    expected = np.ma.stack([array[b, indices[b]] for b in range(3)])
    assert objects_are_equal(gather_along_seq(array, indices), expected)


def test_gather_along_seq_value() -> None:
    assert objects_are_equal(
        gather_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([[2, 4], [1, 3]])),
        np.array([[2, 4], [6, 8]]),
    )


def test_gather_along_seq_extra_dims_value() -> None:
    assert objects_are_equal(
        gather_along_seq(
            np.array(
                [
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ]
            ),
            np.array([[2, 0], [4, 3]]),
        ),
        np.array([[[4, 5], [0, 1]], [[18, 19], [16, 17]]]),
    )


def test_gather_along_seq_negative_index_counts_from_the_end() -> None:
    assert objects_are_equal(
        gather_along_seq(np.array([[0, 1, 2], [3, 4, 5]]), np.array([[-1], [-2]])),
        np.array([[2], [4]]),
    )


def test_gather_along_seq_last_valid_step() -> None:
    lengths = np.array([2, 3])
    assert objects_are_equal(
        gather_along_seq(np.array([[0, 1, 2], [3, 4, 5]]), (lengths - 1)[:, None]),
        np.array([[1], [5]]),
    )


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_gather_along_seq_index_dtypes(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        gather_along_seq(np.array([[0, 1, 2], [3, 4, 5]]), np.array([[2, 0], [1, 1]], dtype=dtype)),
        np.array([[2, 0], [4, 4]]),
    )


def test_gather_along_seq_zero_indices() -> None:
    assert objects_are_equal(
        gather_along_seq(np.ones((2, 3)), np.zeros((2, 0), dtype=int)), np.ones((2, 0))
    )


def test_gather_along_seq_zero_batch_size() -> None:
    assert objects_are_equal(
        gather_along_seq(np.ones((0, 3)), np.zeros((0, 2), dtype=int)), np.ones((0, 2))
    )


def test_gather_along_seq_does_not_modify_inputs() -> None:
    array = np.arange(6).reshape(2, 3)
    indices = np.array([[2, 0], [1, 1]])
    gather_along_seq(array, indices)
    assert objects_are_equal(array, np.arange(6).reshape(2, 3))


def test_gather_along_seq_does_not_modify_indices() -> None:
    array = np.arange(6).reshape(2, 3)
    indices = np.array([[2, 0], [1, 1]])
    gather_along_seq(array, indices)
    assert objects_are_equal(indices, np.array([[2, 0], [1, 1]]))


def test_gather_along_seq_batch_size_mismatch_raises() -> None:
    # take_along_seq(np.arange(6).reshape(2, 3), np.array([[0, 1]])) dropped the row 1.
    with pytest.raises(ValueError, match="batch size of the indices"):
        gather_along_seq(np.arange(6).reshape(2, 3), np.array([[0, 1]]))


@pytest.mark.parametrize("indices", [np.array([0, 1]), np.zeros((2, 1, 1), dtype=int)])
def test_gather_along_seq_indices_not_2d_raises(indices: np.ndarray) -> None:
    with pytest.raises(ValueError, match="indices must be a 2-D array"):
        gather_along_seq(np.ones((2, 3)), indices)


def test_gather_along_seq_1d_array_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        gather_along_seq(np.ones(3), np.zeros((3, 1), dtype=int))


def test_gather_along_seq_float_indices_raises_numpy_error() -> None:
    with pytest.raises(IndexError, match="must be an integer array"):
        gather_along_seq(np.ones((2, 3)), np.zeros((2, 1)))


def test_gather_along_seq_out_of_range_index_raises_numpy_error() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        gather_along_seq(np.ones((2, 3)), np.array([[3], [0]]))
