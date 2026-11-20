from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.nested import (
    gather_along_seq,
    index_select_along_batch,
    index_select_along_seq,
    masked_select_along_batch,
    masked_select_along_seq,
)
from tests.unit.array.test_indexing import INDEX_DTYPES

if TYPE_CHECKING:
    from collections.abc import Callable

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
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            np.array([4, 3, 2, 1, 0], dtype=dtype),
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


###############################################
#     Tests for masked_select_along_batch     #
###############################################


def test_masked_select_along_batch_array() -> None:
    assert objects_are_equal(
        masked_select_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            mask=np.array([False, False, True, False, True]),
        ),
        np.array([[4, 5], [8, 9]]),
    )


def test_masked_select_along_batch_dict() -> None:
    assert objects_are_equal(
        masked_select_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5], [4], [3], [2], [1]]),
                    mask=np.array([[False], [False], [False], [True], [False]]),
                ),
            },
            mask=np.array([False, False, True, False, True]),
        ),
        {
            "a": np.array([[4, 5], [8, 9]]),
            "b": np.ma.masked_array(data=np.array([[3], [1]]), mask=np.array([[False], [False]])),
        },
    )


#############################################
#     Tests for masked_select_along_seq     #
#############################################


def test_masked_select_along_seq_array() -> None:
    assert objects_are_equal(
        masked_select_along_seq(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            mask=np.array([False, False, True, False, True]),
        ),
        np.array([[2, 4], [7, 9]]),
    )


def test_masked_select_along_seq_dict() -> None:
    assert objects_are_equal(
        masked_select_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[5, 4, 3, 2, 1]]),
                    mask=np.array([[False, False, False, True, False]]),
                ),
            },
            mask=np.array([False, False, True, False, True]),
        ),
        {
            "a": np.array([[2, 4], [7, 9]]),
            "b": np.ma.masked_array(data=np.array([[3, 1]]), mask=np.array([[False, False]])),
        },
    )


######################################
#     Tests for gather_along_seq     #
######################################


def test_gather_along_seq_array() -> None:
    assert objects_are_equal(
        gather_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), np.array([[0, 4], [1, 3]])),
        np.array([[0, 4], [6, 8]]),
    )


def test_gather_along_seq_dict() -> None:
    assert objects_are_equal(
        gather_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]),
            },
            np.array([[0, 4], [1, 3]]),
        ),
        {"a": np.array([[0, 4], [6, 8]]), "b": np.array([[4, 0], [8, 6]])},
    )


def test_gather_along_seq_shares_one_index_between_leaves_of_different_feature_shapes() -> None:
    rng = np.random.default_rng(0)
    data = {
        "a": rng.normal(size=(3, 5)),
        "b": rng.normal(size=(3, 5, 2)),
        "c": [rng.normal(size=(3, 5, 2, 4))],
    }
    indices = rng.integers(0, 5, size=(3, 4))
    out = gather_along_seq(data, indices)
    assert objects_are_equal(
        out,
        {
            "a": np.stack([data["a"][b, indices[b]] for b in range(3)]),
            "b": np.stack([data["b"][b, indices[b]] for b in range(3)]),
            "c": [np.stack([data["c"][0][b, indices[b]] for b in range(3)])],
        },
    )


def test_gather_along_seq_masked_array() -> None:
    assert objects_are_equal(
        gather_along_seq(
            {
                "a": np.ma.masked_array(
                    data=np.array([[0, 1, 2], [3, 4, 5]]),
                    mask=np.array([[False, True, False], [False, False, True]]),
                )
            },
            np.array([[1, 2], [0, 2]]),
        ),
        {
            "a": np.ma.masked_array(
                data=np.array([[1, 2], [3, 5]]),
                mask=np.array([[True, False], [False, True]]),
            )
        },
    )


def test_gather_along_seq_batch_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="batch size of the indices"):
        gather_along_seq({"a": np.arange(6).reshape(2, 3)}, np.array([[0, 1]]))


def test_gather_along_seq_indices_not_2d_raises() -> None:
    with pytest.raises(ValueError, match="indices must be a 2-D array"):
        gather_along_seq({"a": np.ones((2, 3))}, np.array([0, 1]))


def test_gather_along_seq_1d_leaf_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        gather_along_seq({"a": np.ones((2, 3)), "b": np.ones(2)}, np.zeros((2, 1), dtype=int))


def test_gather_along_seq_keeps_container_types() -> None:
    out = gather_along_seq(([np.ones((2, 3))], {"a": np.ones((2, 3))}), np.zeros((2, 1), dtype=int))
    assert (type(out), type(out[0]), type(out[1])) == (tuple, list, dict)


@pytest.mark.parametrize("func", [index_select_along_batch, index_select_along_seq])
def test_index_select_indices_not_1d_raises(func: Callable) -> None:
    with pytest.raises(ValueError, match="indices must be a 1-D array"):
        func({"a": np.ones((2, 3))}, np.ones((2, 2), dtype=int))


@pytest.mark.parametrize("func", [masked_select_along_batch, masked_select_along_seq])
def test_masked_select_mask_not_bool_raises(func: Callable) -> None:
    with pytest.raises(ValueError, match="mask must be a boolean array"):
        func({"a": np.ones((3, 3))}, np.array([0, 1, 2]))
