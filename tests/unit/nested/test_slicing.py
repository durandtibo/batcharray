from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from batcharray.nested import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
    slice_along_batch,
    slice_along_seq,
    split_along_batch,
    split_along_seq,
)

#######################################
#     Tests for chunk_along_batch     #
#######################################


def test_chunk_along_batch_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            chunks=3,
        ),
        [
            {"a": np.array([[0, 1], [2, 3]]), "b": np.array([4, 3])},
            {"a": np.array([[4, 5], [6, 7]]), "b": np.array([2, 1])},
            {"a": np.array([[8, 9]]), "b": np.array([0])},
        ],
    )


def test_chunk_along_batch_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            chunks=5,
        ),
        [
            {"a": np.array([[0, 1]]), "b": np.array([4])},
            {"a": np.array([[2, 3]]), "b": np.array([3])},
            {"a": np.array([[4, 5]]), "b": np.array([2])},
            {"a": np.array([[6, 7]]), "b": np.array([1])},
            {"a": np.array([[8, 9]]), "b": np.array([0])},
        ],
    )


#####################################
#     Tests for chunk_along_seq     #
#####################################


def test_chunk_along_seq_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            chunks=3,
        ),
        [
            {"a": np.array([[0, 1], [5, 6]]), "b": np.array([[4, 3]])},
            {"a": np.array([[2, 3], [7, 8]]), "b": np.array([[2, 1]])},
            {"a": np.array([[4], [9]]), "b": np.array([[0]])},
        ],
    )


def test_chunk_along_seq_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            chunks=5,
        ),
        [
            {"a": np.array([[0], [5]]), "b": np.array([[4]])},
            {"a": np.array([[1], [6]]), "b": np.array([[3]])},
            {"a": np.array([[2], [7]]), "b": np.array([[2]])},
            {"a": np.array([[3], [8]]), "b": np.array([[1]])},
            {"a": np.array([[4], [9]]), "b": np.array([[0]])},
        ],
    )


########################################
#     Tests for select_along_batch     #
########################################


def test_select_along_batch_array() -> None:
    assert objects_are_equal(
        select_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), index=2),
        np.array([4, 5]),
    )


def test_select_along_batch_dict() -> None:
    assert objects_are_equal(
        select_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            index=2,
        ),
        {"a": np.array([4, 5]), "b": np.int64(2)},
    )


def test_select_along_batch_nested() -> None:
    assert objects_are_equal(
        select_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
                "list": [np.array([[5], [6], [7], [8], [9]])],
                "dict": {"c": np.ones((5, 2))},
            },
            index=2,
        ),
        {
            "a": np.array([4, 5]),
            "b": np.int64(2),
            "list": [np.array([7])],
            "dict": {"c": np.array([1.0, 1.0])},
        },
    )


######################################
#     Tests for select_along_seq     #
######################################


def test_select_along_seq_array() -> None:
    assert objects_are_equal(
        select_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), index=2),
        np.array([2, 7]),
    )


def test_select_along_seq_dict() -> None:
    assert objects_are_equal(
        select_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            index=2,
        ),
        {"a": np.array([2, 7]), "b": np.array([2])},
    )


def test_select_along_seq_nested() -> None:
    assert objects_are_equal(
        select_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
                "list": [np.array([[5, 6, 7, 8, 9]])],
                "dict": {"c": np.ones((2, 5))},
            },
            index=2,
        ),
        {
            "a": np.array([2, 7]),
            "b": np.array([2]),
            "list": [np.array([7])],
            "dict": {"c": np.array([1.0, 1.0])},
        },
    )


#######################################
#     Tests for slice_along_batch     #
#######################################


def test_slice_along_batch_array() -> None:
    assert objects_are_equal(
        slice_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])),
        np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_array_start_2() -> None:
    assert objects_are_equal(
        slice_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), start=2),
        np.array([[4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_array_stop_3() -> None:
    assert objects_are_equal(
        slice_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), stop=3),
        np.array([[0, 1], [2, 3], [4, 5]]),
    )


def test_slice_along_batch_array_stop_100() -> None:
    assert objects_are_equal(
        slice_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), stop=100),
        np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    )


def test_slice_along_batch_array_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), step=2),
        np.array([[0, 1], [4, 5], [8, 9]]),
    )


def test_slice_along_batch_array_start_1_stop_4_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), start=1, stop=4, step=2
        ),
        np.array([[2, 3], [6, 7]]),
    )


def test_slice_along_batch_dict() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            }
        ),
        {
            "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            "b": np.array([4, 3, 2, 1, 0]),
        },
    )


def test_slice_along_batch_dict_start_2() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            start=2,
        ),
        {"a": np.array([[4, 5], [6, 7], [8, 9]]), "b": np.array([2, 1, 0])},
    )


def test_slice_along_batch_dict_stop_3() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            stop=3,
        ),
        {"a": np.array([[0, 1], [2, 3], [4, 5]]), "b": np.array([4, 3, 2])},
    )


def test_slice_along_batch_dict_stop_100() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            stop=100,
        ),
        {
            "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
            "b": np.array([4, 3, 2, 1, 0]),
        },
    )


def test_slice_along_batch_dict_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            step=2,
        ),
        {"a": np.array([[0, 1], [4, 5], [8, 9]]), "b": np.array([4, 2, 0])},
    )


def test_slice_along_batch_dict_start_1_stop_4_step_2() -> None:
    assert objects_are_equal(
        slice_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            start=1,
            stop=4,
            step=2,
        ),
        {"a": np.array([[2, 3], [6, 7]]), "b": np.array([3, 1])},
    )


#####################################
#     Tests for slice_along_seq     #
#####################################


def test_slice_along_seq_array() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])),
        np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
    )


def test_slice_along_seq_array_start_2() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), start=2),
        np.array([[2, 3, 4], [7, 6, 5]]),
    )


def test_slice_along_seq_array_stop_3() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), stop=3),
        np.array([[0, 1, 2], [9, 8, 7]]),
    )


def test_slice_along_seq_array_stop_100() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), stop=100),
        np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
    )


def test_slice_along_seq_array_step_2() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]), step=2),
        np.array([[0, 2, 4], [9, 7, 5]]),
    )


def test_slice_along_seq_array_start_1_stop_4_step_2() -> None:
    assert objects_are_equal(
        slice_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), start=1, stop=4, step=2),
        np.array([[1, 3], [6, 8]]),
    )


def test_slice_along_seq_dict() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            }
        ),
        {
            "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
            "b": np.array([[4, 3, 2, 1, 0]]),
        },
    )


def test_slice_along_seq_dict_start_2() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            start=2,
        ),
        {
            "a": np.array([[2, 3, 4], [7, 6, 5]]),
            "b": np.array([[2, 1, 0]]),
        },
    )


def test_slice_along_seq_dict_stop_3() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            stop=3,
        ),
        {
            "a": np.array([[0, 1, 2], [9, 8, 7]]),
            "b": np.array([[4, 3, 2]]),
        },
    )


def test_slice_along_seq_dict_stop_100() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            stop=100,
        ),
        {
            "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
            "b": np.array([[4, 3, 2, 1, 0]]),
        },
    )


def test_slice_along_seq_dict_step_2() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            step=2,
        ),
        {"a": np.array([[0, 2, 4], [9, 7, 5]]), "b": np.array([[4, 2, 0]])},
    )


def test_slice_along_seq_dict_start_1_stop_4_step_2() -> None:
    assert objects_are_equal(
        slice_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            start=1,
            stop=4,
            step=2,
        ),
        {"a": np.array([[1, 3], [8, 6]]), "b": np.array([[3, 1]])},
    )


#######################################
#     Tests for split_along_batch     #
#######################################


def test_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        split_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            split_size_or_sections=1,
        ),
        [
            {"a": np.array([[0, 1]]), "b": np.array([4])},
            {"a": np.array([[2, 3]]), "b": np.array([3])},
            {"a": np.array([[4, 5]]), "b": np.array([2])},
            {"a": np.array([[6, 7]]), "b": np.array([1])},
            {"a": np.array([[8, 9]]), "b": np.array([0])},
        ],
    )


def test_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        split_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            split_size_or_sections=2,
        ),
        [
            {"a": np.array([[0, 1], [2, 3]]), "b": np.array([4, 3])},
            {"a": np.array([[4, 5], [6, 7]]), "b": np.array([2, 1])},
            {"a": np.array([[8, 9]]), "b": np.array([0])},
        ],
    )


def test_split_along_batch_split_size_list() -> None:
    assert objects_are_equal(
        split_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
                "b": np.array([4, 3, 2, 1, 0]),
            },
            split_size_or_sections=[2, 2, 1],
        ),
        [
            {"a": np.array([[0, 1], [2, 3]]), "b": np.array([4, 3])},
            {"a": np.array([[4, 5], [6, 7]]), "b": np.array([2, 1])},
            {"a": np.array([[8, 9]]), "b": np.array([0])},
        ],
    )


#####################################
#     Tests for split_along_seq     #
#####################################


def test_split_along_seq_split_size_1() -> None:
    assert objects_are_equal(
        split_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            split_size_or_sections=1,
        ),
        [
            {"a": np.array([[0], [5]]), "b": np.array([[4]])},
            {"a": np.array([[1], [6]]), "b": np.array([[3]])},
            {"a": np.array([[2], [7]]), "b": np.array([[2]])},
            {"a": np.array([[3], [8]]), "b": np.array([[1]])},
            {"a": np.array([[4], [9]]), "b": np.array([[0]])},
        ],
    )


def test_split_along_seq_split_size_2() -> None:
    assert objects_are_equal(
        split_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            split_size_or_sections=2,
        ),
        [
            {"a": np.array([[0, 1], [5, 6]]), "b": np.array([[4, 3]])},
            {"a": np.array([[2, 3], [7, 8]]), "b": np.array([[2, 1]])},
            {"a": np.array([[4], [9]]), "b": np.array([[0]])},
        ],
    )


def test_split_along_seq_split_size_list() -> None:
    assert objects_are_equal(
        split_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            split_size_or_sections=[2, 2, 1],
        ),
        [
            {"a": np.array([[0, 1], [5, 6]]), "b": np.array([[4, 3]])},
            {"a": np.array([[2, 3], [7, 8]]), "b": np.array([[2, 1]])},
            {"a": np.array([[4], [9]]), "b": np.array([[0]])},
        ],
    )
