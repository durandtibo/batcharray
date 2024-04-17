from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from batcharray.array import (
    chunk_along_batch,
    chunk_along_seq,
    select_along_batch,
    select_along_seq,
)

INDEX_DTYPES = [np.int32, np.int64, np.uint32]

#######################################
#     Tests for chunk_along_batch     #
#######################################


def test_chunk_along_batch_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), chunks=3),
        [
            np.array([[0, 1], [2, 3]]),
            np.array([[4, 5], [6, 7]]),
            np.array([[8, 9]]),
        ],
    )


def test_chunk_along_batch_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), chunks=5),
        [
            np.array([[0, 1]]),
            np.array([[2, 3]]),
            np.array([[4, 5]]),
            np.array([[6, 7]]),
            np.array([[8, 9]]),
        ],
    )


#####################################
#     Tests for chunk_along_seq     #
#####################################


def test_chunk_along_seq_chunks_3() -> None:
    assert objects_are_equal(
        chunk_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), chunks=3),
        [
            np.array([[0, 1], [5, 6]]),
            np.array([[2, 3], [7, 8]]),
            np.array([[4], [9]]),
        ],
    )


def test_chunk_along_seq_chunks_5() -> None:
    assert objects_are_equal(
        chunk_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), chunks=5),
        [
            np.array([[0], [5]]),
            np.array([[1], [6]]),
            np.array([[2], [7]]),
            np.array([[3], [8]]),
            np.array([[4], [9]]),
        ],
    )


########################################
#     Tests for select_along_batch     #
########################################


def test_select_along_batch_index_0() -> None:
    assert objects_are_equal(
        select_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), index=0),
        np.array([0, 1]),
    )


def test_select_along_batch_index_2() -> None:
    assert objects_are_equal(
        select_along_batch(np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), index=2),
        np.array([4, 5]),
    )


######################################
#     Tests for select_along_seq     #
######################################


def test_select_along_seq_index_0() -> None:
    assert objects_are_equal(
        select_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), index=0),
        np.array([0, 5]),
    )


def test_select_along_seq_index_2() -> None:
    assert objects_are_equal(
        select_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), index=2),
        np.array([2, 7]),
    )
