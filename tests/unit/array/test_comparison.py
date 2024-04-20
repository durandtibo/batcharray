from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.array import (
    argsort_along_batch,
    argsort_along_seq,
    sort_along_batch,
    sort_along_seq,
)
from batcharray.types import SORT_KINDS

if TYPE_CHECKING:
    from batcharray.types import SortKind


#########################################
#     Tests for argsort_along_batch     #
#########################################


def test_argsort_along_batch() -> None:
    assert objects_are_equal(
        argsort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        np.array([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_argsort_along_batch_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        argsort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), kind=kind),
        np.array([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
    )


def test_argsort_along_batch_masked_array() -> None:
    assert objects_are_equal(
        argsort_along_batch(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 4, 1, 3, 0]]),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            )
        ),
        np.array([[0, 2, 0, 0, 2], [1, 0, 1, 2, 1], [2, 1, 2, 1, 0]]),
    )


#######################################
#     Tests for argsort_along_seq     #
#######################################


def test_argsort_along_seq() -> None:
    assert objects_are_equal(
        argsort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        np.array([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_argsort_along_seq_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        argsort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), kind=kind),
        np.array([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
    )


def test_argsort_along_seq_masked_array() -> None:
    assert objects_are_equal(
        argsort_along_seq(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]]),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            )
        ),
        np.array([[2, 3, 0, 1, 4], [0, 4, 1, 2, 3], [4, 1, 0, 3, 2]]),
    )


######################################
#     Tests for sort_along_batch     #
######################################


def test_sort_along_batch() -> None:
    assert objects_are_equal(
        sort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_sort_along_batch_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        sort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), kind=kind),
        np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]),
    )


def test_sort_along_batch_masked_array() -> None:
    assert objects_are_equal(
        sort_along_batch(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]]),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            )
        ),
        np.ma.masked_array(
            data=np.array([[3, 5, 0, 2, 0], [4, 5, 8, 8, 5], [8, 7, 8, 8, 4]]),
            mask=np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, True, True, True],
                ]
            ),
        ),
    )


####################################
#     Tests for sort_along_seq     #
####################################


def test_sort_along_seq() -> None:
    assert objects_are_equal(
        sort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_sort_along_seq_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        sort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), kind=kind),
        np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]),
    )


def test_sort_along_seq_masked_array() -> None:
    assert objects_are_equal(
        sort_along_seq(
            np.ma.masked_array(
                data=np.array([[3, 5, 0, 2, 4], [4, 7, 8, 8, 5], [8, 5, 8, 8, 0]]),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, False, True, False],
                        [False, False, True, False, False],
                    ]
                ),
            ),
        ),
        np.ma.masked_array(
            data=np.array([[0, 2, 3, 5, 4], [4, 5, 7, 8, 8], [0, 5, 8, 8, 8]]),
            mask=np.array(
                [
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                ]
            ),
        ),
    )
