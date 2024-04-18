from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from batcharray.array import (
    argsort_along_batch,
    argsort_along_seq,
    sort_along_batch,
    sort_along_seq,
)

#########################################
#     Tests for argsort_along_batch     #
#########################################


def test_argsort_along_batch() -> None:
    assert objects_are_equal(
        argsort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        np.array([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]),
    )


#######################################
#     Tests for argsort_along_seq     #
#######################################


def test_argsort_along_seq() -> None:
    assert objects_are_equal(
        argsort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        np.array([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]),
    )


######################################
#     Tests for sort_along_batch     #
######################################


def test_sort_along_batch() -> None:
    assert objects_are_equal(
        sort_along_batch(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
        np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]),
    )


####################################
#     Tests for sort_along_seq     #
####################################


def test_sort_along_seq() -> None:
    assert objects_are_equal(
        sort_along_seq(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]),
    )


# def test_sort_along_seq_masked_array() -> None:
#     assert objects_are_equal(
#         sort_along_seq(
#             np.ma.masked_array(
#                 data=np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]),
#                 mask=np.array(
#                     [[False, False, False, False, False], [True, True, True, False, True]]
#                 ),
#             ),
#         ),
#         np.ma.masked_array(
#             data=np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]),
#             mask=np.array([[False, False, False, False, False], [False, False, True, False, True]]),
#         ),
#     )
