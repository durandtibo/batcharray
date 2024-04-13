from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.array import concatenate_along_batch, concatenate_along_seq

#############################################
#     Tests for concatenate_along_batch     #
#############################################


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        (np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])),
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11, 12]]),
            np.array([[13, 14, 15]]),
        ],
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.ones((0, 3), dtype=int),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_concatenate_along_batch(arrays: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    assert objects_are_equal(
        concatenate_along_batch(arrays),
        np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


# def test_concatenate_along_batch_masked_array() -> None:
#     assert objects_are_equal(
#         concatenate_along_batch(
#             [
#                 np.ma.masked_array(
#                     data=np.array([[0, 1, 2], [4, 5, 6]]),
#                     mask=np.array(
#                         [
#                             [False, False, False],
#                             [False, True, False],
#                         ]
#                     ),
#                 ),
#                 np.ma.masked_array(
#                     data=np.array([[10, 11, 12], [13, 14, 15]]),
#                     mask=np.array(
#                         [
#                             [False, False, True],
#                             [False, False, False],
#                         ]
#                     ),
#                 ),
#             ]
#         ),
#         np.ma.masked_array(
#             data=np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
#             mask=np.array(
#                 [
#                     [False, False, False],
#                     [False, True, False],
#                     [False, False, True],
#                     [False, False, False],
#                 ]
#             ),
#         ),
#     )


###########################################
#     Tests for concatenate_along_seq     #
###########################################


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])],
        (np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])),
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11], [13, 14]]),
            np.array([[12], [15]]),
        ],
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.ones((2, 0), dtype=int),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
    ],
)
def test_concatenate_along_seq(arrays: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    assert objects_are_equal(
        concatenate_along_seq(arrays),
        np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
    )
