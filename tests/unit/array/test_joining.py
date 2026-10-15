from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.array import (
    concatenate_along_batch,
    concatenate_along_seq,
    repeat_along_batch,
    repeat_along_seq,
    tile_along_seq,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

DTYPES = [np.int64, np.float64]

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
def test_concatenate_along_batch(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
) -> None:
    assert objects_are_equal(
        concatenate_along_batch(arrays),
        np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


def test_concatenate_along_batch_masked_array() -> None:
    assert objects_are_equal(
        concatenate_along_batch(
            [
                np.ma.masked_array(
                    data=np.array([[0, 1, 2], [4, 5, 6]]),
                    mask=np.array([[False, False, False], [False, True, False]]),
                ),
                np.ma.masked_array(
                    data=np.array([[10, 11, 12], [13, 14, 15]]),
                    mask=np.array([[False, False, True], [False, False, False]]),
                ),
            ]
        ),
        np.ma.masked_array(
            data=np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
            mask=np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ]
            ),
        ),
    )


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
def test_concatenate_along_seq(
    arrays: list[np.ndarray] | tuple[np.ndarray, ...],
) -> None:
    assert objects_are_equal(
        concatenate_along_seq(arrays),
        np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
    )


def test_concatenate_along_seq_masked_array() -> None:
    assert objects_are_equal(
        concatenate_along_seq(
            [
                np.ma.masked_array(
                    data=np.array([[0, 1, 2], [4, 5, 6]]),
                    mask=np.array([[False, False, False], [False, True, False]]),
                ),
                np.ma.masked_array(
                    data=np.array([[10, 11, 12], [13, 14, 15]]),
                    mask=np.array([[False, False, True], [False, False, False]]),
                ),
            ]
        ),
        np.ma.masked_array(
            data=np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
            mask=np.array(
                [
                    [False, False, False, False, False, True],
                    [False, True, False, False, False, False],
                ]
            ),
        ),
    )


####################################
#     Tests for tile_along_seq     #
####################################


def test_tile_along_seq_reps_0() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), reps=0),
        np.zeros((2, 0)),
    )


def test_tile_along_seq_reps_1() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), reps=1),
        np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    )


def test_tile_along_seq_reps_2() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), reps=2),
        np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
    )


def test_tile_along_seq_reps_3d() -> None:
    assert objects_are_equal(
        tile_along_seq(
            np.array(
                [
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                    [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                ]
            ),
            reps=2,
        ),
        np.array(
            [
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                ],
                [
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                    [10, 11],
                    [12, 13],
                    [14, 15],
                    [16, 17],
                    [18, 19],
                ],
            ]
        ),
    )


def test_tile_along_seq_masked_array() -> None:
    assert objects_are_equal(
        tile_along_seq(
            np.ma.masked_array(
                data=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                mask=np.array(
                    [
                        [False, False, False, False, True],
                        [False, False, True, False, False],
                    ]
                ),
            ),
            reps=2,
        ),
        np.ma.masked_array(
            data=np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
            mask=np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                    ],
                ]
            ),
        ),
    )


@pytest.mark.parametrize("fn", [concatenate_along_batch, concatenate_along_seq])
def test_concatenate_dtype(fn: object) -> None:
    out = fn([np.ones((2, 3), dtype=np.int64), np.ones((2, 3), dtype=np.int64)], dtype=np.float32)
    assert out.dtype == np.float32


def test_tile_along_seq_1d_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        tile_along_seq(np.arange(3), reps=2)


########################################
#     Tests for repeat_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_batch(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), repeats=2
        ),
        np.array(
            [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]],
            dtype=dtype,
        ),
    )


######################################
#     Tests for repeat_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_seq(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), repeats=2),
        np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]], dtype=dtype),
    )


def test_repeat_along_seq_1d_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        repeat_along_seq(np.arange(3), repeats=2)
