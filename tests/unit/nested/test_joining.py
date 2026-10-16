from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.nested import (
    concatenate_along_batch,
    concatenate_along_seq,
    repeat_along_batch,
    repeat_along_seq,
    tile_along_seq,
)
from tests.unit.array.test_joining import DTYPES

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from numpy.typing import DTypeLike

#############################################
#     Tests for concatenate_along_batch     #
#############################################


@pytest.mark.parametrize(
    "data",
    [
        [
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {"a": np.array([[10, 11, 12], [14, 15, 16]]), "b": np.array([[17], [18]])},
        ],
        (
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {"a": np.array([[10, 11, 12], [14, 15, 16]]), "b": np.array([[17], [18]])},
        ),
        [
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {"a": np.array([[10, 11, 12]]), "b": np.array([[17]])},
            {"a": np.array([[14, 15, 16]]), "b": np.array([[18]])},
        ],
    ],
)
def test_concatenate_along_batch(data: Sequence[dict[Hashable, np.ndarray]]) -> None:
    assert objects_are_equal(
        concatenate_along_batch(data),
        {
            "a": np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [14, 15, 16]]),
            "b": np.array([[7], [8], [17], [18]]),
        },
    )


def test_concatenate_along_batch_empty() -> None:
    assert objects_are_equal(concatenate_along_batch([]), {})


def test_concatenate_along_batch_masked_array() -> None:
    assert objects_are_equal(
        concatenate_along_batch(
            [
                {
                    "a": np.ma.masked_array(
                        data=np.array([[0, 1, 2], [4, 5, 6]]),
                        mask=np.array([[False, False, False], [False, True, False]]),
                    ),
                    "b": np.array([[7], [8]]),
                },
                {
                    "a": np.ma.masked_array(
                        data=np.array([[10, 11, 12], [13, 14, 15]]),
                        mask=np.array([[False, False, True], [False, False, False]]),
                    ),
                    "b": np.array([[17], [18]]),
                },
            ]
        ),
        {
            "a": np.ma.masked_array(
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
            "b": np.array([[7], [8], [17], [18]]),
        },
    )


###########################################
#     Tests for concatenate_along_seq     #
###########################################


@pytest.mark.parametrize(
    "data",
    [
        [
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {
                "a": np.array([[10, 11, 12], [13, 14, 15]]),
                "b": np.array([[17, 18], [18, 19]]),
            },
        ],
        (
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {
                "a": np.array([[10, 11, 12], [13, 14, 15]]),
                "b": np.array([[17, 18], [18, 19]]),
            },
        ),
        [
            {"a": np.array([[0, 1, 2], [4, 5, 6]]), "b": np.array([[7], [8]])},
            {"a": np.array([[10, 11], [13, 14]]), "b": np.array([[17], [18]])},
            {"a": np.array([[12], [15]]), "b": np.array([[18], [19]])},
        ],
    ],
)
def test_concatenate_along_seq(data: Sequence[dict[Hashable, np.ndarray]]) -> None:
    assert objects_are_equal(
        concatenate_along_seq(data),
        {
            "a": np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
            "b": np.array([[7, 17, 18], [8, 18, 19]]),
        },
    )


def test_concatenate_along_seq_empty() -> None:
    assert objects_are_equal(concatenate_along_seq([]), {})


def test_concatenate_along_seq_masked_array() -> None:
    assert objects_are_equal(
        concatenate_along_seq(
            [
                {
                    "a": np.ma.masked_array(
                        data=np.array([[0, 1, 2], [4, 5, 6]]),
                        mask=np.array([[False, False, False], [False, True, False]]),
                    ),
                    "b": np.array([[7], [8]]),
                },
                {
                    "a": np.ma.masked_array(
                        data=np.array([[10, 11, 12], [13, 14, 15]]),
                        mask=np.array([[False, False, True], [False, False, False]]),
                    ),
                    "b": np.array([[17, 18], [18, 19]]),
                },
            ]
        ),
        {
            "a": np.ma.masked_array(
                data=np.array([[0, 1, 2, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
                mask=np.array(
                    [
                        [False, False, False, False, False, True],
                        [False, True, False, False, False, False],
                    ]
                ),
            ),
            "b": np.array([[7, 17, 18], [8, 18, 19]]),
        },
    )


####################################
#     Tests for tile_along_seq     #
####################################


def test_tile_along_seq_array_reps_0() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), reps=0),
        np.zeros((2, 0)),
    )


def test_tile_along_seq_array_reps_1() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), reps=1),
        np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    )


def test_tile_along_seq_array_reps_2() -> None:
    assert objects_are_equal(
        tile_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), reps=2),
        np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
    )


def test_tile_along_seq_array_reps_3d() -> None:
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


def test_tile_along_seq_dict_reps_0() -> None:
    assert objects_are_equal(
        tile_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            reps=0,
        ),
        {"a": np.zeros((2, 0), dtype=np.int64), "b": np.zeros((1, 0), dtype=np.int64)},
    )


def test_tile_along_seq_dict_reps_1() -> None:
    assert objects_are_equal(
        tile_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            reps=1,
        ),
        {
            "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            "b": np.array([[4, 3, 2, 1, 0]]),
        },
    )


def test_tile_along_seq_dict_reps_2() -> None:
    assert objects_are_equal(
        tile_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            reps=2,
        ),
        {
            "a": np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
            "b": np.array([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]]),
        },
    )


def test_tile_along_seq_dict_reps_3d() -> None:
    assert objects_are_equal(
        tile_along_seq(
            {
                "a": np.array(
                    [
                        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                    ]
                ),
                "b": np.array([[4, 3, 2, 1, 0]]),
            },
            reps=2,
        ),
        {
            "a": np.array(
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
            "b": np.array([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]]),
        },
    )


def test_tile_along_seq_dict_nested() -> None:
    assert objects_are_equal(
        tile_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                "b": np.ma.masked_array(
                    data=np.array([[4, 3, 2, 1, 0]]),
                    mask=np.array([[False, False, False, True, False]]),
                ),
                "list": [np.array([[5, 6, 7, 8, 9]])],
            },
            reps=2,
        ),
        {
            "a": np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]),
            "b": np.ma.masked_array(
                data=np.array([[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]]),
                mask=np.array(
                    [
                        [
                            False,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                            True,
                            False,
                        ]
                    ]
                ),
            ),
            "list": [np.array([[5, 6, 7, 8, 9, 5, 6, 7, 8, 9]])],
        },
    )


@pytest.mark.parametrize("fn", [concatenate_along_batch, concatenate_along_seq])
def test_concatenate_dtype(fn: object) -> None:
    data = [{"a": np.ones((2, 3), dtype=np.int64)}, {"a": np.ones((2, 3), dtype=np.int64)}]
    assert fn(data, dtype=np.float32)["a"].dtype == np.float32


@pytest.mark.parametrize("func", [concatenate_along_batch, concatenate_along_seq])
@pytest.mark.parametrize(
    "data",
    [
        [{"a": np.zeros((2, 2))}, {"a": np.zeros((2, 2)), "b": np.zeros((2, 2))}],
        [{"a": np.zeros((2, 2)), "b": np.zeros((2, 2))}, {"a": np.zeros((2, 2))}],
    ],
)
def test_concatenate_different_keys_raises(func: Callable, data: list) -> None:
    with pytest.raises(ValueError, match="same keys"):
        func(data)


@pytest.mark.parametrize("func", [concatenate_along_batch, concatenate_along_seq])
def test_concatenate_nested_structure(func: Callable) -> None:
    axis = 0 if func is concatenate_along_batch else 1
    data = [
        {"a": {"b": np.ones((2, 2))}, "c": [np.zeros((2, 2))], "d": (np.ones((2, 2)),)},
        {"a": {"b": np.ones((2, 2)) * 2}, "c": [np.zeros((2, 2)) + 1], "d": (np.ones((2, 2)) * 3,)},
    ]
    assert objects_are_equal(
        func(data),
        {
            "a": {"b": np.concatenate([data[0]["a"]["b"], data[1]["a"]["b"]], axis=axis)},
            "c": [np.concatenate([data[0]["c"][0], data[1]["c"][0]], axis=axis)],
            "d": (np.concatenate([data[0]["d"][0], data[1]["d"][0]], axis=axis),),
        },
    )


@pytest.mark.parametrize("func", [concatenate_along_batch, concatenate_along_seq])
def test_concatenate_nested_different_keys_raises(func: Callable) -> None:
    with pytest.raises(ValueError, match="same keys"):
        func([{"a": {"b": np.zeros((2, 2))}}, {"a": {"c": np.zeros((2, 2))}}])


@pytest.mark.parametrize("func", [concatenate_along_batch, concatenate_along_seq])
def test_concatenate_different_lengths_raises(func: Callable) -> None:
    with pytest.raises(ValueError, match="same length"):
        func([[np.zeros((2, 2))], [np.zeros((2, 2)), np.zeros((2, 2))]])


@pytest.mark.parametrize("func", [concatenate_along_batch, concatenate_along_seq])
@pytest.mark.parametrize(
    "data",
    [
        [{"a": np.zeros((2, 2))}, [np.zeros((2, 2))]],
        [{"a": np.zeros((2, 2))}, {"a": 1}],
        [{"a": 1}, {"a": 2}],
    ],
)
def test_concatenate_incompatible_items_raises(func: Callable, data: list) -> None:
    with pytest.raises(TypeError, match="Expected"):
        func(data)


########################################
#     Tests for repeat_along_batch     #
########################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_batch_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_batch(
            np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype), repeats=2
        ),
        np.array(
            [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]],
            dtype=dtype,
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_batch_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_batch(
            {
                "a": np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=dtype),
                "b": np.array([[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]], dtype=dtype),
            },
            repeats=2,
        ),
        {
            "a": np.array(
                [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]],
                dtype=dtype,
            ),
            "b": np.array(
                [[8, 9], [8, 9], [6, 7], [6, 7], [4, 5], [4, 5], [2, 3], [2, 3], [0, 1], [0, 1]],
                dtype=dtype,
            ),
        },
    )


######################################
#     Tests for repeat_along_seq     #
######################################


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_seq_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_seq(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype), repeats=2),
        np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_along_seq_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        repeat_along_seq(
            {
                "a": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype),
                "b": np.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]], dtype=dtype),
            },
            repeats=2,
        ),
        {
            "a": np.array(
                [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]], dtype=dtype
            ),
            "b": np.array(
                [[4, 4, 3, 3, 2, 2, 1, 1, 0, 0], [9, 9, 8, 8, 7, 7, 6, 6, 5, 5]], dtype=dtype
            ),
        },
    )
