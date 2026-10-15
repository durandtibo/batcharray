from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import nested

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

DTYPES = [np.float32, np.float64, np.int64]
FLOAT_DTYPES = [np.float32, np.float64]
POINTWISE_FUNCTIONS = [
    (np.abs, nested.abs),
    (partial(np.clip, a_min=2, a_max=None), partial(nested.clip, a_min=2)),
    (partial(np.clip, a_min=None, a_max=6), partial(nested.clip, a_max=6)),
    (partial(np.clip, a_min=2, a_max=6), partial(nested.clip, a_min=2, a_max=6)),
    (np.exp, nested.exp),
    (np.exp2, nested.exp2),
    (np.expm1, nested.expm1),
    (np.log, nested.log),
    (np.log2, nested.log2),
    (np.log10, nested.log10),
    (np.log1p, nested.log1p),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_pointwise_function_array(dtype: DTypeLike, functions: tuple[Callable, Callable]) -> None:
    np_fn, nested_fn = functions
    array = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=dtype)
    assert objects_are_equal(nested_fn(array), np_fn(array))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_pointwise_function_dict(dtype: DTypeLike, functions: tuple[Callable, Callable]) -> None:
    np_fn, nested_fn = functions
    assert objects_are_equal(
        nested_fn(
            {
                "a": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=dtype),
                "b": np.array([4, 3, 2, 1], dtype=np.float32),
                "c": [np.array([5, 6, 7, 8, 9], dtype=np.float64)],
                "masked": np.ma.masked_array(
                    data=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                    mask=np.array(
                        [
                            [False, False],
                            [False, False],
                            [False, True],
                            [False, False],
                            [False, True],
                        ]
                    ),
                ),
            },
        ),
        {
            "a": np_fn(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=dtype)),
            "b": np_fn(np.array([4, 3, 2, 1], dtype=np.float32)),
            "c": [np_fn(np.array([5, 6, 7, 8, 9], dtype=np.float64))],
            "masked": np_fn(
                np.ma.masked_array(
                    data=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                    mask=np.array(
                        [
                            [False, False],
                            [False, False],
                            [False, True],
                            [False, False],
                            [False, True],
                        ]
                    ),
                )
            ),
        },
    )


##########################
#     Tests for ceil     #
##########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ceil_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.ceil(np.array([[-1.5, -0.5], [0.2, 1.5]], dtype=dtype)),
        np.array([[-1.0, -0.0], [1.0, 2.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ceil_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.ceil(
            {
                "a": np.array([[-1.5, -0.5], [0.2, 1.5]], dtype=dtype),
                "b": np.array([2.1, -2.1], dtype=dtype),
            }
        ),
        {
            "a": np.array([[-1.0, -0.0], [1.0, 2.0]], dtype=dtype),
            "b": np.array([3.0, -2.0], dtype=dtype),
        },
    )


def test_ceil_nested() -> None:
    out = nested.ceil({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


###########################
#     Tests for floor     #
###########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_floor_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.floor(np.array([[-1.5, -0.5], [0.2, 1.5]], dtype=dtype)),
        np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_floor_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.floor(
            {
                "a": np.array([[-1.5, -0.5], [0.2, 1.5]], dtype=dtype),
                "b": np.array([2.1, -2.1], dtype=dtype),
            }
        ),
        {
            "a": np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=dtype),
            "b": np.array([2.0, -3.0], dtype=dtype),
        },
    )


def test_floor_nested() -> None:
    out = nested.floor({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


###########################
#     Tests for round     #
###########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_round_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.round(np.array([[-1.6, -0.4], [0.2, 1.5]], dtype=dtype)),
        np.array([[-2.0, -0.0], [0.0, 2.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_round_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.round(
            {
                "a": np.array([[-1.6, -0.4], [0.2, 1.5]], dtype=dtype),
                "b": np.array([2.7, -2.7], dtype=dtype),
            }
        ),
        {
            "a": np.array([[-2.0, -0.0], [0.0, 2.0]], dtype=dtype),
            "b": np.array([3.0, -3.0], dtype=dtype),
        },
    )


def test_round_nested() -> None:
    out = nested.round({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


##########################
#     Tests for sign     #
##########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sign_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.sign(np.array([[-4.0, -3.0], [0.0, 1.0], [2.0, 3.0]], dtype=dtype)),
        np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sign_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.sign(
            {
                "a": np.array([[-4.0, -3.0], [0.0, 1.0], [2.0, 3.0]], dtype=dtype),
                "b": np.array([2.0, 0.0, -2.0], dtype=dtype),
            }
        ),
        {
            "a": np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype),
            "b": np.array([1.0, 0.0, -1.0], dtype=dtype),
        },
    )


def test_sign_nested() -> None:
    out = nested.sign({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


##########################
#     Tests for sqrt     #
##########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sqrt_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.sqrt(np.array([[0.0, 1.0], [4.0, 9.0]], dtype=dtype)),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sqrt_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.sqrt(
            {
                "a": np.array([[0.0, 1.0], [4.0, 9.0]], dtype=dtype),
                "b": np.array([16.0, 25.0], dtype=dtype),
            }
        ),
        {
            "a": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=dtype),
            "b": np.array([4.0, 5.0], dtype=dtype),
        },
    )


def test_sqrt_nested() -> None:
    out = nested.sqrt({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


############################
#     Tests for square     #
############################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_square_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.square(np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=dtype)),
        np.array([[4.0, 1.0], [0.0, 1.0], [4.0, 9.0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_square_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.square(
            {
                "a": np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=dtype),
                "b": np.array([2.0, 0.0, -2.0], dtype=dtype),
            }
        ),
        {
            "a": np.array([[4.0, 1.0], [0.0, 1.0], [4.0, 9.0]], dtype=dtype),
            "b": np.array([4.0, 0.0, 4.0], dtype=dtype),
        },
    )


def test_square_nested() -> None:
    out = nested.square({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)


###########################
#     Tests for isnan     #
###########################


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_isnan_array(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.isnan(np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=dtype)),
        np.array([[False, True], [True, False]]),
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_isnan_dict(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        nested.isnan(
            {
                "a": np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=dtype),
                "b": np.array([np.nan, 0.0], dtype=dtype),
            }
        ),
        {"a": np.array([[False, True], [True, False]]), "b": np.array([True, False])},
    )


def test_isnan_nested() -> None:
    out = nested.isnan({"a": [np.ones((2, 2)), (np.ones(2),)], "b": np.ones(3)})
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1], tuple)
    assert out["a"][0].shape == (2, 2)
    assert out["b"].shape == (3,)
