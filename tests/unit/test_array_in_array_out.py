r"""Tests of P12: a function that receives an array returns an array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array as ba, nested as bn
from batcharray.utils import apply_to_arrays

if TYPE_CHECKING:
    from collections.abc import Callable

REDUCTIONS = [
    f"{name}_along_batch"
    for name in (
        "all",
        "any",
        "argmax",
        "argmin",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "std",
        "sum",
        "var",
    )
]
SHAPES = [(4,), (4, 3), (4, 3, 2)]


def make_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


def make_masked_array(shape: tuple[int, ...]) -> np.ma.MaskedArray:
    x = make_array(shape)
    return np.ma.array(x, mask=x % 2 == 0)


#######################################
#     Tests for the array level       #
#######################################


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduction_along_batch_returns_array(name: str, shape: tuple[int, ...]) -> None:
    assert isinstance(getattr(ba, name)(make_array(shape)), np.ndarray)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduction_along_batch_keepdims_returns_array(name: str, shape: tuple[int, ...]) -> None:
    assert isinstance(getattr(ba, name)(make_array(shape), keepdims=True), np.ndarray)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_reduction_along_batch_masked_array_returns_masked_array(
    name: str, shape: tuple[int, ...]
) -> None:
    out = getattr(ba, name)(make_masked_array(shape))
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reduction_along_batch_1d_returns_0d_array(name: str) -> None:
    assert getattr(ba, name)(make_array((4,))).shape == ()


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reduction_along_batch_1d_is_not_a_numpy_scalar(name: str) -> None:
    assert type(getattr(ba, name)(make_array((4,)))) is np.ndarray


@pytest.mark.parametrize("shape", SHAPES)
def test_select_along_batch_returns_array(shape: tuple[int, ...]) -> None:
    assert isinstance(ba.select_along_batch(make_array(shape), 1), np.ndarray)


def test_select_along_batch_1d_returns_0d_array() -> None:
    assert objects_are_equal(ba.select_along_batch(make_array((4,)), 1), np.array(1.0))


def test_sum_along_batch_1d_value() -> None:
    assert objects_are_equal(ba.sum_along_batch(make_array((4,))), np.array(6.0))


def test_any_along_batch_1d_dtype() -> None:
    assert ba.any_along_batch(make_array((4,))).dtype == np.bool_


def test_sum_along_batch_1d_fully_masked_returns_masked_constant() -> None:
    out = ba.sum_along_batch(np.ma.array(make_array((4,)), mask=True))
    assert out is np.ma.masked


#######################################
#     Tests for the nested level      #
#######################################


@pytest.mark.parametrize("name", REDUCTIONS)
def test_nested_reduction_1d_leaf_is_an_array(name: str) -> None:
    out = getattr(bn, name)({"a": make_array((4,))})
    assert isinstance(out["a"], np.ndarray)


@pytest.mark.parametrize("name", REDUCTIONS)
def test_nested_reduction_1d_leaf_is_a_valid_input(name: str) -> None:
    once = getattr(bn, name)({"a": make_array((4,))})
    assert isinstance(apply_to_arrays(once, np.copy), dict)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("shape", [(4, 3), (4, 3, 2)])
def test_nested_reduction_applied_twice(name: str, shape: tuple[int, ...]) -> None:
    func: Callable = getattr(bn, name)
    out = func(func({"a": make_array(shape)}))
    assert isinstance(out["a"], np.ndarray)


def test_nested_sum_applied_twice_value() -> None:
    out = bn.sum_along_batch(bn.sum_along_batch({"a": make_array((4, 3))}))
    assert objects_are_equal(out, {"a": np.array(66.0)})


def test_nested_select_along_batch_1d_leaf_is_an_array() -> None:
    assert isinstance(bn.select_along_batch({"a": make_array((4,))}, 1)["a"], np.ndarray)
