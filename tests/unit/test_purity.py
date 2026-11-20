r"""Tests of P7: inputs are never modified and the output never aliases
the input, except for the views of NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola.equality import objects_are_equal
from coola.iterator import dfs_iterate

from batcharray import array as ba, nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

REDUCTIONS = (
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

# The functions that return a view of the input (the closed list of
# docs/docs/design_principles.md).
VIEWS = {
    "slice_along_batch",
    "slice_along_seq",
    "select_along_batch",
    "select_along_seq",
    "chunk_along_batch",
    "chunk_along_seq",
    "split_along_batch",
    "split_along_seq",
    "flip_along_batch",
    "flip_along_seq",
}

# The (name, the arguments after the data) of each function, for an array
# of shape (4, 5, 2).
CALLS: dict[str, tuple[Any, ...]] = {
    **{f"{name}_along_{axis}": () for name in REDUCTIONS for axis in ("batch", "seq")},
    **{
        f"{name}_along_{axis}": ()
        for name in ("cumsum", "cumprod", "sort", "argsort", "flip", "shuffle")
        for axis in ("batch", "seq")
    },
    "repeat_along_batch": (2,),
    "repeat_along_seq": (2,),
    "tile_along_seq": (2,),
    "select_along_batch": (1,),
    "select_along_seq": (1,),
    "slice_along_batch": (1, 3),
    "slice_along_seq": (1, 3),
    "chunk_along_batch": (2,),
    "chunk_along_seq": (2,),
    "split_along_batch": (2,),
    "split_along_seq": (2,),
    "index_select_along_batch": (np.array([2, 0]),),
    "index_select_along_seq": (np.array([4, 0, 1]),),
    "masked_select_along_batch": (np.array([True, False, True, False]),),
    "masked_select_along_seq": (np.array([True, False, True, False, True]),),
    "permute_along_batch": (np.array([3, 2, 1, 0]),),
    "permute_along_seq": (np.array([4, 3, 2, 1, 0]),),
    "gather_along_seq": (np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),),
}
# Functions that return a new array even if the operation is the identity.
IDENTITY_CALLS: dict[str, tuple[Any, ...]] = {
    "repeat_along_batch": (1,),
    "repeat_along_seq": (1,),
    "tile_along_seq": (1,),
}
INDEX_CALLS = [
    "index_select_along_batch",
    "index_select_along_seq",
    "masked_select_along_batch",
    "masked_select_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "gather_along_seq",
]
CONCATENATE = [
    "concatenate_along_batch",
    "concatenate_along_seq",
    "stack_along_batch",
    "stack_along_seq",
]

# The data passed to the function: an array (array level) or a dict (nested level).
LEVELS: dict[str, tuple[Any, Callable[[np.ndarray], Any]]] = {
    "array": (ba, lambda x: x),
    "nested": (bn, lambda x: {"a": x}),
}


def make_array() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(4, 5, 2))


def make_masked_array() -> np.ma.MaskedArray:
    x = make_array()
    return np.ma.array(x, mask=x > 0.5)


def call(module: Any, name: str, data: Any, args: tuple[Any, ...]) -> Any:
    if name in CONCATENATE:
        return getattr(module, name)([data])
    return getattr(module, name)(data, *args)


def get_arrays(out: Any) -> list[np.ndarray]:
    return list(dfs_iterate(out))


################################################
#     Tests for the inputs (not modified)      #
################################################


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", [*CALLS, *CONCATENATE])
def test_inputs_are_not_modified(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_array()
    call(module, name, wrap(x), CALLS.get(name, ()))
    assert objects_are_equal(x, make_array())


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", [*CALLS, *CONCATENATE])
def test_masked_inputs_are_not_modified(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_masked_array()
    call(module, name, wrap(x), CALLS.get(name, ()))
    assert objects_are_equal(x, make_masked_array())


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", INDEX_CALLS)
def test_index_arguments_are_not_modified(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    args = tuple(a.copy() for a in CALLS[name])
    call(module, name, wrap(make_array()), args)
    assert objects_are_equal(args, CALLS[name])


####################################################
#     Tests for the outputs (no aliasing)          #
####################################################

NEW_ARRAY_CALLS = [name for name in CALLS if name not in VIEWS]
VIEW_CALLS = sorted(VIEWS)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", NEW_ARRAY_CALLS)
def test_output_does_not_alias_input(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_array()
    out = call(module, name, wrap(x), CALLS[name])
    assert not any(np.shares_memory(x, array) for array in get_arrays(out))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", CONCATENATE)
def test_concatenate_of_one_item_does_not_alias_input(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_array()
    out = call(module, name, wrap(x), ())
    assert not any(np.shares_memory(x, array) for array in get_arrays(out))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", IDENTITY_CALLS)
def test_identity_operation_does_not_alias_input(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_array()
    out = call(module, name, wrap(x), IDENTITY_CALLS[name])
    assert not any(np.shares_memory(x, array) for array in get_arrays(out))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", VIEW_CALLS)
def test_view_functions_return_views(level: str, name: str) -> None:
    module, wrap = LEVELS[level]
    x = make_array()
    out = call(module, name, wrap(x), CALLS[name])
    assert all(np.shares_memory(x, array) for array in get_arrays(out))
