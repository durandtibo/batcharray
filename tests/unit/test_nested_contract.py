r"""Tests of the properties that every ``nested`` function must have."""

from __future__ import annotations

import ast
import inspect
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

NESTED_DIR = Path(bn.__file__).parent
NESTED_FILES = sorted(path for path in NESTED_DIR.glob("*.py") if path.name != "__init__.py")
CONTAINER_NAMES = {"dict", "list", "tuple", "Mapping", "Sequence", "set"}


class Point(NamedTuple):
    x: np.ndarray
    y: np.ndarray


REDUCTIONS = [
    f"{name}_along_{axis}"
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
    for axis in ("batch", "seq")
]

# The arguments of each function after the data, enough to reach the leaf check.
MAP_FUNCTIONS: dict[str, tuple[Any, ...]] = {
    **dict.fromkeys(REDUCTIONS, ()),
    **{
        f"{name}_along_{axis}": ()
        for name in ("cumprod", "cumsum", "flip", "sort", "argsort")
        for axis in ("batch", "seq")
    },
    **{f"{name}_along_{axis}": (2,) for name in ("repeat",) for axis in ("batch", "seq")},
    "tile_along_seq": (2,),
    "select_along_batch": (0,),
    "select_along_seq": (0,),
    "slice_along_batch": (),
    "slice_along_seq": (),
    "index_select_along_batch": (np.array([0]),),
    "index_select_along_seq": (np.array([0]),),
    "masked_select_along_batch": (np.array([True]),),
    "masked_select_along_seq": (np.array([True]),),
    "permute_along_batch": (np.array([0]),),
    "permute_along_seq": (np.array([0]),),
    "shuffle_along_batch": (),
    "shuffle_along_seq": (),
}
SPLIT_FUNCTIONS = ["chunk_along_batch", "chunk_along_seq", "split_along_batch", "split_along_seq"]
COMBINE_FUNCTIONS = ["concatenate_along_batch", "concatenate_along_seq"]

LEAVES = [(1, "int"), ("x", "str"), (None, "NoneType"), (np.float64(1.0), "float64")]


def call_with_leaf(name: str, leaf: object) -> object:
    r"""Call a ``nested`` function on a structure with a non-array leaf."""
    func = getattr(bn, name)
    data = {"a": leaf}
    if name in COMBINE_FUNCTIONS:
        return func([data, data])
    if name in SPLIT_FUNCTIONS:
        return func(data, 2)
    return func(data, *MAP_FUNCTIONS[name])


#########################################################
#     Tests for the traversal (P4: one traversal)       #
#########################################################


@pytest.mark.parametrize("path", NESTED_FILES, ids=lambda path: path.name)
def test_nested_module_does_not_import_recursive_apply(path: Path) -> None:
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not imported & {"recursive_apply", "recursive_zip_apply"}


@pytest.mark.parametrize("path", NESTED_FILES, ids=lambda path: path.name)
def test_nested_module_does_not_walk_containers(path: Path) -> None:
    tree = ast.parse(path.read_text())
    checked = {
        ast.unparse(node.args[1])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
    }
    assert not {name for name in checked if CONTAINER_NAMES & set(name.replace(",", " ").split())}


########################################################
#     Tests for the leaf contract (P6: array leaves)    #
########################################################


@pytest.mark.parametrize("name", [*MAP_FUNCTIONS, *SPLIT_FUNCTIONS, *COMBINE_FUNCTIONS])
@pytest.mark.parametrize(("leaf", "type_name"), LEAVES)
def test_non_array_leaf_raises_type_error_with_type_name(
    name: str, leaf: object, type_name: str
) -> None:
    with pytest.raises(TypeError, match=f"object of type {type_name}"):
        call_with_leaf(name, leaf)


#########################################################
#     Tests for the containers (P5: structure kept)     #
#########################################################

CONTAINERS: list[Callable[[np.ndarray], object]] = [
    lambda x: {"a": x},
    lambda x: OrderedDict(a=x),
    lambda x: defaultdict(list, a=x),
    lambda x: [x],
    lambda x: (x,),
    lambda x: Point(x, x),
]
CONTAINER_IDS = ["dict", "OrderedDict", "defaultdict", "list", "tuple", "namedtuple"]


@pytest.mark.parametrize("make", CONTAINERS, ids=CONTAINER_IDS)
def test_map_function_keeps_container_type(make: Callable) -> None:
    data = make(np.ones((4, 3)))
    assert type(bn.sum_along_batch(data)) is type(data)


@pytest.mark.parametrize("make", CONTAINERS, ids=CONTAINER_IDS)
def test_chunk_keeps_container_type(make: Callable) -> None:
    data = make(np.ones((4, 3)))
    assert type(bn.chunk_along_batch(data, 2)[0]) is type(data)


@pytest.mark.parametrize("make", CONTAINERS, ids=CONTAINER_IDS)
def test_split_keeps_container_type(make: Callable) -> None:
    data = make(np.ones((4, 3)))
    assert type(bn.split_along_batch(data, 2)[0]) is type(data)


@pytest.mark.parametrize("make", CONTAINERS, ids=CONTAINER_IDS)
def test_concatenate_keeps_container_type(make: Callable) -> None:
    data = make(np.ones((4, 3)))
    assert type(bn.concatenate_along_batch([data, data])) is type(data)


def test_map_function_keeps_nested_values() -> None:
    data = {"a": (np.ones((2, 3)), [np.zeros((2, 3))]), "b": np.ones((2, 3))}
    assert objects_are_equal(
        bn.sum_along_batch(data),
        {"a": (np.array([2.0, 2.0, 2.0]), [np.array([0.0, 0.0, 0.0])]), "b": np.array([2.0] * 3)},
    )


#########################################################
#     Tests for the annotations (P14: T -> T)           #
#########################################################

PUBLIC_FUNCTIONS = sorted(
    name for name in bn.__all__ if inspect.isfunction(getattr(bn, name)) and name != "NestedArray"
)


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_nested_function_first_parameter_is_t(name: str) -> None:
    first = next(iter(inspect.signature(getattr(bn, name)).parameters.values()))
    assert first.annotation in {"T", "Sequence[T]"}


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_nested_function_returns_t_or_list_of_t(name: str) -> None:
    assert inspect.signature(getattr(bn, name)).return_annotation in {"T", "list[T]"}
