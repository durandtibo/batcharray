r"""Tests of the signatures of the public functions of ``array`` and
``nested`` (P1, P3, P13)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from batcharray import array as ba, nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

# The functions of nested that have no twin in array: they read a shape.
NESTED_ONLY = {"batch_size", "seq_len"}
# The arguments that define the operation: they can be positional.
POSITIONAL_WITH_DEFAULT = {"start", "stop"}

ARRAY_FUNCTIONS = sorted(ba.__all__)
NESTED_FUNCTIONS = sorted(bn.__all__)
BOTH_FUNCTIONS = sorted(set(ba.__all__) & set(bn.__all__))


def get_parameters(func: Callable) -> list[inspect.Parameter]:
    return list(inspect.signature(func).parameters.values())


def get_options(func: Callable) -> list[tuple[str, object, object]]:
    r"""Return the (name, kind, default) of the parameters after the data."""
    return [(p.name, p.kind, p.default) for p in get_parameters(func)[1:]]


#####################################
#     Tests for P1: no `axis`       #
#####################################


@pytest.mark.parametrize("name", ARRAY_FUNCTIONS)
def test_array_function_has_no_axis_parameter(name: str) -> None:
    assert "axis" not in inspect.signature(getattr(ba, name)).parameters


@pytest.mark.parametrize("name", NESTED_FUNCTIONS)
def test_nested_function_has_no_axis_parameter(name: str) -> None:
    assert "axis" not in inspect.signature(getattr(bn, name)).parameters


###################################
#     Tests for P3: parity        #
###################################


def test_api_parity_names() -> None:
    assert set(bn.__all__) - set(ba.__all__) == NESTED_ONLY


def test_api_parity_array_names_are_in_nested() -> None:
    assert set(ba.__all__) - set(bn.__all__) == set()


@pytest.mark.parametrize("name", BOTH_FUNCTIONS)
def test_api_parity_options(name: str) -> None:
    assert get_options(getattr(ba, name)) == get_options(getattr(bn, name))


@pytest.mark.parametrize("name", BOTH_FUNCTIONS)
def test_api_parity_same_number_of_positional_data_parameters(name: str) -> None:
    assert len(get_parameters(getattr(ba, name))) == len(get_parameters(getattr(bn, name)))


#########################################
#     Tests for P13: keyword-only       #
#########################################


def get_positional_with_default(module: object, name: str) -> list[str]:
    return [
        p.name
        for p in get_parameters(getattr(module, name))[1:]
        if p.default is not inspect.Parameter.empty
        and p.kind is not inspect.Parameter.KEYWORD_ONLY
        and p.name not in POSITIONAL_WITH_DEFAULT
    ]


@pytest.mark.parametrize("name", ARRAY_FUNCTIONS)
def test_array_options_are_keyword_only(name: str) -> None:
    assert get_positional_with_default(ba, name) == []


@pytest.mark.parametrize("name", NESTED_FUNCTIONS)
def test_nested_options_are_keyword_only(name: str) -> None:
    assert get_positional_with_default(bn, name) == []


@pytest.mark.parametrize("name", ARRAY_FUNCTIONS)
def test_array_function_has_no_var_args(name: str) -> None:
    kinds = {p.kind for p in get_parameters(getattr(ba, name))}
    assert not kinds & {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}


@pytest.mark.parametrize("name", NESTED_FUNCTIONS)
def test_nested_function_has_no_var_args(name: str) -> None:
    kinds = {p.kind for p in get_parameters(getattr(bn, name))}
    assert not kinds & {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}


def get_parameters_named(name: str) -> list[inspect.Parameter]:
    r"""Return the parameters with this name in all the public functions."""
    return [
        p
        for module in (ba, bn)
        for func_name in module.__all__
        for p in get_parameters(getattr(module, func_name))
        if p.name == name
    ]


OPTIONS = ["keepdims", "ddof", "kind", "step", "rng", "dtype"]


@pytest.mark.parametrize("name", OPTIONS)
def test_known_option_is_used(name: str) -> None:
    assert len(get_parameters_named(name)) > 0


@pytest.mark.parametrize("name", OPTIONS)
def test_known_option_is_keyword_only(name: str) -> None:
    assert {p.kind for p in get_parameters_named(name)} == {inspect.Parameter.KEYWORD_ONLY}
