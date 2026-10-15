from __future__ import annotations

import numpy as np
import pytest

from batcharray.validation import (
    validate_first_array,
    validate_min_ndim,
    validate_same_keys,
    validate_split_sections,
)

#####################################
#     validate_first_array     #
#####################################


def test_validate_first_array_array() -> None:
    array = np.ones((2, 3))
    assert validate_first_array(array) is array


def test_validate_first_array_nested() -> None:
    array = np.ones((2, 3))
    assert validate_first_array({"a": [array, np.zeros(2)]}) is array


@pytest.mark.parametrize("data", [{}, [], {"a": 1}, 1])
def test_validate_first_array_invalid(data: object) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        validate_first_array(data)


#############################
#     validate_min_ndim     #
#############################


def test_validate_min_ndim_valid() -> None:
    validate_min_ndim(np.ones((2, 3)))
    validate_min_ndim(np.ones((2, 3, 4)))
    validate_min_ndim(np.ones(2), min_ndim=1)


def test_validate_min_ndim_invalid() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        validate_min_ndim(np.ones(2))


##############################
#     validate_same_keys     #
##############################


def test_validate_same_keys_valid() -> None:
    validate_same_keys([{"a": 1, "b": 2}, {"b": 3, "a": 4}])


def test_validate_same_keys_empty() -> None:
    validate_same_keys([])


@pytest.mark.parametrize(
    "data", [[{"a": 1}, {"a": 1, "b": 2}], [{"a": 1, "b": 2}, {"a": 1}], [{"a": 1}, {"b": 1}]]
)
def test_validate_same_keys_invalid(data: list) -> None:
    with pytest.raises(ValueError, match="same keys"):
        validate_same_keys(data)


#####################################
#     validate_split_sections     #
#####################################


@pytest.mark.parametrize("sections", [1, 2, 10, [5], [2, 3], [0, 5], (2, 3)])
def test_validate_split_sections_valid(sections: int | list[int]) -> None:
    validate_split_sections(5, sections)


@pytest.mark.parametrize("sections", [0, -1, [2, 2], [3, 3], [-1, 6]])
def test_validate_split_sections_invalid(sections: int | list[int]) -> None:
    with pytest.raises(ValueError, match="split"):
        validate_split_sections(5, sections)
