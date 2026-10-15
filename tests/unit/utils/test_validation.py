from __future__ import annotations

import numpy as np
import pytest

from batcharray.utils.validation import (
    check_first_array,
    check_flat_dict_of_arrays,
    check_min_ndim,
    check_same_keys,
    check_split_sections,
)

#####################################
#     check_first_array     #
#####################################


def test_check_first_array_array() -> None:
    array = np.ones((2, 3))
    assert check_first_array(array) is array


def test_check_first_array_nested() -> None:
    array = np.ones((2, 3))
    assert check_first_array({"a": [array, np.zeros(2)]}) is array


@pytest.mark.parametrize("data", [{}, [], {"a": 1}, 1])
def test_check_first_array_invalid(data: object) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        check_first_array(data)


##########################################
#     check_flat_dict_of_arrays     #
##########################################


def test_check_flat_dict_of_arrays_valid() -> None:
    check_flat_dict_of_arrays({"a": np.ones(2), "b": np.zeros((2, 3))})


@pytest.mark.parametrize("data", [{"a": {"b": np.ones(2)}}, [np.ones(2)], np.ones(2), {"a": 1}])
def test_check_flat_dict_of_arrays_invalid(data: object) -> None:
    with pytest.raises(TypeError, match="flat dictionary"):
        check_flat_dict_of_arrays(data)


#############################
#     check_min_ndim     #
#############################


def test_check_min_ndim_valid() -> None:
    check_min_ndim(np.ones((2, 3)))
    check_min_ndim(np.ones((2, 3, 4)))
    check_min_ndim(np.ones(2), min_ndim=1)


def test_check_min_ndim_invalid() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        check_min_ndim(np.ones(2))


##############################
#     check_same_keys     #
##############################


def test_check_same_keys_valid() -> None:
    check_same_keys([{"a": 1, "b": 2}, {"b": 3, "a": 4}])


def test_check_same_keys_empty() -> None:
    check_same_keys([])


@pytest.mark.parametrize(
    "data", [[{"a": 1}, {"a": 1, "b": 2}], [{"a": 1, "b": 2}, {"a": 1}], [{"a": 1}, {"b": 1}]]
)
def test_check_same_keys_invalid(data: list) -> None:
    with pytest.raises(ValueError, match="same keys"):
        check_same_keys(data)


#####################################
#     check_split_sections     #
#####################################


@pytest.mark.parametrize("sections", [1, 2, 10, [5], [2, 3], [0, 5], (2, 3)])
def test_check_split_sections_valid(sections: int | list[int]) -> None:
    check_split_sections(5, sections)


@pytest.mark.parametrize("sections", [0, -1, [2, 2], [3, 3], [-1, 6]])
def test_check_split_sections_invalid(sections: int | list[int]) -> None:
    with pytest.raises(ValueError, match="split"):
        check_split_sections(5, sections)
