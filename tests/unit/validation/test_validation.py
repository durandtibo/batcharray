from __future__ import annotations

import numpy as np
import pytest

from batcharray.validation import (
    validate_first_array,
    validate_min_ndim,
    validate_same_batch_size,
    validate_same_keys,
    validate_same_seq_size,
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


##########################################
#     validate_same_batch_size / seq     #
##########################################


@pytest.mark.parametrize(
    "data",
    [
        np.ones((2, 3)),
        {"a": np.ones((2, 3)), "b": np.zeros(2), "c": [np.ones((2, 5, 4))], "d": (np.ones(2),)},
    ],
)
def test_validate_same_batch_size_valid(data: object) -> None:
    validate_same_batch_size(data)


@pytest.mark.parametrize(
    "data",
    [
        {"a": np.ones((2, 3)), "b": np.zeros(3)},
        {"a": {"b": np.ones((2, 3))}, "c": [np.ones((4, 3))]},
    ],
)
def test_validate_same_batch_size_different_sizes(data: object) -> None:
    with pytest.raises(ValueError, match="same batch size"):
        validate_same_batch_size(data)


def test_validate_same_batch_size_0d() -> None:
    with pytest.raises(ValueError, match="0 dimension"):
        validate_same_batch_size({"a": np.array(1.0)})


@pytest.mark.parametrize("data", [{"a": 1}, {"a": np.ones(2), "b": "x"}])
def test_validate_same_batch_size_not_array(data: object) -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        validate_same_batch_size(data)


def test_validate_same_batch_size_empty() -> None:
    validate_same_batch_size({})


def test_validate_same_seq_size_valid() -> None:
    validate_same_seq_size({"a": np.ones((2, 3)), "b": np.zeros((5, 3, 4)), "c": [np.ones((1, 3))]})


def test_validate_same_seq_size_different_sizes() -> None:
    with pytest.raises(ValueError, match="same sequence size"):
        validate_same_seq_size({"a": np.ones((2, 3)), "b": np.zeros((2, 4))})


def test_validate_same_seq_size_1d() -> None:
    with pytest.raises(ValueError, match="1 dimension"):
        validate_same_seq_size({"a": np.ones((2, 3)), "b": np.zeros(2)})


def test_validate_same_seq_size_not_array() -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        validate_same_seq_size({"a": 1})
