from __future__ import annotations

import numpy as np
import pytest

from batcharray.validation import (
    validate_min_ndim,
    validate_same_keys,
    validate_split_sections,
)

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


@pytest.mark.parametrize("sections", [np.int64(1), np.int64(2), np.int32(5)])
def test_validate_split_sections_numpy_integer_valid(sections: np.integer) -> None:
    validate_split_sections(5, sections)


@pytest.mark.parametrize("sections", [np.int64(0), np.int64(-1)])
def test_validate_split_sections_numpy_integer_invalid(sections: np.integer) -> None:
    with pytest.raises(ValueError, match="split size must be a positive integer"):
        validate_split_sections(5, sections)


@pytest.mark.parametrize("sections", [True, False])
def test_validate_split_sections_bool_invalid(sections: bool) -> None:
    with pytest.raises(ValueError, match="split size must be a positive integer"):
        validate_split_sections(5, sections)
