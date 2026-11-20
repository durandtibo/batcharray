from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.validation import (
    validate_indices,
    validate_mask,
    validate_min_ndim,
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


############################
#     validate_indices     #
############################


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8, np.float64])
def test_validate_indices_dtype_is_not_checked(dtype: type) -> None:
    validate_indices(np.zeros(3, dtype=dtype), ndim=1)


def test_validate_indices_valid_2d() -> None:
    validate_indices(np.zeros((2, 3), dtype=int), ndim=2)


def test_validate_indices_valid_length() -> None:
    validate_indices(np.zeros(3, dtype=int), ndim=1, length=3)


def test_validate_indices_valid_empty() -> None:
    validate_indices(np.zeros(0, dtype=int), ndim=1, length=0)


@pytest.mark.parametrize("indices", [[0, 1], (0, 1), 1, None, "ab"])
def test_validate_indices_not_array_raises(indices: object) -> None:
    with pytest.raises(TypeError, match=r"indices must be a numpy.ndarray, but got an object of"):
        validate_indices(indices, ndim=1)  # ty: ignore[invalid-argument-type]


def test_validate_indices_not_array_message_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"object of type list"):
        validate_indices([0, 1], ndim=1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("ndim", [1, 2])
def test_validate_indices_bool_raises(ndim: int) -> None:
    with pytest.raises(ValueError, match="indices must be an integer array"):
        validate_indices(np.ones((2,) * ndim, dtype=bool), ndim=ndim)


@pytest.mark.parametrize("indices", [np.array(1), np.zeros((2, 2), dtype=int)])
def test_validate_indices_wrong_ndim_raises(indices: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"indices must be a 1-D array, but got \d dimension"):
        validate_indices(indices, ndim=1)


def test_validate_indices_wrong_ndim_2d_raises() -> None:
    with pytest.raises(ValueError, match=r"indices must be a 2-D array, but got 1 dimension"):
        validate_indices(np.zeros(3, dtype=int), ndim=2)


@pytest.mark.parametrize("length", [0, 2, 4])
def test_validate_indices_wrong_length_raises(length: int) -> None:
    with pytest.raises(ValueError, match=rf"indices must have {length} elements along the first"):
        validate_indices(np.zeros(3, dtype=int), ndim=1, length=length)


def test_validate_indices_does_not_check_the_length_by_default() -> None:
    validate_indices(np.zeros(100, dtype=int), ndim=1)


def test_validate_indices_does_not_modify_indices() -> None:
    indices = np.array([2, 0, 1])
    validate_indices(indices, ndim=1, length=3)
    assert objects_are_equal(indices, np.array([2, 0, 1]))


#######################
#     validate_mask     #
#######################


def test_validate_mask_valid() -> None:
    validate_mask(np.array([True, False, True]))


def test_validate_mask_valid_empty() -> None:
    validate_mask(np.zeros(0, dtype=bool))


@pytest.mark.parametrize("mask", [[True, False], (True, False), 1, None])
def test_validate_mask_not_array_raises(mask: object) -> None:
    with pytest.raises(TypeError, match=r"mask must be a numpy.ndarray, but got an object of"):
        validate_mask(mask)  # ty: ignore[invalid-argument-type]


def test_validate_mask_not_array_message_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"object of type list"):
        validate_mask([True, False])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "mask", [np.array([0, 1]), np.array([1.0, 0.0]), np.array([1, 0], dtype=np.uint8)]
)
def test_validate_mask_not_bool_raises(mask: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mask must be a boolean array, but got dtype"):
        validate_mask(mask)


@pytest.mark.parametrize("mask", [np.array(True), np.ones((2, 3), dtype=bool)])
def test_validate_mask_not_1d_raises(mask: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"mask must be a 1-D array, but got \d dimension"):
        validate_mask(mask)


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
