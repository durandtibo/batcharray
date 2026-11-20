r"""Tests of the error contract: one row per line of the table of
``docs/docs/design_principles.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from batcharray import array as ba

if TYPE_CHECKING:
    from collections.abc import Callable

VECTOR = np.arange(4.0)


###############################################################
#     Tests for the `_along_seq` functions on a 1-D array     #
###############################################################

# Our own validation: the output of NumPy would be wrong or silent.
VALIDATED: dict[str, Callable[[np.ndarray], object]] = {
    "tile_along_seq": lambda x: ba.tile_along_seq(x, 2),
    "shuffle_along_seq": ba.shuffle_along_seq,
    "permute_along_seq": lambda x: ba.permute_along_seq(x, np.arange(4)),
}

# NumPy's AxisError (a ValueError and an IndexError).
AXIS_ERROR: dict[str, Callable[[np.ndarray], object]] = {
    "sum_along_seq": ba.sum_along_seq,
    "cumsum_along_seq": ba.cumsum_along_seq,
    "sort_along_seq": ba.sort_along_seq,
    "argmax_along_seq": ba.argmax_along_seq,
    "flip_along_seq": ba.flip_along_seq,
    "repeat_along_seq": lambda x: ba.repeat_along_seq(x, 2),
    "concatenate_along_seq": lambda x: ba.concatenate_along_seq([x, x]),
    "index_select_along_seq": lambda x: ba.index_select_along_seq(x, np.array([0])),
}

# NumPy's IndexError.
INDEX_ERROR: dict[str, Callable[[np.ndarray], object]] = {
    "slice_along_seq": lambda x: ba.slice_along_seq(x, 0, 2),
    "select_along_seq": lambda x: ba.select_along_seq(x, 0),
    "split_along_seq": lambda x: ba.split_along_seq(x, 2),
    "chunk_along_seq": lambda x: ba.chunk_along_seq(x, 2),
    "masked_select_along_seq": lambda x: ba.masked_select_along_seq(x, x > 1),
}


@pytest.mark.parametrize("name", VALIDATED)
def test_seq_function_on_1d_array_validated_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        VALIDATED[name](VECTOR)


@pytest.mark.parametrize("name", AXIS_ERROR)
def test_seq_function_on_1d_array_axis_error_is_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        AXIS_ERROR[name](VECTOR)


@pytest.mark.parametrize("name", AXIS_ERROR)
def test_seq_function_on_1d_array_axis_error_is_index_error(name: str) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        AXIS_ERROR[name](VECTOR)


@pytest.mark.parametrize("name", INDEX_ERROR)
def test_seq_function_on_1d_array_index_error(name: str) -> None:
    with pytest.raises(IndexError, match=r"too many indices|out of range"):
        INDEX_ERROR[name](VECTOR)
