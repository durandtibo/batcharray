r"""Tests that related public functions behave consistently."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array as ba, nested as bn

ARRAY = np.arange(24, dtype=float).reshape(4, 3, 2)
MASKED = np.ma.masked_array(ARRAY, mask=ARRAY % 5 == 0)


@pytest.mark.parametrize("axis_name", ["batch", "seq"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("x", [ARRAY, MASKED])
def test_amax_amin_equal_max_min(axis_name: str, keepdims: bool, x: np.ndarray) -> None:
    for alias, name in (("amax", "max"), ("amin", "min")):
        out = getattr(ba, f"{alias}_along_{axis_name}")(x, keepdims=keepdims)
        expected = getattr(ba, f"{name}_along_{axis_name}")(x, keepdims=keepdims)
        assert objects_are_equal(out, expected)
        out = getattr(bn, f"{alias}_along_{axis_name}")({"a": x}, keepdims=keepdims)
        expected = getattr(bn, f"{name}_along_{axis_name}")({"a": x}, keepdims=keepdims)
        assert objects_are_equal(out, expected)


@pytest.mark.parametrize("name", ["amax", "amin"])
@pytest.mark.parametrize("axis_name", ["batch", "seq"])
def test_amax_amin_use_computation_model(name: str, axis_name: str) -> None:
    target = "max" if name == "amax" else "min"
    with patch(f"batcharray.array.reduction.cmpt.{target}") as mock:
        getattr(ba, f"{name}_along_{axis_name}")(ARRAY)
    mock.assert_called_once()


@pytest.mark.parametrize("axis_name", ["batch", "seq"])
def test_index_select_equal_take(axis_name: str) -> None:
    indices = np.array([2, 0, 1])
    out = getattr(ba, f"index_select_along_{axis_name}")(ARRAY, indices)
    assert objects_are_equal(out, getattr(ba, f"take_along_{axis_name}")(ARRAY, indices))
    out = getattr(bn, f"index_select_along_{axis_name}")({"a": ARRAY}, indices)
    assert objects_are_equal(out, getattr(bn, f"take_along_{axis_name}")({"a": ARRAY}, indices))


def test_take_along_batch_2d_indices_adds_dimension() -> None:
    assert ba.take_along_batch(ARRAY, np.array([[0, 1], [2, 3]])).shape == (2, 2, 3, 2)


def test_take_along_seq_1d_indices() -> None:
    assert ba.take_along_seq(ARRAY, np.array([2, 0])).shape == (4, 2, 2)


def test_take_along_seq_2d_indices() -> None:
    assert ba.take_along_seq(ARRAY, np.array([[0, 1]] * 4)).shape == (4, 2, 2)


@pytest.mark.parametrize("axis_name", ["batch", "seq"])
def test_chunk_follows_array_split(axis_name: str) -> None:
    axis = 0 if axis_name == "batch" else 1
    x = np.arange(10).reshape(5, 2) if axis == 0 else np.arange(10).reshape(2, 5)
    out = getattr(ba, f"chunk_along_{axis_name}")(x, chunks=3)
    expected = np.array_split(x, 3, axis=axis)
    assert len(out) == len(expected) == 3
    assert [o.shape for o in out] == [e.shape for e in expected]
    assert all(np.array_equal(o, e) for o, e in zip(out, expected, strict=True))


def test_chunk_returns_requested_number_of_chunks() -> None:
    # torch.chunk(torch.arange(5), 4) returns 3 chunks; array_split returns 4.
    assert len(ba.chunk_along_batch(np.arange(5), chunks=4)) == 4
