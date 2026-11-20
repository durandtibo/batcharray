r"""Tests that related public functions behave consistently."""

from __future__ import annotations

import numpy as np
import pytest

from batcharray import array as ba, nested as bn

ARRAY = np.arange(24, dtype=float).reshape(4, 3, 2)
MASKED = np.ma.masked_array(ARRAY, mask=ARRAY % 5 == 0)


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


@pytest.mark.parametrize("name", REDUCTIONS)
def test_array_reduction_keepdims_is_keyword_only(name: str) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        getattr(ba, name)(ARRAY, True)


@pytest.mark.parametrize("name", REDUCTIONS)
def test_nested_reduction_keepdims_is_keyword_only(name: str) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        getattr(bn, name)({"a": ARRAY}, True)
