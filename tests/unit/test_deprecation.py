from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array, nested
from batcharray._deprecation import keepdims_keyword_only

OLD_REDUCTIONS = [
    f"{name}_along_{axis}"
    for name in ("amax", "amin", "argmax", "argmin", "max", "mean", "median", "min", "prod", "sum")
    for axis in ("batch", "seq")
]


@keepdims_keyword_only
def _func(data: np.ndarray, *, keepdims: bool = False) -> tuple:
    return data, keepdims


def test_keepdims_keyword_only_keyword() -> None:
    assert _func(1, keepdims=True) == (1, True)


def test_keepdims_keyword_only_default() -> None:
    assert _func(1) == (1, False)


def test_keepdims_keyword_only_positional_warns() -> None:
    with pytest.warns(DeprecationWarning, match="positional argument"):
        assert _func(1, True) == (1, True)


def test_keepdims_keyword_only_too_many_positional_raises() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        _func(1, True, False)


def test_keepdims_keyword_only_positional_and_keyword_raises() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        _func(1, True, keepdims=True)


def test_keepdims_keyword_only_preserves_metadata() -> None:
    assert _func.__name__ == "_func"


@pytest.mark.parametrize("name", OLD_REDUCTIONS)
def test_array_reduction_positional_keepdims_warns(name: str) -> None:
    x = np.arange(12.0).reshape(4, 3)
    with pytest.warns(DeprecationWarning, match=name):
        out = getattr(array, name)(x, True)
    assert objects_are_equal(out, getattr(array, name)(x, keepdims=True))


@pytest.mark.parametrize("name", OLD_REDUCTIONS)
def test_nested_reduction_positional_keepdims_warns(name: str) -> None:
    data = {"a": np.arange(12.0).reshape(4, 3)}
    with pytest.warns(DeprecationWarning, match=name):
        out = getattr(nested, name)(data, True)
    assert objects_are_equal(out, getattr(nested, name)(data, keepdims=True))


@pytest.mark.parametrize("name", OLD_REDUCTIONS)
def test_reduction_keyword_keepdims_does_not_warn(
    name: str, recwarn: pytest.WarningsRecorder
) -> None:
    getattr(array, name)(np.ones((2, 2)), keepdims=True)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_clip_bounds() -> None:
    data = {"a": np.array([[0, 5], [10, 20]])}
    assert objects_are_equal(
        nested.clip(data, a_min=2, a_max=10), {"a": np.array([[2, 5], [10, 10]])}
    )
    assert objects_are_equal(nested.clip(data, a_max=7), {"a": np.array([[0, 5], [7, 7]])})
