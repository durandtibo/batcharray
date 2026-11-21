from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest

from batcharray import array, nested

if TYPE_CHECKING:
    from collections.abc import Callable

# The layout (P1) is stated once in docs/docs/design_principles.md (P16).
FORBIDDEN = ("assumes the batch axis", "assumes the sequence axis")


def get_functions(module: object) -> dict[str, Callable]:
    return {name: getattr(module, name) for name in module.__all__}


FUNCTIONS = {
    f"{module.__name__}.{name}": func
    for module in (array, nested)
    for name, func in get_functions(module).items()
}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_docstring_does_not_repeat_layout(name: str) -> None:
    doc = " ".join((inspect.getdoc(FUNCTIONS[name]) or "").split())
    assert not any(text in doc for text in FORBIDDEN)
