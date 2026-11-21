from __future__ import annotations

import ast
from pathlib import Path

import pytest

import batcharray.array

ARRAY_MODULES = sorted(Path(batcharray.array.__file__).parent.glob("*.py"))


def find_raises(path: Path) -> list[int]:
    r"""Return the line numbers of the ``raise`` statements of a
    module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Raise)]


@pytest.mark.parametrize("path", ARRAY_MODULES, ids=lambda path: path.name)
def test_no_inline_raise_in_array(path: Path) -> None:
    # Every check of ``array`` is a function of ``batcharray.validation`` (P8).
    assert find_raises(path) == []
