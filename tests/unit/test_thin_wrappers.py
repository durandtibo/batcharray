r"""Test of P2: an ``array`` function is one NumPy call with the axis
fixed, except the functions of the closed list of
``docs/docs/design_principles.md``."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

from batcharray import array as ba

# The reductions of the closed list: ``*_along_batch`` is wrapped in ``np.asanyarray``.
REDUCTION_STEMS = (
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
# The closed list of the functions that are not one NumPy call.
NOT_ONE_NUMPY_CALL = {
    # np.asanyarray around the NumPy call (P12)
    *(f"{name}_along_batch" for name in REDUCTION_STEMS),
    "select_along_batch",
    "split_along_batch",
    "split_along_seq",
    "concatenate_along_batch",
    "concatenate_along_seq",
    "stack_along_batch",
    "stack_along_seq",
    "index_select_along_batch",
    "index_select_along_seq",
    "masked_select_along_batch",
    "masked_select_along_seq",
    "gather_along_seq",
    "tile_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
}
# The name of the callee of a NumPy call: ``np.f``, ``array.f`` (a method of
# the array) or ``get_namespace(array).f`` (``numpy`` or ``numpy.ma``).
CALLEE_PREFIXES = ("np.", "array.", "get_namespace(array).")
THIN_FUNCTIONS = sorted(set(ba.__all__) - NOT_ONE_NUMPY_CALL)


def get_body(name: str) -> list[ast.stmt]:
    r"""Return the statements of a function without its docstring."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ba, name))))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    return func.body[1:]


def get_callees(node: ast.AST) -> list[str]:
    return [ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


@pytest.mark.parametrize("name", THIN_FUNCTIONS)
def test_thin_wrapper_body_is_one_return(name: str) -> None:
    assert [type(stmt) for stmt in get_body(name)] == [ast.Return]


@pytest.mark.parametrize("name", THIN_FUNCTIONS)
def test_thin_wrapper_calls_only_numpy(name: str) -> None:
    callees = get_callees(get_body(name)[0])
    assert all(c.startswith(CALLEE_PREFIXES) or c == "get_namespace" for c in callees)


@pytest.mark.parametrize("name", THIN_FUNCTIONS)
def test_thin_wrapper_makes_at_most_one_numpy_call(name: str) -> None:
    callees = get_callees(get_body(name)[0])
    assert len([c for c in callees if c != "get_namespace"]) <= 1


@pytest.mark.parametrize("name", sorted(NOT_ONE_NUMPY_CALL))
def test_closed_list_only_has_existing_functions(name: str) -> None:
    assert name in ba.__all__


############################################
#     Tests for the table of the docs      #
############################################

DESIGN_PRINCIPLES = Path(__file__).parents[2] / "docs" / "docs" / "design_principles.md"
REDUCTIONS_ROW = "*_along_batch` reductions"


def get_documented_closed_list() -> set[str]:
    r"""Return the functions of the table "Exceptions to the thin
    wrappers".

    The row of the reductions is expanded to the ``*_along_batch``
    reductions, and the ``_*`` forms to the batch and sequence
    functions.
    """
    text = DESIGN_PRINCIPLES.read_text(encoding="utf-8")
    section = text.split("## Exceptions to the thin wrappers", maxsplit=1)[1]
    section = section.split("\n## ", maxsplit=1)[0]
    names: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|")[1]
        if REDUCTIONS_ROW in cell:
            names.update(f"{stem}_along_batch" for stem in REDUCTION_STEMS)
        for name in re.findall(r"`([a-z_*]+)`", cell):
            if name == "*_along_batch":
                continue
            if name.endswith("_*"):
                names.update((f"{name[:-1]}along_batch", f"{name[:-1]}along_seq"))
            else:
                names.add(name)
    return names


def test_closed_list_matches_docs() -> None:
    assert get_documented_closed_list() == NOT_ONE_NUMPY_CALL
