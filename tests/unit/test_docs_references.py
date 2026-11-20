from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs" / "docs"
WORKFLOWS = ROOT / ".github" / "workflows"

WORKFLOW_PATTERN = re.compile(r"actions/workflows/([\w.-]+\.ya?ml)")
MODULE_PATTERN = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)


def get_workflow_references() -> list[tuple[str, str]]:
    r"""Return the ``(file, workflow)`` pairs of the workflows linked in the docs."""
    files = [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]
    return [
        (file.relative_to(ROOT).as_posix(), name)
        for file in files
        for name in sorted(set(WORKFLOW_PATTERN.findall(file.read_text())))
    ]


def get_module_references() -> list[tuple[str, str]]:
    r"""Return the ``(file, module)`` pairs of the modules documented in the API reference."""
    return [
        (file.relative_to(ROOT).as_posix(), name)
        for file in sorted((DOCS / "refs").glob("*.md"))
        for name in MODULE_PATTERN.findall(file.read_text())
    ]


def get_required_dependencies() -> list[str]:
    r"""Return the names of the required dependencies of ``pyproject.toml``."""
    with (ROOT / "pyproject.toml").open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    return [re.split(r"[<>=!~\[ ]", dependency, maxsplit=1)[0] for dependency in dependencies]


###################################
#     Tests for the workflows     #
###################################


@pytest.mark.parametrize(("file", "workflow"), get_workflow_references())
def test_linked_workflow_exists(file: str, workflow: str) -> None:
    assert (WORKFLOWS / workflow).is_file(), f"{file} links the missing workflow {workflow}"


#################################
#     Tests for the modules     #
#################################


@pytest.mark.parametrize(("file", "module"), get_module_references())
def test_documented_module_exists(file: str, module: str) -> None:
    assert importlib.import_module(module) is not None, f"{file} documents {module}"


######################################
#     Tests for the dependencies     #
######################################


def test_required_dependencies() -> None:
    assert sorted(get_required_dependencies()) == ["coola", "numpy"]


@pytest.mark.parametrize("dependency", get_required_dependencies())
def test_index_names_required_dependency(dependency: str) -> None:
    assert dependency.lower() in (DOCS / "index.md").read_text().lower()


def test_index_does_not_claim_zero_dependencies() -> None:
    assert "zero dependencies" not in (DOCS / "index.md").read_text().lower()
