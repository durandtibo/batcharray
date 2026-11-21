from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import batcharray

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs" / "docs"
WORKFLOWS = ROOT / ".github" / "workflows"

WORKFLOW_PATTERN = re.compile(r"actions/workflows/([\w.-]+\.ya?ml)")
MODULE_PATTERN = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)


def get_workflow_references() -> list[tuple[str, str]]:
    r"""Return the ``(file, workflow)`` pairs of the workflows linked in
    the docs."""
    files = [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]
    return [
        (file.relative_to(ROOT).as_posix(), name)
        for file in files
        for name in sorted(set(WORKFLOW_PATTERN.findall(file.read_text())))
    ]


def get_module_references() -> list[tuple[str, str]]:
    r"""Return the ``(file, module)`` pairs of the modules documented in
    the API reference."""
    return [
        (file.relative_to(ROOT).as_posix(), name)
        for file in sorted((DOCS / "refs").glob("*.md"))
        for name in MODULE_PATTERN.findall(file.read_text())
    ]


def get_top_level_modules() -> list[str]:
    r"""Return the modules of the package documented in the API
    reference."""
    return sorted(
        {
            name.split(".")[1]
            for _, name in get_module_references()
            if name.startswith("batcharray.")
        }
    )


def get_requirements() -> list[Requirement]:
    r"""Return the required dependencies of ``pyproject.toml``."""
    with (ROOT / "pyproject.toml").open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    return [Requirement(dependency) for dependency in dependencies]


def get_required_dependencies() -> list[str]:
    r"""Return the names of the required dependencies of
    ``pyproject.toml``."""
    return [requirement.name for requirement in get_requirements()]


def get_tested_versions(name: str) -> list[str]:
    r"""Return the versions of a dependency installed by the dependency
    CI."""
    path = ROOT / "dev" / "config" / "package_versions.json"
    return json.loads(path.read_text())[name]


def get_readme_main_row() -> list[str]:
    r"""Return the cells of the row ``main`` of the table of versions of
    the README."""
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("|") and cells[0] == "`main`":
            return cells
    msg = "The README has no row `main`"
    raise AssertionError(msg)


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


@pytest.mark.parametrize("module", get_top_level_modules())
def test_documented_module_is_in_all(module: str) -> None:
    assert module in batcharray.__all__


@pytest.mark.parametrize("module", get_top_level_modules())
def test_documented_module_is_an_attribute_in_a_fresh_interpreter(module: str) -> None:
    code = f"import batcharray; assert batcharray.{module}.__name__ == 'batcharray.{module}'"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)  # noqa: S603


@pytest.mark.parametrize("module", get_top_level_modules())
def test_index_links_the_reference_of_the_module(module: str) -> None:
    assert f"refs/{module}.md" in (DOCS / "index.md").read_text()


@pytest.mark.parametrize("module", get_top_level_modules())
def test_readme_names_the_module(module: str) -> None:
    assert f"- `{module}`:" in (ROOT / "README.md").read_text()


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


@pytest.mark.parametrize("requirement", get_requirements(), ids=lambda r: r.name)
def test_readme_main_row_has_the_bounds_of_pyproject(requirement: Requirement) -> None:
    header = [
        cell.strip().strip("`")
        for line in (ROOT / "README.md").read_text().splitlines()
        if line.startswith("| `batcharray`")
        for cell in line.strip("|").split("|")
    ]
    cell = get_readme_main_row()[header.index(requirement.name)]
    assert SpecifierSet(cell.strip("`")) == requirement.specifier


@pytest.mark.parametrize("requirement", get_requirements(), ids=lambda r: r.name)
def test_tested_versions_satisfy_pyproject(requirement: Requirement) -> None:
    refused = [
        version
        for version in get_tested_versions(requirement.name)
        if not requirement.specifier.contains(Version(version), prereleases=True)
    ]
    assert refused == []
