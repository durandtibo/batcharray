# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from feu.utils.io import save_json
from feu.utils.mapping import sort_by_keys
from feu.version import (
    fetch_latest_major_versions_map,
    fetch_latest_minor_versions_map,
    fetch_pypi_versions,
    partition_package_bounds,
    read_pyproject_dependencies,
    read_pyproject_optional_dependencies,
)
from packaging.requirements import Requirement
from packaging.version import Version

logger: logging.Logger = logging.getLogger(__name__)


def restrict_to_bounds(
    requirement: Requirement, versions: list[str], available: tuple[str, ...]
) -> list[str]:
    r"""Restrict the versions of a package to the ones its requirement accepts.

    The generator skips the pre-releases, so a pre-release lower bound (for
    example ``coola>=2.0.0a4``) is missing and the versions it returns can be
    older than the lower bound. This function removes the versions that the
    specifier refuses. If the lower bound is a pre-release, it adds the lower
    bound and the latest accepted version, so the CI tests what the package
    declares.

    Args:
        requirement: The requirement read from ``pyproject.toml``.
        versions: The versions returned by the generator.
        available: The versions available on PyPI.

    Returns:
        The sorted versions accepted by the specifier.
    """
    spec = requirement.specifier
    accepted = {v for v in versions if spec.contains(v, prereleases=True)}
    lower_bounds = [Version(s.version) for s in spec if s.operator in {">=", "=="}]
    if any(v.is_prerelease for v in lower_bounds):
        accepted.update(str(v) for v in lower_bounds if v.is_prerelease)
        candidates = [v for v in available if spec.contains(v, prereleases=True)]
        if candidates:
            accepted.add(max(candidates, key=Version))
    return sorted(accepted, key=Version)


def fetch_package_versions(base_dir: Path) -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Args:
        base_dir: Path to the base directory.

    Returns:
        A dictionary with the versions for each package.
    """
    pyproject_path = base_dir.joinpath("pyproject.toml")

    deps = read_pyproject_dependencies(pyproject_path) + read_pyproject_optional_dependencies(
        pyproject_path
    )
    major_deps, minor_deps = partition_package_bounds(deps, [])

    versions = sort_by_keys(
        fetch_latest_major_versions_map(major_deps, include_lower_bound=True)
        | fetch_latest_minor_versions_map(minor_deps, include_lower_bound=True)
    )

    with pyproject_path.open("rb") as file:
        requirements = [Requirement(r) for r in tomllib.load(file)["project"]["dependencies"]]
    for requirement in requirements:
        name = requirement.name
        if name in versions:
            versions[name] = restrict_to_bounds(
                requirement, versions[name], fetch_pypi_versions(name)
            )
    return versions


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    base_dir = Path(__file__).parent.parent
    versions = fetch_package_versions(base_dir)
    logger.info("versions=%s", versions)
    path = base_dir.joinpath("dev/config").joinpath("package_versions.json")
    logger.info("Saving package versions to %s", path)
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
