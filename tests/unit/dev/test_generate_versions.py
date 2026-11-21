from __future__ import annotations

import importlib.util
from pathlib import Path

from coola.equality import objects_are_equal
from packaging.requirements import Requirement

PATH = Path(__file__).parents[3] / "dev" / "generate_versions.py"
SPEC = importlib.util.spec_from_file_location("generate_versions", PATH)
generate_versions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_versions)

restrict_to_bounds = generate_versions.restrict_to_bounds

AVAILABLE = ("1.0.0", "1.1.10", "2.0.0a3", "2.0.0a4")

#########################################
#     Tests for restrict_to_bounds     #
#########################################


def test_restrict_to_bounds_removes_the_versions_below_a_pre_release_bound() -> None:
    result = restrict_to_bounds(
        Requirement("coola>=2.0.0a4,<3.0"), ["1.0.0", "1.1.10"], AVAILABLE
    )
    assert objects_are_equal(result, ["2.0.0a4"])


def test_restrict_to_bounds_adds_the_pre_release_lower_bound() -> None:
    result = restrict_to_bounds(Requirement("coola>=2.0.0a3,<3.0"), [], AVAILABLE)
    assert objects_are_equal(result, ["2.0.0a3", "2.0.0a4"])


def test_restrict_to_bounds_adds_the_latest_accepted_version() -> None:
    result = restrict_to_bounds(Requirement("coola>=2.0.0a3,<3.0"), ["2.0.0a3"], AVAILABLE)
    assert objects_are_equal(result, ["2.0.0a3", "2.0.0a4"])


def test_restrict_to_bounds_keeps_the_versions_of_a_stable_bound() -> None:
    result = restrict_to_bounds(
        Requirement("coola>=1.0,<2.0"), ["1.0.0", "1.1.10"], AVAILABLE
    )
    assert objects_are_equal(result, ["1.0.0", "1.1.10"])


def test_restrict_to_bounds_does_not_add_a_version_for_a_stable_bound() -> None:
    result = restrict_to_bounds(Requirement("coola>=1.0,<3.0"), ["1.0.0"], AVAILABLE)
    assert objects_are_equal(result, ["1.0.0"])


def test_restrict_to_bounds_sorts_the_versions() -> None:
    result = restrict_to_bounds(
        Requirement("coola>=1.0,<2.0"), ["1.1.10", "1.0.0", "1.0.1"], AVAILABLE
    )
    assert objects_are_equal(result, ["1.0.0", "1.0.1", "1.1.10"])


def test_restrict_to_bounds_no_available_version() -> None:
    result = restrict_to_bounds(Requirement("coola>=2.0.0a4,<3.0"), [], ())
    assert objects_are_equal(result, ["2.0.0a4"])


def test_restrict_to_bounds_nothing_accepted() -> None:
    result = restrict_to_bounds(Requirement("coola>=5.0,<6.0"), ["1.0.0"], AVAILABLE)
    assert objects_are_equal(result, [])
