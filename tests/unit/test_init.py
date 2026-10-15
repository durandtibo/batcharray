from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import pytest

import batcharray

##############################
#     Tests for lazy import  #
##############################


@pytest.mark.parametrize("name", ["array", "computation", "nested"])
def test_lazy_subpackage_import(name: str) -> None:
    code = f"import batcharray; assert batcharray.{name}.__name__ == 'batcharray.{name}'"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)  # noqa: S603


def test_getattr_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match=r"has no attribute 'missing'"):
        _ = batcharray.missing


#################################
#     Tests for __version__     #
#################################


def test_version_is_str() -> None:
    assert isinstance(batcharray.__version__, str)
    assert batcharray.__version__


def test_version_matches_installed_metadata() -> None:
    assert batcharray.__version__ == version("batcharray")


def test_version_fallback_when_not_installed() -> None:
    code = (
        "import importlib.metadata as m\n"
        "def fail(name):\n"
        "    raise m.PackageNotFoundError(name)\n"
        "m.version = fail\n"
        "import batcharray\n"
        "assert batcharray.__version__ == '0.0.0'\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)  # noqa: S603
