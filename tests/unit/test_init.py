from __future__ import annotations

import subprocess
import sys

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
