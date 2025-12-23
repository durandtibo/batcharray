from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence

import pytest

from batcharray.recursive2 import TransformerRegistry, get_default_registry
from batcharray.recursive2.transformer import SequenceTransformer


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list):
    r"""Create a custom class that inherits from list."""


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a TransformerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, TransformerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_scalar_types() -> None:
    """Test that scalar types are registered with DefaultTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(int)
    assert registry.has_transformer(float)
    assert registry.has_transformer(complex)
    assert registry.has_transformer(bool)
    assert registry.has_transformer(str)


def test_get_default_registry_sequences() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(list)
    assert registry.has_transformer(tuple)
    assert registry.has_transformer(Sequence)


def test_get_default_registry_sets() -> None:
    """Test that set types are registered with SetTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(set)
    assert registry.has_transformer(frozenset)


def test_register_default_transformers_registers_mappings() -> None:
    """Test that mapping types are registered with
    MappingTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(dict)
    assert registry.has_transformer(Mapping)


def test_register_default_transformers_registers_object() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_transformer(object)


def test_default_registry_can_transform_list() -> None:
    """Test that default registry can transform a list."""
    registry = get_default_registry()
    assert registry.transform([1, 2, 3], str) == ["1", "2", "3"]


def test_default_registry_can_transform_dict() -> None:
    """Test that default registry can transform a dict."""
    registry = get_default_registry()
    assert registry.transform({"a": 1, "b": 2}, lambda x: x * 10) == {"a": 10, "b": 20}


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_transformer(CustomList)
    registry1.register(CustomList, SequenceTransformer())
    assert registry1.has_transformer(CustomList)

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_transformer(CustomList)
