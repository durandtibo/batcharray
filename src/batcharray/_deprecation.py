r"""Contain private helpers to deprecate arguments."""

from __future__ import annotations

__all__ = ["keepdims_keyword_only"]

import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def keepdims_keyword_only(func: F) -> F:
    r"""Deprecate passing ``keepdims`` as a positional argument.

    The decorated function must have the signature
    ``func(data, *, keepdims=False)``. A positional ``keepdims`` is
    still accepted but emits a ``DeprecationWarning``.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(data: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            if len(args) > 1 or "keepdims" in kwargs:
                msg = (
                    f"{func.__name__}() takes 1 positional argument but {len(args) + 1} were given"
                )
                raise TypeError(msg)
            warnings.warn(
                f"Passing `keepdims` as a positional argument to {func.__name__}() is deprecated "
                "and will be removed in a future release. Use `keepdims=...` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["keepdims"] = args[0]
        return func(data, **kwargs)

    return wrapper  # type: ignore[return-value]
