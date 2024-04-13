from __future__ import annotations

__all__ = ["AutoComputationModel", "register_computation_models"]

from typing import ClassVar, TypeVar, TYPE_CHECKING

import numpy as np
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from batcharray.computation.base import BaseComputationModel

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from collections.abc import Sequence

T = TypeVar("T", bound=np.ndarray)


class AutoComputationModel(BaseComputationModel[T]):
    """Implement a computation model that automatically finds the right
    computation model based on the array type.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from batcharray.computation import AutoComputationModel
    >>> comp_model = AutoComputationModel()
    >>> arr = np.ones((2, 3))
    >>> # TODO

    ```
    """

    registry: ClassVar[dict[type, BaseComputationModel]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self.registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_computation_model(
        cls,
        array_type: type[np.ndarray],
        comp_model: BaseComputationModel[T],
        exist_ok: bool = False,
    ) -> None:
        r"""Add a computation model for a given array type.

        Args:
            array_type: The array type.
            comp_model: The computation model to use for the given array
                type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the computation model for a array
                type.

        Raises:
            RuntimeError: if a computation model is already registered for
                the array type and ``exist_ok=False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from batcharray.computation import AutoComputationModel, ArrayComputationModel
        >>> AutoComputationModel.add_computation_model(np.ndarray, ArrayComputationModel(), exist_ok=True)

        ```
        """
        if array_type in cls.registry and not exist_ok:
            msg = (
                f"A computation model {cls.registry[array_type]} is already registered for the "
                f"array type {array_type}. Please use `exist_ok=True` if you want to overwrite "
                "the computation model for this array type"
            )
            raise RuntimeError(msg)
        cls.registry[array_type] = comp_model

    @classmethod
    def has_computation_model(cls, array_type: type[np.ndarray]) -> bool:
        r"""Indicate if a computation model is registered for the given
        array type.

        Args:
            array_type: The array type.

        Returns:
            ``True`` if a computation model is registered,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from batcharray.computation import AutoComputationModel
        >>> AutoComputationModel.has_computation_model(np.ndarray)
        True
        >>> AutoComputationModel.has_computation_model(str)
        False

        ```
        """
        return array_type in cls.registry

    @classmethod
    def find_computation_model(cls, array_type: type[np.ndarray]) -> BaseComputationModel[T]:
        r"""Find the computation model associated to an array type.

        Args:
            array_type: The array type.

        Returns:
            The computation model associated to the array type.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from batcharray.computation import AutoComputationModel
        >>> AutoComputationModel.find_computation_model(np.ndarray)
        LinearSizeFinder()
        >>> AutoComputationModel.find_computation_model(np.ma.MaskedArray)
        BilinearSizeFinder()

        ```
        """
        for object_type in array_type.__mro__:
            comp_model = cls.registry.get(object_type, None)
            if comp_model is not None:
                return comp_model
        msg = f"Incorrect array type: {array_type}"
        raise TypeError(msg)

    def concatenate(
        self, arrays: Sequence[T], axis: int | None = None, *, dtype: DTypeLike = None
    ) -> T:
        pass


def register_computation_models() -> None:
    r"""Register computation models to ``AutoComputationModel``.

    Example usage:

    ```pycon

    >>> from batcharray.computation import AutoComputationModel, register_computation_models
    >>> register_computation_models()
    >>> comp_model = AutoComputationModel()
    >>> comp_model
    AutoComputationModel(
      ...
    )

    ```
    """
    # Local import to avoid cyclic dependency
    from batcharray import computation as cmpt

    comp_models = {
        np.ndarray: cmpt.ArrayComputationModel(),
        np.ma.MaskedArray: cmpt.MaskedArrayComputationModel(),
    }

    for array_type, comp_model in comp_models.items():
        if not AutoComputationModel.has_computation_model(array_type):  # pragma: no cover
            AutoComputationModel.add_computation_model(array_type, comp_model)
