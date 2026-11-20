# Migration guide

This page lists the names that were removed from `batcharray` and their replacement. The package is
pre-1.0: a removal has no deprecation period.

## Pointwise functions and `to_list` of `nested`

The 28 pointwise functions and `to_list` were removed from `batcharray.nested`. They have no `array`
twin and no batch or sequence meaning: each one was `recursive_apply(data, np.f)`. Use
`recursive_apply` of `coola` with the NumPy function:

```python
import numpy as np
from coola.recursive import recursive_apply

data = {"a": np.array([[1.0, 4.0], [9.0, 16.0]])}

out = recursive_apply(data, np.sqrt)
```

| Removed                                                       | Replacement                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------ |
| `nested.abs(data)`                                            | `recursive_apply(data, np.abs)`                              |
| `nested.ceil(data)`                                           | `recursive_apply(data, np.ceil)`                             |
| `nested.clip(data, a_min, a_max)`                             | `recursive_apply(data, lambda x: np.clip(x, a_min, a_max))`  |
| `nested.exp(data)`, `exp2`, `expm1`                           | `recursive_apply(data, np.exp)`, `np.exp2`, `np.expm1`       |
| `nested.floor(data)`                                          | `recursive_apply(data, np.floor)`                            |
| `nested.isnan(data)`                                          | `recursive_apply(data, np.isnan)`                            |
| `nested.log(data)`, `log10`, `log1p`, `log2`                  | `recursive_apply(data, np.log)`, `np.log10`, `np.log1p`, `np.log2` |
| `nested.round(data)`                                          | `recursive_apply(data, np.round)`                            |
| `nested.sign(data)`                                           | `recursive_apply(data, np.sign)`                             |
| `nested.sqrt(data)`, `square`                                 | `recursive_apply(data, np.sqrt)`, `np.square`                |
| `nested.sin(data)`, `cos`, `tan`                              | `recursive_apply(data, np.sin)`, `np.cos`, `np.tan`          |
| `nested.arcsin(data)`, `arccos`, `arctan`                     | `recursive_apply(data, np.arcsin)`, `np.arccos`, `np.arctan` |
| `nested.sinh(data)`, `cosh`, `tanh`                           | `recursive_apply(data, np.sinh)`, `np.cosh`, `np.tanh`       |
| `nested.arcsinh(data)`, `arccosh`, `arctanh`                  | `recursive_apply(data, np.arcsinh)`, `np.arccosh`, `np.arctanh` |
| `nested.to_list(data)`                                        | `recursive_apply(data, np.ndarray.tolist)`                   |

!!! note

    `recursive_apply` does not check the type of a leaf: a leaf that is not an array is passed to the
    function. A later step of the refactoring adds `batcharray.utils.apply_to_arrays`, which
    raises a `TypeError` for a leaf that is not an array. This page will then use it.
