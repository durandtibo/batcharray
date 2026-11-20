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

## The `computation` package

`batcharray.computation` (the computation models, the registry and the 14 functions of the
interface) was removed. A masked array is dispatched by the method of the array (`x.sum(axis=0)`),
or by `numpy.ma` for `median`, `sort`, `argsort` and `concatenate`. The one helper is
`batcharray.utils.get_namespace`, which returns `numpy.ma` for a masked array and `numpy` otherwise.

| Removed                                                          | Replacement                                          |
| ---------------------------------------------------------------- | ---------------------------------------------------- |
| `computation.sum(x, axis=0)`, `prod`, `mean`, `std`, `var`, `min`, `max`, `argmax`, `argmin` | `x.sum(axis=0)`, ... (the method of the array) |
| `computation.median(x, axis=0)`, `sort`, `argsort`              | `get_namespace(x).median(x, axis=0)`, ...            |
| `computation.concatenate(arrays, axis=0)`                        | `get_namespace(arrays[0]).concatenate(arrays, axis=0)` |
| `ArrayComputationModel`, `MaskedArrayComputationModel`, `AutoComputationModel`, `BaseComputationModel`, and `register_computation_models` | none: an array type with other semantics is not supported |

## The `types` module

`batcharray.types` moved to `batcharray.utils.types`. `ArrayT` and `SORT_KINDS` were removed;
`SortKind` stays and is exported by `batcharray.utils`.

## Aliases: `amax_*`, `amin_*` and `take_*`

`amax_*` and `amin_*` were aliases of `max_*` and `min_*` (as in NumPy). `take_*` was documented as
equal to `index_select_*`.

| Removed (`array` and `nested`)                          | Replacement                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `amax_along_batch`, `amax_along_seq`                    | `max_along_batch`, `max_along_seq`                                |
| `amin_along_batch`, `amin_along_seq`                    | `min_along_batch`, `min_along_seq`                                |
| `take_along_batch`, `take_along_seq` with a 1-D index   | `index_select_along_batch`, `index_select_along_seq`              |
| `take_along_seq` with a 2-D index (per-sample gather)   | `index_select_along_seq` for now (a later step adds `gather_along_seq`) |
| `take_along_batch` with a 2-D index                     | `np.take(array, indices, axis=0)`                                 |

## Keyword-only options and the deprecation shim

`keepdims`, `kind`, `rng` and `step` are keyword-only. The `keepdims_keyword_only` shim (and
`batcharray._deprecation`) was removed: a positional `keepdims` is a `TypeError`.

| Before                                    | After                                      |
| ----------------------------------------- | ------------------------------------------ |
| `sum_along_batch(x, True)`                | `sum_along_batch(x, keepdims=True)`        |
| `sort_along_batch(x, "stable")`           | `sort_along_batch(x, kind="stable")`       |
| `argsort_along_seq(x, "stable")`          | `argsort_along_seq(x, kind="stable")`      |
| `shuffle_along_batch(x, rng)`             | `shuffle_along_batch(x, rng=rng)`          |
| `slice_along_batch(x, 0, 10, 2)`          | `slice_along_batch(x, 0, 10, step=2)`      |

`shuffle_along_*` now passes `rng` to `numpy.random.default_rng`: a `Generator`, an integer seed or
`None`. `split_along_*` accepts NumPy integers as the split size and rejects a `bool` with a
`ValueError`.
