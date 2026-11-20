# Migration guide

This page lists the names that were removed from `batcharray` and their replacement. The package is
pre-1.0: a removal has no deprecation period.

## Pointwise functions and `to_list` of `nested`

The 28 pointwise functions and `to_list` were removed from `batcharray.nested`. They have no `array`
twin and no batch or sequence meaning: each one was `apply_to_arrays(data, np.f)`. Use
`apply_to_arrays` of `batcharray.utils` with the NumPy function:

```python
import numpy as np
from batcharray.utils import apply_to_arrays

data = {"a": np.array([[1.0, 4.0], [9.0, 16.0]])}

out = apply_to_arrays(data, np.sqrt)
```

| Removed                                                       | Replacement                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------ |
| `nested.abs(data)`                                            | `apply_to_arrays(data, np.abs)`                              |
| `nested.ceil(data)`                                           | `apply_to_arrays(data, np.ceil)`                             |
| `nested.clip(data, a_min, a_max)`                             | `apply_to_arrays(data, lambda x: np.clip(x, a_min, a_max))`  |
| `nested.exp(data)`, `exp2`, `expm1`                           | `apply_to_arrays(data, np.exp)`, `np.exp2`, `np.expm1`       |
| `nested.floor(data)`                                          | `apply_to_arrays(data, np.floor)`                            |
| `nested.isnan(data)`                                          | `apply_to_arrays(data, np.isnan)`                            |
| `nested.log(data)`, `log10`, `log1p`, `log2`                  | `apply_to_arrays(data, np.log)`, `np.log10`, `np.log1p`, `np.log2` |
| `nested.round(data)`                                          | `apply_to_arrays(data, np.round)`                            |
| `nested.sign(data)`                                           | `apply_to_arrays(data, np.sign)`                             |
| `nested.sqrt(data)`, `square`                                 | `apply_to_arrays(data, np.sqrt)`, `np.square`                |
| `nested.sin(data)`, `cos`, `tan`                              | `apply_to_arrays(data, np.sin)`, `np.cos`, `np.tan`          |
| `nested.arcsin(data)`, `arccos`, `arctan`                     | `apply_to_arrays(data, np.arcsin)`, `np.arccos`, `np.arctan` |
| `nested.sinh(data)`, `cosh`, `tanh`                           | `apply_to_arrays(data, np.sinh)`, `np.cosh`, `np.tanh`       |
| `nested.arcsinh(data)`, `arccosh`, `arctanh`                  | `apply_to_arrays(data, np.arcsinh)`, `np.arccosh`, `np.arctanh` |
| `nested.to_list(data)`                                        | `apply_to_arrays(data, np.ndarray.tolist)`                   |

`apply_to_arrays` raises a `TypeError` that names the type of a leaf that is not an array.

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
| `take_along_seq` with a 2-D index (per-sample gather)   | `gather_along_seq`                                                |
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

## Reductions on a 1-D array and `flip_along_seq`, `repeat_along_seq`

- The `*_along_batch` reductions (`all`, `any`, `argmax`, `argmin`, `max`, `mean`, `median`, `min`,
  `prod`, `std`, `sum`, `var`) and `select_along_batch` return a 0-d array where NumPy returns a
  scalar (`np.float64`, `np.int64`, `np.bool_`), so the output of any function is a valid input of
  `batcharray.nested`. A 0-d array behaves like a scalar in arithmetic, and `float(x)` and
  `x.item()` work.
- `flip_along_seq` and `repeat_along_seq` no longer check the number of dimensions: a 1-D array
  raises NumPy's `AxisError` (a `ValueError` and an `IndexError`) instead of the `ValueError` of
  `batcharray`.

## The index family

- `gather_along_seq(array, indices)` is new at both levels: `out[b, i] = array[b, indices[b, i]]`
  with a 2-D `indices` of shape `(batch_size, n)`. It replaces the 2-D branch of `take_along_seq`
  (which was also reachable with `index_select_along_seq`) and raises a `ValueError` when the batch
  size of the indices differs from the one of the array (it returned fewer rows without an error).
  A typical use is the last valid step of a padded batch:
  `gather_along_seq(x, (lengths - 1)[:, None])`.
- `index_select_along_batch` and `index_select_along_seq` raise a `ValueError` for an index that is
  not 1-D (they returned a higher-rank array or, along the sequence axis, did a per-sample gather).
- `masked_select_along_batch` and `masked_select_along_seq` raise a `ValueError` for a mask that is
  not boolean (an integer mask was read as indices).

## The size validators

`validate_first_array`, `validate_same_batch_size` and `validate_same_seq_size` of
`batcharray.validation` were removed (`validate_min_ndim`, `validate_split_sections` and
`validate_same_keys` stay). `batcharray.nested.batch_size(data)` and `batcharray.nested.seq_len(data)`
do the same check and return the size: all the arrays must have the same size along the axis,
otherwise (or for a structure without array) a `ValueError` is raised, and a leaf that is not an
array raises a `TypeError`.

| Removed                                  | Replacement                              |
| ---------------------------------------- | ---------------------------------------- |
| `validate_same_batch_size(data)`         | `batch_size(data)`                       |
| `validate_same_seq_size(data)`           | `seq_len(data)`                          |
| `validate_first_array(data).shape[0]`    | `batch_size(data)`                       |

The three `import-linter` contracts are replaced by one `layers` contract
(`nested` > `array` > `utils` > `validation` > `constants`).

`permute_along_batch` and `permute_along_seq` raise a `ValueError` for a permutation that is not
1-D (a 2-D permutation with the right first dimension was accepted).
