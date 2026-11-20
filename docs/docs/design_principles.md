# Design principles

This page states the rules of `batcharray` once. The docstrings of the functions describe only what
is specific to a function (its edge cases, its exceptions and its example) and link here for the
rules.

!!! note "Status"

    Each property has a gate: a test or a CI job that fails when it breaks (see the table
    [Gates](#gates)). A property without a gate is given one or removed.
    [`review/target_state.md`](https://github.com/durandtibo/batcharray/blob/main/review/target_state.md)
    records how the code reached this state. The package is pre-1.0: the API can change between
    releases.

## Layout

A batch array has the shape `(batch_size, seq_len, *)`. The batch axis is axis 0 and the sequence
axis is axis 1. The axes come only from `batcharray.constants.BATCH_AXIS` and
`batcharray.constants.SEQ_AXIS`.

No function of `array` or `nested` takes an `axis` argument. Data laid out as `(seq, batch, ...)` is
transposed first. A `_along_seq` function needs an array with at least 2 dimensions.

## Properties

| #   | Property               | Statement                                                                                                                                                                            |
| --- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| P1  | Fixed layout           | Batch is axis 0 and sequence is axis 1. No function takes an `axis`.                                                                                                                 |
| P2  | Thin wrappers          | An `array` function is one NumPy call with the axis fixed: same semantics, same result, no hidden copy or cast. The exceptions are the closed list below.                           |
| P3  | Two symmetric levels   | Every `array` function has a `nested` twin with the same name, options and defaults. The exceptions are `batch_size` and `seq_len` (`nested` only). In `nested`, a `mask`, an `indices` or a `permutation` is one array shared by all the leaves. |
| P4  | One traversal          | Every `nested` function goes through one of the three helpers of `batcharray.utils.structure`. No `nested` function walks a container.                                               |
| P5  | Structure-preserving   | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple` and their subclasses.                                             |
| P6  | Array leaves only      | A leaf that is not a `numpy.ndarray` raises a `TypeError` that names the leaf type. Nothing is skipped silently and no leaf is converted.                                            |
| P7  | Pure                   | Inputs are never modified. The output never aliases the input, except where NumPy returns a view (see the list of views below).                                                      |
| P8  | One error contract     | A bad leaf type is a `TypeError`. An argument that `batcharray` validates is a `ValueError`. Anything else is the NumPy error, unchanged.                                           |
| P9  | Explicit randomness    | Randomness comes only from `rng`, which is passed to `numpy.random.default_rng`. One permutation is shared by all the leaves.                                                        |
| P10 | No cast, no conversion | No function casts or converts what the caller passes. The `dtype` of `concatenate_*` is an explicit option that NumPy defines.                                                       |
| P11 | Masked arrays          | A `numpy.ma.MaskedArray` input gives a `MaskedArray` output with the semantics of `numpy.ma`, except for the functions that return indices (`argsort`, `argmax`, `argmin`).         |
| P12 | Array in, array out    | A function that receives an `ndarray` returns an `ndarray`. Where NumPy returns a scalar, the result is a 0-d array, so the output of any function is a valid input of any other.   |
| P13 | Keyword-only options   | The positional arguments are the data and the arguments that define the operation. Options (`keepdims`, `ddof`, `kind`, `step`, `rng`, `dtype`) are keyword-only.                    |
| P14 | Typed                  | A map function has the type `T -> T` with `T` bound to the nested array type: dict in, dict out, array in, array out.                                                                |
| P15 | Layered                | `constants` <- `validation` <- `utils` <- `array` <- `nested`. No import in the other direction. `numpy` and `coola` are the only required dependencies.                            |
| P16 | Documented by example  | Every public function has a docstring with a runnable `pycon` example. Each rule is stated once, on this page.                                                                       |
| P17 | Defined on edge cases  | Every function has a documented result on empty and degenerate input. "NumPy's behaviour" is an answer, "undefined" is not.                                                          |
| P18 | Documented claims exist | Every file, workflow or module that the README and the docs point to exists, and the description of the package matches `pyproject.toml`.                                           |

## Gates

| Property | Gate                                                                                                                  |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| P1       | `tests/unit/test_signatures.py` (no `axis` parameter)                                                                 |
| P2       | `tests/unit/test_thin_wrappers.py` (the body of a function outside the closed list is one NumPy call)                 |
| P3       | `tests/unit/test_signatures.py` (same names, options and defaults in `array` and `nested`)                            |
| P4       | `tests/unit/test_nested_contract.py` (no `recursive_apply`, no `isinstance` on a container in `nested`)               |
| P5       | `tests/unit/test_nested_contract.py` and `tests/unit/utils/test_structure.py` (the container types are kept)          |
| P6       | `tests/unit/test_nested_contract.py` (every `nested` function raises a `TypeError` for a non-array leaf)              |
| P7       | `tests/unit/test_purity.py` (inputs are not modified, no aliasing except the views)                                   |
| P8       | `tests/unit/test_error_contract.py` (one row per line of the error table)                                             |
| P9       | `tests/unit/test_rng.py`                                                                                              |
| P10      | the dtype tests of `concatenate_*` and of the reductions in `tests/unit/array`                                        |
| P11      | `tests/unit/test_masked_arrays.py` (every function on a masked array equals the `numpy.ma` result)                    |
| P12      | `tests/unit/test_array_in_array_out.py`                                                                               |
| P13      | `tests/unit/test_signatures.py` (options are keyword-only, no `**kwargs`)                                             |
| P14      | `ty` in CI and `tests/unit/test_nested_contract.py` (the annotations are `T -> T`)                                    |
| P15      | the `layers` contract of `import-linter` (`ci-cyclic-imports`)                                                        |
| P16      | `ci-doctest` (the examples run)                                                                                       |
| P17      | `tests/unit/test_edge_case.py` (one test per row of the edge-case table)                                              |
| P18      | `tests/unit/test_docs_references.py`                                                                                  |

**Not guaranteed:**

- Shared references: `sort_along_batch({"a": x, "b": x})` returns two arrays.
- Memory sharing between input and output, except for the views listed below.
- The order of the leaves inside a set.
- Dim-last or sequence-first layouts. Transpose first.
- 0-d arrays: `sum_along_batch(np.array(1.0))` raises NumPy's error. A 0-d array has no batch axis.
- Dataclasses and sets as containers.
- That an index or a `permutation` argument is a valid permutation (only its length is checked).
- A generator or a lazy iterable as `items` of `concatenate_*`. It takes a `Sequence`.

## Containers

The supported structures are `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). Sets and dataclasses are outside the contract.

For `concatenate_*`, the output containers are the ones of the first item and the other items must
have the same keys and lengths. A `list` and a `tuple` can be combined (the output has the type of
the first item), as can a `dict` and an `OrderedDict`. A mapping combined with a sequence raises a
`ValueError`.

For `chunk_*` and `split_*`, all the leaves must have the same size along the split axis, otherwise
the pieces of different leaves are not aligned and a `ValueError` is raised.

## Exceptions to the thin wrappers

A function outside this list is one NumPy call with the axis fixed, and returns a new array or the
view NumPy returns.

| Function                                         | Why it is not one NumPy call                                                                                   |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `*_along_batch` reductions and `select_along_batch` | the NumPy call is wrapped in `np.asanyarray`, because NumPy returns a scalar on a 1-D input (P12)           |
| `split_*`                                        | `np.split` with indices computed from a size or sections (`np.split(x, 2)` means 2 sections, not size 2)       |
| `concatenate_*` with `dtype` on a masked array   | `np.ma.concatenate` has no `dtype`: `astype` after it                                                          |
| `index_select_*`                                 | `np.take` and a check that the index is 1-D (NumPy accepts any shape and returns another rank)                 |
| `masked_select_*`                                | boolean indexing and a check that the mask is boolean (an integer mask is read as indices)                     |
| `gather_along_seq`                               | `np.take_along_axis` with the index reshaped over the feature axes, and a check of the batch size (NumPy broadcasts) |
| `tile_along_seq`                                 | `np.tile` with the repetitions built from the rank, and a check of the rank (NumPy promotes a 1-D array)       |
| `permute_*`                                      | `index_select` and a length check (NumPy would accept a shorter index)                                         |
| `shuffle_*`                                      | draws the permutation with `np.random.default_rng(rng)`, then `permute_*`                                      |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `flip_*`. Everything else
returns a new array. `sort_*`, `cumsum_*`, `repeat_*` and `tile_*` with a count of 1 return a new
array.

## Masked arrays

A reduction, a cumulative function, `argmax`, `argmin`, `any` and `all` use the method of the array,
which `MaskedArray` overrides. `median`, `sort`, `argsort` and `concatenate` use `numpy.ma` for a
masked array and `numpy` otherwise. A subclass of `MaskedArray` is dispatched as a masked array, a
subclass of `ndarray` is a plain array, and an array type with other semantics is not supported.

A function that NumPy does not support on a masked array is not added to the package until it has a
masked implementation.

## Errors

`batcharray` validates only what NumPy cannot see, or what NumPy accepts but means something else.
It never repeats a check NumPy makes.

| Situation                                                                                           | Exception                           |
| --------------------------------------------------------------------------------------------------- | ----------------------------------- |
| A leaf is not an `ndarray`                                                                          | `TypeError`                         |
| `concatenate` receives no item, or structures with a different layout (keys, lengths)               | `ValueError`                        |
| `concatenate` combines a mapping with a sequence                                                    | `ValueError`                        |
| Leaves disagree on the size of the axis of `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len` | `ValueError`                      |
| A split size that is not positive, sections that are negative or do not sum to the size            | `ValueError`                        |
| `index_select_*` with an index that is not 1-D; `masked_select_*` with a mask that is not boolean   | `ValueError`                        |
| `gather_along_seq` with an index that is not 2-D or whose batch size differs                        | `ValueError`                        |
| `permute_*` with a permutation of the wrong rank or length                                          | `ValueError`                        |
| `tile_along_seq`, `shuffle_along_seq`, `permute_along_seq`, `gather_along_seq`, `chunk_along_seq`, `split_along_seq`, `seq_len` on an array with fewer than 2 dimensions | `ValueError` |
| Another `_along_seq` function on a 1-D array (`slice`, `select`, `sum`, `flip`, `repeat`, `sort`, ...) | NumPy's `AxisError` or `IndexError` |
| An out-of-range or a float index, a negative count, a bad `kind`, a bad `ddof`                      | NumPy's own error                   |

A NumPy `AxisError` is both a `ValueError` and an `IndexError`, so a caller can catch either.

## Indices

An index argument is handled the same way in every function, and is never cast or moved:

| Function                 | Index                                       | Checked by `batcharray`                 |
| ------------------------ | ------------------------------------------- | --------------------------------------- |
| `index_select_along_*`   | 1-D integer array                           | `ndim == 1`                             |
| `permute_*`              | 1-D integer array of length = size of axis  | `ndim` and length                       |
| `shuffle_*`              | drawn by `rng.permutation`                  | n/a                                     |
| `gather_along_seq`       | 2-D integer array `(batch_size, n)`         | `ndim` and batch size                   |
| `masked_select_*`        | 1-D boolean array                           | dtype is `bool`                         |

The dtype of the index, the bounds and the negative values are NumPy's rules.

## Edge cases

The result on an empty or degenerate input is NumPy's, and is pinned by a test:

| Function                                                     | Case                         | Result                                  |
| ------------------------------------------------------------ | ---------------------------- | --------------------------------------- |
| `sum_along_batch`, `prod_along_batch`                        | zero batch size              | the identity (`0`, `1`) for each column |
| `mean_along_batch`                                           | zero batch size              | `nan` and NumPy's `RuntimeWarning`      |
| `max_along_batch`, `min_along_batch`, `argmax_*`, `argmin_*` | zero batch size              | NumPy's `ValueError`                    |
| `chunk_*`                                                    | more chunks than items       | empty chunks (`np.array_split`)         |
| `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len`   | a structure without array    | `ValueError`                            |
| `sum_along_batch` and other map functions                    | an empty structure           | the empty structure (`{}`)              |
| `permute_*`                                                  | a repeated index             | not checked: only the length is         |
| `tile_along_seq`                                             | `reps == 0`                  | a zero-length sequence axis             |
| `*_along_batch` reductions and `select_along_batch`           | a 1-D array                  | a 0-d array (P12)                       |
| a masked reduction                                           | a fully masked row or column | what `numpy.ma` returns (`masked`)      |

A function that would need the values of an array to validate it does not validate it.
