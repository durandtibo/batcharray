# Design principles

This page states the rules of `batcharray` once. The docstrings of the functions describe only what
is specific to a function (its edge cases, its exceptions and its example) and link here for the
rules.

!!! note "Status"

    Each property has a gate: a test or a CI job that fails when it breaks (see the
    table [Gates](#gates)). A property without a gate is given one or removed.
    [`review/target_state.md`](https://github.com/durandtibo/batcharray/blob/main/review/target_state.md)
    records how the code reached this state. The package is pre-1.0: the API can
    change between releases.

## Layout

A batch array has the shape `(batch_size, seq_len, *)`. The batch axis is axis 0 and the sequence
axis is axis 1. The axes come only from `batcharray.constants.BATCH_AXIS` and
`batcharray.constants.SEQ_AXIS`.

No function of `array` or `nested` takes an `axis` argument. Data laid out as `(seq, batch, ...)` is
transposed first. A `_along_seq` function needs an array with at least 2 dimensions.

A `nested` function applies the same layout to every leaf. The leaves of a structure are expected to
have the same batch size and the same sequence size, and this is checked only where the alignment of
the output depends on it (`chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len`). A function
that takes a `mask`, an `indices` or a `permutation` (`masked_select_*`, `index_select_*`,
`permute_*`, `gather_along_seq`) takes one array shared by all the leaves.

## Properties

| #   | Property                | Statement                                                                                                                                                                                                                                                                                                                     |
| --- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | Fixed layout            | Batch is axis 0 and sequence is axis 1. No function takes an `axis`.                                                                                                                                                                                                                                                          |
| P2  | Thin wrappers           | An `array` function is one NumPy call with the axis fixed: same semantics, same result, no hidden copy or cast. The exceptions are the closed list below.                                                                                                                                                                     |
| P3  | Two symmetric levels    | Every `array` function has a `nested` twin with the same name, options and defaults. The exceptions are `batch_size` and `seq_len` (`nested` only). In `nested`, a `mask`, an `indices` or a `permutation` is one array shared by all the leaves.                                                                             |
| P4  | One traversal           | Every `nested` function goes through one of the three helpers of `batcharray.utils.structure`. No `nested` function walks a container.                                                                                                                                                                                        |
| P5  | Structure-preserving    | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple` and their subclasses.                                                                                                                                                                                      |
| P6  | Array leaves only       | A leaf that is not a `numpy.ndarray` raises a `TypeError` that names the leaf type. Nothing is skipped silently and no leaf is converted. The `array` level takes a `numpy.ndarray` (the annotation): a non-array argument is outside the contract.                                                                           |
| P7  | Pure                    | Inputs are never modified. The output never aliases the input, except where NumPy returns a view (see the list of views below).                                                                                                                                                                                               |
| P8  | One error contract      | A bad leaf type is a `TypeError`. An argument that `batcharray` validates is a `ValueError`. Anything else is the NumPy error, unchanged. Every check is a function of `batcharray.validation`.                                                                                                                               |
| P9  | Explicit randomness     | Randomness comes only from `rng`, which is passed to `numpy.random.default_rng`. One permutation is shared by all the leaves.                                                                                                                                                                                                 |
| P10 | No cast, no conversion  | No function casts or converts what the caller passes. The `dtype` of `concatenate_*` is an explicit option that NumPy defines.                                                                                                                                                                                                |
| P11 | Masked arrays           | A `numpy.ma.MaskedArray` input gives a `MaskedArray` output with the semantics of `numpy.ma`, except for the functions that return indices (`argsort`, `argmax`, `argmin`). When a function takes several arrays (`concatenate_*`, `stack_*`), one masked array in any position makes the output masked and keeps every mask. |
| P12 | Array in, array out     | A function that receives an `ndarray` returns an `ndarray`. Where NumPy returns a scalar, the result is a 0-d array, so the output of any function is a valid input of any other.                                                                                                                                             |
| P13 | Keyword-only options    | The positional arguments are the data and the arguments that define the operation. Options (`keepdims`, `ddof`, `kind`, `step`, `rng`, `dtype`) are keyword-only.                                                                                                                                                             |
| P14 | Typed                   | A map function has the type `T -> T` with `T` bound to the nested array type: dict in, dict out, array in, array out.                                                                                                                                                                                                         |
| P15 | Layered                 | `constants` <- `validation` <- `utils` <- `array` <- `nested`. No import in the other direction. `numpy` and `coola` are the only required dependencies.                                                                                                                                                                      |
| P16 | Documented by example   | Every public function has a docstring with a runnable `pycon` example. Each rule is stated once, on this page and not repeated in a function docstring: a docstring states what is specific to the function (its edge cases, its `Raises` and its example).                                                                   |
| P17 | Defined on edge cases   | Every function has a documented result on empty and degenerate input. "NumPy's behaviour" is an answer, "undefined" is not. A 0-d array and a non-array argument of an `array` function are outside the contract: the result is NumPy's, which differs from one function to another.                                          |
| P18 | Documented claims exist | Every file, workflow or module that the README and the docs point to exists, the description of the package matches `pyproject.toml`, and every module the docs name as part of the package is reachable as `batcharray.<module>` after `import batcharray`.                                                                  |
| P19 | Unambiguous indices     | An index is an integer array and a mask is a boolean array, and no function reads one as the other. An index or a mask that is not an `ndarray` is a `TypeError`; one with the wrong rank, or a boolean index, is a `ValueError`. Index values (bounds, negative, float dtype) are NumPy's rules.                             |

## Gates

| Property | Gate                                                                                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1       | `tests/unit/test_signatures.py` (no `axis` parameter)                                                                                                                       |
| P2       | `tests/unit/test_thin_wrappers.py` (the body of a function outside the closed list is one NumPy call, and the table below is the closed list)                               |
| P3       | `tests/unit/test_signatures.py` (same names, options and defaults in `array` and `nested`)                                                                                  |
| P4       | `tests/unit/test_nested_contract.py` (no `recursive_apply`, no `isinstance` on a container in `nested`)                                                                     |
| P5       | `tests/unit/test_nested_contract.py` and `tests/unit/utils/test_structure.py` (the container types are kept)                                                                |
| P6       | `tests/unit/test_nested_contract.py` (every `nested` function raises a `TypeError` for a non-array leaf)                                                                    |
| P7       | `tests/unit/test_purity.py` (inputs are not modified, no aliasing except the views)                                                                                         |
| P8       | `tests/unit/test_error_contract.py` (one row per line of the error table) and `tests/unit/test_no_inline_raise_in_array.py` (no `raise` in `array`)                         |
| P19      | `tests/unit/test_error_contract.py` (a boolean index, a 2-D mask, a list index and a list mask)                                                                             |
| P9       | `tests/unit/test_rng.py`                                                                                                                                                    |
| P10      | the dtype tests of `concatenate_*` and of the reductions in `tests/unit/array`                                                                                              |
| P11      | `tests/unit/test_masked_arrays.py` (every function on a masked array equals the `numpy.ma` result, and `concatenate_*` and `stack_*` with a masked array in every position) |
| P12      | `tests/unit/test_array_in_array_out.py`                                                                                                                                     |
| P13      | `tests/unit/test_signatures.py` (options are keyword-only, no `**kwargs`)                                                                                                   |
| P14      | `ty` in CI and `tests/unit/test_nested_contract.py` (the annotations are `T -> T`)                                                                                          |
| P15      | the `layers` contract of `import-linter` (`ci-cyclic-imports`)                                                                                                              |
| P16      | `ci-doctest` (the examples run) and `tests/unit/test_docstring_does_not_repeat_layout.py` (no layout note in a docstring)                                                   |
| P17      | `tests/unit/test_edge_case.py` (one test per row of the edge-case table)                                                                                                    |
| P18      | `tests/unit/test_docs_references.py` and `tests/unit/test_init.py` (the five modules in a fresh interpreter)                                                                |

**Not guaranteed:**

- Shared references: `sort_along_batch({"a": x, "b": x})` returns two arrays.
- Memory sharing between input and output, except for the views listed below.
- The order of the leaves inside a set.
- Dim-last or sequence-first layouts. Transpose first.
- 0-d arrays. A 0-d array has no batch axis, so it is outside the contract and no test pins the
  result. The result is NumPy's and differs from one function to another:
  `sum_along_batch(np.array(1.0))` returns the 0-d array, `cumsum_along_batch` and
  `index_select_along_batch` return an array of shape `(1,)`, `sort_along_batch`, `flip_along_batch`
  and `sum_along_seq` raise `AxisError`, and `select_along_batch`, `slice_along_batch`,
  `masked_select_along_batch` and `chunk_along_batch` raise `IndexError`.
- A non-array argument at the `array` level (a list, a scalar). The annotation is `numpy.ndarray`
  and `ty` rejects the call. No check is made and no test pins the result, which is NumPy's and
  differs from one function to another: `sum_along_batch([[1, 2], [3, 4]])` raises `AttributeError`,
  `sort_along_batch` converts the list to an array, and `slice_along_batch` returns a list. Use
  `nested`, which raises a `TypeError` for a leaf that is not an array.
- Dataclasses and sets as containers.
- That an index or a `permutation` argument is a valid permutation (only its length is checked).
- A generator or a lazy iterable as `items` of `concatenate_*` and `stack_*`. It takes a `Sequence`.

## Containers

The supported structures are `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). Sets and dataclasses are outside the contract.

For `concatenate_*` and `stack_*`, the output containers are the ones of the first item and the
other items must have the same keys and lengths. A `list` and a `tuple` can be combined (the output
has the type of the first item), as can a `dict` and an `OrderedDict`. A mapping combined with a
sequence raises a `ValueError`.

For `chunk_*` and `split_*`, all the leaves must have the same size along the split axis, otherwise
the pieces of different leaves are not aligned and a `ValueError` is raised.

## Exceptions to the thin wrappers

A function outside this list is one NumPy call with the axis fixed, and returns a new array or the
view NumPy returns.

| Function                                            | Why it is not one NumPy call                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `*_along_batch` reductions and `select_along_batch` | the NumPy call is wrapped in `np.asanyarray`, because NumPy returns a scalar on a 1-D input (P12)                                                                       |
| `split_*`                                           | `np.split` with indices computed from a size or sections (`np.split(x, 2)` means 2 sections, not size 2)                                                                |
| `concatenate_*`, `stack_*`                          | `get_namespace(*arrays)` picks `np.ma` when any item is masked (P11); `concatenate_*` with `dtype` on a masked array uses `astype` (`np.ma.concatenate` has no `dtype`) |
| `index_select_*`                                    | `np.take` and the validation of the index (NumPy accepts a boolean and any rank)                                                                                        |
| `masked_select_*`                                   | boolean indexing and the validation of the mask (NumPy accepts any rank, and an integer mask is read as indices)                                                        |
| `gather_along_seq`                                  | `np.take_along_axis` with the index reshaped over the feature axes, and a check of the batch size (NumPy broadcasts)                                                    |
| `tile_along_seq`                                    | `np.tile` with the repetitions built from the rank, and a check of the rank (NumPy promotes a 1-D array)                                                                |
| `permute_*`                                         | `np.take` and the validation of the length (NumPy would accept a shorter index)                                                                                         |
| `shuffle_*`                                         | draws the permutation with `np.random.default_rng(rng)`, then `permute_*`                                                                                               |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `flip_*`. Everything else
returns a new array. `sort_*`, `cumsum_*`, `repeat_*` and `tile_*` with a count of 1 return a new
array.

## Masked arrays

A reduction, a cumulative function, `argmax`, `argmin`, `any` and `all` use the method of the array,
which `MaskedArray` overrides. `median`, `sort`, `argsort`, `concatenate` and `stack` use `numpy.ma`
for a masked array and `numpy` otherwise. `concatenate` and `stack` take several arrays, so
`get_namespace(*arrays)` chooses `numpy.ma` when any of them is masked, whatever its position:
`numpy.concatenate` drops the mask of a masked array that is not first, and so does `numpy.stack`. A
list of plain arrays stays plain (`numpy.ma.stack` would turn it into a masked array). A subclass of
`MaskedArray` is dispatched as a masked array, a subclass of `ndarray` is a plain array, and an
array type with other semantics is not supported.

A function that NumPy does not support on a masked array is not added to the package until it has a
masked implementation.

## Errors

`batcharray` validates only what NumPy cannot see, or what NumPy accepts but means something else.
It never repeats a check NumPy makes.

| Situation                                                                                                                                                                | Exception                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| A leaf is not an `ndarray`                                                                                                                                               | `TypeError`                         |
| `concatenate` or `stack` receives no item, or structures with a different layout (keys, lengths)                                                                         | `ValueError`                        |
| `concatenate` or `stack` combines a mapping with a sequence                                                                                                              | `ValueError`                        |
| Leaves disagree on the size of the axis of `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len`                                                                    | `ValueError`                        |
| A split size that is not positive, sections that are negative or do not sum to the size                                                                                  | `ValueError`                        |
| An index or a mask is not an `ndarray` (a list, ...)                                                                                                                     | `TypeError`                         |
| An index that has the wrong rank or that is boolean (`index_select_*`, `permute_*`, `gather_along_seq`); a mask that is not 1-D boolean                                  | `ValueError`                        |
| `permute_*` with a permutation of the wrong length; `gather_along_seq` with an index whose batch size differs                                                            | `ValueError`                        |
| `tile_along_seq`, `shuffle_along_seq`, `permute_along_seq`, `gather_along_seq`, `chunk_along_seq`, `split_along_seq`, `seq_len` on an array with fewer than 2 dimensions | `ValueError`                        |
| Another `_along_seq` function on a 1-D array (`slice`, `select`, `sum`, `flip`, `repeat`, `sort`, ...)                                                                   | NumPy's `AxisError` or `IndexError` |
| An out-of-range or a float index, a negative count, a bad `kind`, a bad `ddof`                                                                                           | NumPy's own error                   |

A NumPy `AxisError` is both a `ValueError` and an `IndexError`, so a caller can catch either.

## Indices

An index argument is handled the same way in every function, and is never cast or moved:

| Function               | Index or mask                              | Checked by `batcharray`                         | Validator          |
| ---------------------- | ------------------------------------------ | ----------------------------------------------- | ------------------ |
| `index_select_along_*` | 1-D integer array                          | `ndarray`, not boolean, `ndim == 1`             | `validate_indices` |
| `permute_*`            | 1-D integer array of length = size of axis | `ndarray`, not boolean, `ndim == 1`, length     | `validate_indices` |
| `shuffle_*`            | drawn by `rng.permutation`                 | n/a                                             | n/a                |
| `gather_along_seq`     | 2-D integer array `(batch_size, n)`        | `ndarray`, not boolean, `ndim == 2`, batch size | `validate_indices` |
| `masked_select_*`      | 1-D boolean array                          | `ndarray`, dtype is `bool`, `ndim == 1`         | `validate_mask`    |

An index is an integer array and a mask is a boolean array, and no function reads one as the other.
The dtype of the index (an integer or a float), the bounds and the negative values are NumPy's
rules. The only dtype `batcharray` rejects is the boolean one, because NumPy accepts it and reads it
as the integers 0 and 1. An index or a mask is never converted: a list is a `TypeError`.

## Edge cases

The result on an empty or degenerate input is NumPy's, and is pinned by a test:

| Function                                                     | Case                         | Result                                                           |
| ------------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------- |
| `sum_along_batch`, `prod_along_batch`                        | zero batch size              | the identity (`0`, `1`) for each column                          |
| `mean_along_batch`                                           | zero batch size              | `nan` and NumPy's `RuntimeWarning`                               |
| `max_along_batch`, `min_along_batch`, `argmax_*`, `argmin_*` | zero batch size              | NumPy's `ValueError`                                             |
| `chunk_*`                                                    | more chunks than items       | empty chunks (`np.array_split`)                                  |
| `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len`   | a structure without array    | `ValueError`                                                     |
| `sum_along_batch` and other map functions                    | an empty structure           | the empty structure (`{}`)                                       |
| `permute_*`                                                  | a repeated index             | not checked: only the length is                                  |
| `tile_along_seq`                                             | `reps == 0`                  | a zero-length sequence axis                                      |
| `*_along_batch` reductions and `select_along_batch`          | a 1-D array                  | a 0-d array (P12)                                                |
| any function                                                 | a 0-d array                  | outside the contract: NumPy's result, which differs per function |
| any `array` function                                         | a non-array argument         | outside the contract: NumPy's result, which differs per function |
| a masked reduction                                           | a fully masked row or column | what `numpy.ma` returns (`masked`)                               |

A function that would need the values of an array to validate it does not validate it.

## Differences with `batchtensor`

[`batchtensor`](https://github.com/durandtibo/batchtensor) is the package for PyTorch and has the
same properties, the same helpers and the same names where NumPy and PyTorch use the same name. The
rule: a function has the name of the NumPy or PyTorch function it wraps, and a function that has no
equivalent in the other library exists in one package only. The differences at the `nested` level
are not drift:

| Only in `batchtensor`                                                         | Why                                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `cat_*`                                                                       | PyTorch name; `batcharray` has `concatenate_*` (NumPy name)        |
| `amax_*`, `amin_*`                                                            | NumPy `max` and `min` already reduce an axis (removed in #694)     |
| `roll_*`, `pad_*`, `pad_sequences`                                            | not admitted: `roll` is one NumPy call, `pad` drops the mask (P11) |
| `topk_*`, `logsumexp_*`, `masked_{amax,amin,mean,sum}_along_seq`              | no NumPy function; `numpy.ma` covers the masked reductions (P11)   |
| `clone`, `contiguous`, `detach`, `pin_memory`, `to`, `from_numpy`, `to_numpy` | PyTorch device, dtype and graph; `apply_to_arrays` covers a cast   |

| Only in `batcharray` | Why                                                    |
| -------------------- | ------------------------------------------------------ |
| `masked_select_*`    | boolean indexing with the validation of the mask (P19) |
| `tile_along_seq`     | `np.tile` with the repetitions built from the rank     |
| `concatenate_*`      | NumPy name of `cat_*`                                  |

`gather_along_seq` is in `batcharray` and is planned in `batchtensor`.
