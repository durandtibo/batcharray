# `batcharray`: target state and the changes to reach it

**Scope.** This document defines the properties of `batcharray` (`constants`, `validation`,
`utils`, `array`, `nested`), compares them with `main` (`c0a950e`), and lists the changes that
remain to reach the target state. It changes no code. It replaces the previous version, whose steps
1 to 6 are merged (#703 to #708; Appendix B). It follows the structure of
`review/target_state.md` of [`batchtensor`](https://github.com/durandtibo/batchtensor) (`main` at
`8d133c8`) and of `review/recursive.md` of [`coola`](https://github.com/durandtibo/coola): the
properties first, then the findings, the architecture, the steps and the decisions. The two
packages stay aligned: the same properties, the same helpers, and the same names where NumPy and
PyTorch use the same name (Appendix A). Each finding is tagged **[verified]** (a command or a probe
shows it, Section 6.2) or **[reading]** (found by reading the code, not run).

**The aim** is a small package that is easy to maintain: **one NumPy call per `array` function, one
helper call per `nested` function, one validation module, one statement of each rule, and a gate for
every claim**. Breaking changes are accepted and the future state matters more than the past one.

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks). All 19 gates exist on `main`. **`P6`, `P17` and `P18` change** in this version (the change is in the
statement): `P6` says where the leaf contract starts (the `nested` level), `P17` lists the inputs
outside the contract, and `P18` covers the dependency bounds. Section 3 lists where `main` differs
from the properties.

| #   | Property                | Statement                                                                                                                                                                                                                                                                                                                                                                     | Owner                          | Gate                                                                             |
| --- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| P1  | Fixed layout            | Batch is axis 0 and sequence is axis 1. The axes come only from `BATCH_AXIS` and `SEQ_AXIS`. No function of `array` or `nested` takes an `axis`. `utils.structure` is axis-generic (the layer below). Data laid out as `(seq, batch, ...)` is transposed first.                                                                                                                | `constants.py`                 | `test_signatures`                                                                |
| P2  | Thin wrappers           | An `array` function is one NumPy call with the axis fixed: same semantics, same result, no hidden copy or cast. The **closed** list of exceptions is in Section 4.4. A function not in the list is one NumPy call (`np.ma` for a masked array, P11).                                                                                                                          | `array/*`                      | `test_thin_wrappers`, **`test_closed_list_matches_docs` (new)**                  |
| P3  | Two symmetric levels    | Every `array` function has a `nested` twin with the same name, options and defaults, and every `nested` function has an `array` twin. The exceptions are `batch_size` and `seq_len` (`nested` only: they read a shape). In `nested`, a `mask`, an `indices` or a `permutation` is one array shared by all the leaves.                                                         | Parity test                    | `test_signatures`                                                                |
| P4  | One traversal           | Every `nested` function goes through `apply_to_arrays`, `split_arrays` or `combine_arrays` of `batcharray.utils`. No `nested` function walks a container or calls `recursive_apply`.                                                                                                                                                                                          | `utils/structure.py`           | `test_nested_contract`                                                           |
| P5  | Structure-preserving    | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple` and their subclasses (named tuple, `OrderedDict`, `defaultdict`). For `concatenate_*` and `stack_*` the container types of the output are the ones of the first item.                                                                                      | `apply_to_arrays`              | `test_nested_contract`                                                           |
| P6  | Array leaves only       | At the `nested` level, a leaf that is not a `numpy.ndarray` raises a `TypeError` that names the leaf type. Nothing is skipped silently and no leaf is converted. **The `array` level takes `numpy.ndarray` (the annotation); a non-array argument is outside the contract.** |
| P7  | Pure                    | Inputs are never modified. The output never aliases the input, except where NumPy returns a view (`slice`, `select`, `chunk`, `split`, `flip`). A function adds no shortcut that returns its input.                                                                                                                                                                           | Thin wrappers                  | `test_purity`                                                                    |
| P8  | One error contract      | A bad leaf type is a `TypeError`. An argument that `batcharray` validates is a `ValueError`. Anything else is the NumPy error, unchanged. **`batcharray` validates only what NumPy cannot see or what NumPy accepts but means something else, and every check it makes is a function of `batcharray.validation`.** It never repeats a check NumPy makes. The table is in Section 4.6. | **`validation` (changes)**     | `test_error_contract`, **`test_no_inline_raise_in_array` (new)**                 |
| P9  | Explicit randomness     | Randomness comes only from `rng`, which is passed to `numpy.random.default_rng` (a `Generator`, an `int` seed or `None`). One permutation is shared by all the leaves.                                                                                                                                                                                                        | `shuffle_*`                    | `test_rng`                                                                       |
| P10 | No cast, no conversion  | No function casts or converts what the caller passes, data or index. The `dtype` of `concatenate_*` is an explicit option that NumPy defines.                                                                                                                                                                                                                                 | Thin wrappers                  | dtype tests of `array`                                                           |
| P11 | Masked arrays           | A `numpy.ma.MaskedArray` input gives a `MaskedArray` output with the semantics of `numpy.ma`, for every function except the ones that return indices (`argsort`, `argmax`, `argmin`). **When a function takes several arrays (`concatenate_*`, `stack_*`), one masked array in any position makes the output masked and keeps every mask.** One helper does the dispatch.    | `utils/namespace.py`           | `test_masked_arrays` (**extended: every position**)                              |
| P12 | Array in, array out     | A function that receives an `ndarray` returns an `ndarray`. Where NumPy returns a scalar (`np.float64`, for a reduction or a `select` on a 1-D array), the result is a 0-d array. So the output of any function is a valid input of any other.                                                                                                                                | `array/*`                      | `test_array_in_array_out`                                                        |
| P13 | Keyword-only options    | The positional arguments are the data and the arguments that define the operation. Options (`keepdims`, `ddof`, `kind`, `step`, `rng`, `dtype`) are keyword-only. No `**kwargs` forwarding.                                                                                                                                                                                  | Signatures                     | `test_signatures`                                                                |
| P14 | Typed                   | A map function has the type `T -> T` with `T` bound to `NestedArray`: dict in, dict out, array in, array out. Checked by `ty`.                                                                                                                                                                                                                                                | Annotations                    | `ty` in CI, `test_nested_contract`                                               |
| P15 | Layered                 | `constants` <- `validation` <- `utils` <- `array` <- `nested`. No import in the other direction. `numpy` and `coola` are the only required dependencies and no optional library is imported at module level.                                                                                                                                                                  | `import-linter`                | `ci-cyclic-imports`                                                              |
| P16 | Documented once         | Every public function has a docstring with a runnable `pycon` example. **Each rule is stated once, in `docs/docs/design_principles.md`, and is not repeated in a function docstring. A docstring states what is specific to the function: its edge cases, its `Raises` and its example.**                                                                                     | Docstrings, docs               | `ci-doctest`, **`test_docstring_does_not_repeat_layout` (new)**                  |
| P17 | Defined on edge cases   | Every function has a documented result on empty and degenerate input: a zero batch size, a zero sequence length, a structure without array, a fully masked row. "Undefined" is not an answer; "NumPy's behaviour" is. **A 0-d array and a non-array argument of an `array` function are outside the contract: the result is NumPy's and differs from one function to another.**                    | Docstrings, `design_principles` | `test_edge_case`                                                                 |
| P18 | Documented claims exist | Every file, workflow or module that the README and the docs point to exists, the description of the package (dependencies and their bounds, modules, tooling) matches `pyproject.toml`, every version the dependency CI installs satisfies the bounds of `pyproject.toml`, and every module the docs name as part of the package is reachable as `batcharray.<module>`. |
| P19 | Unambiguous indices     | **An index is an integer array and a mask is a boolean array, and no function reads one as the other.** An index or a mask that is not an `ndarray` is a `TypeError`; one with the wrong rank, or a boolean index, is a `ValueError`. Index values (bounds, negative, float dtype) are NumPy's rules.                                                                           | `validation`                   | **`test_error_contract` (rows added)**                                           |

**Not guaranteed:**

- Shared references. `sort_along_batch({"a": x, "b": x})` returns two arrays.
- Memory sharing between input and output, except for the views of Section 4.4.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`chunk_*`, `split_*`,
  `shuffle_*`, `batch_size`, `seq_len`). `slice_*`, `select_*` and `index_select_*` do not compare
  the sizes of the leaves.
- 0-d arrays.
- Dataclasses and sets as containers. Dataclasses work today through `coola`; no test, doc or type
  promises it.
- That an index or a `permutation` argument is a valid permutation (a repeated index is not
  checked, only the length is).
- A generator or a lazy iterable as `items` of `concatenate_*` and `stack_*`. They take a
  `Sequence`.
- The result on a fully masked row, beyond what `numpy.ma` returns.
- Array-likes. An index, a mask or a leaf is an `ndarray` (P6, P19).


## Where the package stands

The architecture is at its target. `src` is 4 862 lines, `ruff`, `ty` and `lint-imports` are clean,
3 126 tests and the 72 tests of the Markdown code blocks pass, and the whole suite also passes on
NumPy 2.0.2 (the lower bound; Python 3.12) **[verified]**. Coverage is 99 %: the only line not
covered is `raise AttributeError` in `batcharray/__init__.py` (F3). Every `nested` function is one
call to one of three helpers (`shuffle_*` is the one composition), the package has one error
contract and one validation module, and every one of the 19 properties has a gate. The two silent
wrong results of the previous version (masks dropped by `concatenate_*`, a boolean index read as
integers) are fixed. What is left is about **the dependencies and the edges of the contract**:

1. **The dependency claims are false.** The README says `coola>=1.0,<2.0` for `main`,
   `pyproject.toml` says `coola>=2.0.0a4,<3.0`, and the dependency CI matrix tests `coola` 1.0.0,
   1.0.1 and 1.1.10, with which `import batcharray.nested` raises `ImportError` (F1).
2. **The `array` level has no stated rule for a non-array argument** and the behaviour differs from
   one function to another (F2).
3. **One line is not covered** (F3).
4. **The release is blocked** by the alpha of `coola` (F4).

This is a pre-1.0 release (`0.4.0a0`). Breaking changes are allowed when they make the package
simpler or the behaviour more predictable.

## 2. Constraints

1. **Simple to maintain.** A rule is implemented in one place. A function is a thin wrapper. Remove
   code instead of adding a workaround.
2. **Breaking changes are allowed.** No deprecation period, no adapter. The future state matters
   more than the past state. The release notes list the changes.
3. **Consistent with `coola` and `batchtensor`.** Container handling (types, keys, order) is the one
   of `coola.recursive`. A name, an option and a rule that exist in both `batcharray` and
   `batchtensor` are the same in both.
4. **Follow `CLAUDE.md`.** Batch is axis 0, sequence is axis 1, `array` and `nested` stay in sync,
   options are keyword-only, shared helpers are public functions, `numpy` and `coola` are the only
   required dependencies, an argument is validated with `batcharray.validation`.
5. **No slower than today.** A nested call is dominated by the NumPy call and by `coola` (Section
   6.1). No step of Section 7 adds work to a hot path except one `isinstance` per item in
   `concatenate_*` and `stack_*`.
6. **A claim has a gate.** A property written in the docs is enforced by a test or a CI job that
   fails when it breaks. A property without a gate is given one or removed.
7. **No hidden conversion.** A function does not cast or reinterpret what the caller passes. It
   forwards it to NumPy and lets NumPy decide (P8, P10). The one rule that goes beyond NumPy is the
   one that NumPy cannot see: a boolean index is read as integers, so it is rejected (P19).


## 3. What differs today

Surface: 58 names in `array`, 60 in `nested`, 4 in `validation`, 8 in `utils` (Appendix A). The
findings are ordered by their effect on users: F1 breaks the dependency CI and misleads the
readers, F2 is an undefined area, F3 is a gap in a gate, F4 is the release.

| #   | Finding | Tag | Property |
| --- | ------- | --- | -------- |
| F1  | **The dependency claims disagree with `pyproject.toml`, and the dependency CI tests versions the package refuses.** `pyproject.toml`: `coola>=2.0.0a4,<3.0`. The README table, row `main`: `coola >=1.0,<2.0`. `dev/config/package_versions.json`: `coola` `["1.0.0", "1.0.1", "1.1.10"]`, which `ci-test-deps` installs. With `coola` 1.1.10, `import batcharray.nested` raises `ImportError: cannot import name 'recursive_zip_apply'` and the suite stops with 28 collection errors. The generator (`dev/generate_versions.py`) cannot return a 2.0 version because PyPI has only `2.0.0a0` to `a4` (pre-releases are skipped). `test_docs_references` checks the names of the dependencies, not their bounds. NumPy is fine: the matrix lists `2.0.2` to `2.5.5` and the whole suite passes on 2.0.2. | [verified] | P18, P15 |
| F2  | **A non-array argument has no rule at the `array` level.** The leaf contract (P6) is implemented by `nested`. A list given to `array` gives four outcomes: `AttributeError` (`sum_along_batch`, `mean`, `argmax`, `cumsum`, `tile_along_seq`), a silent conversion to an `ndarray` (`sort`, `flip`, `select`, `repeat`, `concatenate`), a `list` (`slice_along_batch` returns a list, `chunk_along_batch` returns a list of lists), and a `TypeError` of NumPy. The annotation is `np.ndarray` and `ty` rejects the call, so this is the same situation as a 0-d array: an input outside the contract that the documentation does not name. | [verified] | P6, P12, P17 |
| F3  | **`__getattr__` of `batcharray/__init__.py` has an uncovered branch** (`raise AttributeError` for an unknown name, line 55). `CLAUDE.md` asks for 100 % of the code that is changed; the file was changed in #706. | [verified] | P18 |
| F4  | **The release is blocked by an alpha dependency.** PyPI has `coola` 1.1.10 as the latest stable and `2.0.0a4` as the latest alpha; `batcharray` is `0.3.0` on PyPI and `0.4.0a0` on `main` (classifier `4 - Beta`). The breaking steps of Appendix B have waited since #690. | [verified] | P15 |
| F5  | **The two sibling packages differ in names that look like drift.** `batchtensor` has `cat_*`, `amax_*`, `amin_*`, `roll_*`, `pad_*`, `topk_*`, `logsumexp_*`, the `masked_*_along_seq` reductions, `clone`, `to`, `from_numpy`; `batcharray` has `concatenate_*`, `masked_select_*`, `tile_along_seq`. Each difference has a reason (the name of the NumPy or torch function, or a function that exists in one library only), but the reason is written nowhere, so the next review rediscovers it. Appendix A is the table. | [reading] | P3 |

**Checked and found correct** (no action): `ruff`, `ty`, `lint-imports` and 3 126 tests pass; the
suite passes on NumPy 2.0.2; the masked-array rule holds for 20 functions probed on a masked input
(`gather_along_seq`, `tile_along_seq`, `repeat`, `chunk`, `split`, `permute`, `shuffle`,
`index_select`, `masked_select`, `flip`, `cumsum`, `cumprod`, `sort`, `median`, `sum`, `std`, `any`,
`select`, `slice`, `max`: every one returns a `MaskedArray` with the expected masked cells); a
non-array leaf (`str`, `np.float64`, `int`) at the `nested` level raises a `TypeError` that names the
type; `batch_size` on leaves of different ranks works and `seq_len` on a 1-D leaf raises a
`ValueError`; a mean, a median and a std of integers return `float64`; a sum of booleans returns
`int64`; `stack_*` of arrays with different shapes raises NumPy's `ValueError`; `chunk_along_batch`
and `split_along_batch` of 200 leaves cost 0.9 ms (4 pieces) and 4.3 ms (32 pieces), linear in the
size of the output; `array.sum_along_batch` costs the same as `x.sum(axis=0)` (1.1 us) and a nested
call on two leaves costs 4.8 us.

**Behaviours that must not regress:** the container types of the input are kept; one permutation is
shared by all the leaves of `shuffle_*`; masked arrays keep their mask in every function that keeps
it today, in every position of `concatenate_*` and `stack_*`; a cycle raises `ValueError`; a
non-array leaf raises a `TypeError` that names the type; `chunk_*` and `split_*` raise on leaves of
different sizes; a boolean index raises `ValueError`; the dtype of every function is NumPy's; the
doctests run.

## 4. Architecture

The architecture stays. Five layers, one traversal module, one validation module:

```text
constants          BATCH_AXIS, SEQ_AXIS
   ^
validation         validate_min_ndim, validate_split_sections, validate_indices, validate_mask
   ^
utils              types      NestedArray, T, SortKind
                   namespace  get_namespace(*arrays)
                   structure  apply_to_arrays, split_arrays, combine_arrays, get_size
   ^
array              one NumPy call with the axis fixed (+ the closed list of Section 4.4)
   ^
nested             one helper call per function: partial(ba.f, **options)
```

A module imports only from the layers above it in this list. `validation` imports nothing of the
package, and every check that `array` makes is a function of `validation` (P8). `import-linter`
enforces the order with one `layers` contract, which stays.

### 4.1 A `nested` function is logic-free

```python
def sum_along_batch(data: T, *, keepdims: bool = False) -> T:
    return apply_to_arrays(data, partial(ba.sum_along_batch, keepdims=keepdims))
```

Three shapes exist, and only three. A new operation uses one of them:

| Shape  | Helper                            | Used by                                                                                   |
| ------ | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 1 -> 1 | `apply_to_arrays(data, fn)`       | reductions, indexing, slicing (`select`, `slice`), permutation, sorting, `tile`, `repeat` |
| 1 -> N | `split_arrays(data, fn, *, axis)` | `chunk_*`, `split_*`                                                                      |
| N -> 1 | `combine_arrays(items, fn)`       | `concatenate_*`, **`stack_*` (new)**                                                      |

A function that does not fit one of these shapes is not a `nested` wrapper. It is a helper of
`utils.structure`, like `get_size`. `shuffle_*` is the one composition: `get_size`,
`rng.permutation` and `permute_*`, so the permutation is shared by all the leaves. An argument that
is data of the operation but not a batch (a `mask`, an `indices`, a `permutation`) is one array
shared by all the leaves: a keyword of the `partial`, not a leaf.

`apply_to_arrays(data, fn)` is also the tool for what the package does not wrap: a pointwise
function (`apply_to_arrays(data, np.exp)`), a conversion or a cast. The structure of the output is
the one of the input; the leaves are what `fn` returns.

### 4.2 The container rule

The supported structure is `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). This is exactly `NestedArray`. For `combine_arrays` the output
containers are the ones of the first item and the other items must have the same keys and lengths;
a mapping combined with a sequence raises a `ValueError`; `[list, tuple]` gives a list and
`[dict, OrderedDict]` gives a dict. Sets and dataclasses are outside the contract.

### 4.3 One statement of each rule

- The layout (P1), the container rule, the cross-leaf rule, the empty-structure rule, the masked
  rule (Section 4.5), the error table (Section 4.6) and the index rule (Section 4.7) are in
  `docs/docs/design_principles.md` and nowhere else. The module docstrings of `array` and `nested`
  link to it.
- **The 94 notes `This function assumes the batch axis is the first axis` are removed.** A
  function docstring states only what is specific to the function: its edge cases (P17), its
  `Raises` and its example. A function whose first parameter is not a batch (a `mask` or an
  `indices`) says what that parameter is in `Args`.
- `docs/docs/architecture.md` describes the layers, the three helpers and how to add an operation.
  It does not repeat the properties.

### 4.4 The closed list of exceptions to P2 and P7

A function outside this list is one NumPy call with the axis fixed, and returns a new array or the
view NumPy returns. The table is in `design_principles.md`; **`test_closed_list_matches_docs` makes
the table and `NOT_ONE_NUMPY_CALL` one list**. The list after the change:

| Function                                            | Why it is not one NumPy call                                                                                                      |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `*_along_batch` reductions and `select_along_batch` | the NumPy call wrapped in `np.asanyarray`, because NumPy returns a scalar on a 1-D input (P12)                                    |
| `split_*`                                           | `np.split` with indices computed from a size or sections (`np.split(x, 2)` means 2 sections, not size 2)                          |
| `concatenate_*`, **`stack_*` (new)**                | `get_namespace(*arrays)` picks `np.ma` when any item is masked (P11); `concatenate_*` with `dtype` on a masked array uses `astype` |
| `index_select_*`                                    | `np.take` and the validation of the index (NumPy accepts a boolean and any rank)                                                  |
| `masked_select_*`                                   | boolean indexing and the validation of the mask (NumPy accepts any rank)                                                          |
| `gather_along_seq`                                  | `np.take_along_axis` with the index reshaped over the feature axes, and the validation of the batch size (NumPy broadcasts)       |
| `tile_along_seq`                                    | `np.tile` with the repetitions built from the rank, and a check of the rank (NumPy promotes a 1-D array)                          |
| `permute_*`                                         | `np.take` and the validation of the length (NumPy would accept a shorter index)                                                   |
| `shuffle_*`                                         | draws the permutation with `np.random.default_rng(rng)`, then `permute_*`                                                         |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `flip_*`. Everything else
returns a new array.

### 4.5 Masked arrays (P11)

The dispatch is one helper of `utils`, and it takes the arrays of the call:

```python
def get_namespace(*arrays: np.ndarray) -> ModuleType:
    return np.ma if any(isinstance(a, np.ma.MaskedArray) for a in arrays) else np
```

A function with one array calls `get_namespace(array)`, as it does today. `concatenate_*` and
`stack_*` call `get_namespace(*arrays)`: **one masked item anywhere gives a masked output that keeps
every mask**, and a list of plain arrays stays plain (`np.ma.stack` would turn it into a
`MaskedArray`). `_concatenate` loses its own `isinstance` on `arrays[0]` and the rule exists
once. A subclass of `MaskedArray` is dispatched as a masked array, a subclass of `ndarray` is a
plain array, and an array type with other semantics is not supported.

A reduction, a cumulative function, `argmax`, `argmin`, `any` and `all` call the method of the
array. `median`, `sort`, `argsort`, `concatenate` and `stack` call `get_namespace(...).f(...)`
because they are functions of the module. `take`, `take_along_axis`, `flip`, `tile`, `repeat` and
`roll` keep the mask in NumPy itself (checked).

A function that NumPy does not support on a masked array is not added to the package until it has a
masked implementation. `np.pad` returns a plain `ndarray` for a masked array (checked), which is
one reason why `pad_*` is not added.

### 4.6 The error contract (P8, P19)

`batcharray` validates only what NumPy cannot see, or what NumPy accepts but means something else.
Every check is a function of `batcharray.validation`:

| Validator                                          | Used by                                                                                                    | Checks                                                                                                 |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `validate_min_ndim(array, min_ndim=2)`             | `tile_along_seq`, `shuffle_along_seq`, `permute_along_seq`, `gather_along_seq`                             | the array has a sequence axis (`ValueError`)                                                           |
| `validate_split_sections(size, split_size_or_sections)` | `split_*`                                                                                             | a positive integer that is not a `bool`, or non-negative sections that sum to the size (`ValueError`)  |
| `validate_indices(indices, *, ndim, length=None)` (new) | `index_select_*` (`ndim=1`), `permute_*` (`ndim=1`, `length=size`), `gather_along_seq` (`ndim=2`, `length=batch_size`) | an `ndarray` (`TypeError`), not a boolean dtype, the rank, the length of axis 0 (`ValueError`) |
| `validate_mask(mask)` (new)                        | `masked_select_*`                                                                                          | an `ndarray` (`TypeError`), a boolean dtype, 1-D (`ValueError`)                                         |

The table of situations, in `design_principles.md`:

| Situation                                                                                                                              | Exception                           |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| A leaf is not an `ndarray`                                                                                                             | `TypeError`                         |
| An index or a mask is not an `ndarray`                                                                                                 | `TypeError`                         |
| `concatenate` or `stack` receives no item, or structures with a different layout (keys, lengths)                                       | `ValueError`                        |
| `concatenate` or `stack` combines a mapping with a sequence                                                                            | `ValueError`                        |
| Leaves disagree on the size of the axis of `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len`                                  | `ValueError`                        |
| A split size that is not positive, sections that are negative or do not sum to the size                                                | `ValueError`                        |
| An index that has the wrong rank or that is boolean (`index_select_*`, `permute_*`, `gather_along_seq`); a mask that is not 1-D boolean | `ValueError`                        |
| `permute_*` with a permutation of the wrong length; `gather_along_seq` with an index whose batch size differs                          | `ValueError`                        |
| `tile_along_seq`, `shuffle_along_seq`, `permute_along_seq`, `gather_along_seq`, `seq_len` on an array with fewer than 2 dimensions     | `ValueError`                        |
| Another `_along_seq` function on a 1-D array (`slice`, `select`, `sum`, `flip`, `repeat`, `sort`, ...)                                 | NumPy's `AxisError` or `IndexError` |
| An out-of-range or a float index, a negative count, a bad `kind`, a bad `ddof`                                                         | NumPy's own error                   |

A NumPy `AxisError` is both a `ValueError` and an `IndexError`, so a caller can catch either. The
new rows are the ones of F2 and F3. The validators do not cast or convert their argument (P10).
`validate_same_keys` is removed.

### 4.7 Indices: one rule (P19)

| Function               | Index or mask                              | Checked by `batcharray`                                  | Cast or moved |
| ---------------------- | ------------------------------------------ | -------------------------------------------------------- | ------------- |
| `index_select_along_*` | 1-D integer array                          | `ndarray`, not boolean, `ndim == 1`                      | never         |
| `permute_*`            | 1-D integer array of length = size of axis | `ndarray`, not boolean, `ndim == 1`, length              | never         |
| `shuffle_*`            | drawn by `rng.permutation`                 | n/a                                                      | never         |
| `gather_along_seq`     | 2-D integer array `(batch_size, n)`        | `ndarray`, not boolean, `ndim == 2`, batch size          | never         |
| `masked_select_*`      | 1-D boolean array                          | `ndarray`, dtype is `bool`, `ndim == 1`                  | never         |

The dtype of the index (an integer or a float), the bounds and the negative values are NumPy's
rules. The only dtype `batcharray` rejects is the boolean one, because NumPy accepts it and reads
it as the integers 0 and 1.

### 4.8 Edge cases: document, do not guess (P17)

The result on an empty or degenerate input is NumPy's, and is documented and pinned by
`test_edge_case`. The table of `design_principles.md` stays as it is, and one row changes:

| Function                                                   | Case                         | Result                                                                  |
| ---------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| any function                                               | a 0-d array                  | **outside the contract; NumPy's result, which differs per function** |
| `*_along_batch` reductions and `select_along_batch`        | a 1-D array                  | a 0-d array (P12)                                                       |
| a masked reduction                                         | a fully masked row or column | what `numpy.ma` returns (`masked`)                                      |

All the other rows (zero batch size, `chunk_*` with more chunks than items, a structure without
array, a repeated permutation index, `reps == 0`) are unchanged.

### 4.9 Admission rule for a new operation

An operation enters `array` and `nested` only if all of these hold. It is in `CONTRIBUTING.md` and
`architecture.md`.

1. It needs the fixed axis (`np.f(x, axis=BATCH_AXIS)`), or it composes NumPy calls in a way that
   is easy to get wrong (`gather_along_seq`, `split_*`, **`stack_*`: `np.stack` drops the masks**). A
   pointwise function (`exp`, `clip`) does not: use `apply_to_arrays(data, np.exp)`.
2. It has a `nested` twin of shape 1 -> 1, 1 -> N or N -> 1 (Section 4.1).
3. It comes with the four functions (batch and sequence, `array` and `nested`), or a stated reason
   why only one axis makes sense.
4. It works on a masked array (P11) or the docstring says why it cannot.
5. If it is not one NumPy call, it is added to the list of Section 4.4.
6. It has a runnable example, a documented edge-case result (P17), and a gate for each property it
   touches.

## 5. Public API

No name is added, removed or renamed by the steps of Section 7. The inventory is Appendix A: 58
`array` names, 60 `nested` names, 4 `validation` names and 8 `utils` names. The changes of this
version are to `pyproject`-related files (the README table, `dev/config/package_versions.json`),
to the documentation (`design_principles.md`) and to the tests.

## 6. Dependents and evidence

### 6.0 Dependents

| Package              | Change                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `coola`              | Nothing to request. `recursive_apply` and `recursive_zip_apply` exist in `2.0.0a4`. The stable `2.0` is the blocker of the release.                                      |
| `batchtensor`        | Nothing. It is the model: the `stack_*` names are the ones it has (`stack_along_batch`, `stack_along_seq`), without a `dtype` option. |
| Other local packages | None. No package of `~/workspace/code/2024` imports `batcharray` (`grep` over `pyproject.toml` and `src`). Public users are unknown; `migration.md` lists each removal.        |

### 6.1 Cost

Probes on this machine (Python 3.14.10, NumPy 2.5.5, `coola` 2.0.0a4), 200 keys x 5 arrays of shape
`(4, 3)` = 1 000 leaves:

| Call                                                      | Time per call |
| --------------------------------------------------------- | ------------- |
| `nested.sum_along_batch(data)`                            | 1.61 ms       |
| hand-written comprehension with `v.sum(axis=0)`           | 0.64 ms       |
| `nested.chunk_along_batch(data, 2)`                       | 3.96 ms       |
| `array.sum_along_batch(x)`                                | 0.71 us       |
| `x.sum(axis=0)`                                           | 0.59 us       |
| `nested.chunk_along_batch` of 200 leaves of 64 rows, in 2 / 8 / 64 chunks | 0.72 / 1.48 / 8.61 ms, against 0.60 / 1.18 / 7.01 ms by hand |

The overhead of `nested` is about 1 us per leaf and is the cost of `coola.recursive`; `array` adds
0.12 us to the method. The cost of `split_arrays` grows with the number of pieces
(`pieces + 1` traversals) but stays 1.2 times a hand-written loop up to 64 pieces. The steps of
Section 7 add one `isinstance` per item to `concatenate_*` and `stack_*` and nothing to the others,
so constraint 5 holds. The numbers were taken once, on a laptop; they show an order of magnitude,
not a benchmark.

### 6.2 Probes of behaviour (this review, `main` at `c0a950e`, NumPy 2.5.5 and 2.0.2)

Scripts are kept outside the repository. Results:

| Probe | Result |
| ----- | ------ |
| Full suite on NumPy 2.0.2 / Python 3.12 with `coola` 2.0.0a4 | 3 126 passed |
| Full suite with `coola` 1.1.10 (the version the dependency matrix installs) | `ImportError` at import of `batcharray.nested`: 28 collection errors (F1) |
| `ba.sum_along_batch([[1, 2], [3, 4]])` | `AttributeError: 'list' object has no attribute 'sum'` (F2) |
| `ba.sort_along_batch(list)`, `flip`, `select`, `repeat`, `concatenate` | an `ndarray` (a conversion; F2) |
| `ba.slice_along_batch(list, 0, 1)` | a `list` (F2) |
| `bn.sum_along_batch([[1, 2], [3, 4]])` | `TypeError: Expected an array, but got an object of type int` (P6 holds at `nested`) |
| 20 functions on a masked array | `MaskedArray` with the masked cells of the `numpy.ma` semantics |
| `pytest --cov=batcharray` | 99 %; only `__init__.py:55` is missing (F3) |
| `nested.chunk_along_batch` on 200 leaves, 4 and 32 pieces | 0.9 ms and 4.3 ms |

## 7. Changes to reach the target state

One pull request per step (`CLAUDE.md`), each releasable and small enough to review alone. Step 1
fixes a false claim and a broken CI job and goes first. Steps 2 and 3 change no result.

| Step | Change | Breaking | Status |
| ---- | ------ | -------- | ------ |
| 1 | **Fix F1.** Set the README row of `main` to the bounds of `pyproject.toml`. Make `dev/generate_versions.py` and `package_versions.json` agree with the bound (a pre-release lower bound is tested with `--pre`, or the matrix lists the lower bound explicitly; the version `1.x` is removed). Extend `test_docs_references`: the `main` row of the README table equals the bounds of `pyproject.toml`, and every version of `package_versions.json` satisfies the specifier of its package (one test per dependency). Regression: the new test fails on `main`. Update the table of P18. | No (CI and docs) | Open (#711) |
| 2 | **Fix F2 and F3.** State the rule: the `array` level takes `numpy.ndarray` (the annotation), a non-array argument is outside the contract (P6 starts at `nested`, D1). Add it to P6, P17 and the "Not guaranteed" list of `design_principles.md`. Add a test for the unknown attribute of `batcharray` (`pytest.raises(AttributeError, match=...)`). No check is added to the 58 functions. | No | Open (#712) |
| 3 | **Write the alignment table (F5).** Add Appendix A of this document to `design_principles.md` (a short section "Differences with `batchtensor`"), and keep it in sync in `batchtensor`. Documentation only. | No | Open (#713) |
| 4 | **Release (F4).** Set the version and the classifier together with a stable `coola` lower bound once `coola` 2.0 is stable. Remove "the API is not stable" from `design_principles.md`, `index.md` and the README only when D2 is answered. | No | Blocked: D2 and `coola` 2.0 stable |

Steps 1 to 3 are independent. Estimated size after the steps: `src` unchanged, `tests` +30 lines,
`docs` +25 lines. This is an estimate, not a measure.

## 8. Tests and gates

- **Kept:** the unit tests of `array`, `nested`, `utils` and `validation`, the integration tests of
  `nested` and `utils`, the doctests, `ruff`, `ty`, `lint-imports`, and the gates of Section 1.
  100% coverage of the code added or changed. A bug fix adds a regression test that fails without
  the fix (`git stash push src`).
- **Removed with their code:** the tests of `validate_same_keys`, of the two private validators of
  `array/indexing.py`, and `test_api_consistency.py`.
- **T1. `test_masked_arrays` (extended).** For `concatenate_*` and `stack_*`: a masked array in the
  first, the middle and the last position, plain arrays only (stays plain), at both levels. The
  result equals the `numpy.ma` function on the same data, mask included. Gate for P11.
- **T2. `test_error_contract` (rows added).** One row per line of the table of Section 4.6: a
  boolean index for each of `index_select_*`, `permute_*`, `gather_along_seq`; a 2-D mask for
  `masked_select_*`; a list index and a list mask (`TypeError` with the type name); one row per
  validator of `validation`. Gate for P8, P19.
- **T3. `test_no_inline_raise_in_array`.** No module of `array` contains `raise ValueError` or
  `raise TypeError` (AST). Every check is a call to `batcharray.validation`. Gate for P8.
- **T4. `test_docstring_does_not_repeat_layout`.** No docstring of `array` or `nested` contains
  `assumes the batch axis` or `assumes the sequence axis`. Gate for P16.
- **T5. `test_init` (extended).** In a fresh interpreter (`subprocess` with `-I`), each of the five
  modules is an attribute of `batcharray` after `import batcharray`, and `batcharray.__all__`
  lists them. Gate for P18.
- **T6. `test_closed_list_matches_docs`.** The function names in the table "Exceptions to the thin
  wrappers" of `design_principles.md` equal `NOT_ONE_NUMPY_CALL`, once the `*_along_batch`
  reductions and the `_*` forms are expanded. Gate for P2.
- **T7. `test_stack_*`.** `stack_along_batch` and `stack_along_seq` on 1-D, 2-D and 3-D items, plain
  and masked, equal to `np.stack` (plain) and `np.ma.stack` (masked) on seeded data; an empty list
  raises `ValueError`; the nested twin keeps the container types of the first item. Gate for P2,
  P3, P5, P11.
- **No test** pins the result of a 0-d array (P17) and none says that sets or dataclasses are
  supported.


**Added by this version.** `test_docs_references`: the README row of `main` and the versions of
`dev/config/package_versions.json` agree with `pyproject.toml` (step 1); a test of the unknown
attribute of `batcharray` (step 2). A regression test fails without its fix
(`git stash push src` for code, a revert of the file for the docs).

## 9. Decisions for the maintainer

| #   | Decision | Recommendation |
| --- | -------- | -------------- |
| D1  | A non-array argument at the `array` level (F2): validate (`TypeError`), convert (`np.asarray`), or document? | **Document.** A check in 58 functions repeats `ty` and the annotation, a conversion is a cast (P10), and `nested` already raises the `TypeError` of P6 for a leaf. The result stays NumPy's and no test pins it, as for the 0-d array (D4 of the previous version). Alternative: `validate_array` in every `array` function (58 call sites, about 60 lines). |
| D2  | When is the API stable, and when is the release made? | After `coola` 2.0 is stable and one release with no change. Alternative: release `0.4.0` now with `coola>=2.0.0a4`; the cost is that users of `0.4.0` follow the alpha of `coola`, and `pip` needs `--pre`. |
| D3  | The dependency matrix (F1): test the lower bound of a pre-release, or wait for `coola` 2.0? | **Test the lower bound explicitly** (`2.0.0a4`) and the latest alpha, so the CI tests what the package declares. A matrix of versions the package refuses tests nothing. |
| D4  | Add `roll_*`, `pad_*`, `topk_*`, `logsumexp_*`, `clone`? | **No**, for the reasons of the admission rule (Section 4.9): `roll` is one NumPy call without a trap, `pad` drops the mask (P11), `topk` and `logsumexp` have no NumPy function, `clone` is `x.copy()`. Revisit on a request with a use case. |
| D5  | Keep `concatenate_*` (NumPy name) while `batchtensor` has `cat_*` (torch name)? | **Yes.** The name of the wrapped function wins (P2); the alignment table (Appendix A) records it. |

Decisions taken in the previous target states and kept: `stack_*` has no `dtype`; `validate_same_keys`
is removed; a non-array index or mask is a `TypeError`; mixed masked and plain items are promoted;
dataclasses and sets are outside the contract; the pointwise functions and `to_list` are not
wrapped (`apply_to_arrays(data, np.exp)`); `computation` is removed; `index_select_*` is kept and
`take_*` removed; `amax_*` and `amin_*` are removed; a reduction on a 1-D array returns a 0-d array;
`concatenate_*` follows the first-item container rule; `chunk_*` and `split_*` keep the names of
`batchtensor`; 0-d arrays are outside the contract; fully masked rows follow `numpy.ma`;
the `nested` twins are written, not generated; the module names `comparison.py`, `math.py` and
`joining.py` are kept (Appendix B).

## 10. Not covered

- New operations beyond `stack_*`. The admission rule (Section 4.9) is the process.
- Array libraries other than NumPy (CuPy, JAX, `torch`): `batchtensor` is the package for PyTorch,
  and P11 supports `numpy.ma` only.
- Structured arrays, `object` arrays, sparse arrays and `np.matrix` (`nested.sum_along_batch` of an
  `np.matrix` raises `TypeError: matrix.sum() got an unexpected keyword argument 'keepdims'`).
- Ragged data (lists of arrays of different lengths) and `pad_sequences`.
- A configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`concatenate_*` and `stack_*` are the building
  blocks).
- Python and NumPy version policy beyond `numpy>=2.0,<3.0` and Python 3.11 and above. The probes
  ran on NumPy 2.5.5 only (Section 6.2); the masked behaviour of `np.concatenate` and `np.stack` on
  NumPy 2.0 is not checked here.
- Performance work. Section 6.1 shows no need.
- A `pyright` run of the type alias. Only `ty` runs in CI.
- The workflows, the dependency bots and the documentation theme.

## Appendix A. API inventory and alignment with `batchtensor`

**`array` (58 names).** Reductions (`max`, `min`, `mean`, `median`, `sum`, `prod`, `std`, `var`,
`any`, `all`, `argmax`, `argmin`: 24 names), cumulative (`cumsum`, `cumprod`: 4), sorting (`sort`,
`argsort`: 4), indexing (`index_select`, `masked_select`: 4, and `gather_along_seq`: 1), slicing
(`slice`, `select`, `chunk`, `split`: 8), joining (`concatenate`, `stack`, `repeat`: 6, and
`tile_along_seq`: 1) and permutation (`permute`, `shuffle`, `flip`: 6), each `_along_batch` and
`_along_seq`, except `gather_along_seq` and `tile_along_seq`.

**`nested` (60 names).** The same names as `array`, plus `batch_size` and `seq_len`.

**`utils` (8 names).** `NestedArray`, `T`, `SortKind`, `get_namespace`, `apply_to_arrays`,
`split_arrays`, `combine_arrays`, `get_size`. **`validation` (4 names).** `validate_min_ndim`,
`validate_split_sections`, `validate_indices`, `validate_mask`. **`constants`.** `BATCH_AXIS`,
`SEQ_AXIS`.

**Differences with `batchtensor`** (`nested` level; 81 names there, 60 here). The rule: a function
has the name of the NumPy or torch function it wraps, and a function that has no equivalent in the
other library exists in one package only.

| Only in `batchtensor` | Why |
| --------------------- | --- |
| `cat_*` | torch name; `batcharray` has `concatenate_*` (NumPy name) |
| `amax_*`, `amin_*` | NumPy `max`/`min` are already the reductions of an axis (removed in #694) |
| `roll_*`, `pad_*`, `pad_sequences` | not admitted (D4 and D3 of the previous version) |
| `topk_*`, `logsumexp_*`, `masked_{amax,amin,mean,sum}_along_seq` | no NumPy function; `numpy.ma` covers the masked reductions (P11) |
| `clone`, `contiguous`, `detach`, `pin_memory`, `to`, `from_numpy`, `to_numpy` | torch device, dtype and graph; `apply_to_arrays` covers a cast |

| Only in `batcharray` | Why |
| -------------------- | --- |
| `masked_select_*` | boolean indexing with the validation of P19 |
| `tile_along_seq` | `np.tile` with the repetitions built from the rank |
| `concatenate_*` | NumPy name of `cat_*` |

`gather_along_seq` is in `batcharray` and is planned in `batchtensor` (its target state, #793).

**Size on `main`.** `src`: 4 862 lines; `tests`: see `wc -l`; `docs/docs`: 2 729 lines.

## Appendix B. Record of the previous target states

`review/suggestions.md` (kept, not replaced) is the record of the first review: the sharp edges of
`tile`, `split`, `concatenate`, `shuffle` and `permute`, the keyword-only `keepdims`, the computation
models for `sum`, `prod`, `std` and `var`, the new reductions and the opt-in size validators.

The second version of this document (`main` at `8392e51`) set 18 properties and nine steps, all
merged:

| Step | Change                                                                                      | PR   |
| ---- | ------------------------------------------------------------------------------------------- | ---- |
| 1    | `chunk_*` and `split_*` check the size of every leaf; `concatenate_*` rejects an empty list | #690 |
| 2    | The properties, `design_principles.md`, `architecture.md` and the docs drift                | #691 |
| 3    | The 28 pointwise functions and `to_list` of `nested` removed                                | #692 |
| 4    | The `computation` package removed; `get_namespace` added                                    | #693 |
| 5    | The aliases (`take_*`, `amax_*`, `amin_*`) and the deprecation shim removed                 | #694 |
| 6    | One traversal (`apply_to_arrays`, `split_arrays`, `combine_arrays`), the leaf contract, types | #696 |
| 7    | Array in, array out (0-d arrays) and the 1-D rules                                          | #697 |
| 8    | The index family: `gather_along_seq`, the checks of `index_select_*` and `masked_select_*`  | #698 |
| 9    | `batch_size`, `seq_len`, the layers contract and the gates                                  | #699 |

#700 (NumPy 2.0 as the lower bound) and #701 (format) followed. The next version replaced the open
items of that document. It keeps the architecture and the 18 properties, changes five of them
(P8, P11, P16, P17, P18), adds P19, and replaces step 10 (the release) by step 7 with the finding
F8 (finding numbers of that version). The five findings that reopened steps 1 to 9 were the masked
rule of step 4 (tested for the first item only), the checks of step 8, the claim of step 2 and the
claim of step 7.

The third version (this one, `main` at `c0a950e`) found the steps 1 to 6 of the second one merged:
`concatenate_*` keeps every mask (#703), the validators are in `batcharray.validation` (#704), the
layout notes and the 0-d claim are corrected (#705), the five modules are reachable (#706),
`stack_*` is added (#707), and the overlapping gates are merged (#708). It replaces the open step 7
(the release) by step 4 and adds F1 to F3 and F5.
