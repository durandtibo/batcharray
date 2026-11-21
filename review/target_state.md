# `batcharray`: target state and the changes to reach it

**Scope.** This document defines the properties of `batcharray` (`constants`, `validation`,
`utils`, `array`, `nested`), compares them with `main` (`f1c3654`), and lists the changes that
remain to reach the target state. It changes no code. It replaces the previous version, whose steps
1 to 9 are merged (#690 to #699, plus #700 and #701; Appendix B). It follows the structure of
`review/target_state.md` of [`batchtensor`](https://github.com/durandtibo/batchtensor), the sibling
package for PyTorch tensors, and keeps the two packages aligned: the same properties, the same
names, the same helpers. Each finding is tagged **[verified]** (a command or a probe shows it,
Section 6.2) or **[reading]** (found by reading the code, not run).

**Where the package stands.** The architecture is at its target. `src` is 5 155 lines (7 840 before
the previous steps), `ruff`, `ty` and `lint-imports` are clean, 2 788 tests and the 71 tests of the
Markdown code blocks pass. Every `nested` function is one call to one of three helpers
(`shuffle_*` is the one composition), the package has one error contract, and every one of the
18 properties has a gate. What is left is small, and it is about **trust in the properties**, not
about structure:

1. **Two silent wrong results.** `concatenate_*` drops the masks of the masked arrays that are not
   first (F1). `index_select_*` and `permute_*` read a boolean index as the integers 0 and 1, and
   `masked_select_along_batch` flattens a 2-D mask (F2).
2. **The validation is in three places**, and a validator leaks an `AttributeError` for a list (F3).
   One public validator has no caller (F3).
3. **Two claims of the documentation are false**, and no gate catches them: "each rule is stated
   once" (94 copies of the layout note, F4) and "a 0-d array raises NumPy's error" (F5).
4. **The package entry point and the documentation lag behind the code** (F6).
5. **One operation is missing for the masked-array promise**: `stack_*` (F7). `np.stack` drops
   masks as `np.concatenate` does, so a caller who writes it by hand meets F1 again.
6. **The release is blocked** by an alpha dependency (F8).

This is a pre-1.0 release (`0.4.0a0`). Breaking changes are allowed when they make the package
simpler or the behaviour more predictable.

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks). **`P8`, `P11`, `P16`, `P17` and
`P18` change** (the change is in the statement). **`P19` is new.** The gates exist on `main` unless
they are marked "new" (Section 8). Section 3 lists where `main` differs from the properties.

| #   | Property                | Statement                                                                                                                                                                                                                                                                                                                                                                     | Owner                          | Gate                                                                             |
| --- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| P1  | Fixed layout            | Batch is axis 0 and sequence is axis 1. The axes come only from `BATCH_AXIS` and `SEQ_AXIS`. No function of `array` or `nested` takes an `axis`. `utils.structure` is axis-generic (the layer below). Data laid out as `(seq, batch, ...)` is transposed first.                                                                                                                | `constants.py`                 | `test_signatures`                                                                |
| P2  | Thin wrappers           | An `array` function is one NumPy call with the axis fixed: same semantics, same result, no hidden copy or cast. The **closed** list of exceptions is in Section 4.4. A function not in the list is one NumPy call (`np.ma` for a masked array, P11).                                                                                                                          | `array/*`                      | `test_thin_wrappers`, **`test_closed_list_matches_docs` (new)**                  |
| P3  | Two symmetric levels    | Every `array` function has a `nested` twin with the same name, options and defaults, and every `nested` function has an `array` twin. The exceptions are `batch_size` and `seq_len` (`nested` only: they read a shape). In `nested`, a `mask`, an `indices` or a `permutation` is one array shared by all the leaves.                                                         | Parity test                    | `test_signatures`                                                                |
| P4  | One traversal           | Every `nested` function goes through `apply_to_arrays`, `split_arrays` or `combine_arrays` of `batcharray.utils`. No `nested` function walks a container or calls `recursive_apply`.                                                                                                                                                                                          | `utils/structure.py`           | `test_nested_contract`                                                           |
| P5  | Structure-preserving    | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple` and their subclasses (named tuple, `OrderedDict`, `defaultdict`). For `concatenate_*` and `stack_*` the container types of the output are the ones of the first item.                                                                                      | `apply_to_arrays`              | `test_nested_contract`                                                           |
| P6  | Array leaves only       | A leaf that is not a `numpy.ndarray` raises a `TypeError` that names the leaf type. Nothing is skipped silently and no leaf is converted.                                                                                                                                                                                                                                     | `utils/structure.py`           | `test_nested_contract`                                                           |
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
| P17 | Defined on edge cases   | Every function has a documented result on empty and degenerate input: a zero batch size, a zero sequence length, a structure without array, a fully masked row. "Undefined" is not an answer; "NumPy's behaviour" is. **A 0-d array is outside the contract (it has no batch axis); the result is NumPy's and differs from one function to another (F5).**                    | Docstrings, `design_principles` | `test_edge_case`                                                                 |
| P18 | Documented claims exist | Every file, workflow or module that the README and the docs point to exists, the description of the package (dependencies, modules, tooling) matches `pyproject.toml`, **and every module the docs name as part of the package is reachable as `batcharray.<module>`**.                                                                                                       | Docs, `__init__`               | `test_docs_references`, **`test_init` (extended)**                               |
| P19 | Unambiguous indices     | **An index is an integer array and a mask is a boolean array, and no function reads one as the other.** An index or a mask that is not an `ndarray` is a `TypeError`; one with the wrong rank, or a boolean index, is a `ValueError`. Index values (bounds, negative, float dtype) are NumPy's rules.                                                                           | `validation`                   | **`test_error_contract` (rows added)**                                           |

**Not guaranteed:**

- Shared references. `sort_along_batch({"a": x, "b": x})` returns two arrays.
- Memory sharing between input and output, except for the views of Section 4.4.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`chunk_*`, `split_*`,
  `shuffle_*`, `batch_size`, `seq_len`). `slice_*`, `select_*` and `index_select_*` do not compare
  the sizes of the leaves.
- 0-d arrays (F5, D4).
- Dataclasses and sets as containers. Dataclasses work today through `coola`; no test, doc or type
  promises it.
- That an index or a `permutation` argument is a valid permutation (a repeated index is not
  checked, only the length is).
- A generator or a lazy iterable as `items` of `concatenate_*` and `stack_*`. They take a
  `Sequence`.
- The result on a fully masked row, beyond what `numpy.ma` returns.
- Array-likes. An index, a mask or a leaf is an `ndarray` (P6, P19).

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
   fails when it breaks. A property without a gate is given one or removed (F4, F5).
7. **No hidden conversion.** A function does not cast or reinterpret what the caller passes. It
   forwards it to NumPy and lets NumPy decide (P8, P10). The one rule that goes beyond NumPy is the
   one that NumPy cannot see: a boolean index is read as integers, so it is rejected (P19).

## 3. What differs today

Surface: 56 names in `array`, 58 in `nested` (the 56 of `array`, `batch_size` and `seq_len`), 3 in
`validation` and 8 in `utils` (Appendix A). 5 155 lines in `src`, 9 542 in `tests`, 2 399 in
`docs/docs`. The findings are ordered by their effect on a result: F1 and F2 are wrong results
without an error, F3 to F6 are consistency and trust, F7 is the one missing operation, F8 is the
release.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Tag        | Property         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------- |
| F1  | **`concatenate_*` drops the masks of the masked arrays that are not first.** `_concatenate` picks `np.ma` only when `arrays[0]` is masked. `concatenate_along_batch([x, m])` with `m` a `MaskedArray` with 3 masked cells returns a `MaskedArray` with **0** masked cells (`np.concatenate` wraps the result in a `MaskedArray` and loses the mask), and `[m, x]` keeps 3. The nested twin does the same: `nested.concatenate_along_batch([{"a": x}, {"a": m}])["a"]` has 0 masked cells. `test_masked_arrays` only tests a masked array in the first position, so P11 is green and wrong. The docstring of the removed `AutoComputationModel` documented this ("a plain array first silently drops the masks"); the property P11 does not.                                                                                                                                                                                                                                                                | [verified] | P11              |
| F2  | **An index or a mask is read as something else, without an error.** `index_select_along_batch(x, np.array([True, False, True]))` returns rows 1, 0, 1 (a boolean array is read as the integers 0 and 1, with the shape of the index); `permute_along_batch(x, np.array([True, False, True]))` accepts it as a permutation of length 3 and returns rows `[1, 0, 1]`. `masked_select_along_batch(x, np.ones((3, 2), bool))` accepts a 2-D mask of the shape of the array and returns a flat array of shape `(6,)` (a 2-D mask indexes both axes), while `masked_select_along_seq` with a 2-D mask raises `IndexError: too many indices`. The mask check is `dtype == bool` only, and the index check is `ndim == 1` only. `design_principles.md` says "1-D boolean array" for the mask and the checks of the table do not do it.                                                                                                                                                                                                       | [verified] | P8, P19          |
| F3  | **The validation is in three places, and one validator is dead.** `validation` has three functions. `array/indexing.py` has two private validators (`_validate_1d_indices`, `_validate_bool_mask`) and two inline `raise ValueError` in `gather_along_seq`, and `array/permutation.py` has two inline `raise ValueError` (`permute_*`): 6 `raise` sites in `array` that `CLAUDE.md` asks to be in `batcharray.validation`. They call `.ndim` on their argument, so `index_select_along_batch(x, [0, 1])` raises `AttributeError: 'list' object has no attribute 'ndim'`, while `np.take(x, [0, 1], axis=0)` accepts a list. `validation.validate_same_keys` has no caller in `src` (`grep`): `combine_arrays` delegates the key check to `coola`. P8 names `validation` as the owner; it owns the `ndim` check and the split check, not the index, mask, permutation and gather checks.                                                                                                                                                                                                            | [verified] | P8               |
| F4  | **"Each rule is stated once" is false and has no gate.** `design_principles.md` P16 says each rule is stated on that page and not repeated in the docstrings. 94 docstrings in `src` repeat `Note: This function assumes the batch axis is the first axis` (or the sequence one). `batchtensor` plans the gate `test_docstring_does_not_repeat_layout` for the same property. The repeated note is the rule P1, which `array/__init__.py` and `design_principles.md` already state.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | [verified] | P16              |
| F5  | **"A 0-d array raises NumPy's error" is false.** `design_principles.md` says `sum_along_batch(np.array(1.0))` raises NumPy's error. It returns `array(1.)`. On a 0-d array: `sum_along_batch` returns the 0-d array, `cumsum_along_batch` and `index_select_along_batch` return an array of shape `(1,)`, `sort_along_batch`, `flip_along_batch` and `sum_along_seq` raise `AxisError`, and `select_along_batch`, `slice_along_batch`, `masked_select_along_batch` and `chunk_along_batch` raise `IndexError` (the last one `tuple index out of range`). Four outcomes for the same input; NumPy accepts `axis=0` on a 0-d array in some functions.                                                                                                                                                                                                                                                                                                                                                                      | [verified] | P17              |
| F6  | **The entry point and the docs do not list the same modules.** `batcharray/__init__.py` lazily imports `array`, `nested` and `utils` only. In a fresh interpreter `import batcharray; batcharray.validation` and `batcharray.constants` raise `AttributeError` (they work only after another import loaded them). The module docstring and the README list three modules (`array`, `nested`, `utils`), while the docs have a page for `constants` and `validation` too, and `docs/docs/index.md` links `constants` and `utils` but not `validation`. `index.md` has two bullets for masked arrays ("Masked Arrays" and "Masked arrays: the dispatch ... is done by the array itself"). The user guides and the references cover every name (checked: 0 missing).                                                                                                                                                                                                                                                                               | [verified] | P18              |
| F7  | **There is no `stack_*`, and a hand-written one loses the masks.** `batchtensor` has `stack_along_batch` and `stack_along_seq`; `batcharray` has `concatenate_*` only. A per-sample collate (`np.stack(items, axis=0)`) is the most common way to build a batch, and `combine_arrays` already has the shape (N -> 1). `np.stack` has the bug of F1 in every position: `np.stack([m, m])` returns a `MaskedArray` with 0 masked cells, while `np.ma.stack([m, m])` keeps them (6 for 3 masked cells twice). `np.ma.stack` turns a list of plain arrays into a `MaskedArray`, so the dispatch must be "one masked item", not "`np.ma` always".                                                                                                                                                                                                                                                                                                                                                                 | [verified] | P11, admission   |
| F8  | **The release is blocked by an alpha dependency, and the metadata is ahead of PyPI.** `dependencies = ["coola>=2.0.0a4,<3.0", ...]`; PyPI has `coola` 1.1.10 as the latest stable and `2.0.0` only as `a0` to `a4`. The specifier names a pre-release, so `pip` accepts it, but a breaking change in a later `coola` alpha reaches the users of a stable `batcharray`. PyPI has `batcharray` 0.3.0; `main` is `0.4.0a0` with the classifier `4 - Beta`, after nine breaking steps (Appendix B). `migration.md` lists them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | [verified] | P15, P18         |
| F9  | **Some gates overlap or cannot see a drift between the docs and the tests.** `test_api_consistency.py` (59 lines) tests `keepdims` keyword-only, which `test_signatures.py` also tests (P13), and `chunk` against `np.array_split`. The closed list of the exceptions to P2 is written twice: the table of `design_principles.md` and `NOT_ONE_NUMPY_CALL` in `test_thin_wrappers.py`. Nothing checks that they agree, so a function added to one list only passes. The Indices table of `design_principles.md` and the rows of `test_error_contract` have the same problem: the table says "1-D boolean array" for a mask, no test checks the 1-D part, and F2 passed both.                                                                                                                                                                                                                                                                                                                                                                                                         | [reading]  | P2, P8           |
| F10 | **`apply_to_arrays` is typed `T -> T` for a function that changes the leaf type.** The docs advertise `apply_to_arrays(data, np.ndarray.tolist)` and `np.exp`. For `tolist` the output has lists at the leaves, not arrays, and `ty` still sees `T`. The structure is the one of the input (P5), the leaves are not. This is accepted: a precise type needs a higher-kinded type. It is a sentence in the docstring.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | [reading]  | P14              |

**Checked and found correct** (no action): `ruff`, `ty`, `lint-imports` (one `layers` contract) and
the 2 788 tests pass; `chunk_*` and `split_*` raise on leaves of different sizes (`ValueError`,
checked for equal and for different piece counts); a non-array leaf (`int`, `str`, `None`,
`np.float64`) raises a `TypeError` that names the type; a named tuple is kept by `chunk_*`;
`concatenate_*` on two dicts with the same keys in another order keeps the order of the first, and
raises `ValueError` on a key mismatch, an empty list or a shape mismatch; `sort`, `median`,
`cumsum`, `index_select`, `take`, `take_along_axis`, `flip`, `roll`, `tile` and `repeat` keep the
mask of a masked array; every reduction along batch on a 1-D array returns a 0-d `ndarray` (12
functions probed); `shuffle_*` on a zero batch size returns a zero batch size and `rng="a"` raises
the `TypeError` of `numpy.random.default_rng`; `chunk_*` and `split_*` of the `nested` level on a
leaf with fewer dimensions than the axis raise a `ValueError` that names the item; `chunk_*` costs
1.2 times a hand-written loop (Section 6.1).

**Behaviours that must not regress:** the container types of the input are kept; one permutation is
shared by all the leaves of `shuffle_*`; masked arrays keep their mask in every function that keeps
it today; a cycle raises `ValueError`; a non-array leaf raises a `TypeError` that names the type;
`chunk_*` and `split_*` raise on leaves of different sizes; the dtype of every function is NumPy's;
the doctests run.

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
the one of the input; the leaves are what `fn` returns (F10).

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
- **The 94 notes `This function assumes the batch axis is the first axis` are removed (F4).** A
  function docstring states only what is specific to the function: its edge cases (P17), its
  `Raises` and its example. A function whose first parameter is not a batch (a `mask` or an
  `indices`) says what that parameter is in `Args`.
- `docs/docs/architecture.md` describes the layers, the three helpers and how to add an operation.
  It does not repeat the properties.

### 4.4 The closed list of exceptions to P2 and P7

A function outside this list is one NumPy call with the axis fixed, and returns a new array or the
view NumPy returns. The table is in `design_principles.md`; **`test_closed_list_matches_docs` makes
the table and `NOT_ONE_NUMPY_CALL` one list (F9)**. The list after the change:

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
`MaskedArray`, F7). `_concatenate` loses its own `isinstance` on `arrays[0]` and the rule exists
once. A subclass of `MaskedArray` is dispatched as a masked array, a subclass of `ndarray` is a
plain array, and an array type with other semantics is not supported.

A reduction, a cumulative function, `argmax`, `argmin`, `any` and `all` call the method of the
array. `median`, `sort`, `argsort`, `concatenate` and `stack` call `get_namespace(...).f(...)`
because they are functions of the module. `take`, `take_along_axis`, `flip`, `tile`, `repeat` and
`roll` keep the mask in NumPy itself (checked).

A function that NumPy does not support on a masked array is not added to the package until it has a
masked implementation. `np.pad` returns a plain `ndarray` for a masked array (checked), which is
one reason why `pad_*` is not added (D3).

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
`validate_same_keys` is removed (D5).

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
| any function                                               | a 0-d array                  | **outside the contract; NumPy's result, which differs per function (F5)** |
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

## 5. Public API after the change

Compared with `main` (56 `array` names, 58 `nested` names, 3 `validation` names). After: `array` 58
(+2), `nested` 60 (+2), `validation` 4, and the type `NestedArray`. The full inventory is Appendix A.

| Name                                                                         | Change                                                                                                                                                                                                                                                       | Breaking         |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| `array.concatenate_*`, `nested.concatenate_*`                                | One masked item in any position gives a masked output that keeps every mask (F1). `_concatenate` uses `get_namespace(*arrays)`.                                                                                                                              | Yes (silent bug) |
| `utils.get_namespace`                                                        | Takes `*arrays`: `numpy.ma` if any array is masked (F1, F7). A call with one array is unchanged.                                                                                                                                                             | No               |
| `array.index_select_*`, `array.permute_*`, `array.gather_along_seq`          | A boolean index raises `ValueError`; a non-array index raises `TypeError` (F2, F3).                                                                                                                                                                          | Yes (rare)       |
| `array.masked_select_*`                                                      | A mask that is not 1-D raises `ValueError` (F2); a non-array mask raises `TypeError`.                                                                                                                                                                        | Yes (rare)       |
| `validation.validate_indices`, `validation.validate_mask` (new)              | The checks of `array/indexing.py` and `array/permutation.py` move here (F3). The private validators and the 6 `raise` sites of `array` are removed.                                                                                                         | No               |
| `validation.validate_same_keys`                                              | Removed: no caller; `coola` checks the keys of `combine_arrays` (F3, D5).                                                                                                                                                                                    | Yes              |
| `array.stack_along_batch`, `stack_along_seq`, `nested.stack_*` (new, 4 names) | `np.stack` with the axis fixed and the dispatch of P11. `stack_along_batch` creates the batch axis from items without one; `stack_along_seq` creates the sequence axis from items `(batch_size, *)`. No `dtype` option (D1). Shape N -> 1 on `combine_arrays`. | No               |
| `batcharray` (module)                                                        | Lazily imports `constants` and `validation` as well as `array`, `nested` and `utils`, and lists them in `__all__` and in the module docstring (F6).                                                                                                          | No               |
| 94 docstrings                                                                | The layout note is removed (F4).                                                                                                                                                                                                                             | No               |
| `design_principles.md`, `index.md`, `README.md`, `architecture.md`           | 0-d claim, the indices table, the validators table, the closed list (`stack_*`) and the module lists are corrected (F4, F5, F6). The duplicate "Masked arrays" bullet is removed.                                                                             | No               |
| `pyproject.toml`                                                             | Version and classifier are set together at the release; the `coola` lower bound is a stable release (F8).                                                                                                                                                    | No               |

## 6. Dependents and evidence

### 6.0 Dependents

| Package              | Change                                                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `coola`              | Nothing to request. `recursive_apply` and `recursive_zip_apply` exist in `2.0.0a4`. The stable `2.0` is the blocker of the release (F8).                                      |
| `batchtensor`        | Nothing. It is the model: the `stack_*` names are the ones it has (`stack_along_batch`, `stack_along_seq`), without a `dtype` option (D1). |
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

### 6.2 Probes of behaviour (this review, `main` at `f1c3654`, NumPy 2.5.5)

| Probe                                                                                                                               | Result                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `array.concatenate_along_batch([x, m])`, `([m, x])` with 3 masked cells in `m`                                                      | `MaskedArray` with 0 masked cells; 3 (F1)                                                                 |
| `nested.concatenate_along_batch([{"a": x}, {"a": m}])["a"]`                                                                         | 0 masked cells (F1)                                                                                       |
| `np.concatenate([x, m])`, `np.ma.concatenate([x, m])`                                                                               | 0 masked cells; 3 (F1)                                                                                    |
| `np.stack([x, m])`, `np.stack([m, m])`, `np.ma.stack([x, m])`                                                                       | 0; 0; 3 masked cells per `m` (F7)                                                                         |
| `type(np.ma.stack([x, x]))`                                                                                                         | `MaskedArray`, although no item is masked (F7: the dispatch is "one masked item")                         |
| `np.roll(m, 1, axis=0)`, `np.take`, `np.take_along_axis`, `np.flip` on a masked array                                              | `MaskedArray`, mask kept                                                                                  |
| `np.pad(m, ((0, 1), (0, 0)))`                                                                                                       | plain `ndarray`: the mask is lost (D3)                                                                    |
| `array.index_select_along_batch(x, np.array([True, False, True]))`                                                                  | shape `(3, 2)`, rows 1, 0, 1 (F2)                                                                         |
| `array.permute_along_batch(x, np.array([True, False, True]))`                                                                       | accepted, rows 1, 0, 1 (F2)                                                                               |
| `array.masked_select_along_batch(x, np.ones((3, 2), bool))`, `masked_select_along_seq(x, np.ones((4, 3), bool))`                    | shape `(6,)`; `IndexError: too many indices` (F2)                                                         |
| `array.index_select_along_batch(x, [0, 1])`, `np.take(x, [0, 1], axis=0)`                                                           | `AttributeError: 'list' object has no attribute 'ndim'`; works (F3)                                       |
| `grep -rn validate_same_keys src`                                                                                                   | the definition and `validation/__init__.py` only (F3)                                                     |
| `grep -rn "assumes the" src \| wc -l`                                                                                               | 94 (F4)                                                                                                   |
| `array.sum_along_batch(np.array(1.))`                                                                                               | `array(1.)`, no error (F5)                                                                                |
| the 0-d array on `cumsum`, `index_select`, `sort`, `flip`, `sum_along_seq`, `select`, `slice`, `masked_select`, `chunk`             | shape `(1,)`; shape `(1,)`; `AxisError` x3; `IndexError` x4 (F5)                                          |
| fresh interpreter: `import batcharray; batcharray.validation`, `batcharray.constants`                                               | `AttributeError` for both (F6)                                                                            |
| PyPI: `batcharray`; `coola`                                                                                                         | latest 0.3.0; latest stable 1.1.10, `2.0.0a0` to `2.0.0a4` (F8)                                           |
| `nested.sum_along_batch({"a": np.float64(1)})`, `None`, `"abc"`, `{b"x"}`                                                           | `TypeError: Expected an array, but got an object of type float64` (and `NoneType`, `str`, `bytes`)        |
| `nested.chunk_along_batch({"a": arange(5), "b": arange(3)}, 2)`; `split_along_batch({"a": arange(5), "b": arange(6)}, 3)`          | `ValueError: All the arrays must have the same batch size, but got [3, 5]`; `[5, 6]`                      |
| `nested.chunk_along_seq({"a": x, "b": arange(4)}, 2)`                                                                               | `ValueError: Cannot get the sequence size of item 1: it has 1 dimension(s)`                               |
| `nested.shuffle_along_batch({"a": x, "b": x[:2]})`, `({})`                                                                          | `ValueError` (sizes `[2, 4]`); `ValueError: ... at least one array`                                       |
| `nested.concatenate_along_batch([{"a": x}, {"b": x}])`, `[]`                                                                        | `ValueError: The mappings must have the same keys ...`; `ValueError` from NumPy                           |
| `array.shuffle_along_batch(x, rng="a")`                                                                                             | `TypeError` from `numpy.random.default_rng`                                                               |
| `array.chunk_along_batch(x, True)`, `(x, 2.0)`, `(x, 0)`; `split_along_batch(x, True)`                                              | one chunk; two chunks; `ValueError`; `ValueError` from `validate_split_sections` (NumPy's rule, D6)       |
| `array.sort_along_batch(x, kind="foo")`, `std_along_batch(x, ddof=10)`                                                              | NumPy's `ValueError`; `inf` and NumPy's `RuntimeWarning`                                                  |
| `array.tile_along_seq(x, 0)`, `(x, [2, 2])`                                                                                         | shape `(4, 0)`; NumPy's broadcast `ValueError`                                                            |
| `array.gather_along_seq(x, np.array([[-1]] * 4))`, with a float index                                                               | shape `(4, 1)` (negative from the end); NumPy's `IndexError`                                              |

Commands: `ruff check src tests` clean, `ty check` clean, `lint-imports` 1 contract kept,
`pytest tests -q` 2 788 passed, `pytest docs README.md -q` 71 passed. The probes ran on NumPy 2.5.5
only; the behaviour of F1, F2 and F7 on NumPy 2.0, the lowest supported version, is **not**
checked here. The CI matrix runs the tests, so the new gates (Section 8) are the check.

## 7. Changes to reach the target state

One pull request per step, each releasable and small enough to review alone. Step 1 fixes a wrong
result and goes first. Step 2 is the other wrong result. Steps 3 to 6 do not change a result.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Breaking         | Status                           |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------- |
| 1    | **Fix F1.** `get_namespace(*arrays)`; `_concatenate` uses it and loses its `arrays[0]` test. Extend `test_masked_arrays` so that a masked array is tested in the first, the last and the middle position, at both levels. Regression tests (`[x, m]`, `[{"a": x}, {"a": m}]`, three items) that fail without the fix (`git stash push src`). Update P11 and the masked section of `design_principles.md`.                                                                              | Yes (silent bug) | PR #703                          |
| 2    | **Fix F2 and F3: one validation module.** Add `validate_indices` and `validate_mask`, use them in `index_select_*`, `permute_*`, `gather_along_seq` and `masked_select_*`, delete the two private validators and the inline `raise` sites (6 in total), delete `validate_same_keys`. Add `test_no_inline_raise_in_array` and the rows of `test_error_contract` (boolean index, 2-D mask, list index, one row per validator). Regression tests for the three cases of F2 fail without the fix. Update the tables of Sections 4.6 and 4.7 in `design_principles.md`. | Yes (rare)       | PR #704                          |
| 3    | **Fix F4 and F5.** Remove the 94 layout notes from the docstrings; add `test_docstring_does_not_repeat_layout`. Correct the 0-d claim in `design_principles.md` (P17, edge-case table); add the "not guaranteed" line. Add the sentence of F10 to the docstring of `apply_to_arrays`.                                                                                                                                                                                                     | No               | PR #705                          |
| 4    | **Fix F6.** `_SUBPACKAGES` and `__all__` list the five modules; the module docstring, the README features and `index.md` (the `validation` link, the duplicate bullet) follow. Extend `test_init` (every documented module is an attribute of `batcharray` in a fresh interpreter) and `test_docs_references`.                                                                                                                                                                         | No               | PR #706                          |
| 5    | **Add `stack_*` (F7).** `array.stack_along_batch`, `stack_along_seq` and the `nested` twins on `combine_arrays`, with `get_namespace(*arrays)`. Add the functions to the closed list and to the user guides (`migration.md` is unchanged: nothing is removed). Gate: `test_masked_arrays` (every position), `test_signatures`, `test_purity` (a new array), the edge-case rows (empty list raises `ValueError`).                                                                                                    | No               | PR #707                          |
| 6    | **Merge the overlapping gates (F9).** Move the two tests of `test_api_consistency.py` that are not in `test_signatures.py` (`chunk` against `np.array_split`) to `tests/unit/array/test_slicing.py`, delete the file. Add `test_closed_list_matches_docs`: the table "Exceptions to the thin wrappers" of `design_principles.md` equals `NOT_ONE_NUMPY_CALL`.                                                                                                                              | No               | PR #708                          |
| 7    | **Release (F8).** Set the version and the classifier together, with a stable `coola` lower bound. Remove "the API is not stable" from `design_principles.md` and `index.md` only when D7 is answered.                                                                                                                                                                                                                                                                                   | No               | Blocked: D7 and `coola` 2.0 stable |

Steps 1 and 2 are independent and can ship now. Steps 3, 4 and 6 are documentation and tests and
can ship in any order. Step 5 needs step 1 (it uses the helper that step 1 changes). Estimated size
after the steps: `src` about 5 100 lines (the 94 notes are 188 lines, the validators move, `stack_*`
adds about 200 lines with docstrings), `tests` about 9 800. This is an estimate, not a measure.

## 8. Tests and gates

- **Kept:** the unit tests of `array`, `nested`, `utils` and `validation`, the integration tests of
  `nested` and `utils`, the doctests, `ruff`, `ty`, `lint-imports`, and the gates of Section 1.
  100% coverage of the code added or changed. A bug fix adds a regression test that fails without
  the fix (`git stash push src`).
- **Removed with their code:** the tests of `validate_same_keys`, of the two private validators of
  `array/indexing.py`, and `test_api_consistency.py` (step 6).
- **T1. `test_masked_arrays` (extended).** For `concatenate_*` and `stack_*`: a masked array in the
  first, the middle and the last position, plain arrays only (stays plain), at both levels. The
  result equals the `numpy.ma` function on the same data, mask included. Gate for P11, F1, F7.
- **T2. `test_error_contract` (rows added).** One row per line of the table of Section 4.6: a
  boolean index for each of `index_select_*`, `permute_*`, `gather_along_seq`; a 2-D mask for
  `masked_select_*`; a list index and a list mask (`TypeError` with the type name); one row per
  validator of `validation`. Gate for P8, P19, F2, F3.
- **T3. `test_no_inline_raise_in_array`.** No module of `array` contains `raise ValueError` or
  `raise TypeError` (AST). Every check is a call to `batcharray.validation`. Gate for P8, F3.
- **T4. `test_docstring_does_not_repeat_layout`.** No docstring of `array` or `nested` contains
  `assumes the batch axis` or `assumes the sequence axis`. Gate for P16, F4.
- **T5. `test_init` (extended).** In a fresh interpreter (`subprocess` with `-I`), each of the five
  modules is an attribute of `batcharray` after `import batcharray`, and `batcharray.__all__`
  lists them. Gate for P18, F6.
- **T6. `test_closed_list_matches_docs`.** The function names in the table "Exceptions to the thin
  wrappers" of `design_principles.md` equal `NOT_ONE_NUMPY_CALL`, once the `*_along_batch`
  reductions and the `_*` forms are expanded. Gate for P2, F9.
- **T7. `test_stack_*`.** `stack_along_batch` and `stack_along_seq` on 1-D, 2-D and 3-D items, plain
  and masked, equal to `np.stack` (plain) and `np.ma.stack` (masked) on seeded data; an empty list
  raises `ValueError`; the nested twin keeps the container types of the first item. Gate for P2,
  P3, P5, P11.
- **No test** pins the result of a 0-d array (P17, D4) and none says that sets or dataclasses are
  supported.

## 9. Decisions for the maintainer

| #   | Decision                                                                                                       | Recommendation                                                                                                                                                                                                                                                                                                                                                                     |
| --- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Add `stack_along_batch` and `stack_along_seq`? With a `dtype` option?                                          | **Yes, without `dtype`.** Admission rule 1 holds because `np.stack` loses the masks (F7), `batchtensor` has the same names, and the nested twin is one `combine_arrays` call. `np.ma.stack` has no useful `dtype`, and a cast is `astype` (P10); `concatenate_*` keeps its `dtype` only because it exists today.                                                                    |
| D2  | Add `roll_along_*` (`batchtensor` has it)?                                                                     | **No.** It is one NumPy call (`np.roll(x, shift, axis=0)`) that keeps the mask and has no trap, so rule 1 does not hold. Revisit on a request with a use case.                                                                                                                                                                                                                     |
| D3  | Add `pad_along_*`, `pad_sequences`, `lengths_to_mask` (they are in `batchtensor`)?                             | **Not now.** `np.pad` drops the mask (probe), so `pad_*` needs a masked implementation to satisfy P11. `lengths_to_mask` is `np.arange(n) < lengths[:, None]`. `pad_sequences` is the one function with no NumPy equivalent. Revisit with a use case, a masked implementation and a gate; it would be `combine_arrays` with a padding function.                                    |
| D4  | 0-d arrays: validate (`ValueError`) or document?                                                               | **Document.** Four outcomes today (F5), but a check in 28 functions repeats what NumPy does in most of them and the input has no meaning. The claim of the docs is corrected, P17 says "outside the contract", and no test pins an accident of NumPy. Alternative: `validate_min_ndim(array, 1)` in every `_along_batch` function, which adds 28 call sites.                       |
| D5  | Remove `validate_same_keys`?                                                                                   | **Yes.** No caller in `src` or in the docs; `coola` raises the `ValueError` of the key mismatch of `combine_arrays`. A public function with no caller is a promise the package does not need. A caller of the package can import it from `coola`.                                                                                                                                   |
| D6  | Make `chunk_*` strict on its argument (`True`, `2.0`) as `split_*` is?                                         | **No.** `chunk_*` is `np.array_split`, and the number of sections is NumPy's argument (P8). `split_*` validates because it computes indices from the size, which NumPy does not do. The docstring of `chunk_*` says that it follows `np.array_split`.                                                                                                                              |
| D7  | When is the API stable, and when is the release made?                                                          | After `coola` 2.0 is stable and one release with no change. Until then keep "pre-1.0". Alternative: release `0.4.0` now with `coola>=2.0.0a4` as the dependency, as the breaking steps (Appendix B) have waited since #690; the cost is that users of `0.4.0` follow the alpha of `coola`.                                                                                          |
| D8  | A non-array index or mask: `TypeError`, or accept array-likes with `np.asarray`?                               | **`TypeError`.** The annotation is `np.ndarray`, the leaf rule (P6) is `TypeError`, and a conversion is a cast (P10): `np.asarray([True, 0])` changes the dtype that the check reads. The message names the type.                                                                                                                                                                  |
| D9  | Mixed masked and plain items in `concatenate_*` and `stack_*`: promote, or raise?                              | **Promote** (P11): the output is masked and every mask is kept; a plain item has no masked cell. Raising would reject data that has one obvious meaning, and the previous behaviour depended on the order of the items (F1).                                                                                                                                                         |
| D10 | Keep the module names `comparison.py` (sort, argsort), `math.py` (cumulative), `joining.py` (tile, repeat)?    | **Yes.** They are the names of `batchtensor`, so a function has the same module in both packages. A rename costs the docs and the gates for no change of behaviour.                                                                                                                                                                                                              |
| D11 | Generate the `nested` twins instead of writing them?                                                           | **No.** Generation hides the signature from the type checkers and the docs, and a `nested` function is already one line.                                                                                                                                                                                                                                                          |

Decisions taken in the previous target states and kept: dataclasses and sets are outside the
contract; the pointwise functions and `to_list` are not wrapped (`apply_to_arrays(data, np.exp)`);
`computation` is removed; `index_select_*` is kept and `take_*` removed; `amax_*` and `amin_*`
are removed; a reduction on a 1-D array returns a 0-d array; `concatenate_*` follows the first-item
container rule; `chunk_*` and `split_*` keep the names of `batchtensor`; fully masked rows follow
`numpy.ma` (Appendix B).

## 10. Not covered

- New operations beyond `stack_*`. The admission rule (Section 4.9) is the process (D2, D3).
- Array libraries other than NumPy (CuPy, JAX, `torch`): `batchtensor` is the package for PyTorch,
  and P11 supports `numpy.ma` only.
- Structured arrays, `object` arrays, sparse arrays and `np.matrix` (`nested.sum_along_batch` of an
  `np.matrix` raises `TypeError: matrix.sum() got an unexpected keyword argument 'keepdims'`).
- Ragged data (lists of arrays of different lengths) and `pad_sequences` (D3).
- A configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`concatenate_*` and `stack_*` are the building
  blocks).
- Python and NumPy version policy beyond `numpy>=2.0,<3.0` and Python 3.11 and above. The probes
  ran on NumPy 2.5.5 only (Section 6.2); the masked behaviour of `np.concatenate` and `np.stack` on
  NumPy 2.0 is not checked here.
- Performance work. Section 6.1 shows no need.
- A `pyright` run of the type alias. Only `ty` runs in CI.
- The workflows, the dependency bots and the documentation theme.

## Appendix A. API inventory

**`array` (56 names on `main`, 58 after).** Reductions (`max`, `min`, `mean`, `median`, `sum`,
`prod`, `std`, `var`, `any`, `all`, `argmax`, `argmin`: 24 names), cumulative (`cumsum`, `cumprod`: 4),
sorting (`sort`, `argsort`: 4), indexing (`index_select`, `masked_select`: 4, and `gather_along_seq`:
1), slicing (`slice`, `select`, `chunk`, `split`: 8), joining (`concatenate`, `repeat`: 4, and
`tile_along_seq`: 1) and permutation (`permute`, `shuffle`, `flip`: 6), each `_along_batch` and
`_along_seq`, except `gather_along_seq` and `tile_along_seq`. Added: `stack_along_batch` and
`stack_along_seq`, which makes 7 joining names.

**`nested` (58 names on `main`, 60 after).** The same names as `array`, plus `batch_size` and
`seq_len` (the two of `nested.size`). The type `NestedArray` is in `utils`. Added:
`stack_along_batch` and `stack_along_seq`.

**`utils` (8 names).** `types` (`NestedArray`, `T`, `SortKind`), `namespace` (`get_namespace`, which
takes `*arrays` after step 1), `structure` (`apply_to_arrays`, `split_arrays`, `combine_arrays`,
`get_size`).

**`validation` (3 names on `main`, 4 after).** `validate_min_ndim`, `validate_split_sections`, and
the new `validate_indices` and `validate_mask`. Removed: `validate_same_keys`.

**`constants`.** `BATCH_AXIS`, `SEQ_AXIS`.

**Size on `main`.** `src`: 5 155 lines (`nested` 2 440, `array` 2 223, `utils` 301, `validation`
127, the top-level modules 64); `tests`: 9 542 lines; `docs/docs`: 2 399 lines.

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

#700 (NumPy 2.0 as the lower bound) and #701 (format) followed. This version replaces the open
items of that document. It keeps the architecture and the 18 properties, changes five of them
(P8, P11, P16, P17, P18), adds P19, and replaces step 10 (the release) by step 7 with the finding
F8. The results of steps 1 to 9 were checked again in this review (Section 3, "Checked and found
correct"); the five findings that reopen them are F1 (the masked rule of step 4 was tested for the
first item only), F2 and F3 (the checks of step 8), F4 (the claim of step 2) and F5 (the claim of
step 7).
