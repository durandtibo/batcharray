# `batcharray`: target state and the changes to reach it

**Scope.** This document defines the properties of `batcharray` (`constants`, `validation`,
`utils`, `array`, `nested`), compares them with `main` (`8392e51`), and lists the changes to reach
the target state. It changes no code. It follows the structure of `review/target_state.md` of
[`batchtensor`](https://github.com/durandtibo/batchtensor), the sibling package for PyTorch
tensors, and keeps the two packages aligned: the same properties, the same names, the same
helpers. Each finding is tagged **[verified]** (a command or a probe shows it, Section 6.2) or
**[reading]** (found by reading the code, not run).

**Where the package stands.** The quality gates are green on `main`: `ruff`, `ty` and
`lint-imports` are clean and 1 725 tests pass. The `array` level is small and consistent, and masked
arrays work in every function that was probed (Section 6.2). The problems are in the layers around
it:

1. **Too much code for what it does.** The `computation` package (1 569 lines, four classes, a
   global registry) dispatches between `np` and `np.ma`. That is one expression (F9). 28 pointwise
   functions and `to_list` (844 lines) have no `array` twin and no batch or sequence meaning
   (F8). Six names are aliases of other names (F5).
2. **No leaf contract.** A non-array leaf is silently processed, rejected with the message of the
   computation registry, or rejected with a `TypeError` of Python, depending on the function (F1).
   The output of a reduction on a 1-D array is a NumPy scalar that the next `nested` call rejects
   (F2).
3. **Two wrong results without an error.** `chunk_*` and `split_*` return misaligned pieces when
   the leaves have different sizes (F3), and `nested.concatenate_*([])` returns `{}` (F4).
4. **No single traversal, no types.** 90 call sites of `recursive_apply`, one hand-written walk
   (`concatenate_structures`), and a `nested` API typed with `Any` (F10, F13).
5. **No written properties, no gates.** The documentation describes the API but not the rules.
   Its claims drifted (F12) and no test checks the rules (Section 8).

This is a pre-1.0 release (`0.4.0a0`). Breaking changes are allowed when they make the package
simpler or the behaviour more predictable.

## 1. Properties of the package

These are the properties the package guarantees by construction. Each has an owner (the code that
enforces it) and a gate (the test or job that fails when it breaks). **Every gate marked "new" does
not exist today** (Section 8). On `main`, the unit tests check each function one by one, and only
part of P3 is checked across modules (`tests/unit/test_api_consistency.py`); there is no test for
aliasing, signatures or the axis parameter. The properties are the target. Section 3 lists where
`main` differs.

| #   | Property                 | Statement                                                                                                                                                                                                                                                                                                | Owner                                  | Gate                                           |
| --- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------- |
| P1  | Fixed layout             | Batch is axis 0 and sequence is axis 1. The axes come only from `BATCH_AXIS` and `SEQ_AXIS`. No function of `array` or `nested` takes an `axis`. Data laid out as `(seq, batch, ...)` is transposed first.                                                                                               | `constants.py`                         | `test_no_axis_parameter` (new)                 |
| P2  | Thin wrappers            | An `array` function is one NumPy call with the axis fixed: same semantics, same result, no hidden copy or cast. The **closed** list of exceptions is in Section 3.4. A function not in the list is one NumPy call (`np.ma` for a masked array, P11).                                                      | `array/*`                              | `test_thin_wrappers` (new)                     |
| P3  | Two symmetric levels     | Every `array` function has a `nested` twin with the same name, options and defaults, and every `nested` function has an `array` twin. The exceptions are `batch_size` and `seq_len` (`nested` only: they read a shape). In `nested`, a `mask`, an `indices` or a `permutation` is one array shared by all the leaves. | Parity test                            | `test_api_parity` (new, extends today's test)  |
| P4  | One traversal            | Every `nested` function goes through `apply_to_arrays`, `split_arrays` or `combine_arrays` of `batcharray.utils.structure`. No `nested` function walks a container or calls `recursive_apply`.                                                                                                           | `utils/structure.py`                   | `test_nested_module_does_not_walk_containers` (new) |
| P5  | Structure-preserving     | A map function returns the input structure with every leaf replaced. Supported containers: `dict`, `list`, `tuple` and their subclasses (named tuple, `OrderedDict`, `defaultdict`). For `concatenate_*` the container types of the output are the ones of the first item.                              | `apply_to_arrays`                      | `test_containers` (new)                        |
| P6  | Array leaves only        | A leaf that is not a `numpy.ndarray` raises a `TypeError` that names the leaf type. Nothing is skipped silently and no leaf is converted.                                                                                                                                                              | `utils/structure.py`                   | `test_non_array_leaf` (new)                    |
| P7  | Pure                     | Inputs are never modified. The output never aliases the input, except where NumPy returns a view (`slice`, `select`, `chunk`, `split`, `flip`; Section 3.4). A function adds no shortcut that returns its input.                                                                                         | Thin wrappers                          | `test_inputs_are_not_modified`, `test_output_does_not_alias_input` (new) |
| P8  | One error contract       | A bad leaf type is a `TypeError`. An argument that `batcharray` validates is a `ValueError`. Anything else is the NumPy error, unchanged. **`batcharray` validates only what NumPy cannot see or what NumPy accepts but means something else. It never repeats a check NumPy makes.** The table is in Section 3.6. | `validation`, `array`                  | `test_error_contract` (new, one row per line)  |
| P9  | Explicit randomness      | Randomness comes only from `rng`, which is passed to `numpy.random.default_rng` (a `Generator`, an `int` seed or `None`). One permutation is shared by all the leaves.                                                                                                                                   | `shuffle_*`                            | `test_rng` (new)                               |
| P10 | No cast, no conversion   | No function casts or converts what the caller passes, data or index. The `dtype` of `concatenate_*` is an explicit option that NumPy defines.                                                                                                                                                           | Thin wrappers                          | `test_output_dtype_is_the_one_of_numpy` (new)  |
| P11 | Masked arrays            | A `numpy.ma.MaskedArray` input gives a `MaskedArray` output with the semantics of `numpy.ma`, for every function except the ones that return indices (`argsort`, `argmax`, `argmin`). The dispatch is `np.ma` for a masked array and `np` otherwise, in one helper.                                      | `utils/namespace.py`                   | `test_masked_arrays` (new, parametrized over all functions) |
| P12 | Array in, array out      | A function that receives an `ndarray` returns an `ndarray`. Where NumPy returns a scalar (`np.float64`, for a reduction or a `select` on a 1-D array), the result is a 0-d array. So the output of any function is a valid input of any other.                                                          | `array/*`                              | `test_array_in_array_out` (new)                |
| P13 | Keyword-only options     | The positional arguments are the data and the arguments that define the operation. Options (`keepdims`, `ddof`, `kind`, `step`, `rng`, `dtype`) are keyword-only. No `**kwargs` forwarding.                                                                                                              | Signatures                             | `test_signatures` (new)                        |
| P14 | Typed                    | A map function has the type `T -> T` with `T` bound to `NestedArray`: dict in, dict out, array in, array out. Checked by `ty`.                                                                                                                                                                           | Annotations                            | `ty` in CI, `test_typing` (new)                |
| P15 | Layered                  | `constants` <- `validation` <- `utils` <- `array` <- `nested`. No import in the other direction. `numpy` and `coola` are the only required dependencies and no optional library is imported at module level.                                                                                              | `import-linter`                        | `ci-cyclic-imports` (contract replaced, new)   |
| P16 | Documented by example    | Every public function has a docstring with a runnable `pycon` example. Each rule is stated once, in `docs/docs/design_principles.md`, and is not repeated in the function docstrings.                                                                                                                  | Docstrings, docs                       | `ci-doctest`, `test_docstring_does_not_repeat_layout` (new) |
| P17 | Defined on edge cases    | Every function has a documented result on empty and degenerate input: a zero batch size, a zero sequence length, a structure without array, a fully masked row. "Undefined" is not an answer; "NumPy's behaviour" is.                                                                                     | Docstrings (`Note`/`Returns`)          | `test_edge_case` (new)                         |
| P18 | Documented claims exist  | Every file, workflow or module that the README and the docs point to exists, and the description of the package (dependencies, modules, tooling) matches `pyproject.toml`.                                                                                                                              | Docs                                   | `test_docs_references` (new)                   |

**Not guaranteed:**

- Shared references. `sort_along_batch({"a": x, "b": x})` returns two arrays.
- Memory sharing between input and output, except for the views listed in Section 3.4.
- The order of the leaves inside a set (`coola` rule).
- Dim-last or sequence-first layouts. Transpose first.
- Cross-leaf validation, except where the output alignment depends on it (`chunk_*`, `split_*`,
  `shuffle_*`, `batch_size`, `seq_len`).
- 0-d arrays. `sum_along_batch(np.array(1.0))` raises NumPy's error. A 0-d array has no batch axis.
- Dataclasses and sets (D1). Dataclasses work today through `coola`; no test, doc or type promises
  it.
- That an index or a `permutation` argument is a valid permutation (a repeated index is not
  checked, only the length is).
- A generator or a lazy iterable as `items` of `concatenate_*`. It takes a `Sequence`.
- The result on a fully masked row, beyond what `numpy.ma` returns (D9).

## 2. Constraints

1. **Simple to maintain.** A rule is implemented in one place. A function is a thin wrapper. Remove
   code instead of adding a workaround.
2. **Breaking changes are allowed.** No deprecation period, no adapter. The future state matters
   more than the past state.
3. **Consistent with `coola` and `batchtensor`.** Container handling (types, keys, order) is the
   one of `coola.recursive`. A name, an option and a rule that exist in both `batcharray` and
   `batchtensor` are the same in both.
4. **Follow `CLAUDE.md`.** Batch is axis 0, sequence is axis 1, `array` and `nested` stay in sync,
   options are keyword-only, shared helpers are public functions, `numpy` and `coola` are the only
   required dependencies, an argument is validated with `batcharray.validation`.
5. **No slower than today.** A nested call is dominated by the NumPy call (Section 6.1).
6. **A claim has a gate.** A property written in the docs is enforced by a test or a CI job that
   fails when it breaks. A property without a gate is given one or removed.
7. **No hidden conversion.** A function does not cast or reinterpret what the caller passes. It
   forwards it to NumPy and lets NumPy decide (P8, P10).

## 3. What differs today

Surface: 61 names in `array`, 90 in `nested`, 18 in `computation`, 6 in `validation`. 7 840 lines
in `src`, 12 755 in `tests`. The findings are ordered by their effect on a result: F3 and F4 are
wrong results, F1 and F2 break composition, the others are simplifications and consistency.

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Tag        | Property       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------- |
| F1  | **There is no leaf contract.** A non-array leaf gives a different outcome per function. `nested.sqrt({"a": a, "n": 4})` returns `np.float64(2.0)` for `n` (silent). `nested.sum_along_batch({"a": a, "n": 1})` raises `TypeError: Incorrect array type: <class 'int'>. Registered array types: MaskedArray, ndarray` (the message of the computation registry, which does not say what is wrong). `nested.slice_along_batch(..., stop=2)` raises `'int' object is not subscriptable`. `nested.to_list({"n": (1, 2)})` returns the tuple unchanged. The rule depends on the function that is called, not on the data.                                                          | [verified] | P6, P8         |
| F2  | **A reduction on a 1-D array returns a NumPy scalar, and the next call rejects it.** `sum_along_batch`, `mean_along_batch`, `max_along_batch`, `argmax_along_batch`, `any_along_batch`, `median_along_batch` and `std_along_batch` return `np.float64`, `np.int64` or `np.bool` for a 1-D input, and so does `select_along_batch`. They are annotated `-> np.ndarray`. `nested.sum_along_batch(nested.sum_along_batch({"a": v}))` raises `TypeError: Incorrect array type: <class 'numpy.float64'>`. The output of a function is not always a valid input of another.                                                                                    | [verified] | P12            |
| F3  | **`chunk_*` and `split_*` return misaligned pieces without an error.** `nested.chunk_along_batch({"a": arange(5), "b": arange(3)}, 2)` returns `[{"a": [0, 1, 2], "b": [0, 1]}, {"a": [3, 4], "b": [2]}]`: row 3 of `a` is paired with row 2 of `b`. `nested.split_along_batch({"a": arange(5), "b": arange(6)}, 3)` returns `{"a": [0, 1, 2], "b": [0, 1, 2]}` then `{"a": [3, 4], "b": [3, 4, 5]}`: the pieces have different sizes. `split_structure` checks the **number of pieces**, not the **size** of the axis (the error `All the arrays must be split into the same number of pieces` appears only when the counts differ). `shuffle_*` is correct for the same case, because it reads the size of the first leaf and `permute_*` checks the length of every leaf.      | [verified] | P3, cross-leaf |
| F4  | **`nested.concatenate_*` accepts an empty list and is inconsistent on the container kind.** `nested.concatenate_along_batch([])` returns `{}`; `array.concatenate_along_batch([])` raises `ValueError: Cannot concatenate an empty sequence of arrays`. `[dict, OrderedDict]` returns a `dict` (first item), but `[list, tuple]` raises `TypeError: Expected all the items to be of type list` and `[dict, list]` raises `TypeError`. A mismatch of layout (keys, lengths) is a `ValueError`, a mismatch of container kind is a `TypeError`, a mismatch of leaf shape is NumPy's `ValueError`. A generator raises `'list_iterator' object is not subscriptable`. | [verified] | P5, P8         |
| F5  | **Six names of each level are aliases.** `take_along_batch`/`take_along_seq` are documented as equal to `index_select_*`. `amax_*` and `amin_*` (4 names) are equal to `max_*` and `min_*` (a test checks it: `test_amax_amin_equal_max_min`). In NumPy `amax` is an alias of `max`. `take_along_seq` with a 2-D index is a different operation under the same name (a per-sample gather), and **it does not check the batch size of the index**: `take_along_seq(np.arange(6).reshape(2, 3), np.array([[0, 1]]))` returns `[[0, 1]]` (batch size 1 for an array with batch size 2: row 1 is dropped). `index_select_along_batch(x, np.ones((2, 2), int))` accepts a 2-D index and returns a 3-D array, and `masked_select_along_batch(x, np.array([0, 1, 2]))` reads an integer mask as indices. | [verified] | P2, P3, P8     |
| F6  | **Some arguments fail with an unrelated error.** `split_along_batch(x, np.int64(2))` raises `TypeError: 'numpy.int64' object is not iterable` (the check is `isinstance(value, int)`), and `True` is accepted as a size of 1. `shuffle_along_batch(x, rng=1)` raises `AttributeError: 'int' object has no attribute 'permutation'`. `sort_along_batch(x, "stable")` and `shuffle_along_batch(x, rng)` take their option positionally (`kind`, `rng`, `step` of `slice_*`), while `keepdims` is keyword-only.                                                                                                                                                     | [verified] | P9, P13        |
| F7  | **The checks on a 1-D input are inconsistent and half of them are redundant.** On a 1-D array, the `_along_seq` functions raise three different exceptions. `ValueError` from `validate_min_ndim` in `tile`, `repeat`, `permute`, `shuffle` and `flip`. `numpy.exceptions.AxisError` (a subclass of both `ValueError` and `IndexError`) in `sum`, `cumsum`, `sort`, `argmax`, `concatenate`, `take` and `index_select`. `IndexError` in `slice`, `select`, `split`, `chunk` and `masked_select`. The validation of `flip_along_seq` and `repeat_along_seq` repeats what NumPy does (`np.flip(v, axis=1)` and `np.repeat(v, 2, axis=1)` raise `AxisError`). The validation of `tile_along_seq` is needed: `np.tile(v, (1, 2))` accepts a 1-D array and returns shape `(1, 8)`. | [verified] | P8             |
| F8  | **28 pointwise functions and `to_list` have no `array` twin and no batch or sequence meaning.** `abs`, `ceil`, `clip`, `exp`, `exp2`, `expm1`, `floor`, `isnan`, `log`, `log10`, `log1p`, `log2`, `round`, `sign`, `sqrt`, `square` (`pointwise.py`, 479 lines) and the 12 trigonometric functions (`trigo.py`, 326 lines) are each `recursive_apply(data, np.f)`. `nested.round` has no `decimals` option (`nested.round(x, decimals=1)` raises `TypeError`) and `nested.clip` keeps the old `a_min`/`a_max` names. `batchtensor` removed its pointwise functions for the same reason: `apply_to_tensors(data, torch.exp)` is the API. P3 does not hold: `nested` has 29 names that `array` does not have. They are 29 of the 90 names of `nested`. | [verified] | P3             |
| F9  | **`computation` is 1 569 lines for one expression.** A base class (522 lines), an interface of 14 functions (524), `ArrayComputationModel`, `MaskedArrayComputationModel` and `AutoComputationModel` with a global mutable registry, to choose between `np` and `np.ma`. The method of the array already does the dispatch: for `sum`, `prod`, `mean`, `std`, `var`, `min`, `max`, `argmax`, `argmin`, `any` and `all`, `x.sum(axis=0, keepdims=...)` gives the same result as the current function on plain and masked arrays (checked on both axes, with and without `keepdims`, on data with a fully masked cell). Only `median`, `sort`, `argsort` and `concatenate` need `np.ma` for a masked array (`np.ma.median`, ...), and `np.ma` gives the same results as the models. No other package of the workspace imports `batcharray`. The user guide example of a custom model implements two of the 14 abstract methods. | [verified] | P11, constraint 1 |
| F10 | **There is no single traversal.** The functions of `nested` call `recursive_apply` 82 times in 10 modules, so the leaf check has no place to live (F1). `nested/structure.py` walks containers by hand in `concatenate_structures` (`isinstance(first, dict)`, `(list, tuple)`), and `split_structure` calls `recursive_apply` once to build the pieces and once per piece to regroup them (`pieces` traversals for `pieces` outputs). `batchtensor` has the same three helpers on `coola.recursive` (`apply_to_tensors`, `split_tensors`, `combine_tensors` on `recursive_zip_apply`). | [verified] | P4             |
| F11 | **A deprecation shim wraps 42 reductions.** `keepdims_keyword_only` (`_deprecation.py`) accepts `keepdims` positionally with a `DeprecationWarning`, so every reduction is a `*args` wrapper around the real function. It was meant to be removed in "a later release".                                                                                                                                                                                                                                                                                                                                                                                | [verified] | P13            |
| F12 | **The documentation drifted.** `docs/docs/index.md` and `README.md` link badges of `workflows/docs.yaml`, `workflows/docs-dev.yaml` and `workflows/nightly-tests.yaml`; the workflows are `release-docs*.yaml` and `nightly-test-package.yaml` (`ls .github/workflows`). `index.md` lists "Zero Dependencies: Lightweight with only NumPy as a core dependency", while `coola` is required (`pyproject.toml`). A `black` badge stays although the project formats with `ruff`. The user guides omit 12 names of `array` (`all`, `any`, `flip`, `repeat`, `std`, `var`, each along batch and sequence) and 19 of `nested` (the same 12 and `ceil`, `floor`, `isnan`, `round`, `sign`, `sqrt`, `square`). No page states a property or a rule.                                                                                        | [verified] | P16, P18       |
| F13 | **The `nested` API is typed with `Any`.** Every `nested` function is `(data: Any, ...) -> Any`, so `ty` cannot see that a dict goes in and a dict comes out. `review/suggestions.md` left it open ("worth doing once `ty` handles recursive aliases well"). `batchtensor` has `NestedTensor` and `T` and passes `ty`.                                                                                                                                                                                                                                                                                                                                      | [verified] | P14            |
| F14 | **The layer rules are not enforced and most properties have no gate.** The three `import-linter` contracts forbid three pairs of imports. They say nothing about `utils` (which does not exist) or the direction `array` <- `nested`. `tests/` has no test for P1, P4, P6, P7, P8, P11, P12, P13 and P17. `test_api_consistency.py` checks `amax`/`amin` equality and a few pairs, not the parity of the two modules.                                                                                                                                                                                                                                     | [reading]  | P3, P15        |

**Checked and found correct** (no action): all the 25 functions probed with a masked array return
a `MaskedArray` with the mask kept, except `argsort` and `argmax`, which return indices (P11 holds
today); `slice`, `select` and `flip` return views and every other function returns a new array
(`concatenate_along_batch([b])` too); `shuffle_*` applies one permutation to all the leaves
(`{"a": arange(6), "b": arange(6)}` gives the same order twice); a cycle raises `ValueError: Cycle
detected`; `chunk_*` keeps a named tuple; `concatenate_*` keeps `tuple` of `dict` types; a `str`,
`None` and a list of `int` leaf raise a `TypeError` (with the message of F1); empty structures give
`{}` for `sum_along_batch` and `[]` for `split_*`; the dtype of `mean`, `sum` and `cumsum` is
NumPy's; `std_along_batch(x, ddof=1)` forwards `ddof`; `permute_*` rejects a short permutation and
a float one.

**Behaviours that must not regress:** the container types of the input are kept; one permutation
is shared by all the leaves of a `shuffle_*`; masked arrays keep their mask in every function that
keeps it today; a cycle raises `ValueError`; a non-array leaf raises a `TypeError` that names the
type (F1 makes it true everywhere); `chunk_*` and `split_*` raise on leaves of different sizes (F3
makes this true); the doctests run.

## 4. Architecture

The package gets one more layer (`utils`) and loses one (`computation`).

```text
constants          BATCH_AXIS, SEQ_AXIS
   ^
validation         validate_min_ndim, validate_split_sections, validate_same_keys
utils              types      NestedArray, T, SortKind
                   namespace  get_namespace
                   structure  apply_to_arrays, split_arrays, combine_arrays, get_size
   ^
array              one NumPy call with the axis fixed (+ the closed list of Section 4.4)
   ^
nested             one helper call per function: partial(ba.f, **options)
```

A module imports only from the layers above it in this list (`utils.structure` uses
`validate_same_keys`; `validation` imports nothing of the package). `import-linter` enforces the
order with one `layers` contract (it replaces the three `forbidden` contracts).

### 4.1 A `nested` function is logic-free

```python
def sum_along_batch(data: T, *, keepdims: bool = False) -> T:
    return apply_to_arrays(data, partial(ba.sum_along_batch, keepdims=keepdims))
```

Three shapes exist, and only three. A new operation uses one of them:

| Shape  | Helper                       | Used by                                                       |
| ------ | ---------------------------- | ------------------------------------------------------------- |
| 1 -> 1 | `apply_to_arrays(data, fn)`  | reductions, math, indexing, slicing (`select`, `slice`), permutation, sorting, `tile`, `repeat` |
| 1 -> N | `split_arrays(data, fn, *, axis)` | `chunk_*`, `split_*`                                      |
| N -> 1 | `combine_arrays(items, fn)`  | `concatenate_*`                                               |

A function that does not fit one of these shapes is not a `nested` wrapper and does not belong in
`nested`. It is a helper of `utils.structure`, like `get_size`. `shuffle_*` is the one composition:
`get_size` + `rng.permutation` + `permute_*`, so the permutation is shared by all the leaves.

`apply_to_arrays(data, fn)` is the public tool for what the package no longer wraps: a pointwise
function (`apply_to_arrays(data, np.exp)`), a conversion
(`apply_to_arrays(data, np.ndarray.tolist)`) or a cast. It has the leaf rule (P6) and the container
rule (P5) for free.

### 4.2 The container rule (D1)

The supported structure is `dict`, `list`, `tuple` and their subclasses (named tuple,
`OrderedDict`, `defaultdict`). This is exactly `NestedArray`. For `combine_arrays` the output
containers are the ones of the first item and the other items must have the same keys and lengths;
a mapping combined with a sequence raises a `ValueError`; `[list, tuple]` gives a list and
`[dict, OrderedDict]` gives a dict (F4). Sets and dataclasses are outside the contract.

### 4.3 One statement of each rule

- The layout (P1), the container rule, the cross-leaf rule, the empty-structure rule, the masked
  rule (Section 4.5) and the error table (Section 4.6) are in `docs/docs/design_principles.md` and
  nowhere else. The module docstrings of `array` and `nested` link to it. The repeated
  `Note: This function assumes the batch axis is the first axis` of every docstring is removed:
  it is P1.
- A function docstring states only what is specific to the function: its edge cases (P17), its
  `Raises`, and its example.
- `docs/docs/architecture.md` describes the layers, the three helpers and how to add an operation.
  It does not repeat the properties.

### 4.4 The closed list of exceptions to P2 and P7

A function outside this list is one NumPy call with the axis fixed, and returns a new array or the
view NumPy returns. The list of `main` has no document; this is the list after the change.

| Function                                         | Why it is not one NumPy call                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `*_along_batch` reductions and `select_along_batch` | the NumPy call wrapped in `np.asanyarray`, because NumPy returns a scalar on a 1-D input (P12)           |
| `split_*`                                        | `np.split` with indices computed from a size or sections (`np.split(x, 2)` means 2 sections, not size 2)      |
| `concatenate_*` with `dtype` on a masked array   | `np.ma.concatenate` has no `dtype`: `astype` after it                                                         |
| `index_select_*`                                 | `np.take` and a check that the index is 1-D (NumPy accepts any shape and returns another rank)               |
| `masked_select_*`                                | boolean indexing and a check that the mask is boolean (an integer mask is read as indices)                    |
| `gather_along_seq` (new)                         | `np.take_along_axis` with the index reshaped over the feature axes, and a check of the batch size (NumPy broadcasts) |
| `tile_along_seq`                                 | `np.tile` with the repetitions built from the rank, and a check of the rank (NumPy promotes a 1-D array)      |
| `permute_*`                                      | `index_select` and a length check (NumPy would accept a shorter index)                                        |
| `shuffle_*`                                      | draws the permutation with `np.random.default_rng(rng)`, then `permute_*`                                     |

Views (aliasing allowed): `slice_*`, `select_*`, `chunk_*`, `split_*`, `flip_*`. Everything else
returns a new array. `sort_*`, `cumsum_*`, `repeat_*` and `tile_*` with a count of 1 return a new
array.

### 4.5 Masked arrays (P11)

The dispatch is one helper of `utils`:

```python
def get_namespace(array: np.ndarray) -> ModuleType:
    return np.ma if isinstance(array, np.ma.MaskedArray) else np
```

A reduction, a cumulative function, `argmax`, `argmin`, `any` and `all` call the method of the array
(`array.sum(axis=BATCH_AXIS, keepdims=keepdims)`), which `MaskedArray` overrides. `median`, `sort`,
`argsort` and `concatenate` call `get_namespace(array).f(...)` because they are functions of the
module and not methods. `take`, `take_along_axis`, `flip`, `tile` and `repeat` keep the mask in
NumPy itself (checked, Section 6.2). The registry of computation models disappears: a subclass of
`MaskedArray` is dispatched by `isinstance`, a subclass of `ndarray` is a plain array, and an array
type with other semantics is not supported.

A function that NumPy does not support on a masked array is not added to the package until it has
a masked implementation. `np.pad` returns a plain `ndarray` for a masked array (checked), which is
one reason why `pad_*` is not added (D8).

### 4.6 The error contract (P8)

`batcharray` validates only what NumPy cannot see, or what NumPy accepts but means something else.
The table is `docs/docs/design_principles.md`:

| Situation                                                                   | Exception                 |
| --------------------------------------------------------------------------- | ------------------------- |
| A leaf is not an `ndarray`                                                  | `TypeError`               |
| `concatenate` receives no item, or structures with a different layout (keys, lengths) | `ValueError`    |
| `concatenate` combines a mapping with a sequence                            | `ValueError`              |
| Leaves disagree on the size of the axis of `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len` | `ValueError` |
| A split size that is not positive, sections that are negative or do not sum to the size | `ValueError`  |
| `index_select_*` with an index that is not 1-D; `masked_select_*` with a mask that is not boolean | `ValueError` |
| `gather_along_seq` with an index that is not 2-D or whose batch size differs | `ValueError`             |
| `permute_*` with a permutation of the wrong rank or length                  | `ValueError`              |
| `tile_along_seq`, `shuffle_along_seq`, `permute_along_seq`, `gather_along_seq`, `seq_len` on an array with fewer than 2 dimensions | `ValueError` |
| Another `_along_seq` function on a 1-D array (`slice`, `select`, `sum`, `flip`, `repeat`, `sort`, ...) | NumPy's `AxisError` or `IndexError` |
| An out-of-range or a float index, a negative count, a bad `kind`, a bad `ddof` | NumPy's own error      |

A numpy `AxisError` is both a `ValueError` and an `IndexError`, so a caller can catch either.
Compared with `main`: the `validate_min_ndim` of `flip_along_seq` and `repeat_along_seq` is removed
(F7), and the three checks of `index_select_*`, `masked_select_*` and `gather_along_seq` are new
(F5).

### 4.7 Indices: one rule (F5)

An index argument is handled the same way in every function:

| Function                 | Index                                       | Checked by `batcharray`                 | Cast or moved |
| ------------------------ | ------------------------------------------- | --------------------------------------- | ------------- |
| `index_select_along_*`   | 1-D integer array                           | `ndim == 1` (`ValueError`)              | never         |
| `permute_*`              | 1-D integer array of length = size of axis  | `ndim` and length (`ValueError`)        | never         |
| `shuffle_*`              | drawn by `rng.permutation`                  | n/a                                     | never         |
| `gather_along_seq` (new) | 2-D integer array `(batch_size, n)`         | `ndim` and batch size (`ValueError`)    | never         |
| `masked_select_*`        | 1-D boolean array                           | dtype is `bool` (`ValueError`)          | never         |

`gather_along_seq(array, indices)` returns `out[b, i] = array[b, indices[b, i]]` with shape
`(batch_size, n, *array.shape[2:])`. It is
`np.take_along_axis(array, indices.reshape(...), axis=1)`.
The dtype of the index, the bounds and the negative values are NumPy's rules (a negative index
counts from the end). There is no `gather_along_batch`: a per-sample gather along the batch axis
has no meaning, and `index_select_along_batch` is the operation of that axis. It replaces the 2-D
branch of `take_along_seq`. A typical use is the last valid step of a padded batch:
`gather_along_seq(x, (lengths - 1)[:, None])`.

`index_select` stays and `take` goes (D4): `index_select` is the name `batchtensor` uses for the
same operation, so code moves between the two packages without a rename.

### 4.8 Edge cases: document, do not guess (P17)

The result on an empty or degenerate input is NumPy's, and is documented and pinned by
`test_edge_case`. The table is in `design_principles.md`:

| Function                          | Case                                      | Result                                      |
| --------------------------------- | ----------------------------------------- | ------------------------------------------- |
| `sum_along_batch`, `prod_along_batch` | zero batch size                       | the identity (`0`, `1`) for each column     |
| `mean_along_batch`                | zero batch size                           | `nan` and NumPy's `RuntimeWarning`          |
| `max_along_batch`, `min_along_batch`, `argmax_*`, `argmin_*` | zero batch size | NumPy's `ValueError`                  |
| `chunk_*`                         | more chunks than items                    | empty chunks (`np.array_split`)             |
| `chunk_*`, `split_*`, `shuffle_*`, `batch_size`, `seq_len` | a structure without array | `ValueError`                  |
| `sum_along_batch` and other `apply_to_arrays` functions | an empty structure   | the empty structure (`{}`)                  |
| `permute_*`                       | a repeated index                          | not checked: only the length is             |
| `tile_along_seq`                  | `reps == 0`                               | a zero-length sequence axis                 |
| a masked reduction                | a fully masked row or column              | what `numpy.ma` returns (`masked`) (D9)     |

A function that would need the values of an array to validate it does not validate it.

### 4.9 Admission rule for a new operation

An operation enters `array` and `nested` only if all of these hold. It is in `CONTRIBUTING.md` and
`architecture.md`.

1. It needs the fixed axis (`np.f(x, axis=BATCH_AXIS)`), or it composes NumPy calls in a way that
   is easy to get wrong (`gather_along_seq`, `split_*`). A pointwise function (`exp`, `clip`) does
   not: use `apply_to_arrays(data, np.exp)`.
2. It has a `nested` twin of shape 1 -> 1, 1 -> N or N -> 1 (Section 4.1).
3. It comes with the four functions (batch and sequence, `array` and `nested`), or a stated reason
   why only one axis makes sense (`gather_along_seq`).
4. It works on a masked array (P11) or the docstring says why it cannot.
5. If it is not one NumPy call, it is added to the list of Section 4.4.
6. It has a runnable example, a documented edge-case result (P17), and a gate for each property it
   touches.

## 5. Public API after the change

Compared with `main` (61 `array` names, 90 `nested` names). After: `array` 56 (61, minus 6
aliases, plus `gather_along_seq`) and `nested` 58 (the 56 of `array`, plus `batch_size` and
`seq_len`), and the type `NestedArray`. The full inventory is Appendix A.

| Name                                                              | Change                                                                                                                                                                                                       | Breaking              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| `nested.chunk_*`, `nested.split_*`                                | Raise a `ValueError` when the leaves have different sizes along the axis, whatever the number of pieces (F3), and on a structure without array (`chunk_*` and `split_*` return `[]` today). `split_arrays(data, fn, *, axis)` calls `get_size(data, axis)` first; the check of the piece counts is removed. | Yes (silent bug)      |
| `nested.concatenate_*`                                            | An empty list raises `ValueError`. The output container is the one of the first item; a mapping with a sequence is a `ValueError` (F4). A `Sequence` is required.                                            | Yes                   |
| every `nested` function                                           | A leaf that is not an `ndarray` raises a `TypeError` that names the type (F1). Typed `T -> T` with `T` bound to `NestedArray` (F13). Goes through `apply_to_arrays`, `split_arrays` or `combine_arrays` (F10). | Yes                   |
| reductions along batch, `select_along_batch`                      | Return a 0-d array where NumPy returns a scalar (F2).                                                                                                                                                       | Yes (rare)            |
| `nested.abs`, `ceil`, `clip`, `exp`, `exp2`, `expm1`, `floor`, `isnan`, `log`, `log10`, `log1p`, `log2`, `round`, `sign`, `sqrt`, `square`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` (28) | Removed. Use `apply_to_arrays(data, np.exp)` (F8, D2).                                                                                                         | Yes                   |
| `nested.to_list`                                                  | Removed. Use `apply_to_arrays(data, np.ndarray.tolist)` (F8, D2).                                                                                                                                           | Yes                   |
| `batcharray.computation` (the whole package, 18 names)            | Removed. The dispatch between `np` and `np.ma` is `utils.get_namespace` (F9, D3).                                                                                                                           | Yes                   |
| `batcharray.types`                                                | Moved to `batcharray.utils.types`. `ArrayT` and `SORT_KINDS` are removed; `SortKind` stays; `NestedArray` and `T` are added.                                                                                | Yes                   |
| `amax_*`, `amin_*` (`array` and `nested`, 8 names)                | Removed. Use `max_*` and `min_*` (F5, D5).                                                                                                                                                                   | Yes                   |
| `take_along_batch`, `take_along_seq` (4 names)                    | Removed. Use `index_select_*` (1-D index) or `gather_along_seq` (2-D index) (F5, D4).                                                                                                                       | Yes                   |
| `index_select_*`, `masked_select_*`                               | `index_select_*` checks that the index is 1-D; `masked_select_*` checks that the mask is boolean (F5).                                                                                                      | Yes (rare)            |
| `gather_along_seq` (new, both levels)                             | Per-sample gather on `np.take_along_axis`, with a check of the batch size. In `nested` the `indices` is one array shared by all the leaves.                                                                | No                    |
| `kind`, `rng`, `step` (`sort_*`, `argsort_*`, `shuffle_*`, `slice_*`) | Keyword-only (F6).                                                                                                                                                                                      | Yes                   |
| `shuffle_*`                                                       | `rng` is passed to `np.random.default_rng`: it accepts a `Generator`, an `int` seed or `None` (F6).                                                                                                          | No                    |
| `split_*`                                                         | The size is checked with `numbers.Integral` and rejects a `bool` (F6).                                                                                                                                       | No                    |
| `keepdims_keyword_only`, `batcharray._deprecation`                | Removed. `keepdims` is keyword-only without a shim (F11).                                                                                                                                                    | Yes                   |
| `flip_along_seq`, `repeat_along_seq`                              | No `validate_min_ndim`: NumPy's `AxisError` on a 1-D input (F7).                                                                                                                                             | Yes (rare)            |
| `nested.batch_size`, `nested.seq_len` (new)                       | `get_size(data, axis)` with the first-leaf rule: leaves must agree, a structure without array raises `ValueError`.                                                                                          | No                    |
| `validation.validate_first_array`, `validate_same_batch_size`, `validate_same_seq_size` | Removed: `get_size` (and `batch_size`, `seq_len`) replace them.                                                                                                                              | Yes                   |
| `validation.validate_same_keys`, `validate_min_ndim`, `validate_split_sections` | Kept.                                                                                                                                                                                       | No                    |
| `design_principles.md`, `architecture.md`, `migration.md` (new)   | Written from Sections 1, 4 and 5.                                                                                                                                                                            | No                    |
| `README.md`, `docs/docs/index.md`, `uguide/*`                     | Badges, dependency claim, function lists and module description fixed (F12). `uguide/computation.md` and `refs/computation.md` removed.                                                                     | No                    |

## 6. Dependents and evidence

### 6.0 Dependents

| Package               | Change                                                                                                                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `coola`               | Nothing to request. `recursive_zip_apply` and `recursive_apply` exist in `2.0.0a4`.                                                                                                    |
| `batchtensor`         | Nothing. It is the model: the names, the options and the rules of this document are the ones it has.                                                                                                  |
| Other local packages  | None. No package of `~/workspace/code/2024` imports `batcharray` (`grep` over `.py`, `.toml` and `.md`). Public users are unknown; `migration.md` lists each removal with its replacement.             |

### 6.1 Cost

Probes on this machine (Python 3.14, NumPy 2.5.5, CPU), 200 keys x 5 arrays of shape `(4, 3)` =
1 000 leaves:

| Call                                                           | Time per call |
| -------------------------------------------------------------- | ------------- |
| `nested.sum_along_batch(data)`                                 | 1.95 ms       |
| hand-written dict/list comprehension with `x.sum(axis=0)`      | 0.63 ms       |
| `array.sum_along_batch(x)` on `main` (computation registry)    | 1.54 us       |
| `np.sum(x, axis=0)`                                            | 1.16 us       |
| `x.sum(axis=0)` (the method, target)                           | 0.59 us       |
| `isinstance` dispatch + `np.sum(x, axis=0)`                    | 1.24 us       |

The overhead of `nested` is about 1.3 us per leaf and is the cost of `coola.recursive`. The target
is not slower: the method call is faster than the registry lookup (0.59 us against 1.54 us), and
`np.asanyarray` adds about 0.04 us (0.59 us to 0.63 us for `x.sum(axis=0)`, measured) to the
reductions along batch. Constraint 5 holds. The numbers were taken once, on a laptop; they show an
order of magnitude, not a benchmark.

### 6.2 Probes of behaviour (this review, `main` at `8392e51`, NumPy 2.5.5)

| Probe                                                                                              | Result                                                                            |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `nested.sqrt({"a": a, "n": 4})`                                                                    | `n` becomes `np.float64(2.0)` (F1)                                                |
| `nested.sum_along_batch({"a": a, "n": 1})`, `({"a": "x"})`, `({"a": None})`                        | `TypeError: Incorrect array type: <class 'int'> ...` (F1)                         |
| `nested.slice_along_batch([1, 2, 3], stop=2)`                                                      | `TypeError: 'int' object is not subscriptable` (F1)                               |
| `nested.to_list({"a": a[:1], "n": (1, 2)})`                                                        | the tuple is unchanged (F1)                                                       |
| `type(array.sum_along_batch(np.arange(4.)))` and the same for `mean`, `max`, `argmax`, `any`, `median`, `std`, `select_along_batch` | `float64`, `int64` or `bool`, not `ndarray` (F2)                  |
| `nested.sum_along_batch(nested.sum_along_batch({"a": v}))`                                         | `TypeError: Incorrect array type: <class 'numpy.float64'>` (F2)                   |
| `nested.chunk_along_batch({"a": arange(5), "b": arange(3)}, 2)`                                    | works, misaligned pieces (F3)                                                     |
| `nested.chunk_along_batch({"a": arange(5), "b": arange(4)}, 2)`                                    | works: `a` pieces of 3 and 2, `b` pieces of 2 and 2 (F3)                          |
| `nested.split_along_batch({"a": arange(5), "b": arange(6)}, 3)`                                    | works: `a` pieces of 3 and 2, `b` pieces of 3 and 3 (F3)                          |
| `nested.split_along_batch({"a": arange(5), "b": arange(3)}, 2)`                                    | `ValueError` on the piece counts `[2, 3]`                                         |
| `nested.concatenate_along_batch([])`, `array.concatenate_along_batch([])`                          | `{}`; `ValueError` (F4)                                                           |
| `nested.concatenate_along_batch` on `[dict, OrderedDict]`, `[list, tuple]`, `[dict, list]`, a generator | `dict`; `TypeError`; `TypeError`; `TypeError: ... not subscriptable` (F4)    |
| `array.take_along_seq(np.arange(6).reshape(2, 3), np.array([[0, 1]]))`                             | `[[0, 1]]`: batch size 1, no error (F5)                                           |
| `array.index_select_along_batch(b, np.ones((2, 2), int))`, `masked_select_along_batch(b, np.array([0, 1, 2]))` | a 3-D array; rows 0, 1, 2 as if indices (F5)                           |
| `array.split_along_batch(b, np.int64(2))`, `(b, True)`                                             | `TypeError: 'numpy.int64' object is not iterable`; works, size 1 (F6)             |
| `array.shuffle_along_batch(b, rng=1)`                                                              | `AttributeError: 'int' object has no attribute 'permutation'` (F6)                |
| `np.random.default_rng(g) is g` for a `Generator`; `default_rng(3)`                                | `True`; a `Generator` (P9 needs one line)                                         |
| the `_along_seq` functions on a 1-D array                                                          | `ValueError`: tile, repeat, permute, shuffle, flip. `AxisError`: sum, cumsum, sort, argmax, concatenate, take, index_select. `IndexError`: slice, select, split, chunk, masked_select (F7) |
| `np.flip(v, axis=1)`, `np.repeat(v, 2, axis=1)`, `np.take(v, [0], axis=1)` on a 1-D array         | `AxisError` (F7)                                                                  |
| `np.tile(v, (1, 2)).shape` on a 1-D array                                                          | `(1, 8)`: NumPy promotes (the check of `tile_along_seq` is needed)                |
| `numpy.exceptions.AxisError.__mro__`                                                               | `AxisError`, `ValueError`, `IndexError`, ...                                      |
| `x.f(axis=...)` against `array.f_along_*` for `sum`, `prod`, `mean`, `std`, `var`, `min`, `max`, `argmax`, `argmin`, `any`, `all`, both axes, `keepdims` on and off, plain and masked (one fully masked cell) | identical for all 88 combinations (F9) |
| `np.ma.median`, `np.ma.sort`, `np.ma.argsort`, `np.ma.concatenate` against the `array` functions on masked and plain data | identical (F9)                                                    |
| `np.take`, `np.cumsum`, `np.sort`, `np.take_along_axis` on a masked array                          | `MaskedArray`, mask kept (P11)                                                    |
| `np.pad` on a masked array                                                                         | plain `ndarray`: the mask is lost (D8)                                            |
| 25 `array` functions on a masked array                                                             | `MaskedArray` with the mask kept, except `argsort` and `argmax` (ndarray) (P11)   |
| `np.ma.sum(np.ma.array(v, mask=v > 1), axis=0)` on a 1-D array; the same fully masked              | `float64`; `MaskedConstant` (P12: `np.asanyarray` keeps it)                       |
| `np.take_along_axis(x, idx_with_batch_1, axis=1)` for `x` with batch 2                             | works and broadcasts to batch 2 (the check of `gather_along_seq` is needed)       |
| `nested.sum_along_batch(D(x))` for a dataclass; `({1})` for a set; a self-referencing list         | works; `TypeError` (the `int` leaf); `ValueError: Cycle detected`                 |
| `array.sum_along_batch(b, True)`                                                                   | works with a `DeprecationWarning` (F11)                                           |
| `grep -c "recursive_apply("` in `src/batcharray/nested`                                            | 82 in 10 modules, plus 3 in `structure.py` (F10)                                  |

Commands: `ruff check src tests` clean, `ty check` clean, `lint-imports` 3 contracts kept,
`pytest tests -q` 1 725 passed. The markdown code blocks of `docs/` and `README.md` were not run in
this review (they are not changed).

## 7. Changes to reach the target state

One pull request per step, each releasable and small enough to review alone. The two steps that fix
a wrong result come first.

| Step | Change                                                                                                                                                                                                                                                               | Breaking   | Status                    |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------- |
| 1    | **Fix F3 and F4 in place.** `split_structure` checks the size of the axis of every leaf (not the number of pieces); `concatenate_*` raises on an empty list and follows the first-item container rule. Regression tests with sizes `(5, 3)`, `(5, 4)` and `(5, 6)` with `split_size=3`, and `[]`, which fail without the fix (`git stash push src`). | Yes (bug)  | Done in #690              |
| 2    | **Write the properties.** Add `docs/docs/design_principles.md` (Sections 1, 4.2, 4.4, 4.6, 4.8), `architecture.md` and `CONTRIBUTING.md` admission rule (Section 4.9). Fix the docs drift (F12): badges, "Zero Dependencies", module description, function lists in the guides. Add `test_docs_references` (T12). | No         | Done in #691              |
| 3    | **Remove the pointwise functions and `to_list`** (F8, D2). Delete `nested/pointwise.py`, `trigo.py`, `conversion.py`, their tests and guide entries. Add `migration.md` with the replacement of each name. `apply_to_arrays` is not yet public: the migration uses `coola.recursive.recursive_apply` until step 6. | Yes        | Done in #692              |
| 4    | **Remove `computation`** (F9, D3). Add `utils.namespace.get_namespace`. `array` functions call the method of the array, or the namespace for `median`, `sort`, `argsort`, `concatenate`. Delete `computation/`, its tests, guide and reference page, and the `ArrayT` and `SORT_KINDS` types. Add `test_masked_arrays` first and check that it passes before and after. | Yes        | Done in #693              |
| 5    | **Remove the aliases and the deprecation shim** (F5, F11, D4, D5). Remove `take_*`, `amax_*`, `amin_*` and `_deprecation.py`. Options keyword-only: `kind`, `rng`, `step` (F6). `shuffle_*` accepts a seed. `split_*` accepts NumPy integers and rejects a `bool` (F6). | Yes        | Done in #694              |
| 6    | **One traversal, a leaf contract and types** (F1, F10, F13). Add `utils.structure` (`apply_to_arrays`, `split_arrays`, `combine_arrays`, `get_size`) and `utils.types` (`NestedArray`, `T`, `SortKind`). Move every `nested` function to a helper and annotate it `T -> T`. Delete `nested/structure.py`. Add `test_non_array_leaf`, `test_nested_module_does_not_walk_containers`, `test_containers` and `test_typing`. Check with `ty` that the recursive alias works; if not, keep `T` and bind it to `Any` for one step. | Yes        | Done in #696              |
| 7    | **Array in, array out and the 1-D rules** (F2, F7). `np.asanyarray` in the reductions along batch and in `select_along_batch`. Remove the redundant `validate_min_ndim` of `flip_along_seq` and `repeat_along_seq`. Add `test_array_in_array_out` and the 1-D rows of `test_error_contract`. | Yes (rare) | Done in #697              |
| 8    | **The index family** (F5). `index_select_*` checks that the index is 1-D, `masked_select_*` that the mask is boolean, and `gather_along_seq` is added at both levels with a check of the batch size. Regression tests for the three cases of F5 fail without the fix. Update the table of Section 4.7 in `design_principles.md`. | Yes        | Done in #698              |
| 9    | **`batch_size`, `seq_len`, the layers and the gates** (F14). Add `nested.batch_size` and `nested.seq_len`, delete `validate_first_array`, `validate_same_batch_size`, `validate_same_seq_size`. Replace the three `import-linter` contracts with one `layers` contract. Add the remaining gates of Section 8. | Yes (small) | Done in #699              |
| 10   | **Release.** Set the version and the classifier together. Remove "the API is not stable" from `design_principles.md` and `index.md` only when D10 is answered.                                                                                                      | No         | Blocked: D10 and `coola` 2.0 stable |

Steps 1 and 2 are independent of the others and can ship now. Steps 3, 4 and 5 are independent of
each other and each shrinks the package (steps 3 and 4 delete about 2 400 lines of `src`). Step 6
is the largest and goes after them because it touches every `nested` function and fewer functions
remain. Step 1 is the only one that fixes a wrong result and goes first.

Estimated size of `src` after the steps: 7 840 lines now, about 2 460 removed by steps 3, 4 and 5
(`computation` 1 569, `pointwise` 479, `trigo` 326, `conversion` 39, `_deprecation` 47), about 500
added by steps 6 to 9 (`utils`, `gather_along_seq`, `batch_size`, `seq_len`). The result is about
5 900 lines, a quarter less, and the tests shrink by about the same share. This is an estimate, not
a measure.

## 8. Tests and gates

- **Kept:** the unit tests of `array` and `nested`, `test_api_consistency.py` (extended into
  `test_api_parity`), the integration tests of `nested`, the doctests, `ruff`, `ty`,
  `lint-imports`. 100% coverage of the code added or changed. A bug fix adds a regression test
  that fails without the fix (`git stash push src`).
- **Removed with their code:** the tests of `computation`, `pointwise`, `trigo`, `conversion`,
  `_deprecation`, `take_*`, `amax_*`, `amin_*` and the three `validate_*` of step 9.
- **T1. `test_no_axis_parameter`.** No public function of `array` or `nested` has a parameter named
  `axis`. Gate for P1.
- **T2. `test_thin_wrappers`.** Each `array` function not in the list of Section 4.4 has a body of
  one `return` that calls one NumPy function (checked on the AST). A new function that is neither
  one NumPy call nor in the list fails. Gate for P2.
- **T3. `test_api_parity`.** The names of `array` and `nested` are equal except `batch_size` and
  `seq_len`, and the signatures (names, kinds, defaults) are equal except the first parameter.
  Gate for P3.
- **T4. `test_nested_module_does_not_walk_containers`.** No module of `nested` imports
  `recursive_apply` or uses `isinstance` on a container. Gate for P4.
- **T5. `test_non_array_leaf`.** Every `nested` function raises a `TypeError` that contains the
  type name for an `int`, a `str`, `None` and a `np.float64` leaf. Gate for F1, P6.
- **T6. `test_containers`.** A `dict`, an `OrderedDict`, a `defaultdict`, a `list`, a `tuple` and a
  named tuple keep their type through a map function, `chunk_*`, `split_*` and `concatenate_*`.
  Gate for P5.
- **T7. `test_inputs_are_not_modified` and `test_output_does_not_alias_input`.** One parametrized
  test per property over all the functions, with `np.shares_memory` and the list of views of
  Section 4.4. Gate for P7.
- **T8. `test_error_contract`.** One row per line of the table of Section 4.6, with the exception
  type and a `match`. Gate for P8.
- **T9. `test_masked_arrays`.** Every function, on a masked array with a fully masked cell,
  returns a `MaskedArray` (an `ndarray` for the three index functions) and equals the result of the
  `numpy.ma` function on the same data. Gate for P11, F9. It runs before step 4 and is the safety
  net of that step.
- **T10. `test_array_in_array_out`.** Every function on a 1-D, a 2-D and a 3-D array returns an
  `ndarray`, and `nested.f(nested.f(x))` works for the reductions. Gate for P12, F2.
- **T11. `test_signatures`.** Every optional parameter is keyword-only, except the arguments that
  define the operation (`start`, `stop`), and no function takes `**kwargs`. Gate for P13.
- **T12. `test_docs_references`.** Every `actions/workflows/<name>.yaml` linked in the README and
  `docs/docs/*.md` exists in `.github/workflows`, every module in `refs/*.md` exists, and the
  dependencies named in `index.md` are the ones of `pyproject.toml`. Gate for F12, P18.
- **T13. `test_misaligned_chunk`.** `chunk_*` and `split_*` raise on leaves of size `(5, 3)`,
  `(5, 4)` and, for `split_size=3`, `(5, 6)` (equal piece counts), at the `nested` level. Gate for
  F3.
- **T14. `test_gather_along_seq`.** Equal to `np.stack([x[b, idx[b]] for b in ...])` on seeded
  random data, for 2-D, 3-D and 4-D `x`, plain and masked; a float index and an out-of-range index
  raise NumPy's error; a batch size mismatch raises `ValueError`; the nested twin shares one index
  between leaves of different feature shapes. Gate for P2, P3, P11.
- **T15. `test_edge_case`.** The table of Section 4.8, one row per test. Gate for P17.
- **T16. `test_rng`.** A `Generator`, an `int` seed and `None` are accepted, a seed is
  reproducible, one permutation is shared by all the leaves. Gate for P9.
- **No test** says that sets or dataclasses are supported (D1).

## 9. Decisions for the maintainer

| #   | Decision                                                                                     | Recommendation                                                                                                                                                                                                                                                                                                                       |
| --- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | Dataclasses and sets: supported or outside the contract?                                     | **Outside the contract.** A dataclass works through `coola` and a set fails on its `int` leaf. A promise needs a type (`NestedArray` cannot express a dataclass) and a gate in every family, for a use case where batches are dicts. Same decision as `batchtensor`.                                                               |
| D2  | Remove the 28 pointwise functions and `to_list` from `nested`?                               | **Yes.** They have no `array` twin and no axis (admission rule 1); `apply_to_arrays(data, np.exp)` is shorter than `nested.exp(data)` plus a lookup. They cost 844 lines of `src`, their tests and their docs, and two of them have lost options (`round` has no `decimals`). Alternative: keep them and add 29 `array` functions to restore P3, which adds code for no gain. |
| D3  | Remove the `computation` package?                                                            | **Yes.** `get_namespace` is one line and the methods of the array do the rest (F9). The loss is the registry for a custom array type, which no package of the workspace uses. Alternative: keep `BaseComputationModel` and `AutoComputationModel` and drop only the interface functions; it keeps the registry, the global state and about 1 000 lines. |
| D4  | Keep `index_select` and drop `take`, or the reverse?                                         | **Keep `index_select`.** It is the name of `batchtensor` for the same operation, and `take` has an n-D meaning in NumPy that the package does not support. The 2-D branch of `take_along_seq` becomes `gather_along_seq`.                                                                                                        |
| D5  | Remove `amax_*` and `amin_*`?                                                                | **Yes.** In NumPy they are aliases of `max` and `min`; a test checks that they are equal. In `batchtensor` they differ from `max`/`min` (torch returns indices with `max`); here they do not, and a name for an alias is a maintenance cost with no feature.                                                                       |
| D6  | A reduction on a 1-D array: return a 0-d array, or accept `np.generic` as a leaf?            | **Return a 0-d array (P12).** One leaf type, so `NestedArray` stays an array, a mapping, a list or a tuple, and the output of any function is a valid input. A 0-d array behaves like a scalar in arithmetic, `float(x)` and `x.item()` work. Alternative: accept `np.generic` leaves; it makes every helper and the type handle two leaf kinds. |
| D7  | Make `concatenate_*` strict on the container kind (all items the same type)?                  | **No.** It needs a check on top of `coola`'s. The first-item rule is documented and pinned (P5), as in `batchtensor` (its D10). A mapping with a sequence is the one case that raises.                                                                                                                                              |
| D8  | Add `pad_along_*`, `pad_sequences`, `lengths_to_mask` and `masked_*_along_seq` (they are in `batchtensor`)? | **No, not now.** `np.pad` drops the mask of a masked array, so `pad_*` would need its own masked implementation to satisfy P11. NumPy has `np.ma` for masked reductions, and `lengths_to_mask` is `np.arange(n) < lengths[:, None]`. `pad_sequences` is the one function with no NumPy equivalent; revisit it on a request with a use case, with a masked implementation and a gate. |
| D9  | Fully masked row or column: raise, return a sentinel, or return what `numpy.ma` returns?     | **What `numpy.ma` returns** (`masked`). The package does not add a rule on top of `numpy.ma` (constraint 7). It is documented in P17. A caller who needs a guard tests `np.ma.getmaskarray(x).all(axis=...)`.                                                                                                                       |
| D10 | When is the API stable?                                                                      | After `coola` 2.0 is stable (the dependency is `coola>=2.0.0a4,<3.0`, an alpha) and one release with no change. Until then keep "pre-1.0".                                                                                                                                                                  |
| D11 | Generate the `nested` twins instead of writing them?                                         | **No.** Generation hides the signature from the type checkers and the docs, and a `nested` function is already one line. The cost is docstrings (`suggestions.md`, Section 7), which are the product.                                                                                                                              |
| D12 | Rename `chunk_*` and `split_*`, which follow `np.array_split` and torch (not `np.split`)?    | **No.** The two packages use the same names with the same meaning (`split` takes a size or sections, `chunk` takes a number of chunks); the closed list says why `split_*` is not one NumPy call. The docstrings already state the difference with NumPy.                                                                          |
| D13 | Keep `validation` as a public package?                                                       | **Yes**, with three functions (`validate_min_ndim`, `validate_split_sections`, `validate_same_keys`), as `CLAUDE.md` asks for an argument to be validated there. The two opt-in size validators go: `batch_size` and `seq_len` do the same check and return the size.                                                                  |

## 10. Not covered

- New operations beyond `gather_along_seq`, `batch_size` and `seq_len`. The admission rule
  (Section 4.9) is the process (D8).
- Array libraries other than NumPy (CuPy, JAX, `torch`): `batchtensor` is the package for PyTorch,
  and P11 supports `numpy.ma` only.
- Structured arrays, `object` arrays, sparse arrays and `np.matrix`.
- Ragged data (lists of arrays of different lengths) and `pad_sequences` (D8).
- A configurable layout (P1 says no).
- Data loaders, datasets or collate functions (`concatenate_*` is the building block).
- Python and NumPy version policy beyond `numpy>=2.0,<3.0` and Python 3.11 and above. The probes
  ran on NumPy 2.5.5 only: the equivalence of F9 and the masked behaviour are **not** checked on
  `numpy` 2.0, the lowest supported version. The CI matrix of `package_versions.json` runs the
  tests, so `test_masked_arrays` (T9) is the check; it must pass on the whole matrix before step 4
  is merged.
- Performance work. Section 6.1 shows no need.
- The type alias on `pyright`. Only `ty` runs in CI.

## Appendix A. API inventory

**`array` (61 names on `main`, 56 after).** Reductions (`max`, `min`, `mean`, `median`, `sum`,
`prod`, `std`, `var`, `any`, `all`, `argmax`, `argmin`), cumulative (`cumsum`, `cumprod`), sorting
(`sort`, `argsort`), indexing (`index_select`, `masked_select`, `gather_along_seq`), slicing
(`slice`, `select`, `chunk`, `split`), joining (`concatenate`, `repeat`, `tile_along_seq`),
permutation (`permute`, `shuffle`, `flip`), each `_along_batch` and `_along_seq` except
`gather_along_seq` and `tile_along_seq`. Removed: `amax`, `amin` (4 names) and `take` (2 names).
Added: `gather_along_seq`. Count: 24 reductions, 4 cumulative, 4 sorting, 5 indexing, 8 slicing,
5 joining and 6 permutation, which is 56.

**`nested` (90 names on `main`, 58 functions after, plus the type `NestedArray`).** The same names
as `array`, plus `batch_size` and `seq_len`. Removed: the 28 pointwise functions, `to_list`, and
the six names of F5. Added: `gather_along_seq`, `batch_size`, `seq_len`.

**`utils` (new).** `types` (`NestedArray`, `T`, `SortKind`), `namespace` (`get_namespace`),
`structure` (`apply_to_arrays`, `split_arrays`, `combine_arrays`, `get_size`).

**`validation` (6 names on `main`, 3 after).** `validate_min_ndim`, `validate_split_sections`,
`validate_same_keys`.

**`constants`.** `BATCH_AXIS`, `SEQ_AXIS`.

**Removed packages and modules.** `batcharray.computation` (18 names), `batcharray.types`
(moved), `batcharray._deprecation`, `batcharray.nested.{pointwise,trigo,conversion,structure}`.

**Size on `main`.** `src`: 7 840 lines (`nested` 3 560 including `pointwise` 479 and `trigo` 326,
`array` 2 341, `computation` 1 569, `validation` 245, the top-level modules 125); `tests`:
12 755 lines.

## Appendix B. Record of the previous review

`review/suggestions.md` (kept, not replaced) is the record of the first review. Its sections 1 to 3
are done: the sharp edges of `tile`, `split`, `concatenate`, `shuffle` and `permute`, the
keyword-only `keepdims` with a shim, the computation models for `sum`, `prod`, `std` and `var`, the
new reductions and pointwise functions, and the opt-in size validators. This document replaces its
open items: the `NestedArray` alias (its Section 5, now P14 and step 6), the registry cache of
`AutoComputationModel` (its Section 4, closed by removing the registry, step 4) and the docstring
duplication (its Section 7, D11). It reverses two of its results because the target state values
the future state over the past: the pointwise functions (Section 3 of the old review) and the
computation models for `std` and `var` (step 4). Its "Repository hygiene" section
(Section 6) needs no action.
