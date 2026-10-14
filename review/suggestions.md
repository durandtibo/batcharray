# Review of `src/batcharray`

Scope: structure, computation layer, and a sample of `array/` and `nested/`.
Tests and type checker were not run.

The package is clean and consistent (typed signatures, doctests, `py.typed`,
`__all__` everywhere). The suggestions below mostly reduce repetition and
tighten behavior.

## Suggested starting order

#3 (likely correctness gap), #4 (empty-sequence error, lookup cache), then #7.
#1 was dropped and #2 is done.

## 1. Repeated `_along_batch` / `_along_seq` pairs (dropped)

Each operation exists 4 times (`array`/`nested` x batch/seq), but the bodies are
already single-line delegations that differ only in the axis constant, e.g.
`np.sum(array, axis=BATCH_AXIS, keepdims=keepdims)`. A private helper taking
`axis` would add a function and a call layer per operation without removing
anything. The real duplication is in docstrings/doctests, which each public
function needs; fixing that requires a docstring template/generator, which
hurts readability and IDE/mkdocstrings output. Not worth doing unless docstring
maintenance becomes a real cost.

## 2. Redundant `partial` in `nested/math.py` (done)

Removed the argument-less `partial` wrappers and the unused import. Other
`partial(...)` calls in `nested/` bind real arguments and were left alone.

## 3. Inconsistent reduction backends in `array/reduction.py`

`amax`/`amin` call `np.amax`/`np.amin` directly, while `argmax`, `argmin`, etc.
go through `cmpt`. `amax_along_batch` therefore may not dispatch for
`MaskedArray` or custom registered types. Route all through `cmpt`, or
document the difference.

## 4. `AutoComputationModel` registry (`computation/auto.py`)

- `registry` is a mutable, global class-level `ClassVar`; registration via
  `register_computation_models()` with `# pragma: no cover` on the guard.
- Lookup walks `__mro__` on every call; cache with `functools.lru_cache`,
  invalidated in `add_computation_model`.
- `concatenate` dispatches on `type(arrays[0])`: an empty sequence raises a
  bare `IndexError` (should be a clear `ValueError`), and mixed
  `ndarray`/`MaskedArray` inputs silently use the first type.
- The unknown-type `TypeError` could list the registered types.

## 5. API parity gaps

- `computation/interface.py` has `mean`, `median`, `max`, `min` but no `sum`,
  `std`, `var`.
- `concatenate` accepts `dtype`; check whether the `*_along_*` wrappers in
  `array/joining.py` expose it.

## 6. Input validation in `nested`

- Docs say all arrays must share the batch size, but nothing checks it;
  mismatches fail deep in NumPy. Add an opt-in/default batch-size check helper.
- `to_list` docs say each item must be an `ndarray`, but non-arrays pass
  through unchanged. Fix the docstring or the behavior.

## 7. Types

- `nested` functions are `Any -> Any`. A recursive alias
  (`NestedArray = np.ndarray | Mapping[str, "NestedArray"] | Sequence["NestedArray"]`)
  plus TypeVars would let `ty` check output structure.
- `SORT_KINDS` is a mutable `list`; use a tuple or `typing.get_args(SortKind)`
  to avoid a duplicated source of truth.
- `T = TypeVar("T", bound=np.ndarray)` is defined in several files; define once.

## 8. Shadowed builtins in `nested/pointwise.py`

`abs` is defined with `# noqa: A001`. `from batcharray.nested import *` would
shadow the builtin; document it or keep it namespace-only.

## 9. `__init__.py`

Only `__version__` is exported. Lazily importing subpackages would make
`batcharray.array.slice_along_batch` work after a single `import batcharray`.

## 10. Performance and tests

- Confirm `nested` permutation/indexing generate the random permutation/index
  once and reuse it across arrays; add a test, since it is easy to get wrong
  with `recursive_apply`.
- Add property-based tests (hypothesis) for batch/seq axis equivalence:
  `fn_along_batch(x) == fn(x, axis=0)`.
