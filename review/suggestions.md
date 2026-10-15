# Review of `src/batcharray`

Scope: structure, computation layer, and a sample of `array/` and `nested/`.
Tests and type checker were not run.

The package is clean and consistent (typed signatures, doctests, `py.typed`,
`__all__` everywhere). The suggestions below mostly reduce repetition and
tighten behavior.

## Suggested starting order

#7.
#1 dropped, #2 done, #3 not a bug, #4 and #5 partly done.

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

## 3. Inconsistent reduction backends in `array/reduction.py` (not a bug)
`amax`/`amin`/`mean`/`prod`/`sum` call NumPy directly while `argmax`, `max`,
etc. go through `cmpt`. I suspected this broke `MaskedArray` support, but
checked it: `np.amax`/`np.mean`/`np.sum`/... on an `ndarray` subclass delegate
to the array's own method, so masked inputs return correct `MaskedArray`
results (verified for all seven batch reductions). Only a cosmetic
inconsistency remains; `prod`/`sum` have no `cmpt` equivalent anyway.
No change made.

## 4. `AutoComputationModel` registry (`computation/auto.py`) (partly done)
Done:
- `concatenate([])` now raises a clear `ValueError` instead of `IndexError`.
- The unknown-type `TypeError` lists the registered array types.
- `find_computation_model` checks the exact type first, so the `__mro__` walk
  only happens for subclasses.

Not done:
- A real lookup cache: tests (and users) mutate `registry` directly
  (`patch.dict`), which would leave a separate cache stale. The exact-type
  fast path gets most of the benefit without that risk.
- Mixed `ndarray`/`MaskedArray` inputs still use the first element's type;
  raising there would be a behavior change.
- The mutable global registry / `# pragma: no cover` guard is unchanged.

## 5. API parity gaps (partly done)
Done: `concatenate_along_batch` / `concatenate_along_seq` (both `array` and
`nested`) now accept a keyword-only `dtype`, forwarded to `cmpt.concatenate`.

Not done: adding `sum`, `std`, `var` to the computation models. As with #3,
`np.sum`/`np.std`/`np.var` already work on `MaskedArray` through the array's own
methods, so this is an API-surface choice, not a bug; it would touch the base
class, both models, `AutoComputationModel`, the interface and their tests.

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
