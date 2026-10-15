# Review of `src/batcharray`

Scope: structure, computation layer, and a sample of `array/` and `nested/`.
Tests and type checker were not run.

The package is clean and consistent (typed signatures, doctests, `py.typed`,
`__all__` everywhere). The suggestions below mostly reduce repetition and
tighten behavior.

## Suggested starting order

#7.
#1 dropped, #2 done, #3 not a bug, #4, #5 and #6 and #7 partly done, #9 and #10 done.

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

## 6. Input validation in `nested` (partly done)
Done: `to_list` docstring now says non-array items are returned unchanged
(matching the implementation); added a test.

Not done: a batch-size check. Checked behavior on `{"a": (3, 2), "b": (5,)}`:
`sum`, `cumsum` and `slice_along_batch` work fine per array, so a default check
would reject currently valid calls. Index-based ops (`permute`, `take`) already
fail with understandable `RuntimeError`/`IndexError` messages. If wanted, add an
opt-in helper rather than a default check.

## 7. Types (partly done)
Done:
- `SORT_KINDS` is now `typing.get_args(SortKind)` (an immutable tuple), so the
  literal is the single source of truth.
- The array `TypeVar` is defined once as `batcharray.types.ArrayT` and used by
  `computation/base.py`, `auto.py` and `interface.py` (was three copies of `T`).
  The unbounded `T` in `nested/indexing.py` is a different TypeVar and was kept.

Not done: a recursive `NestedArray` alias for the `nested` API. It changes the
public signatures of ~60 functions and needs per-function TypeVars to preserve
structure; worth a dedicated change once `ty` handles recursive aliases well.

## 8. Shadowed builtins in `nested/pointwise.py`

`abs` is defined with `# noqa: A001`. `from batcharray.nested import *` would
shadow the builtin; document it or keep it namespace-only.

## 9. `__init__.py` (done)
`array`, `computation` and `nested` are now lazily imported through a module
`__getattr__` (PEP 562), so `import batcharray; batcharray.array...` works while
`import batcharray` stays cheap. They are listed in `__all__` and imported under
`TYPE_CHECKING` for type checkers. Tests in `tests/unit/test_init.py`.

## 10. Performance and tests (done)
- Checked `nested.shuffle_along_*`: the permutation is drawn once from the first
  array and reused for every array. Added tests that shuffle several arrays
  (including a nested dict) and assert they stay aligned.
- Added axis-equivalence tests: every `array` `*_along_batch` / `*_along_seq`
  reduction equals the NumPy function with `axis=0` / `axis=1`, with and
  without `keepdims`, over several random inputs.
- Used seeded NumPy randomness instead of `hypothesis`, which is not a
  dependency; adding it would be a separate decision.
