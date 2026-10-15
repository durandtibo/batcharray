# Review of `src/batcharray`

Scope: every module in `src/batcharray` was read (code, not each doctest), and
the behaviors below were checked by running small probes. Tests, docs and CI
were only skimmed (file layout). The test suite and type checker were not run.

The package is small, consistent and well documented (typed signatures,
doctests, `__all__` everywhere, lazy submodule imports). Items from the previous
review that are done, dropped or not bugs were removed. Findings are ordered by
value.

## 1. Bugs and sharp edges (verified)

1. **`tile_along_seq` on a 1-D array** raises a bare `IndexError: list
   assignment index out of range` (`array/joining.py`, `repeats[1] = reps`).
   Raise a clear `ValueError` (the array needs a sequence axis), as for any
   `*_along_seq` function on a 1-D input (`shuffle_along_seq` also raises
   `IndexError: tuple index out of range`).
2. **`split_along_batch` / `split_along_seq` do not validate sections.**
   With `[1, 1]` on 4 rows, or `[3, 3]`, the call silently returns two uneven
   chunks instead of failing. Check `sum(sections) == shape[axis]` (and
   non-negative sizes).
3. **`nested.chunk_*` / `nested.split_*` only support a flat dict of arrays.**
   They are annotated `dict[Hashable, np.ndarray]`, unlike the rest of `nested`
   which takes any nested structure. A nested dict gives `AttributeError: 'dict'
   object has no attribute 'shape'` and a list gives `no attribute 'keys'`. Either
   document the flat-dict restriction or implement it recursively (split every
   leaf, then rebuild the structure per chunk).
4. **`nested.concatenate_*` silently drops keys.** Keys come from the first
   item only: extra keys in later items are ignored, missing keys give a bare
   `KeyError`. Same flat-dict limitation as above (a nested dict fails with
   `TypeError: Incorrect array type: dict`). Validate that all items have the
   same keys and raise a descriptive `ValueError`.
5. **`nested.shuffle_*` on odd input.** `next(dfs_iterate(data))` raises a bare
   `StopIteration` on empty data (`{}`) and `AttributeError` when the first leaf
   is not an array. Validate and raise `ValueError`/`TypeError`.
6. **`MaskedArrayComputationModel.concatenate`** (`computation/masked_array.py`):
   `if dtype:` should be `if dtype is not None:`. When `dtype` is given, the
   result is rebuilt from `.data`/`.mask`, so `fill_value` and hard-mask are
   lost; without `dtype` the fill value also falls back to the default. Use
   `out.astype(dtype)` instead of rebuilding.
7. **Error type inconsistency.** `permute_*` raises `RuntimeError` for a shape
   mismatch, where `ValueError` is the conventional type. Changing it is a
   (small) breaking change; mention in the changelog if done.

## 2. API consistency

- `keepdims` is a positional parameter in `array`/`nested` reductions but
  keyword-only in the computation layer. Making it keyword-only in the public
  functions avoids `f(x, True)` misuse (breaking; do it with a deprecation).
- `amax_*`/`amin_*` are behaviorally equal to `max_*`/`min_*` but go through
  NumPy directly while `max_*`/`min_*` go through the computation models.
  Either document them as aliases or route them identically.
- `index_select_*` is a plain alias of `take_*` (same signature), and
  `take_along_seq` accepts 1-D or 2-D indices while `take_along_batch` takes
  any shape (a 2-D index silently yields an extra dimension). Document accepted
  index shapes.
- `chunk_*` uses `np.array_split` (uneven sizes, like `torch.tensor_split`),
  not `torch.chunk` semantics, which the name suggests. State it in the
  docstring.
- `nested.clip` uses the `a_min`/`a_max` keywords; NumPy 2.1 renamed them to
  `min`/`max`. It works today (no warning raised in the probe), but the
  parameter names are tied to the legacy spelling.

## 3. Features worth considering

- Computation models: `sum`, `std`, `var`, `prod`. `np.sum`/`np.std`/`np.var`
  already work on `MaskedArray` through the array's own methods, so this is an
  API-surface choice, not a bug; it touches the base class, both models,
  `AutoComputationModel`, the interface and tests.
- Reductions and pointwise ops that users of batch/sequence data commonly want:
  `std`/`var`, `any`/`all`, `sqrt`, `square`, `floor`/`ceil`/`round`, `sign`,
  `isnan`, and `flip_along_*`/`repeat_along_*`.
- An opt-in batch-size consistency check for `nested` (a default check would
  reject valid calls, e.g. `sum` on arrays with different batch sizes).

## 4. `AutoComputationModel` (`computation/auto.py`)

- Mixed `ndarray`/`MaskedArray` inputs to `concatenate` pick the model from the
  first element. Document this, or raise on mixed types.
- The registry is a mutable class-level dict that tests mutate with
  `patch.dict`; a lookup cache would go stale. The exact-type fast path already
  covers most of the benefit. Leave as is.
- `register_computation_models` guards with `# pragma: no cover`; the guard
  could be tested by clearing the registry in a test.

## 5. Typing

- The `nested` API is typed with `Any`. A recursive `NestedArray` alias would
  change ~60 signatures and needs per-function TypeVars to preserve structure;
  worth doing once `ty` handles recursive aliases well.
- `nested/indexing.py` defines an unused `T = TypeVar("T")`; remove it.
- `computation.max`/`min` and `nested.abs` shadow builtins (`# noqa: A001`/
  `A004`). `from batcharray.nested import *` would shadow `abs`; document it or
  keep these namespace-only.

## 6. Repository hygiene

- `coverage.xml` and `htmlcov/` are git-ignored; keep them out of any
  sdist/docs build.
- CI already tests several numpy versions (`ci-test-deps.yaml`); just confirm
  the matrix includes the declared minimum, `numpy>=1.22`.

## 7. Docstring duplication (optional)

Each operation exists in four variants (`array`/`nested` x batch/seq). The
bodies are one-line delegations, so a shared helper would not help; the
duplication is in docstrings/doctests. Only revisit if maintaining them becomes
costly.
