# Review of `src/batcharray`

Scope: every module in `src/batcharray` was read (code, not each doctest), and
the behaviors below were checked by running small probes. Tests, docs and CI
were only skimmed (file layout). The test suite and type checker were not run.

The package is small, consistent and well documented (typed signatures,
doctests, `__all__` everywhere, lazy submodule imports). Items from the previous
review that are done, dropped or not bugs were removed. Findings are ordered by
value.

## 1. Bugs and sharp edges (fixed)

All verified issues from the previous pass are fixed, with tests:

- `tile_along_seq`, `shuffle_along_seq` and `permute_along_seq` raise a clear
  `ValueError` on arrays with fewer than 2 dimensions.
- `split_along_*` validate the split size / sections (positive int, or
  non-negative sections summing to the axis size) and raise `ValueError`.
- `nested.chunk_*` / `nested.split_*` raise a `TypeError` for anything but a
  flat dict of arrays instead of a confusing `AttributeError`.
- `nested.concatenate_*` raise a `ValueError` when the dicts have different
  keys instead of dropping keys or raising `KeyError`.
- `nested.shuffle_*` raise a `ValueError` for empty data or a non-array first
  leaf instead of `StopIteration`/`AttributeError`.
- `MaskedArrayComputationModel.concatenate` uses `dtype is not None` and
  `astype`, which keeps the mask. `fill_value` is dropped by `np.ma.concatenate`
  itself (even without `dtype`), so it is not fixable here.
- `permute_*` raise `ValueError` instead of `RuntimeError` on a shape mismatch
  (a breaking change; note it in the changelog).

Still open from this group: supporting nested structures in `nested.chunk_*` /
`nested.split_*` / `nested.concatenate_*`; they now fail clearly instead.

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
