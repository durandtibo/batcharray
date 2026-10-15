# Review of `src/batcharray`

Scope: every module in `src/batcharray` was read (code, not each doctest), and
the behaviors below were checked by running small probes. The unit tests,
doctests and `ruff` pass. The type checker was not run.

The package is small, consistent and well documented (typed signatures,
doctests, `__all__` everywhere, lazy submodule imports). Findings are ordered
by value. Each section says what was done and what was deliberately left.

## 1. Bugs and sharp edges (fixed)

- `tile_along_seq`, `shuffle_along_seq` and `permute_along_seq` raise a clear
  `ValueError` on arrays with fewer than 2 dimensions.
- `split_along_*` validate the split size / sections (positive int, or
  non-negative sections summing to the axis size) and raise `ValueError`.
- `nested.concatenate_*` raise a `ValueError` when the dicts have different
  keys instead of dropping keys or raising `KeyError`.
- `nested.shuffle_*` raise a `ValueError` for empty data or a non-array first
  leaf instead of `StopIteration`/`AttributeError`.
- `MaskedArrayComputationModel.concatenate` uses `dtype is not None` and
  `astype`, which keeps the mask. `fill_value` is dropped by `np.ma.concatenate`
  itself (even without `dtype`), so it is not fixable here.
- `permute_*` raise `ValueError` instead of `RuntimeError` on a shape mismatch
  (a breaking change; note it in the changelog).
- `nested.chunk_*`, `nested.split_*` and `nested.concatenate_*` now support
  nested structures (dicts, lists, tuples of arrays), through the
  helpers in `nested/structure.py`. Non-array leaves raise a `TypeError`,
  mismatched keys / sequence lengths a `ValueError`. The now unused
  `validate_flat_dict_of_arrays` was removed.

## 2. API consistency (done)

- `keepdims` is keyword-only in all the new reductions. In the existing
  `array`/`nested` reductions it is now keyword-only too, with a deprecation
  period: passing it positionally still works but emits a `DeprecationWarning`
  (`batcharray/_deprecation.py`). Remove the shim in a later release.
- `amax_*`/`amin_*` go through the computation models like `max_*`/`min_*`
  (verified equal by `tests/unit/test_api_consistency.py`).
- `index_select_*` is documented as an alias of `take_*`, and the accepted
  `indices` shapes are documented (1-D for the batch axis, 1-D or 2-D for the
  sequence axis).
- `chunk_*` documents that it follows `np.array_split` semantics.
- `nested.clip` passes the bounds positionally to `np.clip`, so it no longer
  depends on the `a_min`/`a_max` keyword spelling (renamed in NumPy 2.1). The
  public parameter names are unchanged.

## 3. Features (done)

- Computation models: `sum`, `prod`, `std` and `var` (with `ddof`) on the base
  class, both models, `AutoComputationModel` and the `computation` interface.
  This adds abstract methods, so third-party `BaseComputationModel` subclasses
  must implement them (breaking). `array.mean/sum/prod/std/var_*` now go
  through the models.
- New reductions: `std_*`, `var_*`, `any_*`, `all_*` (`array` and `nested`).
- New pointwise functions (`nested`): `sqrt`, `square`, `floor`, `ceil`,
  `round`, `sign`, `isnan`.
- New `flip_along_*` and `repeat_along_*` (`array` and `nested`).
- Opt-in batch-size checks: `validation.validate_same_batch_size` and
  `validate_same_seq_size`. `nested` does not call them by default, because a
  default check would reject valid calls (e.g. `sum` on arrays with different
  batch sizes).

## 4. `AutoComputationModel` (`computation/auto.py`)

- Mixed `ndarray`/`MaskedArray` inputs to `concatenate` pick the model from
  the first element. This is documented: a masked array first keeps the masks,
  but a plain array first silently drops the masks of the following masked
  arrays. Raising on mixed types was not chosen, to avoid a breaking change.
- The registry is a mutable class-level dict that tests mutate with
  `patch.dict`; a lookup cache would go stale. The exact-type fast path already
  covers most of the benefit. Left as is.
- The `register_computation_models` guard is tested by clearing the registry,
  and its `# pragma: no cover` was removed.

## 5. Typing

- `nested/indexing.py`: the unused `T = TypeVar("T")` was removed.
- `computation.max`/`min`/`sum` and `nested.abs`/`round` shadow builtins
  (`# noqa: A001`/`A004`). This is documented in the `nested` and
  `computation` module docstrings: use them through the namespace and avoid
  `import *`.
- Not done: the `nested` API is typed with `Any`. A recursive `NestedArray`
  alias would change ~60 signatures and needs per-function TypeVars to
  preserve structure. Worth doing once `ty` handles recursive aliases well.

## 6. Repository hygiene (nothing to change)

- `coverage.xml` and `htmlcov/` are git-ignored and the sdist only includes
  `src` (`only-include = ["src"]`), so they cannot leak into a build.
- The CI dependency matrix comes from `dev/config/package_versions.json` and
  starts at numpy `1.22.4`, which matches the declared `numpy>=1.22`.

## 7. Docstring duplication (not done)

Each operation exists in four variants (`array`/`nested` x batch/seq). The
bodies are one-line delegations, so a shared helper would not help; the
duplication is in docstrings/doctests. Only revisit if maintaining them
becomes costly.
