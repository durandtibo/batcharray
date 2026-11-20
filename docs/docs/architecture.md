# Architecture

This page describes the layers of `batcharray`, the three helpers that every `nested` function uses
and how to add an operation. The rules that the layers implement are on the
[design principles](design_principles.md) page and are not repeated here.

## Layers

A module imports only from the layers above it in this list:

```text
constants          BATCH_AXIS, SEQ_AXIS
   ^
validation         validate_min_ndim, validate_split_sections, validate_same_keys
utils              types      NestedArray, T, SortKind
                   namespace  get_namespace
                   structure  apply_to_arrays, split_arrays, combine_arrays, get_size
   ^
array              one NumPy call with the axis fixed
   ^
nested             one helper call per function
```

!!! note "Current state"

    `utils` has `types` (`NestedArray`, `T`, `SortKind`), `namespace` (`get_namespace`) and
    `structure` (the three helpers and `get_size`), and every `nested` function goes through a
    helper. `nested.batch_size` and `nested.seq_len`, and the `layers` contract of `import-linter`,
    are not done yet. The steps in
    [`review/target_state.md`](https://github.com/durandtibo/batcharray/blob/main/review/target_state.md)
    bring the code to the layers above.

## A `nested` function is logic-free

A `nested` function calls one helper with the `array` function of the same name:

```python notest
def sum_along_batch(data: T, *, keepdims: bool = False) -> T:
    return apply_to_arrays(data, partial(ba.sum_along_batch, keepdims=keepdims))
```

Three shapes exist, and only three. A new operation uses one of them:

| Shape  | Helper                            | Used by                                                                  |
| ------ | --------------------------------- | ------------------------------------------------------------------------ |
| 1 -> 1 | `apply_to_arrays(data, fn)`       | reductions, indexing, slicing (`select`, `slice`), permutation, sorting, `tile`, `repeat` |
| 1 -> N | `split_arrays(data, fn, *, axis)` | `chunk_*`, `split_*`                                                     |
| N -> 1 | `combine_arrays(items, fn)`       | `concatenate_*`                                                          |

A function that does not fit one of these shapes is not a `nested` wrapper and does not belong in
`nested`. It is a helper of `utils.structure`, like `get_size`. `shuffle_*` is the one composition:
`get_size`, `rng.permutation` and `permute_*`, so the permutation is shared by all the leaves.

`apply_to_arrays(data, fn)` is also the tool for what the package does not wrap: a pointwise
function (`apply_to_arrays(data, np.exp)`), a conversion (`apply_to_arrays(data, np.ndarray.tolist)`)
or a cast.

## Adding an operation

See the admission rule in [CONTRIBUTING.md](https://github.com/durandtibo/batcharray/blob/main/CONTRIBUTING.md).
An operation that is accepted:

1. is added to `array` with its `_along_batch` and `_along_seq` versions, as one NumPy call with the
   axis fixed (or an entry of the list of exceptions on the
   [design principles](design_principles.md) page);
2. has a `nested` twin that uses one of the three helpers;
3. is exported in the `__init__.py` of its package and added to the API reference and the user
   guide;
4. has a docstring with a runnable example and a documented edge-case result, and a test for each
   property it touches.
