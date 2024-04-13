from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from batcharray import computation as cmpt

#######################
#     concatenate     #
#######################


def test_concatenate_array_axis_0() -> None:
    out = cmpt.concatenate(
        [np.array([[0, 1, 2], [4, 5, 6]]), np.array([[10, 11, 12], [13, 14, 15]])], axis=0
    )
    assert objects_are_equal(out, np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


def test_concatenate_masked_array_axis_0() -> None:
    out = cmpt.concatenate(
        [
            np.ma.masked_array(
                data=np.array([[0, 1, 2], [4, 5, 6]]),
                mask=np.array([[False, False, False], [False, True, False]]),
            ),
            np.ma.masked_array(
                data=np.array([[10, 11, 12], [13, 14, 15]]),
                mask=np.array([[False, False, True], [False, False, False]]),
            ),
        ],
        axis=0,
    )
    assert objects_are_equal(
        out,
        np.ma.masked_array(
            data=np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
            mask=np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ]
            ),
        ),
    )
