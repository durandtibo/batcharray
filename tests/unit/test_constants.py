from __future__ import annotations

from batcharray.constants import BATCH_AXIS, SEQ_AXIS

#################################
#     Tests for BATCH_AXIS      #
#################################


def test_batch_axis_value() -> None:
    assert BATCH_AXIS == 0


def test_batch_axis_type() -> None:
    assert isinstance(BATCH_AXIS, int)


###############################
#     Tests for SEQ_AXIS      #
###############################


def test_seq_axis_value() -> None:
    assert SEQ_AXIS == 1


def test_seq_axis_type() -> None:
    assert isinstance(SEQ_AXIS, int)


def test_batch_seq_axes_different() -> None:
    assert BATCH_AXIS != SEQ_AXIS
