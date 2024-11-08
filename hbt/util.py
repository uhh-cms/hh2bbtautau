# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = []

from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column


@deferred_column
def IF_NANO_V9(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 9 else None


@deferred_column
def IF_NANO_V11(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 11 else None


@deferred_column
def IF_NANO_V12(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 12 else None


@deferred_column
def IF_RUN_2(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 2 else None


@deferred_column
def IF_RUN_3(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 3 else None


@deferred_column
def IF_DATASET_HAS_LHE_WEIGHTS(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return None if func.dataset_inst.has_tag("no_lhe_weights") else self.get()


@deferred_column
def IF_DATASET_IS_DY(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("is_dy") else None


def hash_events(arr: np.ndarray) -> np.ndarray:
    """
    Helper function to create a hash value from the event, run and luminosityBlock columns.
    The values are padded to specific lengths and concatenated to a single integer.
    """
    import awkward as ak
    import numpy as np

    def assert_value(arr: np.ndarray, field: str, max_value: int) -> None:
        """
        Helper function to check if a column does not exceed a maximum value.
        """
        digits = len(str(arr[field].to_numpy().max()))
        assert digits <= max_value, f"{field} digit count is {digits} and exceed max value {max_value}"

    max_digits_run = 6
    max_digits_luminosityBlock = 5
    max_digits_event = 7

    assert_value(arr, "run", max_digits_run)
    assert_value(arr, "luminosityBlock", max_digits_luminosityBlock)
    assert_value(arr, "event", max_digits_event)
    # upcast to int64 to avoid overflow
    arr = ak.values_astype(arr, np.int64)
    return (
        arr.event * 10**(max_digits_luminosityBlock + max_digits_run) +
        arr.luminosityBlock * 10**max_digits_run +
        arr.run
    )
