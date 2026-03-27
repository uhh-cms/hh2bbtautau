# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = []

from columnflow.columnar_util import (  # noqa: F401
    IF_DATA, IF_MC, IF_DATASET_HAS_TAG, IF_DATASET_NOT_HAS_TAG, EMPTY_FLOAT, ArrayFunction, deferred_column,
    ak_concatenate_safe,
)
from columnflow.util import maybe_import
from columnflow.types import Any, Sequence

np = maybe_import("numpy")
ak = maybe_import("awkward")


@deferred_column
def IF_NANO_V9(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version == 9:
        return self.get()
    return None


@deferred_column
def IF_NANO_V11(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version == 11:
        return self.get()
    return None


@deferred_column
def IF_NANO_V12(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version == 12:
        return self.get()
    return None


@deferred_column
def IF_NANO_V14(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version == 14:
        return self.get()
    return None


@deferred_column
def IF_NANO_V15(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version == 15:
        return self.get()
    return None


@deferred_column
def IF_NANO_GE_V10(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version >= 10:
        return self.get()
    return None


@deferred_column
def IF_NANO_GE_V14(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.version >= 14:
        return self.get()
    return None


@deferred_column
def IF_RUN_2(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 2:
        return self.get()
    return None


@deferred_column
def IF_RUN_3(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3:
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2022(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year == 2022:
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2022_PRE(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if (
        func.config_inst.campaign.x.run == 3 and
        func.config_inst.campaign.x.year == 2022 and
        func.config_inst.campaign.has_tag("preEE")
    ):
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2022_POST(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if (
        func.config_inst.campaign.x.run == 3 and
        func.config_inst.campaign.x.year == 2022 and
        func.config_inst.campaign.has_tag("postEE")
    ):
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2023(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year == 2023:
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2023_PRE(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if (
        func.config_inst.campaign.x.run == 3 and
        func.config_inst.campaign.x.year == 2023 and
        func.config_inst.campaign.has_tag("preBPix")
    ):
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2023_POST(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if (
        func.config_inst.campaign.x.run == 3 and
        func.config_inst.campaign.x.year == 2023 and
        func.config_inst.campaign.has_tag("postBPix")
    ):
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2024(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year == 2024:
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2022_2023(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year in {2022, 2023}:
        return self.get()
    return None


@deferred_column
def IF_RUN_3_2023_2024(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year in {2023, 2024}:
        return self.get()
    return None


IF_DATASET_HAS_LHE_WEIGHTS = IF_DATASET_NOT_HAS_TAG("no_lhe_weights")
IF_DATASET_HAS_TOP = IF_DATASET_HAS_TAG("has_top")
IF_DATASET_HAS_HIGGS = IF_DATASET_HAS_TAG("has_higgs")
IF_DATASET_IS_TT = IF_DATASET_HAS_TAG("ttbar")
IF_DATASET_IS_DY = IF_DATASET_HAS_TAG("dy")
IF_DATASET_IS_DY_MADGRAPH = IF_DATASET_HAS_TAG("dy_madgraph")
IF_DATASET_IS_DY_AMCATNLO = IF_DATASET_HAS_TAG("dy_amcatnlo")
IF_DATASET_IS_DY_POWHEG = IF_DATASET_HAS_TAG("dy_powheg")
IF_DATASET_IS_W_LNU = IF_DATASET_HAS_TAG("w_lnu")


@deferred_column
def IF_QUADJET_APPLIES_TO_DATASET(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if any(
        trigger_inst.applies_to_dataset(func.dataset_inst)
        for trigger_inst in func.config_inst.x.triggers
        if trigger_inst.has_tag("cross_quadjet")
    ):
        return self.get()
    return None


@deferred_column
def MET_COLUMN(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    met_name = func.config_inst.x("met_name", None)
    if not met_name:
        raise Exception("'met_name' has not been configured")
    return f"{met_name}.{self.get()}"


@deferred_column
def RAW_MET_COLUMN(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    raw_met_name = func.config_inst.x("raw_met_name", None)
    if not raw_met_name:
        raise Exception("'raw_met_name' has not been configured")
    return f"{raw_met_name}.{self.get()}"


def hash_events(arr: np.ndarray) -> np.ndarray:
    """
    Helper function to create a hash value from the event, run and luminosityBlock columns.
    The values are padded to specific lengths and concatenated to a single integer.
    """
    import awkward as ak

    def assert_value(arr: np.ndarray, field: str, max_value: int) -> None:
        """
        Helper function to check if a column does not exceed a maximum value.
        """
        digits = len(str(arr[field].to_numpy().max()))
        assert digits <= max_value, f"{field} digit count is {digits} and exceed max value {max_value}"

    max_digits_run = 6
    max_digits_luminosityBlock = 6
    max_digits_event = 8
    assert_value(arr, "run", max_digits_run)
    assert_value(arr, "luminosityBlock", max_digits_luminosityBlock)
    assert_value(arr, "event", max_digits_event)

    max_digits_hash = max_digits_event + max_digits_luminosityBlock + max_digits_run
    assert max_digits_hash <= 20, "sum of digits exceeds uint64"

    # upcast to uint64 to avoid overflow
    return (
        ak.values_astype(arr.run, np.uint64) * 10**(max_digits_luminosityBlock + max_digits_event) +
        ak.values_astype(arr.luminosityBlock, np.uint64) * 10**max_digits_event +
        ak.values_astype(arr.event, np.uint64)
    )


def with_type(type_name: str, data: ak.Array | dict[str, ak.Array], behavior: dict | None = None) -> ak.Array:
    """
    Attaches a named behavior *type_name* to the structured *data* and returns an array with that behavior. The source
    behavior is extracted from the *behavior* mapping, which is extracted from the first data column if not provided.

    :param type_name: The name of the type to attach.
    :param data: The structured data or array to attach the behavior to.
    :param behavior: The behavior to attach, defaults to the first data column's behavior.
    :return: Array with the specified behavior.
    """
    # extract the behavior from the first data column
    if behavior is None:
        behavior = next(iter(data.values())).behavior if isinstance(data, dict) else data.behavior
    return ak.Array(data, with_name=type_name, behavior=behavior)


def create_lvector_exyz(
    e: ak.Array,
    px: ak.Array,
    py: ak.Array,
    pz: ak.Array,
    depth_limit: int | None = None,
    behavior: dict | None = None,
) -> ak.Array:
    """
    Creates a Lorentz vector with the given energy and momentum components.

    :param e: Energy component.
    :param px: x-component of momentum.
    :param py: y-component of momentum.
    :param pz: z-component of momentum.
    :param depth_limit: Limit to pass to ak.zip.
    :param behavior: Behavior to attach to the resulting array.
    :return: Lorentz vector as an awkward array.
    """
    data = {
        "e": e,
        "px": px,
        "py": py,
        "pz": pz,
    }
    arr = ak.zip(data, depth_limit=depth_limit)
    return with_type("LorentzVector", arr, behavior=behavior)


def create_lvector_xyz(
    px: ak.Array,
    py: ak.Array,
    pz: ak.Array,
    depth_limit: int | None = None,
    behavior: dict | None = None,
) -> ak.Array:
    """
    Creates a Lorentz vector with the given momentum components and zero mass.

    :param px: x-component of momentum.
    :param py: y-component of momentum.
    :param pz: z-component of momentum.
    :return: Lorentz vector as an awkward array.
    """
    p = (px**2 + py**2 + pz**2)**0.5
    return create_lvector_exyz(p, px, py, pz, depth_limit=depth_limit, behavior=behavior)


def stack_lvectors(lvectors: Sequence[ak.Array]) -> ak.Array:
    """
    Stack multiple Lorentz vectors.

    :param lvectors: Sequence of Lorentz vectors to stack.
    :return: Stacked Lorentz vectors.
    """
    # convert to same lvector types, add new dimension if necessary
    lvectors = [
        lv if lv.ndim == 2 else lv[:, None]
        for lv in (create_lvector_exyz(lv.e, lv.px, lv.py, lv.pz) for lv in lvectors)
    ]

    # concatenate
    stacked = ak_concatenate_safe(lvectors, axis=1)

    return stacked


def rotate_px_py(
    px: ak.Array | np.ndarray,
    py: ak.Array | np.ndarray,
    ref_phi: ak.Array | np.ndarray,
) -> ak.Array | np.ndarray:
    new_phi = np.arctan2(py, px) + ref_phi  # mind the "+"
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


def delta_r12(vectors: ak.Array) -> ak.Array:
    # delta r between first two elements
    dr = ak.firsts(vectors[:, :1], axis=1).delta_r(ak.firsts(vectors[:, 1:2], axis=1))
    return ak.fill_none(dr, EMPTY_FLOAT)


_uppercase_wps = {
    "vvvvloose": "VVVVLoose",
    "vvvloose": "VVVLoose",
    "vvloose": "VVLoose",
    "vloose": "VLoose",
    "loose": "Loose",
    "medium": "Medium",
    "tight": "Tight",
    "vtight": "VTight",
    "vvtight": "VVTight",
    "vvvtight": "VVVTight",
    "vvvvtight": "VVVVTight",
}


def uppercase_wp(wp: str) -> str:
    """
    Converts a working point string to uppercase format.

    :param wp: Working point string.
    :return: Uppercase working point string.
    """
    wp = wp.lower()
    if wp not in _uppercase_wps:
        raise ValueError(f"unknown working point for uppercase conversion: {wp}")
    return _uppercase_wps[wp]
