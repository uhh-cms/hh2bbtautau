# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = []

from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@deferred_column
def IF_DATA(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.dataset_inst.is_data else None


@deferred_column
def IF_MC(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.dataset_inst.is_mc else None


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
def IF_NANO_V14(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 14 else None


@deferred_column
def IF_NANO_GE_V10(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version >= 10 else None


@deferred_column
def IF_RUN_2(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 2 else None


@deferred_column
def IF_RUN_3(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 3 else None


@deferred_column
def IF_RUN_3_2022(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if (func.config_inst.campaign.x.run == 3 and func.config_inst.campaign.x.year == 2022) else None


@deferred_column
def IF_DATASET_HAS_LHE_WEIGHTS(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if not func.dataset_inst.has_tag("no_lhe_weights") else None


@deferred_column
def IF_DATASET_HAS_TOP(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("has_top") else None


@deferred_column
def IF_DATASET_IS_TT(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("ttbar") else None


@deferred_column
def IF_DATASET_IS_DY(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("dy") else None


@deferred_column
def IF_DATASET_IS_DY_MADGRAPH(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("dy_madgraph") else None


@deferred_column
def IF_DATASET_IS_DY_AMCATNLO(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("dy_amcatnlo") else None


@deferred_column
def IF_DATASET_IS_DY_POWHEG(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("dy_powheg") else None


@deferred_column
def IF_DATASET_IS_W_LNU(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    return self.get() if func.dataset_inst.has_tag("w_lnu") else None


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


def with_type(type_name: str, data: dict[str, ak.Array], behavior: dict | None = None) -> ak.Array:
    """
    Attaches a named behavior *type_name* to the structured *data* and returns an array with that behavior. The source
    behavior is extracted from the *behavior* mapping, which is extracted from the first data column if not provided.

    :param type_name: The name of the type to attach.
    :param data: The structured data to attach the behavior to.
    :param behavior: The behavior to attach, defaults to the first data column's behavior.
    :return: Array with the specified behavior.
    """
    # extract the behavior from the first data column
    if behavior is None:
        behavior = next(iter(data.values())).behavior
    return ak.Array(data, with_name=type_name, behavior=behavior)


def create_lvector_exyz(e: ak.Array, px: ak.Array, py: ak.Array, pz: ak.Array, behavior: dict | None = None) -> ak.Array:
    """
    Creates a Lorentz vector with the given energy and momentum components.

    :param e: Energy component.
    :param px: x-component of momentum.
    :param py: y-component of momentum.
    :param pz: z-component of momentum.
    :return: Lorentz vector as an awkward array.
    """
    data = {
        "e": e,
        "px": px,
        "py": py,
        "pz": pz,
    }
    return with_type("PtEtaPhiMLorentzVector", data, behavior=behavior)


def create_lvector_xyz(px: ak.Array, py: ak.Array, pz: ak.Array, behavior: dict | None = None) -> ak.Array:
    """
    Creates a Lorentz vector with the given momentum components and zero mass.

    :param px: x-component of momentum.
    :param py: y-component of momentum.
    :param pz: z-component of momentum.
    :return: Lorentz vector as an awkward array.
    """
    p = (px**2 + py**2 + pz**2)**0.5
    return create_lvector_exyz(p, px, py, pz, behavior=behavior)
