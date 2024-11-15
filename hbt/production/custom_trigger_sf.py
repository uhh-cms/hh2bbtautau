# coding: utf-8

"""
Custom trigger scale factor production.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from colmnflow.production.cms.muon import muon_weights
from columnflow.production.cms.electron import electron_weights

from hbt.production.tau import tau_trigger_weights
# TODO: check if tau_trigger_weights really do what we want, looks like way too much for me, maybe need
# to recreate it and remove some stuff

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# subclass the electron weights producer to create the electron efficiencies
single_electron_trigger_data_effs = electron_weights.derive(
    "single_electron_trigger_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_electron_trigger_data_effs_names),
        "weight_name": "single_electron_trigger_data_effs",
    },
)

single_electron_trigger_mc_effs = electron_weights.derive(
    "single_electron_trigger_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_electron_trigger_mc_effs_names),
        "weight_name": "single_electron_trigger_mc_effs",
    },
)

cross_electron_trigger_data_effs = electron_weights.derive(
    "cross_electron_trigger_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_electron_trigger_data_effs_names),
        "weight_name": "cross_electron_trigger_data_effs",
    },
)

cross_electron_trigger_mc_effs = electron_weights.derive(
    "cross_electron_trigger_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_electron_trigger_mc_effs_names),
        "weight_name": "cross_electron_trigger_mc_effs",
    },
)

# subclass the tau weights producer to create the tau efficiencies
cross_etau_trigger_data_effs = tau_trigger_weights.derive(
    "cross_etau_trigger_data_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_etau_trigger_data_effs_names),
        "weight_name": "cross_etau_trigger_data_effs",
    },
)

cross_etau_trigger_mc_effs = tau_trigger_weights.derive(
    "cross_etau_trigger_mc_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_etau_trigger_mc_effs_names),
        "weight_name": "cross_etau_trigger_mc_effs",
    },
)

# subclass the muon weights producer to create the muon efficiencies
single_muon_trigger_data_effs = muon_weights.derive(
    "single_muon_trigger_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_muon_trigger_data_effs_names),
        "weight_name": "single_muon_trigger_data_effs",
    },
)

single_muon_trigger_mc_effs = muon_weights.derive(
    "single_muon_trigger_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_muon_trigger_mc_effs_names),
        "weight_name": "single_muon_trigger_mc_effs",
    },
)

cross_muon_trigger_data_effs = muon_weights.derive(
    "cross_muon_trigger_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_muon_trigger_data_effs_names),
        "weight_name": "cross_muon_trigger_data_effs",
    },
)

cross_muon_trigger_mc_effs = muon_weights.derive(
    "cross_muon_trigger_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_muon_trigger_mc_effs_names),
        "weight_name": "cross_muon_trigger_mc_effs",
    },
)

# subclass the tau weights producer to create the tau efficiencies
cross_mutau_trigger_data_effs = tau_trigger_weights.derive(
    "cross_mutau_trigger_data_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_mutau_trigger_data_effs_names),
        "weight_name": "cross_mutau_trigger_data_effs",
    },
)

cross_mutau_trigger_mc_effs = tau_trigger_weights.derive(
    "cross_mutau_trigger_mc_effs",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_config": (lambda self: self.config_inst.x.cross_mutau_trigger_mc_effs_names),
        "weight_name": "cross_mutau_trigger_mc_effs",
    },
)


def reshape_masked_to_oneslike_original(masked_array: ak.Array, mask: ak.Array) -> ak.Array:
    """
    Reshape a masked array to a numpy.ones_like array of the original shape.
    """
    oneslike_original = np.ones_like(mask)
    oneslike_original[mask] = masked_array
    return oneslike_original


def calculate_ditrigger_efficiency(
    single_triggered,
    cross_triggered,
    single_trigger_effs,
    cross_trigger_effs,
    cross_tau_trigger_effs_nom,
) -> ak.Array:
    """
    Calculate the combination of the single and cross trigger efficiencies.
    """

    # compute the trigger weights
    trigger_efficiency = (
        (single_trigger_effs * single_triggered) +
        (cross_tau_trigger_effs_nom * cross_trigger_effs * cross_triggered) -
        (
            single_triggered *
            cross_triggered *
            cross_tau_trigger_effs_nom *
            np.minimum(
                single_trigger_effs,
                cross_trigger_effs,
            )
        )
    )
    return trigger_efficiency


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",  # "trigger_ids"
        single_electron_trigger_data_effs, cross_electron_trigger_data_effs,
        single_electron_trigger_mc_effs, cross_electron_trigger_mc_effs,
        cross_etau_trigger_data_effs, cross_etau_trigger_mc_effs,
        single_muon_trigger_data_effs, cross_muon_trigger_data_effs,
        single_muon_trigger_mc_effs, cross_muon_trigger_mc_effs,
        cross_mutau_trigger_data_effs, cross_mutau_trigger_mc_effs,
    },
    produces={
        "etau_trigger_weight", "mutau_trigger_weight",
    } | {
        f"mutau_trigger_weight_{direction}"
        for direction in ["up", "down"]
    } | {
        f"etau_trigger_weight_{direction}"
        for direction in ["up", "down"]
    },
)
def build_etau_mutau_trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for muon trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``muon_trigger_sf`` and ``cross_muon_trigger_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
            "cross_muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
        })
    """
    from IPython import embed; embed(header="trigger scale factors")

    # TODO: TOCHECK: if necessary, make it an object mask using e.g. "& events.Muon.pt > 0.0"
    single_electron_triggered = (events.channel_id == self.config_inst.channels.n.etau.id) & events.single_triggered
    single_muon_triggered = (events.channel_id == self.config_inst.channels.n.mutau.id) & events.single_triggered
    cross_electron_triggered = (events.channel_id == self.config_inst.channels.n.etau.id) & events.cross_triggered
    cross_muon_triggered = (events.channel_id == self.config_inst.channels.n.mutau.id) & events.cross_triggered

    # get efficiencies from the correctionlib producers
    # TODO: check if it works with event level mask (creation of new column with less entries?)
    # else use object level mask and use the mulitplication in calculate_ditrigger_efficiency to make it
    # a 1d array in a new column
    events = self[single_muon_trigger_data_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_muon_trigger_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[cross_mutau_trigger_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_electron_trigger_data_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_electron_trigger_data_effs](events, cross_electron_triggered, **kwargs)
    events = self[cross_etau_trigger_data_effs](events, cross_electron_triggered, **kwargs)
    single_muon_trigger_data_efficiencies = events.single_muon_trigger_data_effs(
        events,
        single_muon_triggered,
        **kwargs,
    )
    cross_muon_trigger_data_efficiencies = events.cross_muon_trigger_data_effs(events, cross_muon_triggered, **kwargs)
    cross_mutau_trigger_data_efficiencies = events.cross_mutau_trigger_data_effs(events, cross_muon_triggered, **kwargs)

    # compute the trigger weights
    muon_trigger_efficiency_data = calculate_ditrigger_efficiency(
        single_muon_triggered,
        cross_muon_triggered,
        single_muon_trigger_data_efficiencies,
        cross_muon_trigger_data_efficiencies,
        cross_mutau_trigger_data_efficiencies,
    )

    # TODO: same for MC

    # TODO: calculate SFs

    # TODO: same for electrons

    return events
