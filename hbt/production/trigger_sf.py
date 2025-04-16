# coding: utf-8

"""
Custom trigger scale factor production.

Note : The trigger weights producers multiply the sfs for all objects in an event to get the total
trigger scale factor of the event. Since we might want to use different objects in different channels,
we will derive the trigger weights producers for each channel separately to apply the correct masks.
"""

import functools

import order as od

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route
from columnflow.production.cms.muon import muon_trigger_weights as cf_muon_trigger_weights
from columnflow.production.cms.electron import electron_trigger_weights as cf_electron_trigger_weights

from hbt.production.tau import tau_trigger_weights
from hbt.production.jet import jet_trigger_effs

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# subclass the electron trigger weights producer to create the electron trigger weights
electron_trigger_weights = cf_electron_trigger_weights.derive(
    "electron_trigger_weights",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.electron),
    },
)
muon_trigger_weights = cf_muon_trigger_weights.derive(
    "muon_trigger_weights",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.muon),
    },
)

# subclass the electron weights producer to create the electron efficiencies
single_trigger_electron_data_effs = electron_trigger_weights.derive(
    "single_trigger_electron_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.electron),
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_data_effs_cfg),
        "weight_name": "single_trigger_electron_data_effs",
    },
)

single_trigger_electron_mc_effs = electron_trigger_weights.derive(
    "single_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.electron),
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_mc_effs_cfg),
        "weight_name": "single_trigger_electron_mc_effs",
    },
)

cross_trigger_electron_data_effs = electron_trigger_weights.derive(
    "cross_trigger_electron_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.cross_electron),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_data_effs_cfg),
        "weight_name": "cross_trigger_electron_data_effs",
    },
)

cross_trigger_electron_mc_effs = electron_trigger_weights.derive(
    "cross_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.cross_electron),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_mc_effs_cfg),
        "weight_name": "cross_trigger_electron_mc_effs",
    },
)

# subclass the muon weights producer to create the muon efficiencies
single_trigger_muon_data_effs = muon_trigger_weights.derive(
    "single_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.muon),
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_data_effs_cfg),
        "weight_name": "single_trigger_muon_data_effs",
    },
)

single_trigger_muon_mc_effs = muon_trigger_weights.derive(
    "single_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.muon),
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_mc_effs_cfg),
        "weight_name": "single_trigger_muon_mc_effs",
    },
)

cross_trigger_muon_data_effs = muon_trigger_weights.derive(
    "cross_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_data_effs_cfg),
        "weight_name": "cross_trigger_muon_data_effs",
    },
)

cross_trigger_muon_mc_effs = muon_trigger_weights.derive(
    "cross_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_mc_effs_cfg),
        "weight_name": "cross_trigger_muon_mc_effs",
    },
)

# subclass the tau weights producer to use the cclub tau efficiencies
tau_trigger_sf_and_effs_cclub = tau_trigger_weights.derive(
    "tau_trigger_sf_and_effs_cclub",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.trigger_sf.tau),
        "get_tau_corrector": (lambda self: self.config_inst.x.tau_trigger_corrector_cclub),
    },
)


def reshape_masked_to_oneslike_original(masked_array: ak.Array, mask: ak.Array) -> ak.Array:
    """
    Reshape a masked array to a numpy.ones_like array of the original shape.
    """
    oneslike_original = np.ones_like(mask)
    oneslike_original[mask] = masked_array
    return oneslike_original


def calculate_correlated_ditrigger_efficiency(
    first_trigger_matched,
    second_trigger_matched,
    first_trigger_effs,
    second_trigger_common_object_effs,
    second_trigger_other_object_effs,
) -> ak.Array:
    """
    Calculate the combination of the single and cross trigger efficiencies.
    """

    # compute the trigger weights
    trigger_efficiency = (
        (first_trigger_effs * first_trigger_matched) +
        (second_trigger_other_object_effs * second_trigger_common_object_effs * second_trigger_matched) -
        (
            first_trigger_matched *
            second_trigger_matched *
            second_trigger_other_object_effs *
            np.minimum(
                first_trigger_effs,
                second_trigger_common_object_effs,
            )
        )
    )
    return trigger_efficiency


def create_trigger_weights(
    events: ak.Array,
    first_trigger_eff_data: ak.Array,
    first_trigger_eff_mc: ak.Array,
    second_trigger_common_object_eff_data: ak.Array,
    second_trigger_common_object_eff_mc: ak.Array,
    second_trigger_other_object_eff_data: ak.Array,
    second_trigger_other_object_eff_mc: ak.Array,
    channel: od.Channel,
    single_triggered: ak.Array,
    cross_triggered: ak.Array,
    postfix: str,
) -> ak.Array:
    """
    Create the trigger weights for a given channel.
    """

    trigger_efficiency_data = calculate_correlated_ditrigger_efficiency(
        single_triggered,
        cross_triggered,
        first_trigger_eff_data,
        second_trigger_common_object_eff_data,
        second_trigger_other_object_eff_data,
    )

    trigger_efficiency_mc = calculate_correlated_ditrigger_efficiency(
        single_triggered,
        cross_triggered,
        first_trigger_eff_mc,
        second_trigger_common_object_eff_mc,
        second_trigger_other_object_eff_mc,

    )

    # calculate SFs
    trigger_sf = trigger_efficiency_data / trigger_efficiency_mc

    # nan happens for all events not in the specific channel, due to efficiency == 0
    # add a failsafe here in case of efficiency 0 for an event actually in the channel
    nan_mask = np.isnan(trigger_sf)
    if np.any(nan_mask & (events.channel_id == channel.id) & (single_triggered | cross_triggered)):
        raise ValueError(f"Found nan in {channel.name} trigger weights")
    trigger_sf_no_nan = np.nan_to_num(trigger_sf, nan=1.0)

    return set_ak_column_f32(events, f"{channel.name}_trigger_weights{postfix}", trigger_sf_no_nan)


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",  # "matched_trigger_ids"
        single_trigger_electron_data_effs, cross_trigger_electron_data_effs,
        single_trigger_electron_mc_effs, cross_trigger_electron_mc_effs,
        single_trigger_muon_data_effs, cross_trigger_muon_data_effs,
        single_trigger_muon_mc_effs, cross_trigger_muon_mc_effs,
        tau_trigger_sf_and_effs_cclub,
    },
    produces={
        "{e,mu}tau_trigger_weights",
        "{e,mu}tau_trigger_weights_{e,mu,jet}_{up,down}",
        "{e,mu}tau_trigger_weights_tau_dm{0,1,10,11}_{up,down}",
    },
)
def etau_mutau_trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for mutau and etau trigger scale factors derived by Jona Motta. Requires several external files and
    SF configs in the analysis config, e.g.:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
            "cross_trigger_muon_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
        })
    """

    single_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        events.single_triggered &
        (ak.local_index(events.Electron.pt) == 0)
    )
    single_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        events.single_triggered &
        (ak.local_index(events.Muon.pt) == 0)
    )
    cross_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        events.cross_triggered &
        (ak.local_index(events.Electron.pt) == 0)
    )
    cross_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        events.cross_triggered &
        (ak.local_index(events.Muon.pt) == 0)
    )

    # get efficiencies from the correctionlib producers

    # first, create the efficiencies for the leptons in data
    # create the columns
    events = self[single_trigger_muon_data_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_trigger_muon_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_trigger_electron_data_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_trigger_electron_data_effs](events, cross_electron_triggered, **kwargs)

    # do the same for MC efficiencies
    # create the columns
    events = self[single_trigger_muon_mc_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_trigger_muon_mc_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_trigger_electron_mc_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_trigger_electron_mc_effs](events, cross_electron_triggered, **kwargs)

    # create all tau weights
    events = self[tau_trigger_sf_and_effs_cclub](events, **kwargs)

    for postfix in ["", "_up", "_down"]:
        # create the nominal case
        if postfix == "":
            for channel_name in ["etau", "mutau"]:
                if channel_name == "etau":
                    channel = self.config_inst.channels.n.etau
                    single_trigger_lepton_data_efficiencies = events.single_trigger_electron_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_electron_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_electron_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_electron_mc_effs
                else:
                    channel = self.config_inst.channels.n.mutau
                    single_trigger_lepton_data_efficiencies = events.single_trigger_muon_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_muon_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_muon_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_muon_mc_effs

                # tau efficiencies
                cross_trigger_tau_data_efficiencies = events.tau_trigger_eff_data_weight
                cross_trigger_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight

                single_triggered = (events.channel_id == channel.id) & events.single_triggered
                cross_triggered = (events.channel_id == channel.id) & events.cross_triggered

                events = create_trigger_weights(
                    events,
                    single_trigger_lepton_data_efficiencies,
                    single_trigger_lepton_mc_efficiencies,
                    cross_trigger_lepton_data_efficiencies,
                    cross_trigger_lepton_mc_efficiencies,
                    cross_trigger_tau_data_efficiencies,
                    cross_trigger_tau_mc_efficiencies,
                    channel,
                    single_triggered,
                    cross_triggered,
                    postfix,
                )
        else:
            # create all variations
            for uncert in ["_e", "_mu", "_tau_mu", "_tau_e"]:
                if uncert == "_e" or uncert == "_tau_e":
                    channel = self.config_inst.channels.n.etau
                else:
                    channel = self.config_inst.channels.n.mutau
                single_triggered = (events.channel_id == channel.id) & events.single_triggered
                cross_triggered = (events.channel_id == channel.id) & events.cross_triggered

                # deal with the electron and muon variation, as there is no additional dm separation
                if uncert == "_e" or uncert == "_mu":
                    if uncert == "_e":
                        single_trigger_lepton_data_efficiencies = Route(
                            f"single_trigger_electron_data_effs{postfix}",
                        ).apply(events)
                        cross_trigger_lepton_data_efficiencies = Route(
                            f"cross_trigger_electron_data_effs{postfix}",
                        ).apply(events)
                        single_trigger_lepton_mc_efficiencies = Route(
                            f"single_trigger_electron_mc_effs{postfix}",
                        ).apply(events)
                        cross_trigger_lepton_mc_efficiencies = Route(
                            f"cross_trigger_electron_mc_effs{postfix}",
                        ).apply(events)

                        # tau efficiencies
                        cross_trigger_tau_data_efficiencies = events.tau_trigger_eff_data_weight
                        cross_trigger_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight

                    else:
                        single_trigger_lepton_data_efficiencies = Route(
                            f"single_trigger_muon_data_effs{postfix}",
                        ).apply(events)
                        cross_trigger_lepton_data_efficiencies = Route(
                            f"cross_trigger_muon_data_effs{postfix}",
                        ).apply(events)
                        single_trigger_lepton_mc_efficiencies = Route(
                            f"single_trigger_muon_mc_effs{postfix}",
                        ).apply(events)
                        cross_trigger_lepton_mc_efficiencies = Route(
                            f"cross_trigger_muon_mc_effs{postfix}",
                        ).apply(events)

                        # tau efficiencies
                        cross_trigger_tau_data_efficiencies = events.tau_trigger_eff_data_weight
                        cross_trigger_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight

                    events = create_trigger_weights(
                        events,
                        single_trigger_lepton_data_efficiencies,
                        single_trigger_lepton_mc_efficiencies,
                        cross_trigger_lepton_data_efficiencies,
                        cross_trigger_lepton_mc_efficiencies,
                        cross_trigger_tau_data_efficiencies,
                        cross_trigger_tau_mc_efficiencies,
                        channel,
                        single_triggered,
                        cross_triggered,
                        uncert + postfix,
                    )

                # deal with the tau variations
                else:
                    dm_variations_dict = {}
                    if uncert == "_tau_e":
                        single_trigger_lepton_data_efficiencies = events.single_trigger_electron_data_effs
                        cross_trigger_lepton_data_efficiencies = events.cross_trigger_electron_data_effs
                        single_trigger_lepton_mc_efficiencies = events.single_trigger_electron_mc_effs
                        cross_trigger_lepton_mc_efficiencies = events.cross_trigger_electron_mc_effs

                        for dm in [0, 1, 10, 11]:
                            dm_variations_dict[dm] = {
                                "data": Route(
                                    f"tau_trigger_eff_data_weight_dm_{dm}_{channel.name}{postfix}",
                                ).apply(events),
                                "mc": Route(
                                    f"tau_trigger_eff_mc_weight_dm_{dm}_{channel.name}{postfix}",
                                ).apply(events),
                            }

                    else:
                        single_trigger_lepton_data_efficiencies = events.single_trigger_muon_data_effs
                        cross_trigger_lepton_data_efficiencies = events.cross_trigger_muon_data_effs
                        single_trigger_lepton_mc_efficiencies = events.single_trigger_muon_mc_effs
                        cross_trigger_lepton_mc_efficiencies = events.cross_trigger_muon_mc_effs

                        for dm in [0, 1, 10, 11]:
                            dm_variations_dict[dm] = {
                                "data": Route(
                                    f"tau_trigger_eff_data_weight_dm_{dm}_{channel.name}{postfix}",
                                ).apply(events),
                                "mc": Route(
                                    f"tau_trigger_eff_mc_weight_dm_{dm}_{channel.name}{postfix}",
                                ).apply(events),
                            }

                    for dm, dm_variated_effs in dm_variations_dict.items():
                        events = create_trigger_weights(
                            events,
                            single_trigger_lepton_data_efficiencies,
                            single_trigger_lepton_mc_efficiencies,
                            cross_trigger_lepton_data_efficiencies,
                            cross_trigger_lepton_mc_efficiencies,
                            dm_variated_effs["data"],
                            dm_variated_effs["mc"],
                            channel,
                            single_triggered,
                            cross_triggered,
                            f"_tau_dm{dm}{postfix}",
                        )
    for postfix in ["_up", "_down"]:
        for variation in ["e", "jet"]:
            trigger_sf = events.mutau_trigger_weights
            events = set_ak_column_f32(events, f"mutau_trigger_weights_{variation}{postfix}", trigger_sf)
    for postfix in ["_up", "_down"]:
        for variation in ["mu", "jet"]:
            trigger_sf = events.etau_trigger_weights
            events = set_ak_column_f32(events, f"etau_trigger_weights_{variation}{postfix}", trigger_sf)
    return events


@producer(
    uses={
        "channel_id", "matched_trigger_ids",
        tau_trigger_sf_and_effs_cclub, jet_trigger_effs,
    },
    produces={
        "tautau_trigger_weights",
    } | {
        f"tautau_trigger_weights_tau_dm{dm}_{direction}"
        for direction in ["up", "down"]
        for dm in [0, 1, 10, 11]
    } | {
        f"tautau_trigger_weights_jet_{direction}"
        for direction in ["up", "down"]
    } | {
        f"tautau_trigger_weights_e_{direction}"
        for direction in ["up", "down"]
    } | {
        f"tautau_trigger_weights_mu_{direction}"
        for direction in ["up", "down"]
    },
)
def tautau_trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for tautau trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``muon_trigger_sf`` and ``cross_trigger_muon_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "muon_trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
            "cross_trigger_muon_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/muon_z.json.gz",  # noqa
        })
    """

    # create all tau weights (might already exist from the etau/mutau trigger weights)
    events = self[tau_trigger_sf_and_effs_cclub](events, **kwargs)

    channel = self.config_inst.channels.n.tautau

    # find out which tautau triggers are passed
    tautau_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    tautaujet_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    tautauvbf_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("cross_tau_tau"):
            tautau_trigger_passed = tautau_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_tau_tau_jet"):
            tautaujet_trigger_passed = tautaujet_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        if trigger.has_tag("cross_tau_tau_vbf"):
            tautauvbf_trigger_passed = tautauvbf_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa

    ditau_triggered = ((events.channel_id == channel.id) & tautau_trigger_passed)
    ditaujet_triggered = ((events.channel_id == channel.id) & tautaujet_trigger_passed)

    sorted_hhbjet_indices = ak.argsort(events.HHBJet.pt, axis=1, ascending=False)
    leading_HHBJet_mask = (ak.zeros_like(events.HHBJet.pt, dtype=int) == ak.local_index(events.HHBJet.pt)[sorted_hhbjet_indices])  # noqa
    jet_mask = (ditaujet_triggered & leading_HHBJet_mask)
    # create jet trigger weights
    events = self[jet_trigger_effs](events, jet_mask, **kwargs)

    # add phase space requirements
    # vbf_triggered = (
    #     (events.channel_id == self.config_inst.channels.n.tautau.id) &
    #     tautauvbf_trigger_passed
    # )

    for postfix in ["", "_up", "_down"]:
        if postfix == "":
            ditau_data_efficiencies = events.tau_trigger_eff_data_weight
            ditau_mc_efficiencies = events.tau_trigger_eff_mc_weight
            ditaujet_tau_data_efficiencies = events.tau_trigger_eff_data_weight_tautaujet
            ditaujet_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight_tautaujet

            # jet efficiencies
            jet_data_efficiencies = events.ditaujet_trigger_jet_weight_eff_data
            jet_mc_efficiencies = events.ditaujet_trigger_jet_weight_eff_mc

            # make jet efficiencies to event level quantity
            # there should be only one such efficiency for the tautaujet trigger
            jet_data_efficiencies = ak.prod(jet_data_efficiencies, axis=1, mask_identity=False)
            jet_mc_efficiencies = ak.prod(jet_mc_efficiencies, axis=1, mask_identity=False)

            events = create_trigger_weights(
                events,
                ditau_data_efficiencies,
                ditau_mc_efficiencies,
                ditaujet_tau_data_efficiencies,
                ditaujet_tau_mc_efficiencies,
                jet_data_efficiencies,
                jet_mc_efficiencies,
                channel=channel,
                single_triggered=ditau_triggered,
                cross_triggered=ditaujet_triggered,
                postfix=postfix,
            )
        else:
            # jet variations
            # tau efficiencies
            ditau_data_efficiencies = events.tau_trigger_eff_data_weight
            ditau_mc_efficiencies = events.tau_trigger_eff_mc_weight
            ditaujet_tau_data_efficiencies = events.tau_trigger_eff_data_weight_tautaujet
            ditaujet_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight_tautaujet

            # jet efficiencies
            jet_data_efficiencies = Route(f"ditaujet_trigger_jet_weight_eff_data{postfix}").apply(events)
            jet_mc_efficiencies = Route(f"ditaujet_trigger_jet_weight_eff_mc{postfix}").apply(events)

            # make jet efficiencies to event level quantity
            # there should be only one such efficiency for the tautaujet trigger
            jet_data_efficiencies = ak.prod(jet_data_efficiencies, axis=1, mask_identity=False)
            jet_mc_efficiencies = ak.prod(jet_mc_efficiencies, axis=1, mask_identity=False)

            events = create_trigger_weights(
                events,
                ditau_data_efficiencies,
                ditau_mc_efficiencies,
                ditaujet_tau_data_efficiencies,
                ditaujet_tau_mc_efficiencies,
                jet_data_efficiencies,
                jet_mc_efficiencies,
                channel=channel,
                single_triggered=ditau_triggered,
                cross_triggered=ditaujet_triggered,
                postfix="_jet" + postfix,
            )

            # tau variations

            # jet efficiencies
            jet_data_efficiencies = events.ditaujet_trigger_jet_weight_eff_data
            jet_mc_efficiencies = events.ditaujet_trigger_jet_weight_eff_mc

            # make jet efficiencies to event level quantity
            # there should be only one such efficiency for the tautaujet trigger
            jet_data_efficiencies = ak.prod(jet_data_efficiencies, axis=1, mask_identity=False)
            jet_mc_efficiencies = ak.prod(jet_mc_efficiencies, axis=1, mask_identity=False)

            for dm in [0, 1, 10, 11]:
                # tau efficiencies
                dm_variations_dict = {
                    "ditau_data": Route(f"tau_trigger_eff_data_weight_dm_{dm}_tautau{postfix}").apply(events),
                    "ditau_mc": Route(f"tau_trigger_eff_mc_weight_dm_{dm}_tautau{postfix}").apply(events),
                    "ditaujet_data": Route(f"tau_trigger_eff_data_weight_dm_{dm}_tautaujet{postfix}").apply(events),
                    "ditaujet_mc": Route(f"tau_trigger_eff_mc_weight_dm_{dm}_tautaujet{postfix}").apply(events),
                }

                events = create_trigger_weights(
                    events,
                    dm_variations_dict["ditau_data"],
                    dm_variations_dict["ditau_mc"],
                    dm_variations_dict["ditaujet_data"],
                    dm_variations_dict["ditaujet_mc"],
                    jet_data_efficiencies,
                    jet_mc_efficiencies,
                    channel=channel,
                    single_triggered=ditau_triggered,
                    cross_triggered=ditaujet_triggered,
                    postfix=f"_tau_dm{dm}{postfix}",
                )

    for postfix in ["_up", "_down"]:
        for variation in ["e", "mu"]:
            trigger_sf = events.tautau_trigger_weights
            events = set_ak_column_f32(events, f"tautau_trigger_weights_{variation}{postfix}", trigger_sf)
    return events


ee_trigger_weights = electron_trigger_weights.derive(
    "ee_trigger_weights",
    cls_dict={
        "weight_name": "ee_trigger_weights",
    },
)

mumu_trigger_weights = muon_trigger_weights.derive(
    "mumu_trigger_weights",
    cls_dict={
        "weight_name": "mumu_trigger_weights",
    },
)

emu_e_trigger_weights = electron_trigger_weights.derive(
    "emu_e_trigger_weights",
    cls_dict={
        "weight_name": "emu_e_trigger_weights",
    },
)

emu_mu_trigger_weights = muon_trigger_weights.derive(
    "emu_mu_trigger_weights",
    cls_dict={
        "weight_name": "emu_mu_trigger_weights",
    },
)


@producer(
    uses={
        "channel_id", "matched_trigger_ids",
        emu_e_trigger_weights, emu_mu_trigger_weights,
    },
    produces={
        "emu_trigger_weights",
    } | {
        f"emu_trigger_weights_mu_{direction}"
        for direction in ["up", "down"]
    } | {
        f"emu_trigger_weights_e_{direction}"
        for direction in ["up", "down"]
    } | {
        f"emu_trigger_weights_tau_dm{dm}_{direction}"
        for direction in ["up", "down"]
        for dm in [0, 1, 10, 11]
    } | {
        f"emu_trigger_weights_jet_{direction}"
        for direction in ["up", "down"]
    },
)
def emu_trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for emu trigger scale factors.
    """

    # find out which triggers are passed
    mu_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    e_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("single_mu"):
            mu_trigger_passed = mu_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("single_e"):
            e_trigger_passed = e_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)

    pass_muon = (
        (events.channel_id == self.config_inst.channels.n.emu.id) & mu_trigger_passed
    )

    pass_electron = (
        (events.channel_id == self.config_inst.channels.n.emu.id) & e_trigger_passed
    )

    # create object masks for the triggered objects
    muon_object_mask = pass_muon & (ak.local_index(events.Muon.pt) == 0)
    electron_object_mask = pass_electron & (ak.local_index(events.Electron.pt) == 0)

    # calculate the scale factors for the triggered objects, if the object was not triggered, the SF is 1
    # therefore we can just multiply the SFs for the objects in the event
    events = self[emu_e_trigger_weights](events, electron_mask=electron_object_mask, **kwargs)
    events = self[emu_mu_trigger_weights](events, muon_mask=muon_object_mask, **kwargs)

    for postfix in ["", "_e_up", "_e_down", "_mu_up", "_mu_down"]:
        # start with the nominal case
        if postfix == "":
            trigger_sf = events.emu_e_trigger_weights * events.emu_mu_trigger_weights
        elif postfix.startswith("_e"):
            shift = postfix.split("_")[-1]
            # electron shifts
            trigger_sf = Route(f"emu_e_trigger_weights_{shift}").apply(events) * events.emu_mu_trigger_weights
        elif postfix.startswith("_mu"):
            shift = postfix.split("_")[-1]
            trigger_sf = events.emu_e_trigger_weights * Route(f"emu_mu_trigger_weights_{shift}").apply(events)
        else:
            raise ValueError(f"Unknown postfix {postfix}")
        events = set_ak_column_f32(events, f"emu_trigger_weights{postfix}", trigger_sf)

    for postfix in ["_up", "_down"]:
        for variation in ["tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet"]:
            trigger_sf = events.emu_trigger_weights
            events = set_ak_column_f32(events, f"emu_trigger_weights_{variation}{postfix}", trigger_sf)
    return events


@producer(
    uses={
        etau_mutau_trigger_weights,
        tautau_trigger_weights,
        ee_trigger_weights,
        mumu_trigger_weights, emu_trigger_weights,
    },
    produces={
        "trigger_weight",
    } | {
        f"trigger_weight_mu_{direction}"
        for direction in ["up", "down"]
    } | {
        f"trigger_weight_e_{direction}"
        for direction in ["up", "down"]
    } | {
        f"trigger_weight_tau_dm{dm}_{direction}"
        for direction in ["up", "down"]
        for dm in [0, 1, 10, 11]
    } | {
        f"trigger_weight_jet_{direction}"
        for direction in ["up", "down"]
    },
)
def trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for trigger scale factors.
    """

    # create the columns
    # etau and mutau
    events = self[etau_mutau_trigger_weights](events, **kwargs)

    # tautau
    events = self[tautau_trigger_weights](events, **kwargs)

    # ee and mumu
    ee_mask = (events.channel_id == self.config_inst.channels.n.ee.id) & (ak.local_index(events.Electron.pt) == 0)
    mumu_mask = (events.channel_id == self.config_inst.channels.n.mumu.id) & (ak.local_index(events.Muon.pt) == 0)
    events = self[ee_trigger_weights](events, electron_mask=ee_mask, **kwargs)
    events = self[mumu_trigger_weights](events, muon_mask=mumu_mask, **kwargs)

    # rename ee and mumu variations for consistency
    for variation in ["_mu_up", "_mu_down"]:
        events = set_ak_column_f32(
            events,
            f"mumu_trigger_weights{variation}",
            Route(f"mumu_trigger_weights{variation.replace('_mu', '')}").apply(events),
        )
    for variation in ["_e_up", "_e_down"]:
        events = set_ak_column_f32(
            events,
            f"ee_trigger_weights{variation}",
            Route(f"ee_trigger_weights{variation.replace('_e', '')}").apply(events),
        )

    # create the variations for non varying objects in ee and mumu
    for postfix in ["_up", "_down"]:
        for variation in ["mu", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet"]:
            trigger_sf = events.ee_trigger_weights
            events = set_ak_column_f32(events, f"ee_trigger_weights_{variation}{postfix}", trigger_sf)

    for postfix in ["_up", "_down"]:
        for variation in ["e", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet"]:
            trigger_sf = events.mumu_trigger_weights
            events = set_ak_column_f32(events, f"mumu_trigger_weights_{variation}{postfix}", trigger_sf)

    # emu
    events = self[emu_trigger_weights](events, **kwargs)

    # create the total trigger scale factor
    for postfix in ["", "_up", "_down"]:
        if postfix == "":
            trigger_sf = (
                events.etau_trigger_weights *
                events.mutau_trigger_weights *
                events.tautau_trigger_weights *
                events.ee_trigger_weights *
                events.mumu_trigger_weights *
                events.emu_trigger_weights
            )
            events = set_ak_column_f32(events, "trigger_weight", trigger_sf)
        else:
            channels = ["ee", "mumu", "emu", "etau", "mutau", "tautau"]
            for object_ in ["e", "mu", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet"]:
                trigger_sf = events.trigger_weight
                for channel in channels:
                    # for all variations, the default is the nominal trigger weights
                    variation = Route(f"{channel}_trigger_weights_{object_}{postfix}").apply(events)
                    # update the trigger weights with the value for the variation
                    channel_mask = (events.channel_id == self.config_inst.channels.get(channel).id)
                    trigger_sf = ak.where(channel_mask, variation, trigger_sf)
                events = set_ak_column_f32(events, f"trigger_weight_{object_}{postfix}", trigger_sf)
    return events
