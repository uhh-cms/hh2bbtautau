# coding: utf-8

"""
Custom trigger scale factor production.

Note : The trigger weight producers multiply the sfs for all objects in an event to get the total
trigger scale factor of the event. Since we might want to use different objects in different channels,
we will derive the trigger weight producers for each channel separately to apply the correct masks.
"""

import functools

import order as od

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.cms.muon import muon_trigger_weights as cf_muon_trigger_weight
from columnflow.production.cms.electron import electron_trigger_weights as cf_electron_trigger_weight

from hbt.production.tau import tau_trigger_efficiencies
from hbt.production.jet import jet_trigger_efficiencies

ak = maybe_import("awkward")
np = maybe_import("numpy")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# subclass the electron trigger weight producer to create the electron trigger weight
electron_trigger_weight = cf_electron_trigger_weight.derive(
    "electron_trigger_weight",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.electron),
    },
)
muon_trigger_weight = cf_muon_trigger_weight.derive(
    "muon_trigger_weight",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.muon),
    },
)

# subclass the electron weight producer to create the electron efficiencies
single_trigger_electron_data_effs = electron_trigger_weight.derive(
    "single_trigger_electron_data_effs",
    cls_dict={
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_data_effs_cfg),
        "weight_name": "single_trigger_e_data_effs",
    },
)

single_trigger_electron_mc_effs = electron_trigger_weight.derive(
    "single_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_mc_effs_cfg),
        "weight_name": "single_trigger_e_mc_effs",
    },
)

cross_trigger_electron_data_effs = electron_trigger_weight.derive(
    "cross_trigger_electron_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.cross_electron),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_data_effs_cfg),
        "weight_name": "cross_trigger_e_data_effs",
    },
)

cross_trigger_electron_mc_effs = electron_trigger_weight.derive(
    "cross_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf.cross_electron),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_mc_effs_cfg),
        "weight_name": "cross_trigger_e_mc_effs",
    },
)

# subclass the muon weight producer to create the muon efficiencies
single_trigger_muon_data_effs = muon_trigger_weight.derive(
    "single_trigger_muon_data_effs",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_data_effs_cfg),
        "weight_name": "single_trigger_mu_data_effs",
    },
)

single_trigger_muon_mc_effs = muon_trigger_weight.derive(
    "single_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_mc_effs_cfg),
        "weight_name": "single_trigger_mu_mc_effs",
    },
)

cross_trigger_muon_data_effs = muon_trigger_weight.derive(
    "cross_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_data_effs_cfg),
        "weight_name": "cross_trigger_mu_data_effs",
    },
)

cross_trigger_muon_mc_effs = muon_trigger_weight.derive(
    "cross_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_mc_effs_cfg),
        "weight_name": "cross_trigger_mu_mc_effs",
    },
)

# subclass the tau weight producer to use the cclub tau efficiencies
tau_trigger_effs_cclub = tau_trigger_efficiencies.derive(
    "tau_trigger_effs_cclub",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.trigger_sf.tau),
        "get_tau_corrector": (lambda self: self.config_inst.x.tau_trigger_corrector_cclub),
    },
)

ee_trigger_weight = electron_trigger_weight.derive(
    "ee_trigger_weight",
    cls_dict={
        "weight_name": "ee_trigger_weight",
    },
)

mumu_trigger_weight = muon_trigger_weight.derive(
    "mumu_trigger_weight",
    cls_dict={
        "weight_name": "mumu_trigger_weight",
    },
)

emu_e_trigger_weight = electron_trigger_weight.derive(
    "emu_e_trigger_weight",
    cls_dict={
        "weight_name": "emu_e_trigger_weight",
    },
)

emu_mu_trigger_weight = muon_trigger_weight.derive(
    "emu_mu_trigger_weight",
    cls_dict={
        "weight_name": "emu_mu_trigger_weight",
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
    first_trigger_matched: ak.Array,
    second_trigger_matched: ak.Array,
    first_trigger_effs: ak.Array,
    second_trigger_common_object_effs: ak.Array,
    second_trigger_other_object_effs: ak.Array,
) -> ak.Array:
    """
    Calculate the combination of the single and cross trigger efficiencies.
    """
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


def create_trigger_weight(
    events: ak.Array,
    first_trigger_eff_data: ak.Array,
    first_trigger_eff_mc: ak.Array,
    second_trigger_common_object_eff_data: ak.Array,
    second_trigger_common_object_eff_mc: ak.Array,
    second_trigger_other_object_eff_data: ak.Array,
    second_trigger_other_object_eff_mc: ak.Array,
    channel: od.Channel,
    first_trigger_matched: ak.Array,
    second_trigger_matched: ak.Array,
) -> ak.Array:
    """
    Create the trigger weight for a given channel.
    """
    trigger_eff_data = calculate_correlated_ditrigger_efficiency(
        first_trigger_matched,
        second_trigger_matched,
        first_trigger_eff_data,
        second_trigger_common_object_eff_data,
        second_trigger_other_object_eff_data,
    )
    trigger_eff_mc = calculate_correlated_ditrigger_efficiency(
        first_trigger_matched,
        second_trigger_matched,
        first_trigger_eff_mc,
        second_trigger_common_object_eff_mc,
        second_trigger_other_object_eff_mc,
    )

    # calculate the ratio
    trigger_weight = trigger_eff_data / trigger_eff_mc

    # nan happens for all events not in the specific channel, due to efficiency == 0
    # add a failsafe here in case of efficiency 0 for an event actually in the channel
    nan_mask = np.isnan(trigger_weight)
    if np.any(nan_mask & (events.channel_id == channel.id) & (first_trigger_matched | second_trigger_matched)):
        raise ValueError(f"Found nan in {channel.name} trigger weight")
    trigger_weight_no_nan = np.nan_to_num(trigger_weight, nan=1.0)

    return trigger_weight_no_nan


@producer(
    uses={
        "channel_id", "single_triggered", "cross_triggered",  # "matched_trigger_ids"
        single_trigger_electron_data_effs, cross_trigger_electron_data_effs,
        single_trigger_electron_mc_effs, cross_trigger_electron_mc_effs,
        single_trigger_muon_data_effs, cross_trigger_muon_data_effs,
        single_trigger_muon_mc_effs, cross_trigger_muon_mc_effs,
        tau_trigger_effs_cclub,
    },
    produces={
        "{e,mu}tau_trigger_weight",
        "etau_trigger_weight_e_{up,down}",
        "mutau_trigger_weight_mu_{up,down}",
        "{e,mu}tau_trigger_weight_tau_dm{0,1,10,11}_{up,down}",
    },
)
def etau_mutau_trigger_weight(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Produces trigger weight for events that fall into the etau and mutau categories. Requires several external files
    and configs in the analysis config.
    """
    # create e/mu object-level masks for events in the etau and mutau channel, selecting only the leading lepton for
    # which the trigger efficiencies were initially calculated (we used the same lepton for matching in the selection)
    single_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        events.single_triggered &
        (ak.local_index(events.Electron) == 0)
    )
    single_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        events.single_triggered &
        (ak.local_index(events.Muon) == 0)
    )
    cross_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        events.cross_triggered &
        (ak.local_index(events.Electron) == 0)
    )
    cross_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        events.cross_triggered &
        (ak.local_index(events.Muon) == 0)
    )

    # get efficiencies from the correctionlib producers

    # first, create the efficiencies for the leptons in data
    events = self[single_trigger_muon_data_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_trigger_muon_data_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_trigger_electron_data_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_trigger_electron_data_effs](events, cross_electron_triggered, **kwargs)

    # do the same for MC efficiencies
    events = self[single_trigger_muon_mc_effs](events, single_muon_triggered, **kwargs)
    events = self[cross_trigger_muon_mc_effs](events, cross_muon_triggered, **kwargs)
    events = self[single_trigger_electron_mc_effs](events, single_electron_triggered, **kwargs)
    events = self[cross_trigger_electron_mc_effs](events, cross_electron_triggered, **kwargs)

    # create all tau efficiencies at object-level
    events = self[tau_trigger_effs_cclub](events, **kwargs)

    # create the nominal case
    for lepton, channel_name in [("e", "etau"), ("mu", "mutau")]:
        channel = self.config_inst.get_channel(channel_name)
        single_trigger_lepton_data_effs = events[f"single_trigger_{lepton}_data_effs"]
        cross_trigger_lepton_data_effs = events[f"cross_trigger_{lepton}_data_effs"]
        single_trigger_lepton_mc_effs = events[f"single_trigger_{lepton}_mc_effs"]
        cross_trigger_lepton_mc_effs = events[f"cross_trigger_{lepton}_mc_effs"]

        # make tau efficiencies to event level quantity
        cross_trigger_tau_data_effs = ak.prod(
            events[f"tau_trigger_eff_data_{channel_name}"],
            axis=1,
            mask_identity=False,
        )
        cross_trigger_tau_mc_effs = ak.prod(
            events[f"tau_trigger_eff_mc_{channel_name}"],
            axis=1,
            mask_identity=False,
        )

        trigger_weight = create_trigger_weight(
            events,
            single_trigger_lepton_data_effs,
            single_trigger_lepton_mc_effs,
            cross_trigger_lepton_data_effs,
            cross_trigger_lepton_mc_effs,
            cross_trigger_tau_data_effs,
            cross_trigger_tau_mc_effs,
            channel,
            ((events.channel_id == channel.id) & events.single_triggered),
            ((events.channel_id == channel.id) & events.cross_triggered),
        )
        events = set_ak_column_f32(events, f"{channel_name}_trigger_weight", trigger_weight)

    # create the variations
    for direction in ["up", "down"]:
        for lepton, channel_name in [("e", "etau"), ("mu", "mutau")]:
            # e and mu variations

            channel = self.config_inst.get_channel(channel_name)
            single_triggered = (events.channel_id == channel.id) & events.single_triggered
            cross_triggered = (events.channel_id == channel.id) & events.cross_triggered

            single_trigger_lepton_data_effs = events[f"single_trigger_{lepton}_data_effs_{direction}"]
            cross_trigger_lepton_data_effs = events[f"cross_trigger_{lepton}_data_effs_{direction}"]
            single_trigger_lepton_mc_effs = events[f"single_trigger_{lepton}_mc_effs_{direction}"]
            cross_trigger_lepton_mc_effs = events[f"cross_trigger_{lepton}_mc_effs_{direction}"]
            cross_trigger_tau_data_effs = events[f"tau_trigger_eff_data_{channel_name}"]
            cross_trigger_tau_mc_effs = events[f"tau_trigger_eff_mc_{channel_name}"]

            # make tau efficiencies to event level quantity
            cross_trigger_tau_data_effs = ak.prod(
                cross_trigger_tau_data_effs,
                axis=1,
                mask_identity=False,
            )
            cross_trigger_tau_mc_effs = ak.prod(
                cross_trigger_tau_mc_effs,
                axis=1,
                mask_identity=False,
            )

            trigger_weight = create_trigger_weight(
                events,
                single_trigger_lepton_data_effs,
                single_trigger_lepton_mc_effs,
                cross_trigger_lepton_data_effs,
                cross_trigger_lepton_mc_effs,
                cross_trigger_tau_data_effs,
                cross_trigger_tau_mc_effs,
                channel,
                single_triggered,
                cross_triggered,
            )
            events = set_ak_column_f32(events, f"{channel.name}_trigger_weight_{lepton}_{direction}", trigger_weight)

            # tau variations
            single_trigger_lepton_data_effs = events[f"single_trigger_{lepton}_data_effs"]
            cross_trigger_lepton_data_effs = events[f"cross_trigger_{lepton}_data_effs"]
            single_trigger_lepton_mc_effs = events[f"single_trigger_{lepton}_mc_effs"]
            cross_trigger_lepton_mc_effs = events[f"cross_trigger_{lepton}_mc_effs"]

            for dm in [0, 1, 10, 11]:
                trigger_weight = create_trigger_weight(
                    events,
                    single_trigger_lepton_data_effs,
                    single_trigger_lepton_mc_effs,
                    cross_trigger_lepton_data_effs,
                    cross_trigger_lepton_mc_effs,
                    ak.prod(events[f"tau_trigger_eff_data_{channel.name}_dm{dm}_{direction}"], axis=1, mask_identity=False),  # noqa: E501
                    ak.prod(events[f"tau_trigger_eff_mc_{channel.name}_dm{dm}_{direction}"], axis=1, mask_identity=False),  # noqa: E501
                    channel,
                    single_triggered,
                    cross_triggered,
                )
                events = set_ak_column_f32(events, f"{channel.name}_trigger_weight_tau_dm{dm}_{direction}", trigger_weight)  # noqa: E501

    return events


@producer(
    uses={
        "channel_id", "matched_trigger_ids",
        tau_trigger_effs_cclub, jet_trigger_efficiencies,
    },
    produces={
        "tautau_trigger_weight",
        "tautau_trigger_weight_jet_{up,down}",
        "tautau_trigger_weight_tau_dm{0,1,10,11}_{up,down}",
    },
)
def tautau_trigger_weight(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Produces trigger weight for events that fall into the tautau category. Requires several external file and configs
    in the analysis config.
    """
    channel = self.config_inst.channels.n.tautau

    # create all tau efficiencies
    events = self[tau_trigger_effs_cclub](events, **kwargs)

    # find out which tautau triggers are passed
    tt_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    ttj_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    # ttv_trigger_passed = ak.zeros_like(events.channel_id, dtype=np.bool)
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("cross_tau_tau"):
            tt_trigger_passed = tt_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_tau_tau_jet"):
            ttj_trigger_passed = ttj_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        # if trigger.has_tag("cross_tau_tau_vbf"):
        #     ttv_trigger_passed = ttv_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa

    tt_triggered = ((events.channel_id == channel.id) & tt_trigger_passed)
    ttj_triggered = ((events.channel_id == channel.id) & ttj_trigger_passed)
    # ttv_triggered = ((events.channel_id == channel.id) & ttv_trigger_passed)  # vbf treatment left out from here on

    sorted_hhbjet_indices = ak.argsort(events.HHBJet.pt, axis=1, ascending=False)
    leading_HHBJet_mask = (ak.zeros_like(events.HHBJet.pt, dtype=int) == ak.local_index(events.HHBJet.pt)[sorted_hhbjet_indices])  # noqa
    jet_mask = (ttj_triggered & leading_HHBJet_mask)
    # create jet trigger efficiencies
    events = self[jet_trigger_efficiencies](events, jet_mask, **kwargs)

    # tau efficiencies
    # make ditau efficiencies to event level quantity
    tt_data_effs = ak.prod(events.tau_trigger_eff_data_tautau, axis=1, mask_identity=False)
    tt_mc_effs = ak.prod(events.tau_trigger_eff_mc_tautau, axis=1, mask_identity=False)
    ttj_tau_data_effs = ak.prod(events.tau_trigger_eff_data_tautaujet, axis=1, mask_identity=False)
    ttj_tau_mc_effs = ak.prod(events.tau_trigger_eff_mc_tautaujet, axis=1, mask_identity=False)

    # jet efficiencies
    # make jet efficiencies to event level quantity
    # there should be only one such efficiency for the tautaujet trigger
    ttj_jet_data_effs = ak.prod(events.jet_trigger_eff_data, axis=1, mask_identity=False)
    ttj_jet_mc_effs = ak.prod(events.jet_trigger_eff_mc, axis=1, mask_identity=False)

    trigger_weight = create_trigger_weight(
        events,
        tt_data_effs,
        tt_mc_effs,
        ttj_tau_data_effs,
        ttj_tau_mc_effs,
        ttj_jet_data_effs,
        ttj_jet_mc_effs,
        channel=channel,
        first_trigger_matched=tt_triggered,
        second_trigger_matched=ttj_triggered,
    )
    events = set_ak_column_f32(events, "tautau_trigger_weight", trigger_weight)

    for direction in ["up", "down"]:
        # jet variations

        # tau efficiencies
        # make ditau efficiencies to event level quantity
        tt_data_effs = ak.prod(events.tau_trigger_eff_data_tautau, axis=1, mask_identity=False)
        tt_mc_effs = ak.prod(events.tau_trigger_eff_mc_tautau, axis=1, mask_identity=False)
        ttj_tau_data_effs = ak.prod(events.tau_trigger_eff_data_tautaujet, axis=1, mask_identity=False)
        ttj_tau_mc_effs = ak.prod(events.tau_trigger_eff_mc_tautaujet, axis=1, mask_identity=False)

        # jet efficiencies
        # make jet efficiencies to event level quantity
        # there should be only one such efficiency for the tautaujet trigger
        ttj_jet_data_effs = ak.prod(events[f"jet_trigger_eff_data_{direction}"], axis=1, mask_identity=False)
        ttj_jet_mc_effs = ak.prod(events[f"jet_trigger_eff_mc_{direction}"], axis=1, mask_identity=False)

        trigger_weight = create_trigger_weight(
            events,
            tt_data_effs,
            tt_mc_effs,
            ttj_tau_data_effs,
            ttj_tau_mc_effs,
            ttj_jet_data_effs,
            ttj_jet_mc_effs,
            channel=channel,
            first_trigger_matched=tt_triggered,
            second_trigger_matched=ttj_triggered,
        )
        events = set_ak_column_f32(events, f"tautau_trigger_weight_jet_{direction}", trigger_weight)

        # tau variations

        # jet efficiencies
        # make jet efficiencies to event level quantity
        # there should be only one such efficiency for the tautaujet trigger
        ttj_jet_data_effs = ak.prod(events.jet_trigger_eff_data, axis=1, mask_identity=False)
        ttj_jet_mc_effs = ak.prod(events.jet_trigger_eff_mc, axis=1, mask_identity=False)

        for dm in [0, 1, 10, 11]:
            trigger_weight = create_trigger_weight(
                events,
                ak.prod(events[f"tau_trigger_eff_data_tautau_dm{dm}_{direction}"], axis=1, mask_identity=False),
                ak.prod(events[f"tau_trigger_eff_mc_tautau_dm{dm}_{direction}"], axis=1, mask_identity=False),
                ak.prod(events[f"tau_trigger_eff_data_tautaujet_dm{dm}_{direction}"], axis=1, mask_identity=False),
                ak.prod(events[f"tau_trigger_eff_mc_tautaujet_dm{dm}_{direction}"], axis=1, mask_identity=False),
                ttj_jet_data_effs,
                ttj_jet_mc_effs,
                channel=channel,
                first_trigger_matched=tt_triggered,
                second_trigger_matched=ttj_triggered,
            )
            events = set_ak_column_f32(events, f"tautau_trigger_weight_tau_dm{dm}_{direction}", trigger_weight)

    return events


@producer(
    uses={
        "channel_id", "matched_trigger_ids",
        emu_e_trigger_weight, emu_mu_trigger_weight,
    },
    produces={
        "emu_trigger_weight",
        "emu_trigger_weight_{e,mu}_{up,down}",
    },
)
def emu_trigger_weight(
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

    # create e/mu object-level masks for events in the emu channel, selecting only the leading lepton for
    # which the trigger efficiencies were initially calculated (we used the same lepton for matching in the selection)
    muon_object_mask = (
        (events.channel_id == self.config_inst.channels.n.emu.id) &
        mu_trigger_passed &
        (ak.local_index(events.Muon) == 0)
    )
    electron_object_mask = (
        (events.channel_id == self.config_inst.channels.n.emu.id) &
        e_trigger_passed &
        (ak.local_index(events.Electron) == 0)
    )

    # calculate the scale factors for the triggered objects, if the object was not triggered, the SF is 1
    # therefore we can just multiply the SFs for the objects in the event
    events = self[emu_e_trigger_weight](events, electron_mask=electron_object_mask, **kwargs)
    events = self[emu_mu_trigger_weight](events, muon_mask=muon_object_mask, **kwargs)

    # nominal case
    trigger_weight = events.emu_e_trigger_weight * events.emu_mu_trigger_weight
    events = set_ak_column_f32(events, "emu_trigger_weight", trigger_weight)

    # e and mu variations
    for direction in ["up", "down"]:
        # e
        trigger_weight = events[f"emu_e_trigger_weight_{direction}"] * events.emu_mu_trigger_weight
        events = set_ak_column_f32(events, f"emu_trigger_weight_e_{direction}", trigger_weight)
        # mu
        trigger_weight = events.emu_e_trigger_weight * events[f"emu_mu_trigger_weight_{direction}"]
        events = set_ak_column_f32(events, f"emu_trigger_weight_mu_{direction}", trigger_weight)

    return events


@producer(
    uses={
        "channel_id",
        ee_trigger_weight,
        mumu_trigger_weight,
    },
    produces={
        "ee_trigger_weight",
        "mumu_trigger_weight",
        "mumu_trigger_weight_mu_{up,down}",
        "ee_trigger_weight_e_{up,down}",
    },
)
def ee_mumu_trigger_weight(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for ee and mumu trigger scale factors.
    """

    ee_mask = (events.channel_id == self.config_inst.channels.n.ee.id) & (ak.local_index(events.Electron) == 0)
    mumu_mask = (events.channel_id == self.config_inst.channels.n.mumu.id) & (ak.local_index(events.Muon) == 0)
    events = self[ee_trigger_weight](events, electron_mask=ee_mask, **kwargs)
    events = self[mumu_trigger_weight](events, muon_mask=mumu_mask, **kwargs)

    # create the columns
    events = self[ee_trigger_weight](events, **kwargs)
    events = self[mumu_trigger_weight](events, **kwargs)

    # rename ee and mumu variations for consistency
    for direction in ["up", "down"]:
        events = set_ak_column_f32(
            events,
            f"mumu_trigger_weight_mu_{direction}",
            events[f"mumu_trigger_weight_{direction}"],
        )
        events = set_ak_column_f32(
            events,
            f"ee_trigger_weight_e_{direction}",
            events[f"ee_trigger_weight_{direction}"],
        )

    return events


@producer(
    uses={
        etau_mutau_trigger_weight,
        tautau_trigger_weight,
        ee_mumu_trigger_weight,
        emu_trigger_weight,
    },
    produces={
        "trigger_weight",
        "trigger_weight_{e,mu,jet}_{up,down}",
        "trigger_weight_tau_dm{0,1,10,11}_{up,down}",
    },
)
def trigger_weight(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for trigger scale factors.
    """
    # etau and mutau
    events = self[etau_mutau_trigger_weight](events, **kwargs)

    # tautau
    events = self[tautau_trigger_weight](events, **kwargs)

    # ee and mumu
    events = self[ee_mumu_trigger_weight](events, **kwargs)

    # emu
    events = self[emu_trigger_weight](events, **kwargs)

    # get channels
    channels = {
        channel_name: self.config_inst.channels.get(channel_name)
        for channel_name in ["etau", "mutau", "tautau", "ee", "mumu", "emu"]
    }

    # create the total trigger scale factor
    # A multiplication is done here, as every the columns used contain the value 1.0 for events not in the channel
    # and the channels are mutually exclusive
    for channel_name, channel in channels.items():
        channel_mask = (events.channel_id == channel.id)
        if not ak.all(channel_mask | (events[f"{channel_name}_trigger_weight"] == 1.0)):
            raise ValueError(f"trigger weight for {channel_name} not all 1.0 for events not in the channel")
    trigger_weight = (
        events.etau_trigger_weight *
        events.mutau_trigger_weight *
        events.tautau_trigger_weight *
        events.ee_trigger_weight *
        events.mumu_trigger_weight *
        events.emu_trigger_weight
    )
    events = set_ak_column_f32(events, "trigger_weight", trigger_weight)

    # create the variations
    # Do to the choice of triggers, certain channel do not have variations for certain objects
    # e.g. etau does not have a muon or jet dependent trigger, therefore the variations are not defined
    # we check further down that the columns do not exist for only these specific cases
    undefined_variations = {
        ("etau", "mu"), ("etau", "jet"),
        ("mutau", "e"), ("mutau", "jet"),
        ("tautau", "e"), ("tautau", "mu"),
        ("ee", "mu"), ("ee", "jet"),
        ("ee", "tau_dm0"), ("ee", "tau_dm1"), ("ee", "tau_dm10"), ("ee", "tau_dm11"),
        ("mumu", "e"), ("mumu", "jet"),
        ("mumu", "tau_dm0"), ("mumu", "tau_dm1"), ("mumu", "tau_dm10"), ("mumu", "tau_dm11"),
        ("emu", "jet"),
        ("emu", "tau_dm0"), ("emu", "tau_dm1"), ("emu", "tau_dm10"), ("emu", "tau_dm11"),
    }

    for direction in ["up", "down"]:
        for variation in ["e", "mu", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet"]:
            trigger_weight = events.trigger_weight
            weight_name = "trigger_weight"
            varied_weight_name = f"{weight_name}_{variation}_{direction}"
            for channel_name, channel in channels.items():
                # for all variations, the default is the nominal trigger weight
                channel_weight_name = f"{channel_name}_{varied_weight_name}"
                if channel_weight_name not in events.fields:
                    if (channel_name, variation) not in undefined_variations:
                        raise ValueError(f"trigger weight variation {channel_weight_name} not found in events")
                    channel_weight_name = f"{channel_name}_{weight_name}"
                variation_array = events[channel_weight_name]

                # update the trigger weight with the value for the variation
                channel_mask = (events.channel_id == channel.id)
                trigger_weight = ak.where(channel_mask, variation_array, trigger_weight)

            events = set_ak_column_f32(events, varied_weight_name, trigger_weight)
    return events
