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

from hbt.production.tau import tau_trigger_efficiencies, quadjet_tau_trigger_sf
from hbt.production.jet import jet_trigger_efficiencies, quadjet_jet_trigger_sf, vbfjet_trigger_efficiencies
from hbt.util import IF_RUN_3_2024, IF_RUN_3_2023_2024

ak = maybe_import("awkward")
np = maybe_import("numpy")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# subclass the electron trigger weight producer to create the electron trigger weight
electron_trigger_weight = cf_electron_trigger_weight.derive(
    "electron_trigger_weight",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.trigger_sf_single_e),
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

# single muon files in cclub use modified scale_factors keys to reach efficiencies
def add_data_efficiency_postfix(corrector, variable_map, postfix=""):
    # get the efficiency key and modify the value variable map
    variable_map["scale_factors"] = variable_map["scale_factors"] + postfix
    return variable_map


single_trigger_muon_data_effs = muon_trigger_weight.derive(
    "single_trigger_muon_data_effs",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_data_effs_cfg),
        "weight_name": "single_trigger_mu_data_effs",
        "update_corrector_variables": functools.partial(add_data_efficiency_postfix, postfix="_DATAeff"),
    },
)

single_trigger_muon_mc_effs = muon_trigger_weight.derive(
    "single_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_mc_effs_cfg),
        "weight_name": "single_trigger_mu_mc_effs",
        "update_corrector_variables": functools.partial(add_data_efficiency_postfix, postfix="_MCeff"),
    },
)


# cross muon files in cclub use "efficiencies" as input instead of "scale_factors" to reach the efficiencies
def add_efficiencies_fields_to_variable_map(producer, corrector, variable_map):
    # add the efficiency fields to the variable map
    variable_map["efficiencies"] = variable_map["scale_factors"]
    return variable_map


cross_trigger_muon_data_effs = muon_trigger_weight.derive(
    "cross_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_data_effs_cfg),
        "weight_name": "cross_trigger_mu_data_effs",
        "update_corrector_variables": add_efficiencies_fields_to_variable_map,
    },
)

cross_trigger_muon_mc_effs = muon_trigger_weight.derive(
    "cross_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.trigger_sf.cross_muon),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_mc_effs_cfg),
        "weight_name": "cross_trigger_mu_mc_effs",
        "update_corrector_variables": add_efficiencies_fields_to_variable_map,
    },
)

# subclass the tau weight producer to use the cclub tau efficiencies
tau_trigger_effs_cclub = tau_trigger_efficiencies.derive(
    "tau_trigger_effs_cclub",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.trigger_sf_tau),
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

# vbf trigger sf
vbf_ditau_jet_trigger_sf = vbfjet_trigger_efficiencies.derive(
    "vbf_ditau_jet_trigger_sf",
    cls_dict={
        "sf_name": "vbf_ditau_jet_trigger_sf",
    },
)
vbf_incl_jet_trigger_sf_etau = vbfjet_trigger_efficiencies.derive(
    "vbf_incl_jet_trigger_sf_etau",
    cls_dict={
        "get_vbfjet_file": (lambda self, external_files: external_files.trigger_sf.vbf_incl),
        # Note: if triple jet is added, change config to get efficiencies instead of sfs
        "sf_name": "vbf_incl_jet_trigger_sf_etau",
    },
)
vbf_incl_jet_trigger_sf_tautau = vbfjet_trigger_efficiencies.derive(
    "vbf_incl_jet_trigger_sf_tautau",
    cls_dict={
        "get_vbfjet_file": (lambda self, external_files: external_files.trigger_sf.vbf_incl),
        # Note: if triple jet is added, change config to get efficiencies instead of sfs
        "sf_name": "vbf_incl_jet_trigger_sf_tautau",
    },
)
vbf_mu_jet_trigger_sf = vbfjet_trigger_efficiencies.derive(
    "vbf_mu_jet_trigger_sf",
    cls_dict={
        "get_vbfjet_file": (lambda self, external_files: external_files.trigger_sf.vbf_mu),
        "sf_name": "vbf_mu_jet_trigger_sf",
    },
)
vbf_tau_jet_trigger_sf = vbfjet_trigger_efficiencies.derive(
    "vbf_tau_jet_trigger_sf",
    cls_dict={
        "get_vbfjet_file": (lambda self, external_files: external_files.trigger_sf.vbf_tau),
        "sf_name": "vbf_tau_jet_trigger_sf",
    },
)
vbf_e_jet_trigger_sf = vbfjet_trigger_efficiencies.derive(
    "vbf_e_jet_trigger_sf",
    cls_dict={
        "get_vbfjet_file": (lambda self, external_files: external_files.trigger_sf.vbf_e),
        "get_vbfjet_config": (lambda self: self.config_inst.x.vbfjet_e_trigger_config),
        "sf_name": "vbf_e_jet_trigger_sf",
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
        "channel_id", "single_triggered", "cross_triggered", "VBFJet.pt", "matched_trigger_ids",
        single_trigger_electron_data_effs, cross_trigger_electron_data_effs,
        single_trigger_electron_mc_effs, cross_trigger_electron_mc_effs,
        single_trigger_muon_data_effs, cross_trigger_muon_data_effs,
        single_trigger_muon_mc_effs, cross_trigger_muon_mc_effs,
        tau_trigger_effs_cclub,
        IF_RUN_3_2023_2024(vbf_incl_jet_trigger_sf_etau),
        IF_RUN_3_2023_2024(vbf_e_jet_trigger_sf),
        IF_RUN_3_2023_2024(vbf_mu_jet_trigger_sf),
    },
    produces={
        "{e,mu}tau_trigger_weight",
        "etau_trigger_weight_e_{up,down}",
        "mutau_trigger_weight_mu_{up,down}",
        "{e,mu}tau_trigger_weight_tau_dm{0,1,10,11}_{up,down}",
        IF_RUN_3_2023_2024("{e,mu}tau_trigger_weight_vbfjets_{up,down}"),
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

    # find out which etau/mutau triggers are passed
    single_l_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    cross_lt_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    # vbf triggers
    cross_lj_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    # for etau channel
    cross_jj_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("single_e") or trigger.has_tag("single_mu"):
            single_l_trigger_passed = single_l_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_e_tau") or trigger.has_tag("cross_mu_tau"):
            cross_lt_trigger_passed = cross_lt_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_e_vbf") or trigger.has_tag("cross_mu_vbf"):
            cross_lj_trigger_passed = cross_lj_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_vbf"):
            cross_jj_trigger_passed = cross_jj_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)

    single_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        single_l_trigger_passed &
        (ak.local_index(events.Electron) == 0)
    )
    single_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        single_l_trigger_passed &
        (ak.local_index(events.Muon) == 0)
    )
    cross_electron_triggered = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        cross_lt_trigger_passed &
        (ak.local_index(events.Electron) == 0)
    )
    cross_muon_triggered = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        cross_lt_trigger_passed &
        (ak.local_index(events.Muon) == 0)
    )

    cross_e_vbf_triggered_event_mask = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        cross_lj_trigger_passed
    )
    cross_mu_vbf_triggered_event_mask = (
        (events.channel_id == self.config_inst.channels.n.mutau.id) &
        cross_lj_trigger_passed
    )
    cross_jj_triggered_event_mask = (
        (events.channel_id == self.config_inst.channels.n.etau.id) &
        cross_jj_trigger_passed
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

    # create all tau efficiencies at object-level for the non vbf triggers
    events = self[tau_trigger_effs_cclub](events, **kwargs)

    # create the vbf sfs
    if self.config_inst.campaign.x.year in {2023, 2024}:
        events = self[vbf_incl_jet_trigger_sf_etau](events, cross_jj_triggered_event_mask & (events.VBFJet.pt > 0), **kwargs)  # noqa: E501
        events = self[vbf_e_jet_trigger_sf](events, cross_e_vbf_triggered_event_mask & (events.VBFJet.pt > 0), **kwargs)
        events = self[vbf_mu_jet_trigger_sf](events, cross_mu_vbf_triggered_event_mask & (events.VBFJet.pt > 0), **kwargs)  # noqa: E501

        vbf_dict = {
            "e": {
                "mask": [cross_e_vbf_triggered_event_mask, cross_jj_triggered_event_mask],
                "sf": ["vbf_e_jet_trigger_sf", "vbf_incl_jet_trigger_sf_etau"],
            },
            "mu": {
                "mask": [cross_mu_vbf_triggered_event_mask],
                "sf": ["vbf_mu_jet_trigger_sf"],
            },
        }

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
            ((events.channel_id == channel.id) & single_l_trigger_passed),
            ((events.channel_id == channel.id) & cross_lt_trigger_passed),
        )

        if self.config_inst.campaign.x.year in {2023, 2024}:
            # add vbf nominal
            for mask, sf in zip(vbf_dict[lepton]["mask"], vbf_dict[lepton]["sf"]):
                if ak.any(trigger_weight[mask] != 1):
                    raise ValueError(f"Trying to apply vbf trigger sf {sf} to events that are already affected by the single/cross trigger efficiencies in the {channel_name} channel")  # noqa: E501
                trigger_weight = ak.where(
                    mask,
                    events[sf],
                    trigger_weight,
                )

        events = set_ak_column_f32(events, f"{channel_name}_trigger_weight", trigger_weight)

    # create the variations
    for direction in ["up", "down"]:
        for lepton, channel_name in [("e", "etau"), ("mu", "mutau")]:
            # e and mu variations

            channel = self.config_inst.get_channel(channel_name)
            single_triggered = (events.channel_id == channel.id) & single_l_trigger_passed
            cross_triggered = (events.channel_id == channel.id) & cross_lt_trigger_passed

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

            if self.config_inst.campaign.x.year in {2023, 2024}:
                # add vbf nominal
                for mask, sf in zip(vbf_dict[lepton]["mask"], vbf_dict[lepton]["sf"]):
                    trigger_weight = ak.where(
                        mask,
                        events[sf],
                        trigger_weight,
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

                if self.config_inst.campaign.x.year in {2023, 2024}:
                    # add vbf nominal
                    for mask, sf in zip(vbf_dict[lepton]["mask"], vbf_dict[lepton]["sf"]):
                        trigger_weight = ak.where(
                            mask,
                            events[sf],
                            trigger_weight,
                        )

                events = set_ak_column_f32(events, f"{channel.name}_trigger_weight_tau_dm{dm}_{direction}", trigger_weight)  # noqa: E501

                if self.config_inst.campaign.x.year in {2023, 2024}:
                    # vbf variations
                    for i, (mask, sf) in enumerate(zip(vbf_dict[lepton]["mask"], vbf_dict[lepton]["sf"])):
                        if i == 0:
                            trigger_weight = ak.where(
                                mask,
                                events[f"{sf}_{direction}"],
                                events[f"{channel.name}_trigger_weight"],
                            )
                        else:
                            trigger_weight = ak.where(
                                mask,
                                events[f"{sf}_{direction}"],
                                trigger_weight,
                            )
                    events = set_ak_column_f32(events, f"{channel.name}_trigger_weight_vbfjets_{direction}", trigger_weight)  # noqa: E501

    return events


@producer(
    uses={
        "channel_id", "matched_trigger_ids", "HHBJet.pt", "Tau.pt", "VBFJet.pt",
        tau_trigger_effs_cclub, jet_trigger_efficiencies,
        vbf_ditau_jet_trigger_sf,
        IF_RUN_3_2023_2024(vbf_incl_jet_trigger_sf_tautau),
        IF_RUN_3_2023_2024(vbf_tau_jet_trigger_sf),
        IF_RUN_3_2024(quadjet_tau_trigger_sf),
        IF_RUN_3_2024(quadjet_jet_trigger_sf),
    },
    produces={
        "tautau_trigger_weight",
        "tautau_trigger_weight_jet_{up,down}",
        "tautau_trigger_weight_tau_dm{0,1,10,11}_{up,down}",
        IF_RUN_3_2024("tautau_trigger_weight_quadjet_{up,down}"),
        "tautau_trigger_weight_vbfjets_{up,down}",
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
    tt_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    ttj_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    quadjet_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    ttv_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    tv_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    v_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    quadjet_applied = False
    for trigger in self.config_inst.x.triggers:
        if trigger.has_tag("cross_tau_tau"):
            tt_trigger_passed = tt_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("cross_tau_tau_jet"):
            ttj_trigger_passed = ttj_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        if trigger.has_tag("cross_tau_tau_vbf"):
            ttv_trigger_passed = ttv_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        if trigger.has_tag("cross_tau_vbf"):
            tv_trigger_passed = tv_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        if trigger.has_tag("cross_vbf"):
            v_trigger_passed = v_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
        if trigger.has_tag("cross_quadjet"):
            if trigger.applies_to_dataset(self.dataset_inst):
                quadjet_trigger_passed = quadjet_trigger_passed | np.any(events.matched_trigger_ids == trigger.id, axis=-1)  # noqa
                quadjet_applied = True

    tt_triggered = ((events.channel_id == channel.id) & tt_trigger_passed)
    ttj_triggered = ((events.channel_id == channel.id) & ttj_trigger_passed)
    ttv_triggered = ((events.channel_id == channel.id) & ttv_trigger_passed)
    tv_triggered = ((events.channel_id == channel.id) & tv_trigger_passed)
    v_triggered = ((events.channel_id == channel.id) & v_trigger_passed)
    if quadjet_applied:
        quadjet_triggered = ((events.channel_id == channel.id) & quadjet_trigger_passed)
        if np.any(quadjet_triggered & tt_triggered) or np.any(quadjet_triggered & ttj_triggered):
            raise ValueError("Found events matched to the quadjet trigger that are also matched to the other tautau "
            "triggers, this should never happen due to orthogonalization")
        if np.any(quadjet_triggered & ttv_triggered) or np.any(quadjet_triggered & tv_triggered) or np.any(quadjet_triggered & v_triggered):  # noqa
            raise ValueError("Found events matched to the quadjet trigger that are also matched to the vbf triggers,"
            " this should never happen due to orthogonalization")

    # create object masks for the correctionlib producers
    sorted_hhbjet_indices = ak.argsort(events.HHBJet.pt, axis=1, ascending=False)
    leading_HHBJet_mask = (ak.zeros_like(events.HHBJet.pt, dtype=int) == ak.local_index(events.HHBJet.pt)[sorted_hhbjet_indices])  # noqa
    ttj_jet_mask = (ttj_triggered & leading_HHBJet_mask)
    if quadjet_applied:
        quadjet_jet_mask = (quadjet_triggered & leading_HHBJet_mask)
    # indices for sorting taus first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt) or 0.0)) + 2)
    tau_sorting_key = events.Tau[f"raw{self.config_inst.x.tau_tagger}VSjet"] * f + events.Tau.pt
    tau_sorting_indices = ak.argsort(tau_sorting_key, axis=-1, ascending=False)
    leading_tau_mask = (ak.zeros_like(events.Tau.pt, dtype=int) == ak.local_index(events.Tau.pt)[tau_sorting_indices])

    vbf_ditau_jet_mask = (ttv_triggered & (events.VBFJet.pt > 0))
    vbf_incl_jet_mask = (v_triggered & (events.VBFJet.pt > 0))
    vbf_tau_jet_mask = (tv_triggered & (events.VBFJet.pt > 0))

    # HOTFIX vbf ditau: the sfs are only defined for vbfjet 1 pt > 160, vbfjet 2 pt > 70, mjj > 1100 for 2024
    # 140,60,850 for 2022, 2023
    # vbf tau 2024: jet cut put to 65 and mjj cut to 900
    # TODO: remove after new production
    original_vbf_ditau_jet_mask = vbf_ditau_jet_mask
    original_vbf_tau_jet_mask = vbf_tau_jet_mask
    vbf_jet_1 = ak.firsts(events.VBFJet, axis=1)
    vbf_jet_2 = ak.firsts(events.VBFJet[:, 1:], axis=1)
    if self.config_inst.campaign.x.year == 2024:
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & (vbf_jet_2.pt > 70)
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & (vbf_jet_1.pt > 160)
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & ((vbf_jet_1 + vbf_jet_2).mass > 1100)
        vbf_tau_jet_mask = vbf_tau_jet_mask & (vbf_jet_1.pt > 65) & (vbf_jet_2.pt > 65) & ((vbf_jet_1 + vbf_jet_2).mass > 900)  # noqa: E501
    else:
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & (vbf_jet_2.pt > 60)
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & (vbf_jet_1.pt > 140)
        vbf_ditau_jet_mask = vbf_ditau_jet_mask & ((vbf_jet_1 + vbf_jet_2).mass > 850)

    # get jet and quadjet efficiencies/sf
    events = self[jet_trigger_efficiencies](events, ttj_jet_mask, **kwargs)
    if quadjet_applied:
        leading_tau_mask_for_quadjet = leading_tau_mask & quadjet_triggered
        events = self[quadjet_jet_trigger_sf](events, quadjet_jet_mask, **kwargs)
        events = self[quadjet_tau_trigger_sf](events, leading_tau_mask_for_quadjet, **kwargs)

    # vbf jet efficiencies
    events = self[vbf_ditau_jet_trigger_sf](events, vbf_ditau_jet_mask, **kwargs)
    if self.config_inst.campaign.x.year in {2023, 2024}:
        events = self[vbf_incl_jet_trigger_sf_tautau](events, vbf_incl_jet_mask, **kwargs)
        events = self[vbf_tau_jet_trigger_sf](events, vbf_tau_jet_mask, **kwargs)

    # HOTFIX part 2, fill none entries due to vbf cuts with 1.0 TODO: remove after new production
    original_vbf_ditau_jet_mask_event_mask = ak.any(original_vbf_ditau_jet_mask, axis=1)
    vbf_ditau_jet_mask_event_mask = ak.any(vbf_ditau_jet_mask, axis=1)
    original_vbf_tau_jet_mask_event_mask = ak.any(original_vbf_tau_jet_mask, axis=1)
    vbf_tau_jet_mask_event_mask = ak.any(vbf_tau_jet_mask, axis=1)
    for sys in ["", "_up", "_down"]:
        vbf_ditau_jet_sf_column_name = f"vbf_ditau_jet_trigger_sf{sys}"
        vbf_ditau_jet_sf_column = ak.where(
            original_vbf_ditau_jet_mask_event_mask & ~vbf_ditau_jet_mask_event_mask,
            1.0,
            events[vbf_ditau_jet_sf_column_name],
        )
        events = set_ak_column_f32(events, vbf_ditau_jet_sf_column_name, vbf_ditau_jet_sf_column)
        if self.config_inst.campaign.x.year == 2024:
            vbf_tau_jet_sf_column_name = f"vbf_tau_jet_trigger_sf{sys}"
            vbf_tau_jet_sf_column = ak.where(
                original_vbf_tau_jet_mask_event_mask & ~vbf_tau_jet_mask_event_mask,
                1.0,
                events[vbf_tau_jet_sf_column_name],
            )
            events = set_ak_column_f32(events, vbf_tau_jet_sf_column_name, vbf_tau_jet_sf_column)
    vbf_ditau_jet_mask = original_vbf_ditau_jet_mask
    if self.config_inst.campaign.x.year == 2024:
        vbf_tau_jet_mask = original_vbf_tau_jet_mask

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

    # vbf
    trigger_weight = ak.where(
        ttv_triggered,
        ak.prod(events.tau_trigger_sf_tautauvbf, axis=1, mask_identity=False) * events.vbf_ditau_jet_trigger_sf,
        trigger_weight,
    )
    if self.config_inst.campaign.x.year in {2023, 2024}:
        trigger_weight = ak.where(
            tv_triggered,
            ak.prod(events.tau_trigger_sf_tauvbf, axis=1, mask_identity=False) * events.vbf_tau_jet_trigger_sf,
            trigger_weight,
        )
        trigger_weight = ak.where(
            v_triggered,
            events.vbf_incl_jet_trigger_sf_tautau,
            trigger_weight,
        )

    # quadjet with variations, needs to be applied after all the other sfs such that the variations are correct
    if quadjet_applied:
        trigger_weight = ak.where(
            quadjet_triggered,
            events.quadjet_tau_trigger_sf * events.quadjet_jet_trigger_sf,
            trigger_weight,
        )
        for direction in ["up", "down"]:
            trigger_weight_syst = ak.where(
                quadjet_triggered,
                events[f"quadjet_tau_trigger_sf_{direction}"] * events[f"quadjet_jet_trigger_sf_{direction}"],
                trigger_weight,
            )
            events = set_ak_column_f32(events, f"tautau_trigger_weight_quadjet_{direction}", trigger_weight_syst)

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
        if quadjet_applied:
            trigger_weight = ak.where(
                quadjet_triggered,
                events.quadjet_tau_trigger_sf * events.quadjet_jet_trigger_sf,
                trigger_weight,
            )
        trigger_weight = ak.where(
            ttv_triggered,
            ak.prod(events.tau_trigger_sf_tautauvbf, axis=1, mask_identity=False) * events.vbf_ditau_jet_trigger_sf,
            trigger_weight,
        )
        if self.config_inst.campaign.x.year in {2023, 2024}:
            trigger_weight = ak.where(
                tv_triggered,
                ak.prod(events.tau_trigger_sf_tauvbf, axis=1, mask_identity=False) * events.vbf_tau_jet_trigger_sf,
                trigger_weight,
            )
            trigger_weight = ak.where(
                v_triggered,
                events.vbf_incl_jet_trigger_sf_tautau,
                trigger_weight,
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
            if quadjet_applied:
                trigger_weight = ak.where(
                    quadjet_triggered,
                    events.quadjet_tau_trigger_sf * events.quadjet_jet_trigger_sf,
                    trigger_weight,
                )
            trigger_weight = ak.where(
                ttv_triggered,
                ak.prod(events[f"tau_trigger_sf_tautauvbf_dm{dm}_{direction}"], axis=1, mask_identity=False) * events.vbf_ditau_jet_trigger_sf,  # noqa: E501
                trigger_weight,
            )
            if self.config_inst.campaign.x.year in {2023, 2024}:
                trigger_weight = ak.where(
                    tv_triggered,
                    ak.prod(events[f"tau_trigger_sf_tauvbf_dm{dm}_{direction}"], axis=1, mask_identity=False) * events.vbf_tau_jet_trigger_sf,  # noqa: E501
                    trigger_weight,
                )
                trigger_weight = ak.where(
                    v_triggered,
                    events.vbf_incl_jet_trigger_sf_tautau,
                    trigger_weight,
                )

            events = set_ak_column_f32(events, f"tautau_trigger_weight_tau_dm{dm}_{direction}", trigger_weight)

        # vbfjet variations
        # take nominal for all triggers and modify only the vbf entries
        trigger_weight = ak.where(
            ttv_triggered,
            (
                ak.prod(events.tau_trigger_sf_tautauvbf, axis=1, mask_identity=False) *
                events[f"vbf_ditau_jet_trigger_sf_{direction}"]
            ),
            events.tautau_trigger_weight,
        )
        if self.config_inst.campaign.x.year in {2023, 2024}:
            trigger_weight = ak.where(
                tv_triggered,
                (
                    ak.prod(events.tau_trigger_sf_tauvbf, axis=1, mask_identity=False) *
                    events[f"vbf_tau_jet_trigger_sf_{direction}"]
                ),
                trigger_weight,
            )
            trigger_weight = ak.where(
                v_triggered,
                events[f"vbf_incl_jet_trigger_sf_tautau_{direction}"],
                trigger_weight,
            )
        events = set_ak_column_f32(events, f"tautau_trigger_weight_vbfjets_{direction}", trigger_weight)

    return events


@tautau_trigger_weight.init
def tautau_trigger_weight_init(self: Producer, **kwargs) -> None:
    # add column to load the raw tau tagger score
    self.uses.add(f"Tau.raw{self.config_inst.x.tau_tagger}VSjet")


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
    mu_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
    e_trigger_passed = ak.zeros_like(events.channel_id, dtype=bool)
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
        ee_trigger_weight, mumu_trigger_weight,
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
        etau_mutau_trigger_weight, tautau_trigger_weight, ee_mumu_trigger_weight, emu_trigger_weight,
    },
    produces={
        "trigger_weight",
        "trigger_weight_{e,mu,jet,vbfjets}_{up,down}",
        IF_RUN_3_2024("trigger_weight_quadjet_{up,down}"),
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
        ("etau", "mu"), ("etau", "jet"), ("etau", "quadjet"),
        ("mutau", "e"), ("mutau", "jet"), ("mutau", "quadjet"),
        ("tautau", "e"), ("tautau", "mu"),
        ("ee", "mu"), ("ee", "jet"), ("ee", "quadjet"), ("ee", "vbfjets"),
        ("ee", "tau_dm0"), ("ee", "tau_dm1"), ("ee", "tau_dm10"), ("ee", "tau_dm11"),
        ("mumu", "e"), ("mumu", "jet"), ("mumu", "quadjet"), ("mumu", "vbfjets"),
        ("mumu", "tau_dm0"), ("mumu", "tau_dm1"), ("mumu", "tau_dm10"), ("mumu", "tau_dm11"),
        ("emu", "jet"), ("emu", "quadjet"), ("emu", "vbfjets"),
        ("emu", "tau_dm0"), ("emu", "tau_dm1"), ("emu", "tau_dm10"), ("emu", "tau_dm11"),
    }

    if self.config_inst.campaign.x.year == 2022:
        undefined_variations = undefined_variations.union({
            ("etau", "vbfjets"),
            ("mutau", "vbfjets"),
        })

    if self.config_inst.campaign.x.year == 2024:
        variation_list = ["e", "mu", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet", "quadjet", "vbfjets"]
    else:
        variation_list = ["e", "mu", "tau_dm0", "tau_dm1", "tau_dm10", "tau_dm11", "jet", "vbfjets"]

    for direction in ["up", "down"]:
        for variation in variation_list:
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
