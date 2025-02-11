# coding: utf-8

"""
Custom trigger scale factor production.

Note : The trigger weights producers multiply the sfs for all objects in an event to get the total
trigger scale factor of the event. Since we might want to use different objects in different channels,
we will derive the trigger weights producers for each channel separately to apply the correct masks.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.production.cms.muon import muon_weights, muon_trigger_weights
from columnflow.production.cms.electron import electron_weights, electron_trigger_weights


from hbt.production.tau import tau_trigger_weights

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

# subclass the electron weights producer to create the electron efficiencies
single_trigger_electron_data_effs = electron_weights.derive(
    "single_trigger_electron_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_data_effs_names),
        "weight_name": "single_trigger_electron_data_effs",
    },
)

single_trigger_electron_mc_effs = electron_weights.derive(
    "single_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_sf),
        "get_electron_config": (lambda self: self.config_inst.x.single_trigger_electron_mc_effs_names),
        "weight_name": "single_trigger_electron_mc_effs",
    },
)

cross_trigger_electron_data_effs = electron_weights.derive(
    "cross_trigger_electron_data_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_trigger_electron_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_data_effs_names),
        "weight_name": "cross_trigger_electron_data_effs",
    },
)

cross_trigger_electron_mc_effs = electron_weights.derive(
    "cross_trigger_electron_mc_effs",
    cls_dict={
        "get_electron_file": (lambda self, external_files: external_files.cross_trigger_electron_sf),
        "get_electron_config": (lambda self: self.config_inst.x.cross_trigger_electron_mc_effs_names),
        "weight_name": "cross_trigger_electron_mc_effs",
    },
)

# subclass the muon weights producer to create the muon efficiencies
single_trigger_muon_data_effs = muon_weights.derive(
    "single_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_data_effs_names),
        "weight_name": "single_trigger_muon_data_effs",
    },
)

single_trigger_muon_mc_effs = muon_weights.derive(
    "single_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_trigger_sf),
        "get_muon_config": (lambda self: self.config_inst.x.single_trigger_muon_mc_effs_names),
        "weight_name": "single_trigger_muon_mc_effs",
    },
)

cross_trigger_muon_data_effs = muon_weights.derive(
    "cross_trigger_muon_data_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_trigger_muon_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_data_effs_names),
        "weight_name": "cross_trigger_muon_data_effs",
    },
)

cross_trigger_muon_mc_effs = muon_weights.derive(
    "cross_trigger_muon_mc_effs",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.cross_trigger_muon_sf),
        "get_muon_config": (lambda self: self.config_inst.x.cross_trigger_muon_mc_effs_names),
        "weight_name": "cross_trigger_muon_mc_effs",
    },
)

# subclass the tau weights producer to use the cclub tau efficiencies
tau_trigger_sf_and_effs_cclub = tau_trigger_weights.derive(
    "tau_trigger_sf_and_effs_cclub",
    cls_dict={
        "get_tau_file": (lambda self, external_files: external_files.tau_trigger_sf),
        "get_tau_corrector": (lambda self: self.config_inst.x.cclub_tau_corrector),
        "weight_name": "tau_trigger_sf_and_effs_cclub",
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
        single_trigger_electron_data_effs, cross_trigger_electron_data_effs,
        single_trigger_electron_mc_effs, cross_trigger_electron_mc_effs,
        single_trigger_muon_data_effs, cross_trigger_muon_data_effs,
        single_trigger_muon_mc_effs, cross_trigger_muon_mc_effs,
        tau_trigger_sf_and_effs_cclub,
    },
    produces={
        "etau_trigger_weight", "mutau_trigger_weight",
    } | {
        f"mutau_trigger_weight_muon_{direction}"
        for direction in ["up", "down"]
    } | {
        f"etau_trigger_weight_electron_{direction}"
        for direction in ["up", "down"]
    } | {
        f"mutau_trigger_weight_tau_mu_{direction}"
        for direction in ["up", "down"]
    } | {
        f"etau_trigger_weight_tau_e_{direction}"
        for direction in ["up", "down"]
    },
)
def etau_mutau_trigger_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer for muon trigger scale factors derived by Jona Motta. Requires external files in the
    config under ``muon_trigger_sf`` and ``cross_trigger_muon_sf``:

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
            for channel in ["etau", "mutau"]:
                if channel == "etau":
                    channel_id = self.config_inst.channels.n.etau.id
                    single_trigger_lepton_data_efficiencies = events.single_trigger_electron_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_electron_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_electron_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_electron_mc_effs
                else:
                    channel_id = self.config_inst.channels.n.mutau.id
                    single_trigger_lepton_data_efficiencies = events.single_trigger_muon_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_muon_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_muon_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_muon_mc_effs

                # tau efficiencies
                cross_trigger_tau_data_efficiencies = events.tau_trigger_eff_data_weight
                cross_trigger_tau_mc_efficiencies = events.tau_trigger_eff_mc_weight

                single_triggered = (events.channel_id == channel_id) & events.single_triggered
                cross_triggered = (events.channel_id == channel_id) & events.cross_triggered

                trigger_efficiency_data = calculate_correlated_ditrigger_efficiency(
                    single_triggered,
                    cross_triggered,
                    single_trigger_lepton_data_efficiencies,
                    cross_trigger_lepton_data_efficiencies,
                    cross_trigger_tau_data_efficiencies,
                )

                trigger_efficiency_mc = calculate_correlated_ditrigger_efficiency(
                    single_triggered,
                    cross_triggered,
                    single_trigger_lepton_mc_efficiencies,
                    cross_trigger_lepton_mc_efficiencies,
                    cross_trigger_tau_mc_efficiencies,
                )

                # calculate SFs

                # electron
                trigger_sf = trigger_efficiency_data / trigger_efficiency_mc

                trigger_sf_no_nan = np.nan_to_num(trigger_sf, nan=EMPTY_FLOAT)

                events = set_ak_column_f32(events, f"{channel}_trigger_weight{postfix}", trigger_sf_no_nan)

        else:
            # create all variations
            for uncert in ["_electron", "_muon", "_tau_mu", "_tau_e"]:
                if uncert == "_electron" or uncert == "_tau_e":
                    channel = "etau"
                else:
                    channel = "mutau"
                if channel == "etau" and uncert == "_electron":
                    channel_id = self.config_inst.channels.n.etau.id
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

                elif channel == "etau" and uncert == "_tau_e":
                    channel_id = self.config_inst.channels.n.etau.id
                    single_trigger_lepton_data_efficiencies = events.single_trigger_electron_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_electron_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_electron_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_electron_mc_effs

                    # tau efficiencies
                    cross_trigger_tau_data_efficiencies = Route(
                        f"tau_trigger_eff_data_weight_{channel}{postfix}",
                    ).apply(events)
                    cross_trigger_tau_mc_efficiencies = Route(
                        f"tau_trigger_eff_mc_weight_{channel}{postfix}",
                    ).apply(events)

                elif channel == "mutau" and uncert == "_muon":
                    channel_id = self.config_inst.channels.n.mutau.id
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

                elif channel == "mutau" and uncert == "_tau_mu":
                    single_trigger_lepton_data_efficiencies = events.single_trigger_muon_data_effs
                    cross_trigger_lepton_data_efficiencies = events.cross_trigger_muon_data_effs
                    single_trigger_lepton_mc_efficiencies = events.single_trigger_muon_mc_effs
                    cross_trigger_lepton_mc_efficiencies = events.cross_trigger_muon_mc_effs

                    # tau efficiencies
                    cross_trigger_tau_data_efficiencies = Route(
                        f"tau_trigger_eff_data_weight_{channel}{postfix}",
                    ).apply(events)
                    cross_trigger_tau_mc_efficiencies = Route(
                        f"tau_trigger_eff_mc_weight_{channel}{postfix}",
                    ).apply(events)

                else:
                    raise ValueError(f"Unknown channel {channel} and uncertainty {uncert}")

                single_triggered = (events.channel_id == channel_id) & events.single_triggered
                cross_triggered = (events.channel_id == channel_id) & events.cross_triggered

                trigger_efficiency_data = calculate_correlated_ditrigger_efficiency(
                    single_triggered,
                    cross_triggered,
                    single_trigger_lepton_data_efficiencies,
                    cross_trigger_lepton_data_efficiencies,
                    cross_trigger_tau_data_efficiencies,
                )

                trigger_efficiency_mc = calculate_correlated_ditrigger_efficiency(
                    single_triggered,
                    cross_triggered,
                    single_trigger_lepton_mc_efficiencies,
                    cross_trigger_lepton_mc_efficiencies,
                    cross_trigger_tau_mc_efficiencies,
                )

                # calculate SFs

                # electron
                trigger_sf = trigger_efficiency_data / trigger_efficiency_mc

                trigger_sf_no_nan = np.nan_to_num(trigger_sf, nan=EMPTY_FLOAT)

                events = set_ak_column_f32(events, f"{channel}_trigger_weight{uncert}{postfix}", trigger_sf_no_nan)
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
        "channel_id", "trigger_ids",
        emu_e_trigger_weights, emu_mu_trigger_weights,
    },
    produces={
        "emu_trigger_weights",
    } | {
        f"emu_trigger_weights_muon_{direction}"
        for direction in ["up", "down"]
    } | {
        f"emu_trigger_weights_electron_{direction}"
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
            mu_trigger_passed = mu_trigger_passed | np.any(events.trigger_ids == trigger.id, axis=-1)
        if trigger.has_tag("single_e"):
            e_trigger_passed = e_trigger_passed | np.any(events.trigger_ids == trigger.id, axis=-1)

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

    # TODO: check if the SFs variations are correct, calculated differently by CCLUB
    # https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/0bc24612210b2fb7973669ccdd21f0896edde084/src/HHTrigSFinterface.cc#L345
    for postfix in ["", "_electron_up", "_electron_down", "_muon_up", "_muon_down"]:
        # start with the nominal case
        if postfix == "":
            trigger_sf = events.emu_e_trigger_weights * events.emu_mu_trigger_weights
        elif postfix.startswith("_electron"):
            shift = postfix.split("_")[-1]
            # electron shifts
            trigger_sf = Route(f"emu_e_trigger_weights_{shift}").apply(events) * events.emu_mu_trigger_weights
        elif postfix.startswith("_muon"):
            shift = postfix.split("_")[-1]
            trigger_sf = events.emu_e_trigger_weights * Route(f"emu_mu_trigger_weights_{shift}").apply(events)
        else:
            raise ValueError(f"Unknown postfix {postfix}")
        events = set_ak_column_f32(events, f"emu_trigger_weights{postfix}", trigger_sf)

    return events


@producer(
    uses={
        etau_mutau_trigger_weights,
        # tau_tau_trigger_weights,
        ee_trigger_weights,
        mumu_trigger_weights, emu_trigger_weights,
    },
    produces={
        etau_mutau_trigger_weights,
        # tau_tau_trigger_weights,
        ee_trigger_weights,
        mumu_trigger_weights, emu_trigger_weights,
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

    # # tautau
    # events = self[tau_tau_trigger_weights](events, **kwargs)

    # ee and mumu
    ee_mask = (events.channel_id == self.config_inst.channels.n.ee.id) & (ak.local_index(events.Electron.pt) == 0)
    mumu_mask = (events.channel_id == self.config_inst.channels.n.mumu.id) & (ak.local_index(events.Muon.pt) == 0)
    events = self[ee_trigger_weights](events, electron_mask=ee_mask, **kwargs)
    events = self[mumu_trigger_weights](events, muon_mask=mumu_mask, **kwargs)

    # emu
    events = self[emu_trigger_weights](events, **kwargs)

    return events
