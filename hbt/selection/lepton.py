# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import law

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hbt.util import IF_NANO_V9, IF_NANO_GE_V10
from hbt.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    threshold: float = 0.5,
    axis: int = 2,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
    leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. When *return_all_matches* is *True*,
    the matrix with all matching decisions is returned as well.
    """
    # delta_r for all combinations
    dr = vectors1.metric_table(vectors2)

    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)

    return any_match


def update_channel_ids(
    events: ak.Array,
    previous_channel_ids: ak.Array,
    correct_channel_id: int,
    channel_mask: ak.Array,
) -> ak.Array:
    """
    Check if the events in the is_mask can be inside the given channel
    or have already been sorted in another channel before.
    """
    events_not_in_channel = (previous_channel_ids != 0) & (previous_channel_ids != correct_channel_id)
    channel_id_overwrite = events_not_in_channel & channel_mask
    if ak.any(channel_id_overwrite):
        raise ValueError(
            "The channel_ids of some events are being set to two different values. "
            "The first event of this chunk concerned has index",
            ak.where(channel_id_overwrite)[0],
        )
    return ak.where(channel_mask, correct_channel_id, previous_channel_ids)


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.dxy", "Electron.dz",
        "Electron.pfRelIso03_all",
        IF_NANO_V9("Electron.mvaFall17V2Iso_WP80", "Electron.mvaFall17V2Iso_WP90", "Electron.mvaFall17V2noIso_WP90"),
        IF_NANO_GE_V10("Electron.mvaIso_WP80", "Electron.mvaIso_WP90", "Electron.mvaNoIso_WP90"),
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    trigger_fire_list: list[bool] | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Electron selection returning two sets of indices for default and veto electrons.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    If the trigger_fire_list is given, all events that are not fired by the trigger lose their trigger
    matching, i.e. the trigger object matching is set to True. This is useful for the emu channel.
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")
    is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-electron mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])
    elif is_cross:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)

    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        # beware that the available Iso should be mvaFall17V2 for run2 files, not Winter22V1,
        # check this in original root files if necessary
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
        # mva_noniso_wp90 = events.Electron.mvaNoIso_WP90
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        # mva_noniso_wp90 = events.Electron.mvaFall17V2noIso_WP90

    # default electron mask, only required for single and cross triggers with electron leg
    default_mask = None
    default_indices = None

    # add true to the leg mask if the trigger is not fired for the emu channel only case,
    # where trigger_fire_list should be given
    if trigger_fire_list is not None:
        matches_leg0 = ak.where(trigger_fire_list, events.Electron.pt > -1, events.Electron.pt < -1)
    if is_single or is_cross or (trigger_fire_list is not None):
        min_pt = 26.0 if is_2016 else (31.0 if is_single else 25.0)
        default_mask = (
            (mva_iso_wp80 == 1) &
            (abs(events.Electron.eta) < 2.5) if is_single else (abs(events.Electron.eta) < 2.1) &
            (abs(events.Electron.dxy) < 0.045) &
            (abs(events.Electron.dz) < 0.2) &
            (events.Electron.pt > min_pt) &
            matches_leg0
        )
        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)

    # veto electron mask
    veto_mask = (
        (
            (mva_iso_wp90 == 1) |
            False
            # disabled as part of the resonant synchronization effort
            # ((mva_noniso_wp90 == 1) & (events.Electron.pfRelIso03_all < 0.3))
        ) &
        (abs(events.Electron.eta) < 2.5) &
        (abs(events.Electron.dxy) < 0.045) &
        (abs(events.Electron.dz) < 0.2) &
        (events.Electron.pt > 10.0)
    )
    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    return default_indices, veto_indices


@selector(
    uses={
        # nano columns
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mediumId", "Muon.tightId", "Muon.pfRelIso04_all",
        "Muon.dxy", "Muon.dz",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
    },
    exposed=False,
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    select_without_trigger: bool = False,
    mumu_selection: bool = False,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Muon selection returning two sets of indidces for default and veto muons.

    References:

    - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
    - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15

    If mumu_selection is set to True, a second muon is selected and the corresponding indices are additionally returned.
    """
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")
    is_2016 = self.config_inst.campaign.x.year == 2016

    if (is_single or is_cross or mumu_selection) and select_without_trigger:
        raise ValueError("select_without_trigger can only be used for non-triggered muon selections")

    if is_cross and mumu_selection:
        raise ValueError("mumu_selection can only be used for single muon selections")

    # start per-muon mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])
        if mumu_selection:
            # TODO: check with Jona if the first Muon matched to trigger
            # is the one with highest pt before or after selection cuts
            matches_leg0 = ak.where(
                ak.local_index(events.Muon) == 0,
                trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]]),
                False,
            )
            matches_second_muon = ~matches_leg0
    elif is_cross:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])
    elif select_without_trigger:
        matches_leg0 = events.Muon.pt > -1

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)

    # default muon mask, only required for single and cross triggers with muon leg
    default_mask = None
    default_indices = None
    if is_single or is_cross or select_without_trigger:
        if is_2016:
            min_pt = 23.0 if is_single else 20.0
        else:
            min_pt = 26.0 if is_single else 22.0
        default_mask_wo_trigger = (
            (events.Muon.tightId == 1) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.045) &
            (abs(events.Muon.dz) < 0.2) &
            (events.Muon.pfRelIso04_all < 0.15) &
            (events.Muon.pt > min_pt)
        )

        default_mask = default_mask_wo_trigger & matches_leg0

        if mumu_selection:
            default_mask_second_muon = default_mask_wo_trigger & matches_second_muon

        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)
        if mumu_selection:
            default_indices_second_muon = sorted_indices[default_mask_second_muon[sorted_indices]]
            default_indices_second_muon = ak.values_astype(default_indices_second_muon, np.int32)

    # veto muon mask
    veto_mask = (
        ((events.Muon.mediumId == 1) | (events.Muon.tightId == 1)) &
        (abs(events.Muon.eta) < 2.4) &
        (abs(events.Muon.dxy) < 0.045) &
        (abs(events.Muon.dz) < 0.2) &
        (events.Muon.pfRelIso04_all < 0.3) &
        (events.Muon.pt > 10)
    )
    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    if mumu_selection:
        return default_indices, default_indices_second_muon, veto_indices
    else:
        return default_indices, veto_indices


@selector(
    uses={
        # nano columns
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.dz",
        "Tau.decayMode",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
        "Electron.pt", "Electron.eta", "Electron.phi",
        "Muon.pt", "Muon.eta", "Muon.phi",
    },
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    electron_indices: ak.Array,
    muon_indices: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a set of indices for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select the action Medium isolated ones, eventually to separate normal and
    iso inverted taus for QCD estimations.
    """
    # return empty mask if no tagged taus exists in the chunk
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        # convenient definition of empty indices and iso mask
        empty_indices = ak.local_index(events.Tau)
        return empty_indices, empty_indices == 0

    is_single_e = trigger.has_tag("single_e")
    is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_any_cross_tau = is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_run3 = self.config_inst.campaign.x.run == 3
    get_tau_tagger = lambda tag: f"id{self.config_inst.x.tau_tagger}VS{tag}"

    wp_config = self.config_inst.x.tau_id_working_points

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[1].pdg_id) == 15
        # match leg 1
        matches_leg1 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[1]])
    elif is_any_cross_tau:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) >= 2
        assert abs(trigger.legs[0].pdg_id) == 15
        assert abs(trigger.legs[1].pdg_id) == 15
        # match both legs
        matches_leg0 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[0]])
        matches_leg1 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[1]])

    # determine minimum pt and maximum eta
    max_eta = 2.5
    if is_single_e or is_single_mu:
        min_pt = 20.0
    elif is_cross_e:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 35.0
    elif is_cross_mu:
        min_pt = 25.0 if is_2016 else 32.0
    elif is_cross_tau:
        min_pt = 40.0
    elif is_cross_tau_vbf:
        # only existing after 2016
        min_pt = 0.0 if is_2016 else 25.0
    elif is_cross_tau_jet:
        min_pt = None if not is_run3 else 35.0

    # select which decay modes to consider
    decay_mode_mask = reduce(or_,
        [
            events.Tau.decayMode == 0,
            events.Tau.decayMode == 1,
            events.Tau.decayMode == 10,
            events.Tau.decayMode == 11,
        ],
    )

    # base tau mask for default and qcd sideband tau
    base_mask = (
        (abs(events.Tau.eta) < max_eta) &
        (events.Tau.pt > min_pt) &
        (abs(events.Tau.dz) < 0.2) &
        decay_mode_mask &
        (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose) &
        (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight) &
        (events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.vvvloose)
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_indices is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_indices]) > 0.5, axis=2)
    if muon_indices is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_indices]) > 0.5, axis=2)

    # add trigger object masks
    if is_cross_e or is_cross_mu:
        base_mask = base_mask & matches_leg1
    elif is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet:
        # taus need to be matched to at least one leg, but as a side condition
        # each leg has to have at least one match to a tau
        base_mask = base_mask & (
            (matches_leg0 | matches_leg1) &
            ak.any(matches_leg0, axis=1) &
            ak.any(matches_leg1, axis=1)
        )

    # indices for sorting first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt))) + 1)
    sort_key = events.Tau[get_tau_tagger("jet")] * f + events.Tau.pt
    sorted_indices = ak.argsort(sort_key, axis=-1, ascending=False)

    # convert to sorted indices
    base_indices = sorted_indices[base_mask[sorted_indices]]
    base_indices = ak.values_astype(base_indices, np.int32)

    # additional mask to select final, Medium isolated taus
    iso_mask = events.Tau[base_indices][get_tau_tagger("jet")] >= wp_config.tau_vs_jet.medium

    return base_indices, iso_mask


@tau_selection.init
def tau_selection_init(self: Selector) -> None:
    # register tec shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("tec")
    }

    # Add columns for the right tau tagger
    self.uses |= {
        f"Tau.id{self.config_inst.x.tau_tagger}VS{tag}"
        for tag in ("e", "mu", "jet")
    }


@selector(
    uses={
        electron_selection, muon_selection, tau_selection,
        # nano columns
        "event", "Electron.charge", "Muon.charge", "Tau.charge", "Electron.mass", "Muon.mass",
        "Tau.mass",
    },
    produces={
        electron_selection, muon_selection, tau_selection,
        # new columns
        "channel_id", "leptons_os", "tau2_isolated", "single_triggered", "cross_triggered",
    },
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.
    """
    # get channels from the config
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")
    ch_mumu = self.config_inst.get_channel("mumu")
    ch_emu = self.config_inst.get_channel("emu")

    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask
    tau2_isolated = false_mask
    leptons_os = false_mask
    single_triggered = false_mask
    cross_triggered = false_mask
    empty_indices = events.Tau[:, :0].charge * 1  # ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    sel_electron_indices = empty_indices
    sel_muon_indices = empty_indices
    sel_tau_indices = empty_indices
    leading_taus = events.Tau[:, :0]

    # perform each lepton election step separately per trigger, avoid caching
    sel_kwargs = {**kwargs, "call_force": True}
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        is_single = trigger.has_tag("single_trigger")
        is_cross = trigger.has_tag("cross_trigger")

        # electron selection
        electron_indices, electron_veto_indices = self[electron_selection](
            events,
            trigger,
            leg_masks,
            **sel_kwargs,
        )

        # muon selection
        muon_indices, muon_veto_indices = self[muon_selection](
            events,
            trigger,
            leg_masks,
            **sel_kwargs,
        )

        # tau selection
        tau_indices, tau_iso_mask = self[tau_selection](
            events,
            trigger,
            leg_masks,
            electron_indices,
            muon_indices,
            **sel_kwargs,
        )

        # lepton pair selecton per trigger via lepton counting

        if trigger.has_tag({"single_e", "cross_e_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("etau")
        ):
            # expect 1 electron, 1 veto electron (the same one), 0 veto muons, and at least one tau
            is_etau = (
                trigger_fired &
                (ak.num(electron_indices, axis=1) == 1) &
                (ak.num(electron_veto_indices, axis=1) == 1) &
                (ak.num(muon_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 1)
            )
            is_iso = ak.sum(tau_iso_mask, axis=1) >= 1
            # determine the os/ss charge sign relation
            e_charge = ak.firsts(events.Electron[electron_indices].charge, axis=1)
            tau_charge = ak.firsts(events.Tau[tau_indices].charge, axis=1)
            is_os = e_charge == -tau_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_etau.id, is_etau)
            tau2_isolated = ak.where(is_etau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_etau, is_os, leptons_os)
            single_triggered = ak.where(is_etau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_etau & is_cross, True, cross_triggered)
            sel_electron_indices = ak.where(is_etau, electron_indices, sel_electron_indices)
            sel_tau_indices = ak.where(is_etau, tau_indices, sel_tau_indices)
            leading_taus = ak.where(is_etau, events.Tau[tau_indices[:, :1]], leading_taus)

        elif trigger.has_tag({"single_mu", "cross_mu_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("mutau")
        ):
            # expect 1 muon, 1 veto muon (the same one), 0 veto electrons, and at least one tau
            is_mutau = (
                trigger_fired &
                (ak.num(muon_indices, axis=1) == 1) &
                (ak.num(muon_veto_indices, axis=1) == 1) &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 1)
            )
            is_iso = ak.sum(tau_iso_mask, axis=1) >= 1
            # determine the os/ss charge sign relation
            mu_charge = ak.firsts(events.Muon[muon_indices].charge, axis=1)
            tau_charge = ak.firsts(events.Tau[tau_indices].charge, axis=1)
            is_os = mu_charge == -tau_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_mutau.id, is_mutau)
            tau2_isolated = ak.where(is_mutau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_mutau, is_os, leptons_os)
            single_triggered = ak.where(is_mutau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_mutau & is_cross, True, cross_triggered)
            sel_muon_indices = ak.where(is_mutau, muon_indices, sel_muon_indices)
            sel_tau_indices = ak.where(is_mutau, tau_indices, sel_tau_indices)
            leading_taus = ak.where(is_mutau, events.Tau[tau_indices[:, :1]], leading_taus)

        elif trigger.has_tag({"cross_tau_tau", "cross_tau_tau_vbf", "cross_tau_tau_jet"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("tautau")
        ):
            # expect 0 veto electrons, 0 veto muons and at least two taus of which one is isolated
            is_tautau = (
                trigger_fired &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (ak.num(muon_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 2) &
                (ak.sum(tau_iso_mask, axis=1) >= 1)
            )
            # special case for cross tau vbf trigger:
            # to avoid overlap, with non-vbf triggers, only one tau is allowed to have pt > 40
            if trigger.has_tag("cross_tau_tau_vbf"):
                is_tautau = is_tautau & (ak.sum(events.Tau[tau_indices].pt > 40, axis=1) <= 1)
            is_iso = ak.sum(tau_iso_mask, axis=1) >= 2
            # tau_indices are sorted by highest isolation as cond. 1 and highest pt as cond. 2, so
            # the first two indices are exactly those selected by the full-blown pairing algorithm
            # and there is no need here to apply it again :)
            # determine the os/ss charge sign relation
            tau1_charge = ak.firsts(events.Tau[tau_indices].charge, axis=1)
            tau2_charge = ak.firsts(events.Tau[tau_indices].charge[..., 1:], axis=1)
            is_os = tau1_charge == -tau2_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_tautau.id, is_tautau)
            tau2_isolated = ak.where(is_tautau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_tautau, is_os, leptons_os)
            single_triggered = ak.where(is_tautau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_tautau & is_cross, True, cross_triggered)
            sel_tau_indices = ak.where(is_tautau, tau_indices, sel_tau_indices)
            leading_taus = ak.where(is_tautau, events.Tau[tau_indices[:, :2]], leading_taus)

        # control regions
        if trigger.has_tag({"single_mu"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("mumu")
        ):
            # TODO: Ask Jona if trigger should be matched to the muon with highest pt before or after selection cuts
            # muon selection
            first_muon_indices, second_muon_indices, muon_veto_indices = self[muon_selection](
                events,
                trigger,
                leg_masks,
                mumu_selection=True,
                **sel_kwargs,
            )

            mumu_muon_indices = ak.concatenate([first_muon_indices, second_muon_indices], axis=1)

            # expect 2 muons, 2 veto muons, 0 veto electrons, and ignore the taus
            is_mumu = (
                trigger_fired &
                (ak.num(first_muon_indices, axis=1) == 1) &
                (ak.num(second_muon_indices, axis=1) == 1) &
                (ak.num(muon_veto_indices, axis=1) == 2) &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 0)  # to remove?
            )
            # store necessary global variables
            channel_id = update_channel_ids(events, channel_id, ch_mumu.id, is_mumu)
            sel_muon_indices = ak.where(is_mumu, mumu_muon_indices, sel_muon_indices)
            single_triggered = ak.where(is_mumu & is_single, True, single_triggered)
            cross_triggered = ak.where(is_mumu & is_cross, True, cross_triggered)

            # define fake iso regions for mumu, as there is not necessarily a tau to be isolated
            # should be always false, as no tau are used
            is_iso = ak.sum(tau_iso_mask, axis=1) < 0
            # determine the os/ss charge sign relation
            mu1_charge = ak.firsts(events.Muon[muon_indices].charge, axis=1)
            mu2_charge = ak.firsts(events.Muon[muon_indices].charge[..., 1:], axis=1)
            is_os = mu1_charge == -mu2_charge
            # store global variables
            tau2_isolated = ak.where(is_mumu, is_iso, tau2_isolated)
            leptons_os = ak.where(is_mumu, is_os, leptons_os)
            print("number events in mumu channel", ak.sum(is_mumu))

        # emu channel
        if (
            (trigger.has_tag({"single_e"}) and (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_e"))) or
            (trigger.has_tag({"single_mu"}) and (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_mu")))
        ):

            # behavior for Single Muon dataset
            if trigger.has_tag({"single_mu"}) and (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_mu")):
                for trigger_emu, trigger_fired_emu, leg_masks_emu in trigger_results.x.trigger_data:
                    # verify that the single electron trigger is matched if the single electron trigger is fired
                    # if not, the matching is not verified in the selection.
                    # This is done by giving the electron selection the trigger_fired_mask

                    # TODO: handle the case where there are several single e triggers (maybe not necessary?)
                    # as of now, only the last single electron trigger in the list of triggers applied to the dataset
                    # is used
                    if trigger_emu.has_tag("single_e"):
                        electron_indices, electron_veto_indices = self[electron_selection](
                            events,
                            trigger_emu,
                            leg_masks,
                            trigger_fire_list=trigger_fired_emu,
                            **sel_kwargs,
                        )
                not_muon_in_e_trigger_fired = True

            # behavior for Single Electron dataset
            elif trigger.has_tag({"single_e"}) and (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_e")):
                muon_indices, muon_veto_indices = self[muon_selection](
                    events,
                    trigger,
                    leg_masks,
                    select_without_trigger=True,
                    **sel_kwargs,
                )
                muon_triggers_fire_list = []
                for trigger_emu, trigger_fired_emu, leg_masks_emu in trigger_results.x.trigger_data:
                    if trigger_emu.has_tag("single_mu"):
                        muon_triggers_fire_list += [trigger_fired_emu]
                muon_trigger_fired = reduce(
                    or_,
                    muon_triggers_fire_list,
                )
                not_muon_in_e_trigger_fired = ~muon_trigger_fired

            # general emu channel selection
            # expect 1 electron, 1 muon, 1 veto electron, 1 veto muon, and ignore taus
            is_emu = (
                trigger_fired & not_muon_in_e_trigger_fired &
                (ak.num(electron_indices, axis=1) == 1) &
                (ak.num(electron_veto_indices, axis=1) == 1) &
                (ak.num(muon_indices, axis=1) == 1) &
                (ak.num(muon_veto_indices, axis=1) == 1) &
                (ak.num(tau_indices, axis=1) >= 0)  # to remove?
            )

            # store necessary global variables
            channel_id = update_channel_ids(events, channel_id, ch_emu.id, is_emu)
            sel_electron_indices = ak.where(is_emu, electron_indices, sel_electron_indices)
            sel_muon_indices = ak.where(is_emu, muon_indices, sel_muon_indices)
            single_triggered = ak.where(is_emu & is_single, True, single_triggered)
            cross_triggered = ak.where(is_emu & is_cross, True, cross_triggered)

            # define fake iso regions for emu, as there is not necessarily a tau to be isolated
            # should be always false, as no tau are used
            is_iso = ak.sum(tau_iso_mask, axis=1) < 0
            # determine the os/ss charge sign relation
            mu_charge = ak.firsts(events.Muon[muon_indices].charge, axis=1)
            e_charge = ak.firsts(events.Electron[electron_indices].charge, axis=1)
            is_os = mu_charge == -e_charge
            # store global variables
            tau2_isolated = ak.where(is_emu, is_iso, tau2_isolated)
            leptons_os = ak.where(is_emu, is_os, leptons_os)
            print("number events in emu channel", ak.sum(is_emu))

    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    leptons_os = ak.fill_none(leptons_os, False)
    sel_electron_indices = ak.values_astype(sel_electron_indices, np.int32)
    sel_muon_indices = ak.values_astype(sel_muon_indices, np.int32)
    sel_tau_indices = ak.values_astype(sel_tau_indices, np.int32)

    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "tau2_isolated", tau2_isolated)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "cross_triggered", cross_triggered)

    return events, SelectionResult(
        steps={
            "lepton": channel_id != 0,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
            },
            "Muon": {
                "Muon": sel_muon_indices,
            },
            "Tau": {
                "Tau": sel_tau_indices,
            },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    events.Muon[sel_muon_indices] * 1,
                    events.Tau[sel_tau_indices] * 1,
                ],
                axis=1,
            )[:, :2],

            # save the leading taus for the duration of the selection
            # exactly 1 for etau/mutau and exactly 2 for tautau
            "leading_taus": leading_taus,
        },
    )
