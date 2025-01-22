# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import law

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import (
    set_ak_column, sorted_indices_from_mask, flat_np_view, full_like,
)
from columnflow.util import maybe_import

from hbt.util import IF_NANO_V9, IF_NANO_GE_V10
from hbt.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    /,
    *,
    threshold: float = 0.5,
    axis: int = 2,
    event_mask: ak.Array | type(Ellipsis) | None = None,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
    leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. If an *event_mask* is given, the
    the matching is performed only for those events, but a full object mask with the same shape as
    that of *vectors1* is returned, which all objects set to *False* where not matching was done.
    """
    # handle event masks
    used_event_mask = event_mask is not None and event_mask is not Ellipsis
    event_mask = Ellipsis if event_mask is None else event_mask

    # delta_r for all combinations
    dr = vectors1[event_mask].metric_table(vectors2[event_mask])

    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)

    # expand to original shape if an event mask was given
    if used_event_mask:
        full_any_match = full_like(vectors1.pt, False, dtype=bool)
        flat_full_any_match = flat_np_view(full_any_match)
        flat_full_any_match[flat_np_view(full_any_match | event_mask)] = flat_np_view(any_match)
        any_match = full_any_match

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
        "Electron.{pt,eta,phi,dxy,dz,pfRelIso03_all,seediEtaOriX,seediPhiOriY}",
        IF_NANO_V9("Electron.mvaFall17V2{Iso_WP80,Iso_WP90}"),
        IF_NANO_GE_V10("Electron.{mvaIso_WP80,mvaIso_WP90}"),
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array | None, ak.Array]:
    """
    Electron selection returning two sets of masks for default and veto electrons.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_2022_post = (
        self.config_inst.campaign.x.year == 2022 and
        self.config_inst.campaign.has_tag("postEE")
    )
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")

    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        # beware that the available Iso should be mvaFall17V2 for run2 files, not Winter22V1,
        # check this in original root files if necessary
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90

    # default electron mask
    default_mask = None
    if is_single or is_cross:
        min_pt = 26.0 if is_2016 else (31.0 if is_single else 25.0)
        max_eta = 2.5 if is_single else 2.1
        default_mask = (
            (mva_iso_wp80 == 1) &
            (abs(events.Electron.eta) < max_eta) &
            (abs(events.Electron.dxy) < 0.045) &
            (abs(events.Electron.dz) < 0.2) &
            (events.Electron.pt > min_pt)
        )

        # additional cut in 2022 post-EE
        # see https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis?rev=162#From_ECAL_and_EGM
        if is_2022_post:
            default_mask = default_mask & ~(
                (events.Electron.eta > 1.556) &
                (events.Electron.seediEtaOriX < 45) &
                (events.Electron.seediPhiOriY > 72)
            )

    # veto electron mask (must be trigger independent!)
    veto_mask = (
        (mva_iso_wp90 == 1) &
        (abs(events.Electron.eta) < 2.5) &
        (abs(events.Electron.dxy) < 0.045) &
        (abs(events.Electron.dz) < 0.2) &
        (events.Electron.pt > 10.0)
    )

    return default_mask, veto_mask


@selector(
    uses={"{Electron,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def electron_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Electron trigger matching.
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")

    # catch config errors
    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["e"].pdg_id) == 11

    return trigger_object_matching(
        events.Electron,
        events.TrigObj[leg_masks["e"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={"Muon.{pt,eta,phi,mediumId,tightId,pfRelIso04_all,dxy,dz}"},
    exposed=False,
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    **kwargs,
) -> tuple[ak.Array | None, ak.Array]:
    """
    Muon selection returning two sets of masks for default and veto muons.

    References:

    - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
    - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15
    """
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")

    # default muon mask
    default_mask = None
    if is_single or is_cross:
        if is_2016:
            min_pt = 23.0 if is_single else 20.0
        else:
            min_pt = 26.0 if is_single else 22.0
        default_mask = (
            (events.Muon.tightId == 1) &
            (abs(events.Muon.eta) < 2.4) &
            (abs(events.Muon.dxy) < 0.045) &
            (abs(events.Muon.dz) < 0.2) &
            (events.Muon.pfRelIso04_all < 0.15) &
            (events.Muon.pt > min_pt)
        )

    # veto muon mask (must be trigger independent!)
    veto_mask = (
        ((events.Muon.mediumId == 1) | (events.Muon.tightId == 1)) &
        (abs(events.Muon.eta) < 2.4) &
        (abs(events.Muon.dxy) < 0.045) &
        (abs(events.Muon.dz) < 0.2) &
        (events.Muon.pfRelIso04_all < 0.3) &
        (events.Muon.pt > 10)
    )

    return default_mask, veto_mask


@selector(
    uses={"{Muon,TrigObj}.{pt,eta,phi}"},
    exposed=False,
)
def muon_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Muon trigger matching.
    """
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")

    # catch config errors
    assert is_single or is_cross
    assert trigger.n_legs == len(leg_masks) == (1 if is_single else 2)
    assert abs(trigger.legs["mu"].pdg_id) == 13

    return trigger_object_matching(
        events.Muon,
        events.TrigObj[leg_masks["mu"]],
        event_mask=trigger_fired,
    )


@selector(
    uses={
        "Tau.{pt,eta,phi,dz,decayMode}",
        "{Electron,Muon,TrigObj}.{pt,eta,phi}",
    },
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    electron_mask: ak.Array | None,
    muon_mask: ak.Array | None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a masks for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select isolated ones, eventually to separate normal and iso inverted taus
    for QCD estimations.
    """
    # return empty mask if no tagged taus exists in the chunk
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        false_mask = full_like(events.Tau.pt, False, dtype=bool)
        return false_mask, false_mask

    is_single_e = trigger.has_tag("single_e")
    is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_run3 = self.config_inst.campaign.x.run == 3
    get_tau_tagger = lambda tag: f"id{self.config_inst.x.tau_tagger}VS{tag}"
    wp_config = self.config_inst.x.tau_id_working_points

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

    # base tau mask for default and qcd sideband tau
    base_mask = (
        (abs(events.Tau.eta) < max_eta) &
        (events.Tau.pt > min_pt) &
        (abs(events.Tau.dz) < 0.2) &
        reduce(or_, [events.Tau.decayMode == mode for mode in (0, 1, 10, 11)]) &
        (events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.vvvloose)
        # vs e and mu cuts are channel dependent and thus applied in the overall lepton selection
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_mask]) > 0.5, axis=2)
    if muon_mask is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_mask]) > 0.5, axis=2)

    # compute the isolation mask separately as it is used to defined (qcd) categories later on
    iso_mask = events.Tau[get_tau_tagger("jet")] >= wp_config.tau_vs_jet.medium

    return base_mask, iso_mask


@selector(
    uses={"{Tau,TrigObj}.{pt,eta,phi}"},
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    **kwargs,
) -> tuple[ak.Array]:
    """
    Tau trigger matching.
    """
    if ak.all(ak.num(events.Tau) == 0):
        logger.info("no taus found in event chunk")
        return full_like(events.Tau.pt, False, dtype=bool)

    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_any_cross_tau = is_cross_tau or is_cross_tau_vbf or is_cross_tau_jet
    assert is_cross_e or is_cross_mu or is_any_cross_tau

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs["tau"].pdg_id) == 15
        # match leg 1
        return trigger_object_matching(
            events.Tau,
            events.TrigObj[leg_masks["tau"]],
            event_mask=trigger_fired,
        )

    # is_any_cross_tau
    # catch config errors
    assert trigger.n_legs == len(leg_masks) >= 2
    assert abs(trigger.legs["tau1"].pdg_id) == 15
    assert abs(trigger.legs["tau2"].pdg_id) == 15

    # match both legs
    matches_leg0 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau1"]],
        event_mask=trigger_fired,
    )
    matches_leg1 = trigger_object_matching(
        events.Tau,
        events.TrigObj[leg_masks["tau2"]],
        event_mask=trigger_fired,
    )

    # taus need to be matched to at least one leg, but as a side condition
    # each leg has to have at least one match to a tau
    matches = (
        (matches_leg0 | matches_leg1) &
        ak.any(matches_leg0, axis=1) &
        ak.any(matches_leg1, axis=1)
    )

    return matches


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
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
        "event", "{Electron,Muon,Tau}.{charge,mass}",
    },
    produces={
        electron_selection, electron_trigger_matching, muon_selection, muon_trigger_matching,
        tau_selection, tau_trigger_matching,
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
    wp_config = self.config_inst.x.tau_id_working_points
    get_tau_tagger = lambda tag: f"id{self.config_inst.x.tau_tagger}VS{tag}"

    # get channels from the config
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")
    ch_ee = self.config_inst.get_channel("ee")
    ch_mumu = self.config_inst.get_channel("mumu")
    ch_emu = self.config_inst.get_channel("emu")

    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask
    tau2_isolated = false_mask
    leptons_os = false_mask
    single_triggered = false_mask
    cross_triggered = false_mask
    sel_electron_mask = full_like(events.Electron.pt, False, dtype=bool)
    sel_muon_mask = full_like(events.Muon.pt, False, dtype=bool)
    sel_tau_mask = full_like(events.Tau.pt, False, dtype=bool)
    leading_taus = events.Tau[:, :0]

    # indices for sorting taus first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    f = 10**(np.ceil(np.log10(ak.max(events.Tau.pt))) + 2)
    tau_sorting_key = events.Tau[f"raw{self.config_inst.x.tau_tagger}VSjet"] * f + events.Tau.pt
    tau_sorting_indices = ak.argsort(tau_sorting_key, axis=-1, ascending=False)

    # perform each lepton election step separately per trigger, avoid caching
    sel_kwargs = {**kwargs, "call_force": True}
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        is_single = trigger.has_tag("single_trigger")
        is_cross = trigger.has_tag("cross_trigger")

        # electron selection
        electron_mask, electron_veto_mask = self[electron_selection](
            events,
            trigger,
            **sel_kwargs,
        )

        # muon selection
        muon_mask, muon_veto_mask = self[muon_selection](
            events,
            trigger,
            **sel_kwargs,
        )

        # tau selection
        tau_mask, tau_iso_mask = self[tau_selection](
            events,
            trigger,
            electron_mask,
            muon_mask,
            **sel_kwargs,
        )

        # conditions potentially leading to etau channel
        if trigger.has_tag({"single_e", "cross_e_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("etau")
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_tau_mask = (
                tau_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
            )

            # fold trigger matching into the selection
            trig_electron_mask = (
                electron_mask &
                self[electron_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )
            trig_tau_mask = ch_tau_mask
            if trigger.has_tag("cross_e_tau"):
                trig_tau_mask = (
                    trig_tau_mask &
                    self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )

            # check if the most isolated tau among the selected ones is matched
            first_tau_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_tau_mask[tau_sorting_indices]]], axis=1),
                False,
            )

            # expect 1 electron, 1 veto electron (the same one), 0 veto muons, and at least one tau
            # without and with trigger matching on the default objects
            is_etau = (
                trigger_fired &
                (ak.sum(electron_mask, axis=1) == 1) &
                (ak.sum(trig_electron_mask, axis=1) == 1) &
                (ak.sum(electron_veto_mask, axis=1) == 1) &
                (ak.sum(muon_veto_mask, axis=1) == 0) &
                first_tau_matched
            )

            # get selected taus and sort them
            # (this will be correct for events for which is_etau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            e_charge = ak.firsts(events.Electron[trig_electron_mask].charge, axis=1)
            tau_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            is_os = e_charge == -tau_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 1
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_etau.id, is_etau)
            tau2_isolated = ak.where(is_etau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_etau, is_os, leptons_os)
            single_triggered = ak.where(is_etau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_etau & is_cross, True, cross_triggered)
            sel_electron_mask = ak.where(is_etau, trig_electron_mask, sel_electron_mask)
            sel_tau_mask = ak.where(is_etau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_etau, sorted_sel_taus[:, :1], leading_taus)

        # mutau channel
        if trigger.has_tag({"single_mu", "cross_mu_tau"}) and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("mutau")
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_tau_mask = (
                tau_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vvloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.tight)
            )

            # fold trigger matching into the selection
            trig_muon_mask = (
                muon_mask &
                self[muon_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )
            trig_tau_mask = ch_tau_mask
            if trigger.has_tag("cross_e_tau"):
                trig_tau_mask = (
                    trig_tau_mask &
                    self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )

            # check if the most isolated tau among the selected ones is matched
            first_tau_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_tau_mask[tau_sorting_indices]]], axis=1),
                False,
            )

            # expect 1 muon, 1 veto muon (the same one), 0 veto electrons, and at least one tau
            # without and with trigger matching on the default objects
            is_mutau = (
                trigger_fired &
                (ak.sum(muon_mask, axis=1) == 1) &
                (ak.sum(trig_muon_mask, axis=1) == 1) &
                (ak.sum(muon_veto_mask, axis=1) == 1) &
                (ak.sum(electron_veto_mask, axis=1) == 0) &
                first_tau_matched
            )

            # get selected, sorted taus to obtain quantities
            # (this will be correct for events for which is_mutau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            mu_charge = ak.firsts(events.Muon[trig_muon_mask].charge, axis=1)
            tau_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            is_os = mu_charge == -tau_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 1
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_mutau.id, is_mutau)
            tau2_isolated = ak.where(is_mutau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_mutau, is_os, leptons_os)
            single_triggered = ak.where(is_mutau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_mutau & is_cross, True, cross_triggered)
            sel_muon_mask = ak.where(is_mutau, trig_muon_mask, sel_muon_mask)
            sel_tau_mask = ak.where(is_mutau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_mutau, sorted_sel_taus[:, :1], leading_taus)

        # tautau channel
        if (
            trigger.has_tag({"cross_tau_tau", "cross_tau_tau_vbf", "cross_tau_tau_jet"}) and
            (self.dataset_inst.is_mc or self.dataset_inst.has_tag("tautau"))
        ):
            # channel dependent deeptau cuts vs e and mu
            ch_tau_mask = (
                tau_mask &
                (events.Tau[get_tau_tagger("e")] >= wp_config.tau_vs_e.vvloose) &
                (events.Tau[get_tau_tagger("mu")] >= wp_config.tau_vs_mu.vloose)
            )

            # fold trigger matching into the selection
            trig_tau_mask = (
                ch_tau_mask &
                self[tau_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )

            # check if the two leading (most isolated) taus are matched
            leading_taus_matched = ak.fill_none(
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_tau_mask[tau_sorting_indices]]], axis=1) &
                ak.firsts(trig_tau_mask[tau_sorting_indices[ch_tau_mask[tau_sorting_indices]]][:, 1:], axis=1),
                False,
            )

            # expect 0 veto electrons, 0 veto muons and at least two taus of which one is isolated
            is_tautau = (
                trigger_fired &
                (ak.sum(electron_veto_mask, axis=1) == 0) &
                (ak.sum(muon_veto_mask, axis=1) == 0) &
                leading_taus_matched
            )

            # get selected, sorted taus to obtain quantities
            # (this will be correct for events for which is_tautau is actually True)
            sorted_sel_taus = events.Tau[tau_sorting_indices][trig_tau_mask[tau_sorting_indices]]
            # determine the relative charge and tau2 isolation
            tau1_charge = ak.firsts(sorted_sel_taus.charge, axis=1)
            tau2_charge = ak.firsts(sorted_sel_taus.charge[:, 1:], axis=1)
            is_os = tau1_charge == -tau2_charge
            is_iso = ak.sum(tau_iso_mask[trig_tau_mask], axis=1) >= 2
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_tautau.id, is_tautau)
            tau2_isolated = ak.where(is_tautau, is_iso, tau2_isolated)
            leptons_os = ak.where(is_tautau, is_os, leptons_os)
            single_triggered = ak.where(is_tautau & is_single, True, single_triggered)
            cross_triggered = ak.where(is_tautau & is_cross, True, cross_triggered)
            sel_tau_mask = ak.where(is_tautau, trig_tau_mask, sel_tau_mask)
            leading_taus = ak.where(is_tautau, sorted_sel_taus[:, :2], leading_taus)

        # ee channel
        if trigger.has_tag("single_e") and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("ee")
        ):
            # fold trigger matching into the selection
            trig_electron_mask = (
                electron_mask &
                self[electron_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )

            # check if the first (hardest) electron matched
            electron_sorting_indices = ak.argsort(events.Electron.pt, axis=1, ascending=False)
            leading_electron_matched = ak.fill_none(
                ak.firsts(trig_electron_mask[electron_sorting_indices[electron_mask[electron_sorting_indices]]], axis=1),  # noqa: E501
                False,
            )

            # expect 2 electrons, 2 veto electrons, 0 veto muons, and ignore the taus
            is_ee = (
                trigger_fired &
                (ak.sum(electron_mask, axis=1) == 2) &
                leading_electron_matched &
                (ak.sum(electron_veto_mask, axis=1) == 2) &
                (ak.sum(muon_veto_mask, axis=1) == 0)
            )

            # get selected, sorted electrons to obtain quantities
            # (this will be correct for events for which is_ee is actually True)
            sorted_sel_electrons = events.Electron[electron_sorting_indices][electron_mask[electron_sorting_indices]]
            # determine the relative charge
            e1_charge = ak.firsts(sorted_sel_electrons.charge, axis=1)
            e2_charge = ak.firsts(sorted_sel_electrons.charge[:, 1:], axis=1)
            is_os = e1_charge == -e2_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_ee.id, is_ee)
            leptons_os = ak.where(is_ee, is_os, leptons_os)
            single_triggered = ak.where(is_ee & is_single, True, single_triggered)
            cross_triggered = ak.where(is_ee & is_cross, True, cross_triggered)
            sel_electron_mask = ak.where(is_ee, electron_mask, sel_electron_mask)

        # mumu channel
        if trigger.has_tag("single_mu") and (
            self.dataset_inst.is_mc or
            self.dataset_inst.has_tag("mumu")
        ):
            # fold trigger matching into the selection
            trig_muon_mask = (
                muon_mask &
                self[muon_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
            )

            # check if the first (hardest) muon matched
            muon_sorting_indices = ak.argsort(events.Muon.pt, axis=1, ascending=False)
            leading_muon_matched = ak.fill_none(
                ak.firsts(trig_muon_mask[muon_sorting_indices[muon_mask[muon_sorting_indices]]], axis=1),
                False,
            )

            # expect 2 muons, 2 veto muons, 0 veto electrons, and ignore the taus
            is_mumu = (
                trigger_fired &
                (ak.sum(muon_mask, axis=1) == 2) &
                leading_muon_matched &
                (ak.sum(muon_veto_mask, axis=1) == 2) &
                (ak.sum(electron_veto_mask, axis=1) == 0)
            )

            # get selected, sorted muons to obtain quantities
            # (this will be correct for events for which is_mumu is actually True)
            sorted_sel_muons = events.Muon[muon_sorting_indices][muon_mask[muon_sorting_indices]]
            # determine the relative charge
            mu1_charge = ak.firsts(sorted_sel_muons.charge, axis=1)
            mu2_charge = ak.firsts(sorted_sel_muons.charge[:, 1:], axis=1)
            is_os = mu1_charge == -mu2_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_mumu.id, is_mumu)
            leptons_os = ak.where(is_mumu, is_os, leptons_os)
            single_triggered = ak.where(is_mumu & is_single, True, single_triggered)
            cross_triggered = ak.where(is_mumu & is_cross, True, cross_triggered)
            sel_muon_mask = ak.where(is_mumu, muon_mask, sel_muon_mask)

        # emu channel
        if (
            (emu_from_e := (
                trigger.has_tag("single_e") and
                (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_e"))
            )) or (
                trigger.has_tag("single_mu") and
                (self.dataset_inst.is_mc or self.dataset_inst.has_tag("emu_from_mu"))
            )
        ):
            if emu_from_e:
                emu_electron_mask = electron_mask
                # fold trigger matching into the selection
                trig_electron_mask = (
                    electron_mask &
                    self[electron_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )
                # for muons, loop over triggers, find single triggers and make sure none of them
                # fired in order to avoid double counting
                emu_muon_mask = False
                mu_trig_fired = full_like(events.event, False, dtype=bool)
                for _trigger, _trigger_fired, _ in trigger_results.x.trigger_data:
                    if not _trigger.has_tag("single_mu"):
                        continue
                    # evaluate the muon selection once (it is the same for all single triggers)
                    if emu_muon_mask is False:
                        emu_muon_mask, _ = self[muon_selection](events, _trigger, **sel_kwargs)
                    # store the trigger decision
                    mu_trig_fired = mu_trig_fired | _trigger_fired
                # muons obey the trigger rules if no single trigger fired
                trig_muon_mask = emu_muon_mask & ~mu_trig_fired

            else:
                emu_muon_mask = muon_mask
                # fold trigger matching into the selection
                trig_muon_mask = (
                    muon_mask &
                    self[muon_trigger_matching](events, trigger, trigger_fired, leg_masks, **sel_kwargs)
                )
                # for electrons, loop over triggers, find single triggers and check the matching
                # only in case a trigger fired
                emu_electron_mask = False
                e_trig_fired = full_like(events.event, False, dtype=bool)
                e_match_mask = full_like(events.Electron.pt, False, dtype=bool)
                for _trigger, _trigger_fired, _leg_masks in trigger_results.x.trigger_data:
                    if not _trigger.has_tag("single_e"):
                        continue
                    # evaluate the electron selection once (it is the same for all single triggers)
                    if emu_electron_mask is False:
                        emu_electron_mask, _ = self[electron_selection](events, _trigger, **sel_kwargs)
                    # store the trigger decision
                    e_trig_fired = e_trig_fired | _trigger_fired
                    # evaluate the matching
                    e_match_mask = e_match_mask | (
                        self[electron_trigger_matching](events, _trigger, _trigger_fired, _leg_masks, **sel_kwargs) &
                        _trigger_fired
                    )
                # for events in which no single e trigger fired, consider the matching as successful
                e_match_mask = e_match_mask | ~e_trig_fired
                trig_electron_mask = emu_electron_mask & e_match_mask

            # expect 1 electron, 1 muon, 1 veto electron, 1 veto muon, and ignore taus
            is_emu = (
                trigger_fired &
                (ak.sum(emu_electron_mask, axis=1) == 1) &
                (ak.sum(trig_electron_mask, axis=1) == 1) &
                (ak.sum(electron_veto_mask, axis=1) == 1) &
                (ak.sum(emu_muon_mask, axis=1) == 1) &
                (ak.sum(trig_muon_mask, axis=1) == 1) &
                (ak.sum(muon_veto_mask, axis=1) == 1)
            )

            # determine the relative charge
            e_charge = ak.firsts(events.Electron[trig_electron_mask].charge, axis=1)
            mu_charge = ak.firsts(events.Muon[trig_muon_mask].charge, axis=1)
            is_os = e_charge == -mu_charge
            # store global variables
            channel_id = update_channel_ids(events, channel_id, ch_emu.id, is_emu)
            leptons_os = ak.where(is_emu, is_os, leptons_os)
            single_triggered = ak.where(is_emu & is_single, True, single_triggered)
            cross_triggered = ak.where(is_emu & is_cross, True, cross_triggered)
            sel_electron_mask = ak.where(is_emu, trig_electron_mask, sel_electron_mask)
            sel_muon_mask = ak.where(is_emu, trig_muon_mask, sel_muon_mask)

    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    leptons_os = ak.fill_none(leptons_os, False)

    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "tau2_isolated", tau2_isolated)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "cross_triggered", cross_triggered)

    # convert lepton masks to sorted indices (pt for e/mu, iso for tau)
    sel_electron_indices = sorted_indices_from_mask(sel_electron_mask, events.Electron.pt, ascending=False)
    sel_muon_indices = sorted_indices_from_mask(sel_muon_mask, events.Muon.pt, ascending=False)
    sel_tau_indices = sorted_indices_from_mask(sel_tau_mask, tau_sorting_key, ascending=False)

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


@lepton_selection.init
def lepton_selection_init(self: Selector) -> None:
    # add column to load the raw tau tagger score
    self.uses.add(f"Tau.raw{self.config_inst.x.tau_tagger}VSjet")
