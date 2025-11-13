# coding: utf-8

"""
Jet selection methods.
"""

from __future__ import annotations

from operator import or_
from functools import reduce

import law

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.cms.jet import jet_id, fatjet_id
from columnflow.columnar_util import (
    EMPTY_FLOAT, set_ak_column, sorted_indices_from_mask, mask_from_indices, flat_np_view, full_like,
)
from columnflow.util import maybe_import

from hbt.production.hhbtag import hhbtag
from hbt.production.vbfjtag import vbfjtag
from hbt.selection.lepton import trigger_object_matching
from hbt.util import IF_RUN_2
from hbt.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector(
    uses={"{Jet,TrigObj}.{pt,eta,phi}"},
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def jet_trigger_matching(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    trigger_fired: ak.Array,
    leg_masks: dict[str, ak.Array],
    jet_object_mask: ak.Array | None = None,
    **kwargs,
) -> tuple[ak.Array]:
    """
    Jet trigger matching.
    """
    if ak.all(ak.num(events.Jet) == 0):
        logger.info("no jets found in event chunk")
        return full_like(events.Jet.pt, False, dtype=bool)

    is_cross_tau_tau_vbf = trigger.has_tag("cross_tau_tau_vbf")
    is_cross_tau_tau_jet = trigger.has_tag("cross_tau_tau_jet")
    is_cross_tau_vbf = trigger.has_tag("cross_tau_vbf")
    is_cross_vbf = trigger.has_tag("cross_vbf")
    is_cross_e_vbf = trigger.has_tag("cross_e_vbf")
    is_cross_mu_vbf = trigger.has_tag("cross_mu_vbf")
    assert is_cross_e_vbf or is_cross_mu_vbf or is_cross_tau_vbf or is_cross_tau_tau_jet or is_cross_vbf or is_cross_tau_tau_vbf  # noqa: E501

    # define the jet objects to be considered for matching
    if jet_object_mask is not None:
        masked_jets = events.Jet[jet_object_mask]
    else:
        masked_jets = events.Jet

    # define the back mapping to the original jet collection

    def map_to_full_jet_array(matched_mask: ak.Array) -> ak.Array:
        if jet_object_mask is None:
            return matched_mask
        full_mask = full_like(events.Jet.pt, False, dtype=bool)
        flat_full_mask = flat_np_view(full_mask)
        flat_full_mask[flat_np_view(jet_object_mask)] = flat_np_view(matched_mask)
        return full_mask

    # start per-jet mask with trigger object matching per leg
    if is_cross_tau_tau_jet:
        assert trigger.n_legs == len(leg_masks) == 3
        assert abs(trigger.legs["jet"].pdg_id) == 1
        # match jet lag
        match_leg = trigger_object_matching(
            masked_jets,
            events.TrigObj[leg_masks["jet"]],
            event_mask=trigger_fired,
        )
        return map_to_full_jet_array(match_leg)

    # all triggers with two jet legs
    # catch config errors
    if is_cross_vbf:
        assert trigger.n_legs == len(leg_masks) == 2
    elif (is_cross_tau_vbf or is_cross_e_vbf or is_cross_mu_vbf):
        assert trigger.n_legs == len(leg_masks) == 3
    elif is_cross_tau_tau_vbf:
        assert trigger.n_legs == len(leg_masks) == 4
    assert abs(trigger.legs["vbf1"].pdg_id) == 1
    assert abs(trigger.legs["vbf2"].pdg_id) == 1

    assert jet_object_mask is not None, "For dijet triggers, jet_object_mask must be defined to match only the 2 candidate jets"  # noqa: E501

    # match both legs
    matches_leg0 = trigger_object_matching(
        masked_jets,
        events.TrigObj[leg_masks["vbf1"]],
        event_mask=trigger_fired,
    )
    matches_leg1 = trigger_object_matching(
        masked_jets,
        events.TrigObj[leg_masks["vbf2"]],
        event_mask=trigger_fired,
    )

    # jets need to be matched to at least one leg, but as a side condition
    # each leg has to have at least one match to a jet
    matches = (
        (matches_leg0 | matches_leg1) &
        ak.any(matches_leg0, axis=1) &
        ak.any(matches_leg1, axis=1)
    )

    # additional condition: there must be at least two matched trigger objects
    # since the same trigger object could fulfill both legs trigger bits and
    # thus both reconstructed taus could match to the same trigger object

    mask_leg_1 = mask_from_indices(leg_masks["vbf1"], events.TrigObj.pt)
    mask_leg_2 = mask_from_indices(leg_masks["vbf2"], events.TrigObj.pt)
    mask_all_legs = mask_leg_1 | mask_leg_2
    matched_trig_objs = trigger_object_matching(
        events.TrigObj[mask_all_legs],
        masked_jets,
        event_mask=trigger_fired,
    )

    matches = matches & (ak.sum(matched_trig_objs, axis=1) >= 2)

    # bring the mask back to the full jet collection
    matches = map_to_full_jet_array(matches)

    return matches


@selector(
    uses={
        jet_id, fatjet_id, hhbtag, vbfjtag, jet_trigger_matching,
        "fired_trigger_ids", "TrigObj.{pt,eta,phi}",
        "Jet.{pt,eta,phi,mass,jetId}", IF_RUN_2("Jet.puId"),
        "FatJet.{pt,eta,phi,mass,msoftdrop,jetId,particleNet_XbbVsQCD}",
    },
    produces={
        hhbtag, vbfjtag,
        "Jet.hhbtag", "Jet.vbfjtag", "Jet.assignment_bits", "matched_trigger_ids",
    },
    max_chunk_size=20_000,  # limit the chunk size due to hhbtag and vbfjtag being used simultaneously
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Jet selection based on ultra-legacy recommendations.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID?rev=107#nanoAOD_Flags
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL?rev=15#Recommendations_for_the_13_T_AN1
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL?rev=17
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD?rev=100#Jets
    """
    is_2016 = self.config_inst.campaign.x.year == 2016
    is_2023_pre = (
        self.config_inst.campaign.x.year == 2023 and
        self.config_inst.campaign.has_tag("preBPix")
    )
    is_2023_post = (
        self.config_inst.campaign.x.year == 2023 and
        self.config_inst.campaign.has_tag("postBPix")
    )
    ch_tautau = self.config_inst.get_channel("tautau")

    if self.dataset_inst.has_tag("parking_vbf") and not (is_2023_pre or is_2023_post):
        raise ValueError("VBF parking datasets should only be used in 2023")

    # recompute jet ids
    events = self[jet_id](events, **kwargs)
    events = self[fatjet_id](events, **kwargs)

    #
    # default jet selection
    #

    # common ak4 jet mask for normal and vbf jets
    ak4_mask = (
        (events.Jet.jetId & (1 << 1) != 0) &  # tight (2nd bit set)
        # (events.Jet.jetId & (1 << 2) != 0) &  # tight plus lepton veto (3rd bit set)
        ak.all(events.Jet.metric_table(lepton_results.x.leading_taus) > 0.5, axis=2) &
        ak.all(events.Jet.metric_table(lepton_results.x.leading_e_mu) > 0.5, axis=2)
    )

    # puId for run 2
    if self.config_inst.campaign.x.run == 2:
        ak4_mask = (
            ak4_mask &
            ((events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4)))  # flipped in 2016
        )
    else:
        # Horn removal recommendation : remove all jets below 50 GeV if they are in the eta range ]2.5, 3.0[
        # from https://twiki.cern.ch/twiki/bin/view/CMS/JetMET?rev=293#Run3_recommendations
        ak4_mask = ak4_mask & ~((events.Jet.pt <= 50) & (abs(events.Jet.eta) > 2.5) & ((events.Jet.eta) < 3.0))

    # default jets
    default_mask = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.5)
    )

    #
    # hhb-jet identification
    #

    # get the hhbtag values per jet per event
    events = self[hhbtag](events, default_mask, lepton_results.x.lepton_pair, **kwargs)
    hhbtag_scores = events.hhbtag_score
    # create a mask where only the two highest scoring hhbjets are selected
    score_indices = ak.argsort(hhbtag_scores, axis=1, ascending=False)
    hhbjet_mask = mask_from_indices(score_indices[:, :2], hhbtag_scores)

    # deselect jets in events with less than two valid scores
    hhbjet_mask = hhbjet_mask & (ak.sum(hhbtag_scores != EMPTY_FLOAT, axis=1) >= 2)

    # trigger leg matching for tautau events that were triggered by a tau-tau-jet cross trigger;
    # two strategies were studied a) and b) but strategy a) seems to not comply with how trigger
    # matching should be done and should therefore be ignored.

    false_mask = full_like(events.event, False, dtype=bool)

    # create mask for tautau events that fired and matched tautau trigger
    tt_match_mask = (
        (events.channel_id == ch_tautau.id) &
        ak.any(reduce(
            or_,
            [(events.matched_trigger_ids == tid) for tid in self.trigger_ids_tt],
            false_mask,
        ), axis=1)
    )

    # create a mask to select tautau events that were triggered by a tau-tau-jet cross trigger
    # and passed the tautau matching in the lepton selection
    ttj_mask = (
        (events.channel_id == ch_tautau.id) &
        ak.any(reduce(
            or_,
            [(lepton_results.x.lepton_part_trigger_ids == tid) for tid in self.trigger_ids_ttj],
            false_mask,
        ), axis=1)
    )

    # create mask for events that matched taus in any vbf trigger -> not tautau channel specific!
    all_vbf_trigger = (
        self.trigger_ids_ttv +
        self.trigger_ids_tv +
        self.trigger_ids_vbf +
        self.trigger_ids_ev +
        self.trigger_ids_mv
    )
    vbf_lep_trigger_mask = (
        ak.any(reduce(
            or_,
            [(lepton_results.x.lepton_part_trigger_ids == tid) for tid in all_vbf_trigger],
            false_mask,
        ), axis=1)
    )

    # we want to remove tautau events for which after trigger and tau tau matching, only ttj/v
    # triggers are under consideration, but the jet leg cannot be matched, so create a mask that
    # rejects these events
    match_at_least_one_trigger = full_like(events.event, True, dtype=bool)

    # prepare to fill the list of matched trigger ids with the events passing tautaujet and vbf
    matched_trigger_ids_list = [events.matched_trigger_ids]

    # only perform this special treatment when applicable
    if ak.any(ttj_mask):
        # store the leading hhbjet
        sel_hhbjet_mask = hhbjet_mask[ttj_mask]
        pt_sorting_indices = ak.argsort(events.Jet.pt[ttj_mask][sel_hhbjet_mask], axis=1, ascending=False)

        # define mask for matched hhbjets
        # constrain to jets with a score and a minimum pt corresponding to the trigger jet leg
        constraints_mask_matched_hhbjet = (
            (hhbjet_mask[ttj_mask] != EMPTY_FLOAT) &
            (events.Jet.pt[ttj_mask] > 60.0)  # ! Note: hardcoded value
        )

        # check which jets can be matched to any of the jet legs
        matching_mask = full_like(events.Jet.pt[ttj_mask], False, dtype=bool)
        for trigger, _, leg_masks in trigger_results.x.trigger_data:
            if trigger.id in self.trigger_ids_ttj:
                trigger_matching_mask = self[jet_trigger_matching](
                    events=events,
                    trigger=trigger,
                    trigger_fired=ttj_mask,
                    leg_masks=leg_masks,
                    **kwargs,
                )[ttj_mask]

                # update overall matching mask to be used for the hhbjet selection
                matching_mask = (
                    matching_mask |
                    trigger_matching_mask
                )

                # update trigger matching mask with constraints on the jets
                trigger_matching_mask = (
                    trigger_matching_mask &
                    constraints_mask_matched_hhbjet
                )

                # add trigger_id to matched_trigger_ids if the pt-leading jet is matched
                leading_matched = ak.fill_none(
                    ak.firsts(trigger_matching_mask[sel_hhbjet_mask][pt_sorting_indices], axis=1),
                    False,
                )

                # cast leading matched mask to event mask
                leading_matched_all_events = full_like(events.event, False, dtype=bool)
                flat_leading_matched_all_events = flat_np_view(leading_matched_all_events)
                flat_leading_matched_all_events[flat_np_view(ttj_mask)] = flat_np_view(leading_matched)

                # store the matched trigger ids
                ids = ak.where(leading_matched_all_events, np.float32(trigger.id), np.float32(np.nan))
                matched_trigger_ids_list.append(ak.singletons(ak.nan_to_none(ids)))

        # store the matched trigger ids
        matched_trigger_ids = ak.concatenate(matched_trigger_ids_list, axis=1)
        # replace the existing column matched_trigger_ids from the lepton selection with the updated one
        events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids, value_type=np.int32)

        # constrain to jets with a score and a minimum pt corresponding to the trigger jet leg
        matching_mask = (
            matching_mask &
            constraints_mask_matched_hhbjet
        )

        # create a mask to select tautau events that were only triggered by a tau-tau-jet cross trigger
        only_ttj_mask = (
            ttj_mask & ~tt_match_mask & ~vbf_lep_trigger_mask
        )

        #
        # a)
        # two hhb-tagged jets must be selected. The highest scoring jet is always selected.
        #  - If this jet happens to match the trigger leg, then the second highest scoring jet is also selected.
        #  - If this is not the case, then the highest scoring jet that matches the trigger leg is selected.
        # ! Note : Apparently the official recommendation is that trigger matching should only be used
        #          to select full events and not for individual objects selection. Thus, this strategy results in bias.
        #

        # # sort matching masks by score first
        # sel_score_indices = score_indices[ttj_mask]
        # sorted_matching_mask = matching_mask[sel_score_indices]
        # # get the position of the highest scoring _and_ matched hhbjet
        # # (this hhbet is guaranteed to be selected)
        # sel_li = ak.local_index(sorted_matching_mask)
        # matched_idx = ak.firsts(sel_li[sorted_matching_mask], axis=1)
        # # the other hhbjet is not required to be matched and is either at the 0th or 1st position
        # # (depending on whether the matched one had the highest score)
        # other_idx = ak.where(matched_idx == 0, 1, 0)
        # # use comparisons between selected indices and the local index to convert back into a mask
        # # and check again that both hhbjets have a score
        # sel_hhbjet_mask = (
        #     (sel_li == ak.fill_none(sel_score_indices[matched_idx[..., None]][..., 0], -1)) |
        #     (sel_li == ak.fill_none(sel_score_indices[other_idx[..., None]][..., 0], -1))
        # ) & (hhbjet_mask[ttj_mask] != EMPTY_FLOAT)

        #
        # b)
        # two hhb-tagged jets must be selected. The highest and second-highest scoring jets are selected.
        #  - If the jet with the highest pt matches the trigger leg, the event is accepted.
        #  - Otherwise the event is rejected if it was only triggered by the tautaujet trigger.
        #

        # check if the pt-leading jet of the two hhbjets is matched for any tautaujet trigger
        # and fold back into hhbjet_mask
        leading_matched = ak.fill_none(ak.firsts(matching_mask[sel_hhbjet_mask][pt_sorting_indices], axis=1), False)

        # cast full leading matched mask to event mask
        full_leading_matched_all_events = full_like(events.event, False, dtype=bool)
        flat_full_leading_matched_all_events = flat_np_view(full_leading_matched_all_events)
        flat_full_leading_matched_all_events[flat_np_view(ttj_mask)] = flat_np_view(leading_matched)

        # remove all events where the matching did not work if they were only triggered by the tautaujet trigger
        match_at_least_one_trigger = ak.where(
            only_ttj_mask & ~flat_full_leading_matched_all_events,
            False,
            match_at_least_one_trigger,
        )

    # validate that either none or two hhbjets were identified
    assert ak.all(((n_hhbjets := ak.sum(hhbjet_mask, axis=1)) == 0) | (n_hhbjets == 2))

    #
    # fat jets
    #

    # new definition for run3 requires only one AK8 jet, no subjet matching
    fatjet_mask = (
        (events.FatJet.jetId & (1 << 1) != 0) &  # tight
        (events.FatJet.msoftdrop > 30.0) &
        (events.FatJet.pt > 250.0) &  # ParticleNet not trained for lower values
        (abs(events.FatJet.eta) < 2.5) &
        ak.all(events.FatJet.metric_table(lepton_results.x.leading_taus) > 0.8, axis=2)
    )

    # We could also allow for fatjettautau cleaning here, but since we don't have this kind of boosted regime,
    # we do not apply this selection here

    # store fatjet and subjet indices
    fatjet_indices = ak.local_index(events.FatJet.pt)[fatjet_mask]

    # store the one fatjet with the highest ParticleNet_XbbVsQCD score

    # indices for sorting fatjets first by particleNet score, then by pt
    # for this, combine pNet score and pt values, e.g. pNet 255 and pt 32.3 -> 2550032.3
    f = 10**(np.ceil(np.log10(ak.max(events.FatJet.pt) or 0.0)) + 2)
    fatjet_sorting_key = events.FatJet.particleNet_XbbVsQCD * f + events.FatJet.pt
    fatjet_sorting_indices = ak.argsort(fatjet_sorting_key, axis=-1, ascending=False)

    selected_fatjets = ak.firsts(events.FatJet[fatjet_sorting_indices][fatjet_mask[fatjet_sorting_indices]], axis=1)

    #
    # vbf jets
    #

    vbf_mask_with_hhbjets = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 4.7)
    )

    #
    # vbf-jet identification
    #

    # get the vbfjtag values per jet per event
    events = self[vbfjtag](
        events,
        vbf_mask_with_hhbjets,
        lepton_results.x.lepton_pair,
        hhbjet_mask,
        selected_fatjets,
        **kwargs,
    )

    vbfjtag_scores = events.vbfjtag_score
    # create a mask where only the two highest scoring vbfjets are selected
    score_indices = ak.argsort(vbfjtag_scores, axis=1, ascending=False)
    vbfjet_mask = mask_from_indices(score_indices[:, :2], vbfjtag_scores)
    # due to the original jet ordering, applying this mask gives the vbf jets pt ordered

    # remove the jets without a valid score
    vbfjet_mask = vbfjet_mask & (~(vbfjtag_scores == EMPTY_FLOAT))

    # redefine the trigger matched list after it was updated with tautaujet ids
    matched_trigger_ids_list = [events.matched_trigger_ids]

    parking_vbf_double_counting = full_like(events.event, False, dtype=bool)
    if all_vbf_trigger:
        vbf_trigger_fired_all_matched = full_like(events.event, False, dtype=bool)
        for trigger, _, leg_masks in trigger_results.x.trigger_data:
            if trigger.id in all_vbf_trigger:
                # create event-level trigger requirements mask
                pt_jet_1 = trigger.x.offline_cuts.get("pt_jet1", None)
                pt_jet_2 = trigger.x.offline_cuts.get("pt_jet2", None)
                mjj = trigger.x.offline_cuts.get("mjj", None)
                delta_eta_jj = trigger.x.offline_cuts.get("delta_eta_jj", None)

                # create the mask for the trigger requirements, unnecessarily complicated due to the possibility
                # of less than 2 vbf jets being present
                vbf_jet_1 = ak.firsts(events.Jet[vbfjet_mask][:, :1])
                vbf_jet_2 = ak.firsts(events.Jet[vbfjet_mask][:, 1:2])
                trig_req_mask = (
                    (ak.fill_none(vbf_jet_1.pt > pt_jet_1, False)) &
                    (ak.fill_none(vbf_jet_2.pt > pt_jet_2, False) if pt_jet_2 is not None else True) &
                    # add with None in the case of less than 2 vbf jets being present leads to None
                    (ak.fill_none((vbf_jet_1 + vbf_jet_2).mass > mjj, False) if mjj is not None else True) &  # noqa: E501
                    (ak.fill_none(abs(vbf_jet_1.eta - vbf_jet_2.eta) < delta_eta_jj, False) if delta_eta_jj is not None else True)  # noqa: E501
                )

                # event-level mask for events with trigger matched leptons(/fired trigger for dijet trigger)
                trigger_fired_leptons_matched = (
                    ak.any(lepton_results.x.lepton_part_trigger_ids == trigger.id, axis=1)
                )

                # TODO: change vbf jets matching procedure when SF procedure has been decided,
                # not available for now, so define the final mask
                # from the tt matching decision, jet matching and trigger thresholds for now

                # object-level mask with the trigger matching jets
                trigger_matching_jets = self[jet_trigger_matching](
                    events=events,
                    trigger=trigger,
                    trigger_fired=trigger_fired_leptons_matched,
                    leg_masks=leg_masks,
                    jet_object_mask=vbfjet_mask,
                    **kwargs,
                )

                n_required_jets = 2 if pt_jet_2 is not None else 1
                _trigger_fired_all_matched = (
                    trigger_fired_leptons_matched &
                    trig_req_mask &
                    (ak.sum(trigger_matching_jets[vbfjet_mask], axis=1) == n_required_jets)
                )
                vbf_trigger_fired_all_matched = vbf_trigger_fired_all_matched | _trigger_fired_all_matched
                ids = ak.where(_trigger_fired_all_matched, np.float32(trigger.id), np.float32(np.nan))
                matched_trigger_ids_list.append(ak.singletons(ak.nan_to_none(ids)))

        # store the matched trigger ids
        matched_trigger_ids = ak.concatenate(matched_trigger_ids_list, axis=1)
        events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids, value_type=np.int32)

        # remove events if from parking_vbf datasets and matched by another trigger
        if self.dataset_inst.has_tag("parking_vbf"):
            set_trigger_tags_no_parking = {
                "single_e", "single_mu", "cross_e_tau", "cross_mu_tau", "cross_tau_tau", "cross_tau_tau_jet",
            }
            if is_2023_pre:
                set_trigger_tags_no_parking.add("cross_tau_tau_vbf")
            trigger_ids_no_parking = [
                trigger.id for trigger in self.config_inst.x.triggers
                if trigger.has_tag(set_trigger_tags_no_parking)
            ]
            # maybe use reduce instead?
            for tid in trigger_ids_no_parking:
                parking_vbf_double_counting = (
                    parking_vbf_double_counting |
                    ak.any(events.matched_trigger_ids == tid, axis=1)
                )

        # remove all events that fired only vbf trigger but were not matched or
        # that fired vbf and tautaujet triggers and matched the leptons but not the jets
        lep_tid = [
            trigger.id for trigger in self.config_inst.x.triggers
            if trigger.has_tag({"single_e", "single_mu", "cross_e_tau", "cross_mu_tau", "cross_tau_tau"})
        ]
        lep_trigger_matched_mask = (
            ak.any(reduce(
                or_,
                [(events.matched_trigger_ids == tid) for tid in lep_tid],
                false_mask,
            ), axis=1)
        )

        vbf_fired_j_not_matched = (
            # need to match either only vbf or vbf and tautaujet triggers
            ~lep_trigger_matched_mask &  # need to not match any of the lepton triggers
            vbf_lep_trigger_mask &  # need to pass the lepton matching for the vbf triggers
            ~vbf_trigger_fired_all_matched    # need to not match the jet legs in the vbf trigger
        )
        if ak.any(ttj_mask):
            # case where vbf and tautaujet triggers were both fired
            ttjv_fired_j_not_matched = (
                vbf_fired_j_not_matched &
                ttj_mask &
                ~full_leading_matched_all_events
            )
            match_at_least_one_trigger = ak.where(
                ttjv_fired_j_not_matched,
                False,
                match_at_least_one_trigger,
            )
            # case where only vbf trigger was fired
            vbf_fired_j_not_matched = (
                vbf_fired_j_not_matched &
                ~ttj_mask
            )

        match_at_least_one_trigger = ak.where(vbf_fired_j_not_matched, False, match_at_least_one_trigger)

    #
    # final selection and object construction
    #

    # pt sorted indices to convert mask
    jet_indices = sorted_indices_from_mask(default_mask, events.Jet.pt, ascending=False)

    # get indices of the two hhbjets, sorted by pt
    hhbjet_indices = sorted_indices_from_mask(hhbjet_mask, events.Jet.pt, ascending=False)

    # keep indices of default jets that are explicitly not selected as hhbjets for easier handling
    non_hhbjet_indices = sorted_indices_from_mask(
        default_mask & (~hhbjet_mask),
        events.Jet.pt,
        ascending=False,
    )

    # get indices of the two vbfjets, sorted by pt
    vbfjet_indices = sorted_indices_from_mask(vbfjet_mask, events.Jet.pt, ascending=False)

    # final event selection (only looking at number of default jets for now)
    # perform a cut on ≥1 jet and all other cuts first, and then cut on ≥2, resulting in an
    # additional, _skippable_ step
    jet_sel = (
        (ak.sum(default_mask, axis=1) >= 1) &
        match_at_least_one_trigger &
        ~parking_vbf_double_counting
        # add additional cuts here in the future
    )
    jet_sel2 = jet_sel & (ak.sum(default_mask, axis=1) >= 2)

    # create bits for jets
    first_hhbjet = mask_from_indices(hhbjet_indices[:, :1], events.Jet.pt)
    second_hhbjet = mask_from_indices(hhbjet_indices[:, 1:2], events.Jet.pt)
    vbfjet1 = mask_from_indices(vbfjet_indices[:, :1], events.Jet.pt)
    vbfjet2 = mask_from_indices(vbfjet_indices[:, 1:2], events.Jet.pt)
    bits = first_hhbjet + 2 * second_hhbjet + 4 * vbfjet1 + 8 * vbfjet2

    # some final type conversions
    jet_indices = ak.values_astype(ak.fill_none(jet_indices, 0), np.int32)
    hhbjet_indices = ak.values_astype(hhbjet_indices, np.int32)
    non_hhbjet_indices = ak.values_astype(ak.fill_none(non_hhbjet_indices, 0), np.int32)
    fatjet_indices = ak.values_astype(fatjet_indices, np.int32)
    vbfjet_indices = ak.values_astype(ak.fill_none(vbfjet_indices, 0), np.int32)
    bits = ak.values_astype(bits, np.uint8)

    # store some columns
    events = set_ak_column(events, "Jet.hhbtag", hhbtag_scores)
    events = set_ak_column(events, "Jet.vbfjtag", vbfjtag_scores)
    events = set_ak_column(events, "Jet.assignment_bits", bits)

    # build selection results plus new columns (src -> dst -> indices)
    result = SelectionResult(
        steps={
            "jet": jet_sel,
            "jet2": jet_sel2,
            # the btag weight normalization requires a selection with everything but the bjet
            # selection, so add this step here
            # note: there is currently no b-tag discriminant cut at this point, so skip it
            # "bjet_deepjet": jet_sel,
            # "bjet_pnet": jet_sel,  # no need in run 2
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "HHBJet": hhbjet_indices,  # sorted by pt
                "NonHHBJet": non_hhbjet_indices,
                "VBFJet": vbfjet_indices,  # sorted by pt
            },
            "FatJet": {
                "FatJet": fatjet_indices,
            },
        },
        aux={
            # jet mask that lead to the jet_indices
            "jet_mask": default_mask,
            # used to determine sum of weights in increment_stats
            "n_central_jets": ak.num(jet_indices, axis=1),
        },
    )

    return events, result


@jet_selection.init
def jet_selection_init(self: Selector, **kwargs) -> None:
    # register shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }


@jet_selection.setup
def jet_selection_setup(self: Selector, task: law.Task, **kwargs) -> None:
    # store ids of tau-tau cross triggers
    self.trigger_ids_tt = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_tau_tau")
    ]
    self.trigger_ids_ttj = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_tau_tau_jet")
    ]
    self.trigger_ids_ttv = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_tau_tau_vbf")
    ]
    self.trigger_ids_tv = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_tau_vbf")
    ]
    self.trigger_ids_vbf = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_vbf")
    ]
    self.trigger_ids_ev = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_e_vbf")
    ]
    self.trigger_ids_mv = [
        trigger.id for trigger in self.config_inst.x.triggers
        if trigger.has_tag("cross_mu_vbf")
    ]
