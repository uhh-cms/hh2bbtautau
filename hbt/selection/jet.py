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
from hbt.selection.lepton import trigger_object_matching
from hbt.util import IF_RUN_2

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        jet_id, fatjet_id, hhbtag,
        "fired_trigger_ids", "TrigObj.{pt,eta,phi}",
        "Jet.{pt,eta,phi,mass,jetId}", IF_RUN_2("Jet.puId"),
        "FatJet.{pt,eta,phi,mass,msoftdrop,jetId,subJetIdx1,subJetIdx2}",
        "SubJet.{pt,eta,phi,mass,btagDeepB}",
    },
    produces={
        hhbtag,
        "Jet.hhbtag", "matched_trigger_ids",
    },
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
    ch_tautau = self.config_inst.get_channel("tautau")

    # local jet index
    li = ak.local_index(events.Jet)

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

    # create mask for tautau events that matched taus in vbf trigger
    ttv_mask = (
        (events.channel_id == ch_tautau.id) &
        ak.any(reduce(
            or_,
            [(lepton_results.x.lepton_part_trigger_ids == tid) for tid in self.trigger_ids_ttv],
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
                trig_objs = events.TrigObj[leg_masks["jet"]]
                trigger_matching_mask = trigger_object_matching(events.Jet[ttj_mask], trig_objs[ttj_mask])

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
            ttj_mask & ~tt_match_mask & ~ttv_mask
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

    fatjet_mask = (
        (events.FatJet.jetId == 6) &  # tight plus lepton veto
        (events.FatJet.msoftdrop > 30.0) &
        (events.FatJet.pt > 250.0) &  # ParticleNet not trained for lower values
        (abs(events.FatJet.eta) < 2.5) &
        ak.all(events.FatJet.metric_table(lepton_results.x.leading_taus) > 0.8, axis=2) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0)
    )

    # store fatjet and subjet indices
    fatjet_indices = ak.local_index(events.FatJet.pt)[fatjet_mask]
    subjet_indices = ak.concatenate(
        [
            events.FatJet[fatjet_mask].subJetIdx1[..., None],
            events.FatJet[fatjet_mask].subJetIdx2[..., None],
        ],
        axis=2,
    )

    # check subjet btags (only deepcsv available)
    # note: skipped for now as we do not have a final strategy for run 3 yet
    # wp = ...
    # subjets_btagged = ak.all(events.SubJet[ak.firsts(subjet_indices)].btagDeepB > wp, axis=1)

    #
    # vbf jets
    #

    vbf_mask = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 4.7) &
        (~hhbjet_mask) &
        ak.all(events.Jet.metric_table(events.SubJet[subjet_indices[..., 0]]) > 0.4, axis=2) &
        ak.all(events.Jet.metric_table(events.SubJet[subjet_indices[..., 1]]) > 0.4, axis=2)
    )

    # build vectors of vbf jets representing all combinations and apply selections
    vbf1, vbf2 = ak.unzip(ak.combinations(events.Jet[vbf_mask], 2, axis=1))
    vbf_pair = ak.concatenate([vbf1[..., None], vbf2[..., None]], axis=2)
    vbfjj = vbf1 + vbf2
    vbf_pair_mask = (
        (vbfjj.mass > 500.0) &
        (abs(vbf1.eta - vbf2.eta) > 3.0)
    )

    # redefine the trigger matched list after it was updated with tautaujet ids
    matched_trigger_ids_list = [events.matched_trigger_ids]

    # extra requirements for events for which only the tau tau vbf cross trigger fired
    if not self.trigger_ids_ttv:
        cross_vbf_mask = full_like(events.event, False, dtype=bool)
    else:
        ttv_fired_all_matched = full_like(events.event, False, dtype=bool)
        for trigger, _, leg_masks in trigger_results.x.trigger_data:
            if trigger.id in self.trigger_ids_ttv:
                ttv_fired_tt_matched = (
                    (events.channel_id == ch_tautau.id) &
                    ak.any(lepton_results.x.lepton_part_trigger_ids == trigger.id, axis=1)
                )
                # TODO: add vbf jets matching when SF procedure has been decided not available for
                # now, so define the final mask just from the tt matching decision for now
                _ttv_fired_all_matched = ttv_fired_tt_matched
                ttv_fired_all_matched = ttv_fired_all_matched | _ttv_fired_all_matched
                ids = ak.where(_ttv_fired_all_matched, np.float32(trigger.id), np.float32(np.nan))
                matched_trigger_ids_list.append(ak.singletons(ak.nan_to_none(ids)))

        # store the matched trigger ids
        matched_trigger_ids = ak.concatenate(matched_trigger_ids_list, axis=1)
        events = set_ak_column(events, "matched_trigger_ids", matched_trigger_ids, value_type=np.int32)

        # update the "ttv only" mask
        cross_vbf_masks = [events.matched_trigger_ids == tid for tid in self.trigger_ids_ttv]
        cross_vbf_mask = ak.all(reduce(or_, cross_vbf_masks), axis=1)

        # remove all events that fired only vbf trigger but were not matched or
        # that fired vbf and tautaujet triggers and matched the taus but not the jets
        ttv_fired_v_not_matched = (
            # need to match either only vbf or vbf and tautaujet triggers
            (events.channel_id == ch_tautau.id) &  # need to be a tautau event
            ~tt_match_mask &  # need to not match the tautau trigger
            ttv_mask &  # need to match the taus in the vbf trigger
            # need to not match the jet legs in the vbf trigger
            ~ttv_fired_all_matched    # need to not match the jet legs in the vbf trigger
        )
        if ak.any(ttj_mask):
            # case where vbf and tautaujet triggers were both fired
            ttjv_fired_vj_not_matched = (
                ttv_fired_v_not_matched &
                ttj_mask &
                ~full_leading_matched_all_events
            )
            match_at_least_one_trigger = ak.where(
                ttjv_fired_vj_not_matched,
                False,
                match_at_least_one_trigger,
            )
            # case where only vbf trigger was fired
            ttv_fired_v_not_matched = (
                ttv_fired_v_not_matched &
                ~ttj_mask
            )

        match_at_least_one_trigger = ak.where(ttv_fired_v_not_matched, False, match_at_least_one_trigger)

    # impose additional cuts on the vbf pair in case only a ttv trigger fired (and all objects
    # matched), but no other trigger
    vbf_pair_mask = vbf_pair_mask & (
        (~cross_vbf_mask) | (
            (vbfjj.mass > 800) &
            (ak.max(vbf_pair.pt, axis=2) > 140.0) &
            (ak.min(vbf_pair.pt, axis=2) > 60.0)
        )
    )

    # get the index to the pair with the highest pass
    vbf_mass_indices = ak.argsort(vbfjj.mass, axis=1, ascending=False)
    vbf_pair_index = vbf_mass_indices[vbf_pair_mask[vbf_mass_indices]][..., :1]

    # get the two indices referring to jets passing vbf_mask
    # and change them so that they point to jets in the full set, sorted by pt
    vbf_indices_local = ak.concatenate(
        [
            ak.singletons(idx) for idx in
            ak.unzip(ak.firsts(ak.argcombinations(events.Jet[vbf_mask], 2, axis=1)[vbf_pair_index]))
        ],
        axis=1,
    )
    vbfjet_indices = li[vbf_mask][vbf_indices_local]
    vbfjet_indices = vbfjet_indices[ak.argsort(events.Jet[vbfjet_indices].pt, axis=1, ascending=False)]

    #
    # final selection and object construction
    #

    # pt sorted indices to convert mask
    jet_indices = sorted_indices_from_mask(default_mask, events.Jet.pt, ascending=False)

    # get indices of the two hhbjets
    hhbjet_indices = sorted_indices_from_mask(hhbjet_mask, hhbtag_scores, ascending=False)

    # keep indices of default jets that are explicitly not selected as hhbjets for easier handling
    non_hhbjet_indices = sorted_indices_from_mask(
        default_mask & (~hhbjet_mask),
        events.Jet.pt,
        ascending=False,
    )

    # final event selection (only looking at number of default jets for now)
    # perform a cut on ≥1 jet and all other cuts first, and then cut on ≥2, resulting in an
    # additional, _skippable_ step
    jet_sel = (
        (ak.sum(default_mask, axis=1) >= 1) &
        match_at_least_one_trigger
        # add additional cuts here in the future
    )
    jet_sel2 = jet_sel & (ak.sum(default_mask, axis=1) >= 2)

    # some final type conversions
    jet_indices = ak.values_astype(ak.fill_none(jet_indices, 0), np.int32)
    hhbjet_indices = ak.values_astype(hhbjet_indices, np.int32)
    non_hhbjet_indices = ak.values_astype(ak.fill_none(non_hhbjet_indices, 0), np.int32)
    fatjet_indices = ak.values_astype(fatjet_indices, np.int32)
    vbfjet_indices = ak.values_astype(ak.fill_none(vbfjet_indices, 0), np.int32)

    # store some columns
    events = set_ak_column(events, "Jet.hhbtag", hhbtag_scores)

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
                "HHBJet": hhbjet_indices,
                "NonHHBJet": non_hhbjet_indices,
                "VBFJet": vbfjet_indices,
            },
            "FatJet": {
                "FatJet": fatjet_indices,
            },
            "SubJet": {
                "SubJet1": subjet_indices[..., 0],
                "SubJet2": subjet_indices[..., 1],
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
