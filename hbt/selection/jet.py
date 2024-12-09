# coding: utf-8

"""
Jet selection methods.
"""

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.util import IF_RUN_2, IF_RUN_3
from hbt.production.hhbtag import hhbtag
from hbt.selection.lepton import trigger_object_matching

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        hhbtag, IF_RUN_3(jet_veto_map),
        # custom columns created upstream, probably by a selector
        "trigger_ids",
        # nano columns
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
        "Jet.{pt,eta,phi,mass,jetId}",
        IF_RUN_2("Jet.puId"),
        "Jet.btagDeepFlavB",
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass", "FatJet.msoftdrop",
        "FatJet.jetId", "FatJet.subJetIdx1", "FatJet.subJetIdx2",
        "SubJet.pt", "SubJet.eta", "SubJet.phi", "SubJet.mass", "SubJet.btagDeepB",
    },
    produces={
        # new columns
        "Jet.hhbtag",
    },
    # shifts are declared dynamically below in jet_selection_init
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

    # local jet index
    li = ak.local_index(events.Jet)

    # common ak4 jet mask for normal and vbf jets
    ak4_mask = (
        (events.Jet.jetId == 6) &  # tight plus lepton veto
        ak.all(events.Jet.metric_table(lepton_results.x.leading_taus) > 0.4, axis=2)
    )

    if self.config_inst.campaign.x.run == 2:
        ak4_mask = ak4_mask & ((events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4)))  # flipped in 2016

    # default jets
    default_mask = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.4)
    )

    # hhb-jets
    # --------------------------------------------------------------------------------------------
    # get the hhbtag values per jet per event
    hhbtag_scores = self[hhbtag](events, default_mask, lepton_results.x.lepton_pair, **kwargs)

    # get the score indices of each hhbtag value
    score_indices = ak.argsort(hhbtag_scores, axis=1, ascending=False)

    # only consider tautau events for which the tau_tau_jet trigger fired and no other tau tau trigger
    trigger_mask = (events.channel_id == 3) & ~ak.any((events.trigger_ids == 505) | (events.trigger_ids == 603), axis=1) & ak.any(events.trigger_ids == 701, axis=1)

    # score (and local index) of the hhbtag for each jet coming from events which passed the trigger mask
    sel_score_indices = score_indices[trigger_mask]
    sel_score_indices_li = ak.local_index(sel_score_indices, axis=1)

    # ids of the objects which fired the jet leg of the cross_tau_tau_jet trigger
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        if trigger.has_tag("cross_tau_tau_jet"):
            # Zip the trigger legs with their corresponding masks
            for leg, mask in zip(trigger.legs, leg_masks):
                if leg.pdg_id == 1:
                    obj_ids = mask[trigger_mask]

    # trigger objects corresponding to the above ids
    sel_trig_objs = events.TrigObj[trigger_mask][obj_ids]

    # mask that checks wheather or not the selected hhbjets *matches* the trigger object with delta R < 0.5
    matching_mask = trigger_object_matching(events[trigger_mask].Jet[sel_score_indices], sel_trig_objs)

    # index of the highest scored hhbjet *matching* the trigger object
    sel_hhbjet_idx = ak.argmax(matching_mask, axis=1)

    # create mask to remove all jets except the highest scored hhbjet *matching* the trigger object
    unselected_hhbjet_idx = sel_score_indices_li != sel_hhbjet_idx

    # hhbtag score of the highest scored hhbjet *matching* the trigger object
    first_hhbjet_idx = sel_score_indices[~unselected_hhbjet_idx]

    # select the highest scored hhbjet of the remaining *matched and unmatched* hhbjets
    remaining_hhbjets_score_indices = sel_score_indices[unselected_hhbjet_idx]
    second_hhbjet_idx = ak.singletons((ak.firsts(remaining_hhbjets_score_indices)))

    # concatenate both selected hhbjet indices; 1st index is *matched*; 2nd can be *matched* or *unmatched*
    sel_hhbjets_score_indices = ak.concatenate([first_hhbjet_idx, second_hhbjet_idx], axis=1)
    # sort the selected score indices (now the *matched* hhbjet can be in either position)
    sorted_sel_hhbjets_score_indices = ak.sort(sel_hhbjets_score_indices, axis=1)

    # when only 1 hhbjet was found fill the second index with None
    sel_hhbjets_score_indices = ak.pad_none(sel_hhbjets_score_indices, 2, axis=1)
    sorted_sel_hhbjets_score_indices = ak.pad_none(sorted_sel_hhbjets_score_indices, 2, axis=1)

    # all event indices
    event_idx = ak.local_index(score_indices, axis=0)
    # indices of events passing the trigger mask
    sel_event_idx = event_idx[trigger_mask]

    # store the selected hhbjet score indices in the corresponding event indices
    temp_mask = (event_idx[:, None] == sel_event_idx)
    pos_mask = ak.any(temp_mask, axis=1)
    match_index = ak.argmax(temp_mask, axis=1)

    # initializing a None array
    none = ak.Array([[None, None]])
    none_expanded = ak.broadcast_arrays(none, event_idx)[0]

    # save selected hhbjet scores for the selected events and save [None, None] to the remaining events
    padded_hhbjet_indices = ak.where(pos_mask, sel_hhbjets_score_indices[match_index], none_expanded)
    hhbjet_mask = ((li == padded_hhbjet_indices[..., [0]]) | (li == padded_hhbjet_indices[..., [1]]))
    # --------------------------------------------------------------------------------------------

    # get indices for actual book keeping only for events with both lepton candidates and where at
    # least two jets pass the default mask (bjet candidates)
    valid_score_mask = (
        default_mask &
        (ak.sum(default_mask, axis=1) >= 2) &
        (ak.num(lepton_results.x.lepton_pair, axis=1) == 2)
    )
    hhbjet_indices = score_indices[valid_score_mask[score_indices]][..., :2]

    # fat jets
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

    # discard the event in case the (first) fatjet with matching subjets is found
    # but they are not b-tagged (TODO: move to deepjet when available for subjets)
    if self.config_inst.campaign.x.run == 3:
        wp = self.config_inst.x.btag_working_points.particleNet.loose
    else:
        wp = self.config_inst.x.btag_working_points.deepcsv.loose
    subjets_btagged = ak.all(events.SubJet[ak.firsts(subjet_indices)].btagDeepB > wp, axis=1)

    # vbf jets
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

    # extra requirements for events for which only the tau tau vbf cross trigger fired
    cross_vbf_ids = [t.id for t in self.config_inst.x.triggers if t.has_tag("cross_tau_tau_vbf")]
    if not cross_vbf_ids:
        cross_vbf_mask = ak.full_like(1 * events.event, False, dtype=bool)
    else:
        cross_vbf_masks = [events.trigger_ids == tid for tid in cross_vbf_ids]
        # This combines "at least one cross trigger is fired" and "no other triggers are fired"
        cross_vbf_mask = ak.all(reduce(or_, cross_vbf_masks), axis=1)
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

    # pt sorted indices to convert mask
    jet_indices = sorted_indices_from_mask(default_mask, events.Jet.pt, ascending=False)

    # keep indices of default jets that are explicitly not selected as hhbjets for easier handling
    non_hhbjet_indices = sorted_indices_from_mask(
        default_mask & (~hhbjet_mask),
        events.Jet.pt,
        ascending=False,
    )

    # final event selection
    jet_sel = (
        (ak.sum(default_mask, axis=1) >= 2) &
        ak.fill_none(subjets_btagged, True)  # was none for events with no matched fatjet
    )

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
            # the btag weight normalization requires a selection with everything but the bjet
            # selection, so add this step here
            # note: there is currently no b-tag discriminant cut at this point, so take jet_sel
            "bjet": jet_sel,
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

    # additional jet veto map, vetoing entire events
    if self.has_dep(jet_veto_map):
        events, veto_result = self[jet_veto_map](events, **kwargs)
        result += veto_result

    return events, result


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # register shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }
