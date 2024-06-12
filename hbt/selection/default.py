# coding: utf-8

"""
Selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox

from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection
from hbt.production.features import cutflow_features


np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        json_filter, met_filters, trigger_selection, lepton_selection, jet_selection, process_ids,
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, process_ids,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=False,
)
def particle_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    results = SelectionResult()

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # met filter selection
    events, met_filter_results = self[met_filters](events, **kwargs)
    results += met_filter_results

    # trigger selection
    events, trigger_results = self[trigger_selection](events, **kwargs)
    results += trigger_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, trigger_results, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[jet_selection](events, trigger_results, lepton_results, **kwargs)
    results += jet_results

    # create process ids
    events = self[process_ids](events, **kwargs)

    return events, results


@selector(
    uses={
        mc_weight, pdf_weights, murmuf_weights, pu_weight, btag_weights,
    },
    produces={
        mc_weight, pdf_weights, murmuf_weights, pu_weight, btag_weights,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=False,
)
def mc_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    results = SelectionResult()

    # corrected mc weights
    events = self[mc_weight](events, **kwargs)

    # pdf weights
    events = self[pdf_weights](events, **kwargs)

    # renormalization/factorization scale weights
    events = self[murmuf_weights](events, **kwargs)

    # pileup weights
    events = self[pu_weight](events, **kwargs)

        # btag weights
        events = self[btag_weights](
            events,
            ak.fill_none(results.x.jet_mask, False, axis=-1),
            negative_b_score_log_mode="none",
            **kwargs,
        )

    return events, results


@selector(
    uses={
        pu_weight, btag_weights,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=False,
)
def selection_mc_weights(
    self: Selector,
    events: ak.Array,
    selection_results: SelectionResult,
    **kwargs,
) -> tuple[dict, dict, list]:
    event_sel = selection_results.event

    # combined event seleciton after all but the bjet step
    event_sel_nob = selection_results.steps.all_but_bjet = reduce(
        and_,
        [mask for step_name, mask in selection_results.steps.items() if step_name != "bjet"],
    )

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": event_sel,
        "num_events_selected_nobjet": event_sel_nob,
    }
    group_map: dict[str, dict] = {}
    group_combinations: list[tuple] = []
    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_sel)
        weight_map["sum_mc_weight_selected_nobjet"] = (events.mc_weight, event_sel_nob)
        # pu weights with variations
        for name in sorted(self[pu_weight].produces):
            weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)
        # pdf and murmuf weights with variations
        for v in ["", "_up", "_down"]:
            weight_map[f"sum_pdf_weight{v}"] = events[f"pdf_weight{v}"]
            weight_map[f"sum_pdf_weight{v}_selected"] = (events[f"pdf_weight{v}"], event_sel)
            weight_map[f"sum_murmuf_weight{v}"] = events[f"murmuf_weight{v}"]
            weight_map[f"sum_murmuf_weight{v}_selected"] = (events[f"murmuf_weight{v}"], event_sel)
        # btag weights
        for name in sorted(self[btag_weights].produces):
            if not name.startswith("btag_weight"):
                continue
            weight_map[f"sum_{name}"] = events[name]
            weight_map[f"sum_{name}_selected"] = (events[name], event_sel)
            weight_map[f"sum_{name}_selected_nobjet"] = (events[name], event_sel_nob)
            weight_map[f"sum_mc_weight_{name}_selected_nobjet"] = (events.mc_weight * events[name], event_sel_nob)
        # groups
        group_map = {
            **group_map,
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
            # per jet multiplicity
            "njet": {
                "values": selection_results.x.n_central_jets,
                "mask_fn": (lambda v: selection_results.x.n_central_jets == v),
            },
        }
        # combinations
        group_combinations.append(("process", "njet"))

    return weight_map, group_map, group_combinations


@selector(
    uses={
        particle_selection, mc_selection, selection_mc_weights, cutflow_features,
        increment_stats, attach_coffea_behavior,
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, mc_weight,
        pdf_weights, murmuf_weights, pu_weight, btag_weights, process_ids, cutflow_features,
        increment_stats,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    events, particle_results = self[particle_selection](events, **kwargs)
    results += particle_results

    # mc-only functions
    if self.dataset_inst.is_mc:
        events, mc_results = self[mc_selection](events, **kwargs)
        results += mc_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    # some cutflow features
    events = self[cutflow_features](events, results.objects, **kwargs)

    weight_map, group_map, group_combinations = self[selection_mc_weights](
        events,
        results,
        **kwargs,
    )

    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )
    return events, results


@selector(
    uses={
        particle_selection, mc_selection, selection_mc_weights, cutflow_features,
        increment_stats, attach_coffea_behavior,
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, mc_weight,
        pdf_weights, murmuf_weights, pu_weight, btag_weights, process_ids, cutflow_features,
        increment_stats,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=True,
)
def empty(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    events, _ = self[particle_selection](events, **kwargs)

    # mc-only functions
    if self.dataset_inst.is_mc:
        events, _ = self[mc_selection](events, **kwargs)

    # True selection
    all_true_selection = SelectionResult(
        steps={"all": ak.ones_like(events.run, dtype=bool)},
    )
    results += all_true_selection

    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    # some cutflow features
    events = self[cutflow_features](events, results.objects, **kwargs)

    weight_map, group_map, group_combinations = self[selection_mc_weights](
        events,
        results,
        **kwargs,
    )

    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )
    return events, results
