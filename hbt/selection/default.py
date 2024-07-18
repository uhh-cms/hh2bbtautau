# coding: utf-8

"""
Selection methods.
"""

from __future__ import annotations

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
from columnflow.util import maybe_import

from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection
from hbt.production.features import cutflow_features
from hbt.production.processes import process_ids_dy
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        json_filter, met_filters, trigger_selection, lepton_selection, jet_selection, mc_weight,
        pu_weight, btag_weights, process_ids, cutflow_features, increment_stats,
        attach_coffea_behavior,
        IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, mc_weight, pu_weight, btag_weights,
        process_ids, cutflow_features, increment_stats,
        IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
    },
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

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results
    else:
        results += SelectionResult(steps={"json": np.ones(len(events), dtype=bool)})

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

    # mc-only functions
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

        # pdf weights
        if self.has_dep(pdf_weights):
            events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        if self.has_dep(murmuf_weights):
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

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    # combined event seleciton after all but the bjet step
    event_sel_nob = results.steps.all_but_bjet = reduce(
        and_,
        [mask for step_name, mask in results.steps.items() if step_name != "bjet"],
    )

    # create process ids
    if self.process_ids_dy is not None:
        events = self[self.process_ids_dy](events, **kwargs)
    else:
        events = self[process_ids](events, **kwargs)

    # some cutflow features
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": event_sel,
        "num_events_selected_nobjet": event_sel_nob,
    }
    group_map = {}
    group_combinations = []
    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_sel)
        weight_map["sum_mc_weight_selected_nobjet"] = (events.mc_weight, event_sel_nob)
        # pu weights with variations
        for name in sorted(self[pu_weight].produces):
            weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)
        # pdf and murmuf weights with variations
        if not self.dataset_inst.has_tag("no_lhe_weights"):
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
                "values": results.x.n_central_jets,
                "mask_fn": (lambda v: results.x.n_central_jets == v),
            },
        }
        # combinations
        group_combinations.append(("process", "njet"))

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


@default.init
def default_init(self: Selector) -> None:
    if getattr(self, "dataset_inst", None) is None:
        return

    self.process_ids_dy: process_ids_dy | None = None
    if self.dataset_inst.has_tag("is_dy"):
        # check if this dataset is covered by any dy id producer
        for name, dy_cfg in self.config_inst.x.dy_stitching.items():
            dataset_inst = dy_cfg["inclusive_dataset"]
            # the dataset is "covered" if its process is a subprocess of that of the dy dataset
            if dataset_inst.has_process(self.dataset_inst.processes.get_first()):
                self.process_ids_dy = process_ids_dy.derive(f"process_ids_dy_{name}", cls_dict={
                    "dy_inclusive_dataset": dataset_inst,
                    "dy_leaf_processes": dy_cfg["leaf_processes"],
                })

                # add it as a dependency
                self.uses.add(self.process_ids_dy)
                self.produces.add(self.process_ids_dy)

                # stop after the first match
                break
