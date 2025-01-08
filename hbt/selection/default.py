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
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import

from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection
from hbt.production.processes import process_ids_dy
from hbt.production.btag import btag_weights_deepjet, btag_weights_pnet
from hbt.production.features import cutflow_features
from hbt.production.patches import patch_ecalBadCalibFilter
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        json_filter, met_filters, trigger_selection, lepton_selection, jet_selection, mc_weight,
        pu_weight, btag_weights_deepjet, IF_RUN_3(btag_weights_pnet), process_ids, cutflow_features,
        increment_stats, attach_coffea_behavior, patch_ecalBadCalibFilter,
        IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, mc_weight, pu_weight,
        btag_weights_deepjet, IF_RUN_3(btag_weights_pnet), process_ids, cutflow_features,
        increment_stats, IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
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
    # patch for the broken "Flag_ecalBadCalibFilter" MET filter in prompt data (tag set in config)
    if self.dataset_inst.has_tag("broken_ecalBadCalibFilter"):
        # fold decision into met filter results
        events = self[patch_ecalBadCalibFilter](events, **kwargs)
        met_filter_results.steps.met_filter = (
            met_filter_results.steps.met_filter &
            events.patchedEcalBadCalibFilter
        )
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
            events = self[pdf_weights](events, outlier_log_mode="debug", **kwargs)

        # renormalization/factorization scale weights
        if self.has_dep(murmuf_weights):
            events = self[murmuf_weights](events, **kwargs)

        # pileup weights
        events = self[pu_weight](events, **kwargs)

        # btag weights
        btag_weight_jet_mask = ak.fill_none(results.x.jet_mask, False, axis=-1)
        events = self[btag_weights_deepjet](
            events,
            jet_mask=btag_weight_jet_mask,
            negative_b_score_log_mode="none",
            **kwargs,
        )
        if self.has_dep(btag_weights_pnet):
            events = self[btag_weights_pnet](
                events,
                jet_mask=btag_weight_jet_mask,
                negative_b_score_log_mode="none",
                **kwargs,
            )

    # create process ids
    if self.process_ids_dy is not None:
        events = self[self.process_ids_dy](events, **kwargs)
    else:
        events = self[process_ids](events, **kwargs)

    # some cutflow features
    events = self[cutflow_features](events, results.objects, **kwargs)

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    # combined event selection after all but the bjet step
    def event_sel_nob(btag_weight_cls):
        tagger_name = btag_weights_deepjet.tagger_name
        var_sel = results.steps[f"all_but_bjet_{tagger_name}"] = reduce(and_, [
            mask for step_name, mask in results.steps.items()
            if step_name != f"bjet_{tagger_name}"
        ])
        return var_sel

    # increment stats
    events, results = setup_and_increment_stats(
        self,
        events=events,
        results=results,
        stats=stats,
        event_sel=event_sel,
        event_sel_variations={
            "nob_deepjet": event_sel_nob(btag_weights_deepjet),
            "nob_pnet": event_sel_nob(btag_weights_pnet) if self.has_dep(btag_weights_pnet) else None,
        },
        njets=results.x.n_central_jets,
    )

    return events, results


@default.init
def default_init(self: Selector) -> None:
    if getattr(self, "dataset_inst", None) is None:
        return

    self.process_ids_dy: process_ids_dy | None = None
    if self.dataset_inst.has_tag("is_dy"):
        dy_stitching = self.config_inst.x.stitching["is_dy"]
        # check if this dataset is covered by any dy id producer
        for name, dy_cfg in dy_stitching.items():
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


empty = default.derive("empty", cls_dict={})


@empty.init
def empty_init(self: Selector) -> None:
    super(empty, self).init_func()

    # remove unused dependencies
    unused = {
        json_filter,
        met_filters,
        cutflow_features,
        patch_ecalBadCalibFilter,
        jet_selection,
        lepton_selection,
        trigger_selection,
    }
    self.uses -= unused
    self.produces -= unused

    # add custom columns
    self.uses.add("Jet.phi")  # needed by vector behavior for accessing pt in btag_weights
    self.produces |= {"channel_id", "leptons_os", "tau2_isolated", "{single,cross}_triggered"}


@empty.call
def empty_call(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    An empty selection that does not perform selection steps but only invokes producers that are
    necessary to create columns that are required downstream, e.g. for ProduceColumns with our
    "default" producer.
    """
    from columnflow.columnar_util import set_ak_column

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

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
        btag_weight_jet_mask = abs(events.Jet["eta"]) < 2.5
        events = self[btag_weights_deepjet](
            events,
            jet_mask=btag_weight_jet_mask,
            negative_b_score_log_mode="none",
            **kwargs,
        )
        if self.has_dep(btag_weights_pnet):
            events = self[btag_weights_pnet](
                events,
                jet_mask=btag_weight_jet_mask,
                negative_b_score_log_mode="none",
                **kwargs,
            )

    # create process ids
    if self.process_ids_dy is not None:
        events = self[self.process_ids_dy](events, **kwargs)
    else:
        events = self[process_ids](events, **kwargs)

    # fake lepton selection results
    events = set_ak_column(events, "channel_id", np.zeros(len(events), dtype=np.uint8))
    events = set_ak_column(events, "leptons_os", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "tau2_isolated", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "cross_triggered", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "single_triggered", np.zeros(len(events), dtype=bool))

    # trivial selection mask capturing all events
    results.event = np.ones(len(events), dtype=bool)

    # increment stats
    events, results = setup_and_increment_stats(
        self,
        events=events,
        results=results,
        stats=stats,
        event_sel=results.event,
        event_sel_variations={
            "nob_deepjet": results.event,
            "nob_pnet": results.event if self.has_dep(btag_weights_pnet) else None,
        },
        njets=ak.num(events.Jet, axis=1),
    )

    return events, results


def setup_and_increment_stats(
    self: Selector,
    *,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    event_sel: np.ndarray | ak.Array,
    event_sel_variations: dict[str, np.ndarray | ak.Array] | None = None,
    njets: np.ndarray | ak.Array | None = None,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Helper function that sets up the weight and group maps for the increment_stats task, invokes it
    and returns the updated events and results objects.

    :param self: The selector instance.
    :param events: The events array.
    :param results: The current selection results.
    :param stats: The stats dictionary.
    :param event_sel: The general event selection mask.
    :param event_sel_variations: Named variations of the event selection mask for additional stats.
    :param event_sel_nob_pnet: The event selection mask without the bjet step for pnet.
    :param njets: The number of central jets.
    :return: The updated events and results objects in a tuple.
    """
    if event_sel_variations is None:
        event_sel_variations = {}
    event_sel_variations = {n: s for n, s in event_sel_variations.items() if s is not None}

    # start creating a weight, group and group combination map
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": event_sel,
    }
    for var_name, var_sel in event_sel_variations.items():
        weight_map[f"num_events_selected_{var_name}"] = var_sel
    group_map = {}
    group_combinations = []

    # add mc info
    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_sel)
        for var_name, var_sel in event_sel_variations.items():
            weight_map[f"sum_mc_weight_selected_{var_name}"] = (events.mc_weight, var_sel)

        # pu weights with variations
        for route in sorted(self[pu_weight].produced_columns):
            name = str(route)
            weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)

        # pdf weights with variations
        if self.has_dep(pdf_weights):
            for v in ["", "_up", "_down"]:
                weight_map[f"sum_pdf_weight{v}"] = events[f"pdf_weight{v}"]
                weight_map[f"sum_pdf_weight{v}_selected"] = (events[f"pdf_weight{v}"], event_sel)

        # mur/muf weights with variations
        if self.has_dep(murmuf_weights):
            for v in ["", "_up", "_down"]:
                weight_map[f"sum_murmuf_weight{v}"] = events[f"murmuf_weight{v}"]
                weight_map[f"sum_murmuf_weight{v}_selected"] = (events[f"murmuf_weight{v}"], event_sel)

        # btag weights
        for prod in (btag_weights_deepjet, btag_weights_pnet):
            if not self.has_dep(prod):
                continue
            for route in sorted(self[prod].produced_columns):
                weight_name = str(route)
                if not weight_name.startswith(prod.weight_name):
                    continue
                weight_map[f"sum_{weight_name}"] = events[weight_name]
                weight_map[f"sum_{weight_name}_selected"] = (events[weight_name], event_sel)
                for var_name, var_sel in event_sel_variations.items():
                    weight_map[f"sum_{weight_name}_selected_{var_name}"] = (events[weight_name], var_sel)
                    weight_map[f"sum_mc_weight_{weight_name}_selected_{var_name}"] = (events.mc_weight * events[weight_name], var_sel)  # noqa: E501

        # groups
        group_map = {
            **group_map,
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
        }
        # per jet multiplicity
        if njets is not None:
            group_map["njet"] = {
                "values": njets,
                "mask_fn": (lambda v: njets == v),
            }

        # combinations
        group_combinations.append(("process", "njet"))

    return self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )
