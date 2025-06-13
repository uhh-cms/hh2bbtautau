# coding: utf-8

"""
Selection methods.
"""

from __future__ import annotations

from operator import and_
from functools import reduce
from collections import defaultdict

import law
import order as od

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters as cf_met_filters
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import Route, set_ak_column, full_like
from columnflow.hist_util import create_hist_from_variables, fill_hist
from columnflow.util import maybe_import, DotDict
from columnflow.types import Iterable

from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection
import hbt.production.processes as process_producers
from hbt.production.btag import btag_weights_deepjet, btag_weights_pnet
from hbt.production.features import cutflow_features
from hbt.production.patches import patch_ecalBadCalibFilter
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


# updated met_filters selector to define dataset dependent filters
def get_met_filters(self: Selector) -> Iterable[str]:
    met_filters = set(self.config_inst.x.met_filters[self.dataset_inst.data_source])
    if self.dataset_inst.has_tag("broken_ecalBadCalibFilter"):
        met_filters -= {"Flag.ecalBadCalibFilter"}

    return list(met_filters)


met_filters = cf_met_filters.derive("met_filters", cls_dict={"get_met_filters": get_met_filters})


# helper to identify bad events that should be considered missing altogether
def get_bad_events(self: Selector, events: ak.Array) -> ak.Array:
    bad_mask = full_like(events.event, False, dtype=bool)

    # drop events for which we expect lhe infos but that lack them
    # see https://cms-talk.web.cern.ch/t/lhe-weight-vector-empty-for-certain-events/97636/3
    if (
        self.dataset_inst.is_mc and
        self.dataset_inst.has_tag("partial_lhe_weights") and
        self.has_dep(pdf_weights)
    ):
        n_weights = ak.num(events.LHEPdfWeight, axis=1)
        bad_lhe_mask = (n_weights != 101) & (n_weights != 103)
        if ak.any(bad_lhe_mask):
            bad_mask = bad_mask & bad_lhe_mask
            frac = ak.mean(bad_lhe_mask)
            logger.warning(
                f"found {ak.sum(bad_lhe_mask)} events ({frac * 100:.1f}%) with bad LHEPdfWeights",
            )

    return bad_mask


@selector(
    uses={
        json_filter, met_filters, IF_RUN_3(jet_veto_map), trigger_selection, lepton_selection, jet_selection,
        mc_weight, pu_weight, ps_weights, btag_weights_deepjet, IF_RUN_3(btag_weights_pnet), process_ids,
        cutflow_features, attach_coffea_behavior, patch_ecalBadCalibFilter,
        IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
    },
    produces={
        trigger_selection, lepton_selection, jet_selection, mc_weight, pu_weight, ps_weights, btag_weights_deepjet,
        process_ids, cutflow_features, IF_RUN_3(btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(pdf_weights, murmuf_weights),
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # before performing selection steps, drop events that should not be considered at all and
    # maintain a mask "no_sel" that refers to events that are kept
    bad_mask = get_bad_events(self, events)
    no_sel = ~bad_mask
    results += SelectionResult(steps={"bad": no_sel})

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results
    else:
        results += SelectionResult(steps={"json": full_like(events.event, True, dtype=bool)})

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

    # jet veto map
    if self.has_dep(jet_veto_map):
        events, veto_result = self[jet_veto_map](events, **kwargs)
        results += veto_result

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
            events = self[pdf_weights](
                events,
                outlier_log_mode="debug",
                invalid_weights_action="ignore" if self.dataset_inst.has_tag("partial_lhe_weights") else "raise",
                **kwargs,
            )

        # renormalization/factorization scale weights
        if self.has_dep(murmuf_weights):
            events = self[murmuf_weights](events, **kwargs)

        # parton shower weights
        events = self[ps_weights](events, invalid_weights_action="ignore_one", **kwargs)

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
    if self.process_ids_dy_nlo is not None:
        events = self[self.process_ids_dy_nlo](events, **kwargs)
    elif self.process_ids_dy_nnlo is not None:
        events = self[self.process_ids_dy_nnlo](events, **kwargs)
    elif self.process_ids_w_lnu is not None:
        events = self[self.process_ids_w_lnu](events, **kwargs)
    else:
        events = self[process_ids](events, **kwargs)

    # create jet collections for categorization
    events["HHBJet"] = events.Jet[results.objects.Jet.HHBJet]
    events["FatJet"] = events.FatJet[results.objects.FatJet.FatJet]

    # store number of jets for stats and histograms
    events = set_ak_column(events, "n_jets_stats", results.x.n_central_jets, value_type=np.int32)

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
    events, results = increment_stats(
        self,
        events=events,
        task=kwargs["task"],
        results=results,
        stats=stats,
        hists=hists,
        no_sel=no_sel,
        event_sel=event_sel,
        event_sel_variations={
            "nob_deepjet": event_sel_nob(btag_weights_deepjet),
            "nob_pnet": event_sel_nob(btag_weights_pnet) if self.has_dep(btag_weights_pnet) else None,
        },
    )

    return events, results


@default.init
def default_init(self: Selector, **kwargs) -> None:
    # build and store derived process id producers
    for tag in {"dy_nlo", "dy_nnlo", "w_lnu"}:
        prod_name = f"process_ids_{tag}"
        setattr(self, prod_name, None)
        if not self.dataset_inst.has_tag(tag):
            continue
        if not (stitching_cfg := self.config_inst.x(f"{tag}_stitching", None)):
            continue
        # check if the producer was already created and saved in the config
        if (prod := self.config_inst.x(prod_name, None)) is None:
            # check if this dataset is covered by any dy id producer
            for stitch_name, cfg in stitching_cfg.items():
                incl_dataset_inst = cfg["inclusive_dataset"]
                # the dataset is "covered" if its process is a subprocess of that of the dy dataset
                if incl_dataset_inst.has_process(self.dataset_inst.processes.get_first()):
                    base_prod = getattr(process_producers, prod_name)
                    prod = base_prod.derive(f"{prod_name}_{stitch_name}", cls_dict={
                        "leaf_processes": cfg["leaf_processes"],
                    })
                    # cache it
                    self.config_inst.set_aux(prod_name, prod)
                    # stop after the first match
                    break
        if prod is not None:
            # add it as a dependency
            self.uses.add(prod)
            self.produces.add(prod)
            # save it as an attribute
            setattr(self, prod_name, prod)


@default.setup
def default_setup(self: Selector, task: law.Task, **kwargs) -> None:
    # pre-define variable objects for creating stats histograms
    self.hist_vars = [
        od.Variable(
            name="process",
            expression="process_id",
            aux={
                "axis_type": "intcat",
                "axis_kwargs": {"growth": True},
            },
        ),
        od.Variable(
            name="n_jets",
            expression="n_jets_stats",
            binning=list(range(9)),
            aux={
                "axis_type": "int",
                "axis_kwargs": {"growth": True},
            },
        ),
    ]


empty = default.derive("empty", cls_dict={})


@empty.init
def empty_init(self: Selector, **kwargs) -> None:
    super(empty, self).init_func(**kwargs)

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
    hists: DotDict[str, hist.Hist],
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

    # before performing selection steps, drop events that should not be considered at all and
    # maintain a mask "no_sel" that refers to events that are kept
    bad_mask = get_bad_events(self, events)
    no_sel = ~bad_mask
    results += SelectionResult(steps={"bad": no_sel})

    # mc-only functions
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

        # pdf weights
        if self.has_dep(pdf_weights):
            events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        if self.has_dep(murmuf_weights):
            events = self[murmuf_weights](events, **kwargs)

        # parton shower weights
        events = self[ps_weights](events, invalid_weights_action="ignore_one", **kwargs)

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
    if self.process_ids_dy_nlo is not None:
        events = self[self.process_ids_dy_nlo](events, **kwargs)
    elif self.process_ids_dy_nnlo is not None:
        events = self[self.process_ids_dy_nnlo](events, **kwargs)
    elif self.process_ids_w_lnu is not None:
        events = self[self.process_ids_w_lnu](events, **kwargs)
    else:
        events = self[process_ids](events, **kwargs)

    # fake lepton selection results
    events = set_ak_column(events, "channel_id", np.zeros(len(events), dtype=np.uint8))
    events = set_ak_column(events, "leptons_os", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "tau2_isolated", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "cross_triggered", np.zeros(len(events), dtype=bool))
    events = set_ak_column(events, "single_triggered", np.zeros(len(events), dtype=bool))

    # store number of jets for stats and histograms
    events = set_ak_column(events, "n_jets_stats", ak.num(events.Jet, axis=1), value_type=np.int32)

    # trivial selection mask capturing all events
    results.event = np.ones(len(events), dtype=bool)

    # increment stats
    events, results = increment_stats(
        self,
        events=events,
        task=kwargs["task"],
        results=results,
        stats=stats,
        hists=hists,
        no_sel=no_sel,
        event_sel=results.event,
        event_sel_variations={
            "nob_deepjet": results.event,
            "nob_pnet": results.event if self.has_dep(btag_weights_pnet) else None,
        },
    )

    return events, results


def increment_stats(
    self: Selector,
    *,
    events: ak.Array,
    task: law.Task,
    results: SelectionResult,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    no_sel: np.ndarray | ak.Array,
    event_sel: np.ndarray | ak.Array,
    event_sel_variations: dict[str, np.ndarray | ak.Array] | None = None,
) -> tuple[ak.Array, SelectionResult]:
    """
    Helper function that sets up the stats and histograms to bookkeep event counts and weights.

    :param self: The selector instance.
    :param events: The events array.
    :param task: The law task.
    :param results: The current selection results.
    :param stats: The stats dictionary.
    :param hists: Dictionary with histograms that can store stats counts.
    :param event_sel: The general event selection mask.
    :param event_sel_variations: Named variations of the event selection mask for additional stats.
    :return: The updated events and results objects in a tuple.
    """
    if event_sel_variations is None:
        event_sel_variations = {}
    event_sel_variations = {n: s for n, s in event_sel_variations.items() if s is not None}

    # when a shift was requested, skip all other systematic variations
    skip_shifts = task.global_shift_inst != "nominal"

    # start creating a "stats map"
    # - keys: names of histograms to be created
    # - values: (weight array, selection array)
    # note that only a subset of entries end up in the stats dictionary, but all are used for histograms
    stats_map: dict[str, np.ndarray | ak.Array | tuple[np.ndarray | ak.Array, np.ndarray | ak.Array]] = {}
    keys_for_stats = []
    keys_for_hists = []

    def add(key, sel, weight=None, for_stats=False, for_hists=True):
        stats_map[key] = sel if weight is None else (weight, sel)
        if for_stats and key not in keys_for_stats:
            keys_for_stats.append(key)
        if for_hists and key not in keys_for_hists:
            keys_for_hists.append(key)

    # basic event counts
    add("num_events", no_sel, for_stats=True)
    add("num_events_selected", event_sel, for_stats=True)
    for var_name, var_sel in event_sel_variations.items():
        add(f"num_events_selected_{var_name}", var_sel, for_stats=True)

    # add mc info
    if self.dataset_inst.is_mc:
        add("sum_mc_weight", no_sel, events.mc_weight, for_stats=True)
        add("sum_mc_weight_selected", event_sel, events.mc_weight, for_stats=True)
        for var_name, var_sel in event_sel_variations.items():
            add(f"sum_mc_weight_selected_{var_name}", var_sel, events.mc_weight, for_stats=True)

        # pu weights with variations
        for route in sorted(self[pu_weight].produced_columns):
            add(f"sum_mc_weight_{route}", no_sel, events.mc_weight * route.apply(events))

        # pdf weights with variations
        if self.has_dep(pdf_weights):
            for v in (("",) if skip_shifts else ("", "_up", "_down")):
                add(f"sum_pdf_weight{v}", no_sel, events[f"pdf_weight{v}"])
                add(f"sum_pdf_weight{v}_selected", event_sel, events[f"pdf_weight{v}"])

        # mur/muf weights with variations
        if self.has_dep(murmuf_weights):
            for v in (("",) if skip_shifts else ("", "_up", "_down")):
                add(f"sum_murmuf_weight{v}", no_sel, events[f"murmuf_weight{v}"])
                add(f"sum_murmuf_weight{v}_selected", event_sel, events[f"murmuf_weight{v}"])

        # parton shower weights with variations
        if self.has_dep(ps_weights):
            for v in (("",) if skip_shifts else ("", "_up", "_down")):
                add(f"sum_isr_weight{v}", no_sel, events[f"isr_weight{v}"])
                add(f"sum_isr_weight{v}_selected", event_sel, events[f"isr_weight{v}"])
                add(f"sum_fsr_weight{v}", no_sel, events[f"fsr_weight{v}"])
                add(f"sum_fsr_weight{v}_selected", event_sel, events[f"fsr_weight{v}"])

        # btag weights
        for prod in [btag_weights_deepjet, btag_weights_pnet]:
            if not self.has_dep(prod):
                continue
            for route in sorted(self[prod].produced_columns):
                weight_name = str(route)
                if not weight_name.startswith(prod.weight_name):
                    continue
                if skip_shifts and weight_name.endswith(("_up", "_down")):
                    continue
                add(f"sum_{weight_name}", no_sel, events[weight_name])
                add(f"sum_{weight_name}_selected", event_sel, events[weight_name])
                for var_name, var_sel in event_sel_variations.items():
                    add(f"sum_{weight_name}_selected_{var_name}", var_sel, events[weight_name])
                    add(f"sum_mc_weight_{weight_name}_selected_{var_name}", var_sel, events.mc_weight * events[weight_name])  # noqa: E501

        # add num_events_per_process and sum_mc_weight_per_process directly to stats, needed for normalization weight
        if "num_events_per_process" not in stats:
            stats["num_events_per_process"] = defaultdict(float)
        if "sum_mc_weight_per_process" not in stats:
            stats["sum_mc_weight_per_process"] = defaultdict(float)
        for proc_id in np.unique(events.process_id):
            proc_weights = events.mc_weight[events.process_id == proc_id]
            stats["num_events_per_process"][str(proc_id)] += float(len(proc_weights))
            stats["sum_mc_weight_per_process"][str(proc_id)] += float(ak.sum(proc_weights))

    # fill stats and histograms
    for key, val in stats_map.items():
        is_num = key.startswith("num_")
        weight, sel = ((None,) + law.util.make_tuple(val))[-2:]

        if key in keys_for_hists:
            # create the histogram when not existing
            if key not in hists:
                hists[key] = create_hist_from_variables(*self.hist_vars, storage="double" if is_num else "weight")
            # fill it
            fill_data = {
                v.name: Route(v.expression).apply(events)[sel]
                for v in self.hist_vars
            }
            if not is_num:
                fill_data["weight"] = weight[sel]
            fill_hist(hists[key], fill_data, last_edge_inclusive=True)

        if key in keys_for_stats:
            stats[key] += float(ak.sum(sel if is_num else weight[sel]))

    return events, results
