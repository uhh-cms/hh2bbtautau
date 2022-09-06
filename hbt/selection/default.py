# coding: utf-8

"""
Selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict
from typing import Tuple

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.processes import process_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox

from hbt.selection.met import met_filter_selection
from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"mc_weight"})
def increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # apply the main selection mask to obtain selected events
    event_mask = results.main.event
    events_sel = events[event_mask]

    # increment plain counts
    stats["n_events"] += len(events)
    stats["n_events_selected"] += ak.sum(event_mask, axis=0)

    # store sum of event weights for mc events
    if self.dataset_inst.is_mc:
        weights = events.mc_weight

        # sum for all processes
        stats["sum_mc_weight"] += ak.sum(weights)
        stats["sum_mc_weight_selected"] += ak.sum(weights[event_mask])

        # sums per process id and again per jet multiplicity
        stats.setdefault("sum_mc_weight_per_process", defaultdict(float))
        stats.setdefault("sum_mc_weight_selected_per_process", defaultdict(float))
        stats.setdefault("sum_mc_weight_per_process_and_njet", defaultdict(lambda: defaultdict(float)))
        stats.setdefault("sum_mc_weight_selected_per_process_and_njet", defaultdict(lambda: defaultdict(float)))
        unique_process_ids = np.unique(events.process_id)
        unique_n_jets = []
        if results.has_aux("n_central_jets"):
            unique_n_jets = np.unique(results.x.n_central_jets)
        for p in unique_process_ids:
            stats["sum_mc_weight_per_process"][int(p)] += ak.sum(
                weights[events.process_id == p],
            )
            stats["sum_mc_weight_selected_per_process"][int(p)] += ak.sum(
                weights[event_mask][events_sel.process_id == p],
            )
            for n in unique_n_jets:
                stats["sum_mc_weight_per_process_and_njet"][int(p)][int(n)] += ak.sum(
                    weights[(events.process_id == p) & (results.x.n_central_jets == n)],
                )
                stats["sum_mc_weight_selected_per_process_and_njet"][int(p)][int(n)] += ak.sum(
                    weights[event_mask][
                        (events_sel.process_id == p) &
                        (results.x.n_central_jets[event_mask] == n)
                    ],
                )

    return events


@selector(
    uses={
        attach_coffea_behavior, met_filter_selection, trigger_selection, lepton_selection,
        jet_selection, process_ids, increment_stats,
    },
    produces={
        met_filter_selection, trigger_selection, lepton_selection, jet_selection, process_ids,
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
) -> Tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # met filter selection
    events, met_filter_results = self[met_filter_selection](events, **kwargs)
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

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    # create process ids
    events = self[process_ids](events, **kwargs)

    # increment stats
    events = self[increment_stats](events, results, stats, **kwargs)

    return events, results
