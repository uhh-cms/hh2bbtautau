# coding: utf-8

"""
Empty selectors + trigger selection
"""

from operator import and_
from functools import reduce
from collections import defaultdict
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.processes import process_ids
# from columnflow.selection.cms.met_filters import met_filters
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox
from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.production.features import cutflow_features
from hbt.selection.jet import jet_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        process_ids, mc_weight, increment_stats, cutflow_features, trigger_selection,
        lepton_selection, attach_coffea_behavior, category_ids, jet_selection,
    },
    produces={
        process_ids, mc_weight, cutflow_features, trigger_selection, jet_selection,
        lepton_selection, category_ids,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=True,
)
def default_custom(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # add corrected mc weights
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # met filter selection TODO
    # events, met_filter_results = self[met_filters](events, **kwargs)
    # results += met_filter_results

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
    results.event = event_sel

    # write out process/category IDs
    events = self[process_ids](events, **kwargs)
    events = self[category_ids](events, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": Ellipsis,
    }
    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, Ellipsis)

        group_map = {
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results
