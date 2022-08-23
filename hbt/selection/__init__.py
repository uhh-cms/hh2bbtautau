# coding: utf-8

"""
Selection methods.
"""

from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from columnflow.columnar_util import EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt"},
    produces={"cutflow.n_jet", "cutflow.ht", "cutflow.jet1_pt"},
    exposed=True,
)
def cutflow_features(self: Selector, events: ak.Array, **kwargs) -> None:
    set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    set_ak_column(events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))


@selector(uses={"LHEWeight.originalXWGTUP"})
def increment_stats(
    self: Selector,
    events: ak.Array,
    mask: ak.Array,
    stats: dict,
    **kwargs,
) -> None:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # apply the mask to obtain selected events
    events_sel = events[mask]

    # increment plain counts
    stats["n_events"] += len(events)
    stats["n_events_selected"] += ak.sum(mask, axis=0)

    # store sum of event weights for mc events
    if self.dataset_inst.is_mc:
        # TODO: some samples don't have this column, so check if "1" is fine for them
        weights = events.LHEWeight.originalXWGTUP

        # sum for all processes
        stats["sum_mc_weight"] += ak.sum(weights)
        stats["sum_mc_weight_selected"] += ak.sum(weights[mask])

        # sums per process id
        stats.setdefault("sum_mc_weight_per_process", defaultdict(float))
        stats.setdefault("sum_mc_weight_selected_per_process", defaultdict(float))
        for p in np.unique(events.process_id):
            stats["sum_mc_weight_per_process"][int(p)] += ak.sum(
                weights[events.process_id == p],
            )
            stats["sum_mc_weight_selected_per_process"][int(p)] += ak.sum(
                weights[mask][events_sel.process_id == p],
            )


@selector(
    uses={
        "nJet", "Jet.pt", "Jet.eta",
    },
    exposed=True,
)
def jet_selection(self: Selector, events: ak.Array, stats: defaultdict, **kwargs) -> SelectionResult:
    # per jet mask
    mask = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)
    # convert to indices and sort by pt
    jet_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)[mask]

    # per event mask
    jet_sel = ak.num(jet_indices, axis=1) >= 1

    # build and return selection results plus new columns (src -> dst -> indices)
    return SelectionResult(
        steps={"Jet": jet_sel},
        objects={"Jet": {"Jet": jet_indices}},
    )


@selector(
    uses={
        jet_selection, category_ids, process_ids, increment_stats,
    },
    produces={
        jet_selection, category_ids, process_ids, increment_stats,
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> SelectionResult:
    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # jet selection
    jet_results = self[jet_selection](events, stats, **kwargs)
    results += jet_results

    # combined event selection after all steps
    event_sel = (
        jet_results.steps.Jet
        # TODO: others
    )
    results.main["event"] = event_sel

    # build categories
    self[category_ids](events, results=results, **kwargs)

    # create process ids
    self[process_ids](events, **kwargs)

    # increment stats
    self[increment_stats](events, event_sel, stats, **kwargs)

    return results
