# coding: utf-8

"""
Producers for phase-space normalized btag scale factor weights.
"""

from __future__ import annotations

import functools

from columnflow.production import Producer, producer
from columnflow.production.cms.btag import btag_weights
from columnflow.util import maybe_import, safe_div, InsertableDict
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# custom btag weight producer for deepjet and pnet configs
btag_weights_deepjet = btag_weights.derive("btag_weights_deepjet", cls_dict={
    "weight_name": "btag_weight_deepjet",
    "tagger_name": "deepjet",
    "get_btag_config": (lambda self: self.config_inst.x.btag_sf_deepjet),
})
btag_weights_pnet = btag_weights.derive("btag_weights_pnet", cls_dict={
    "weight_name": "btag_weight_pnet",
    "tagger_name": "pnet",
    "get_btag_config": (lambda self: self.config_inst.x.btag_sf_pnet),
})


@producer(
    uses={"process_id", "Jet.{pt,phi,eta}"},
    # only run on mc
    mc_only=True,
    # configurable weight producer class
    btag_weights_cls=None,
)
def _normalized_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for route in self[self.btag_weights_cls].produced_columns:
        weight_name = str(route)
        if not weight_name.startswith(self.weight_name):
            continue

        # create a weight vectors starting with ones for both weight variations, i.e.,
        # nomalization per pid and normalization per pid and jet multiplicity
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)
        norm_weight_per_pid_njet = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            # single value
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]
            # lookup table
            n_jets = ak.to_numpy(ak.num(events[pid_mask].Jet.pt, axis=1))
            norm_weight_per_pid_njet[pid_mask] = self.ratio_per_pid_njet[weight_name][pid][n_jets]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]
        norm_weight_per_pid_njet = norm_weight_per_pid_njet * events[weight_name]

        # store them
        events = set_ak_column_f32(events, f"normalized_{weight_name}", norm_weight_per_pid)
        events = set_ak_column_f32(events, f"normalized_njet_{weight_name}", norm_weight_per_pid_njet)

    return events


@_normalized_btag_weights.init
def _normalized_btag_weights_init(self: Producer) -> None:
    assert self.btag_weights_cls, "btag_weights_cls must be set"

    if not getattr(self, "dataset_inst", None):
        return

    # reuse the weight and tagger names
    self.weight_name = self.btag_weights_cls.weight_name
    self.tagger_name = self.btag_weights_cls.tagger_name

    # add produced columns
    for route in self[self.btag_weights_cls].produced_columns:
        name = str(route)
        if name.startswith(self.weight_name):
            self.produces.add(f"normalized_{{,njet_}}{name}")


@_normalized_btag_weights.requires
def _normalized_btag_weights_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        self.task,
        branch=-1 if self.task.is_workflow() else 0,
    )


@_normalized_btag_weights.setup
def _normalized_btag_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    # load the selection stats
    selection_stats = self.task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    # get the unique process ids in that dataset
    key = f"sum_btag_weight_{self.tagger_name}_selected_nob_{self.tagger_name}_per_process_and_njet"
    self.unique_process_ids = list(map(int, selection_stats[key].keys()))

    # get the maximum numbers of jets
    max_n_jets = max(map(int, sum((list(d.keys()) for d in selection_stats[key].values()), [])))

    # helper to get sums of mc weights per pid and njet, with an optional weight name
    def sum_per_pid(pid, weight_name="", /):
        if weight_name:
            weight_name += "_"
        key = f"sum_mc_weight_{weight_name}selected_nob_{self.tagger_name}_per_process"
        return selection_stats[key].get(str(pid), 0.0)

    def sum_per_pid_njet(pid, n_jets, weight_name="", /):
        if weight_name:
            weight_name += "_"
        key = f"sum_mc_weight_{weight_name}selected_nob_{self.tagger_name}_per_process_and_njet"
        return selection_stats[key].get(str(pid), {}).get(str(n_jets), 0.0)

    # ratio per weight and pid
    # extract the ratio per weight, pid and also the jet multiplicity, using the latter as in index
    self.ratio_per_pid = {}
    self.ratio_per_pid_njet = {}
    for route in self[self.btag_weights_cls].produced_columns:
        weight_name = str(route)
        if not weight_name.startswith(self.btag_weights_cls.weight_name):
            continue
        # normal ratio
        self.ratio_per_pid[weight_name] = {
            pid: safe_div(sum_per_pid(pid), sum_per_pid(pid, weight_name))
            for pid in self.unique_process_ids
        }
        # per jet multiplicity ratio
        self.ratio_per_pid_njet[weight_name] = {
            pid: np.array([
                safe_div(sum_per_pid_njet(pid, n_jets), sum_per_pid_njet(pid, n_jets, weight_name))
                for n_jets in range(max_n_jets + 1)
            ])
            for pid in self.unique_process_ids
        }


# derive for btaggers
normalized_btag_weights_deepjet = _normalized_btag_weights.derive("normalized_btag_weights_deepjet", cls_dict={
    "btag_weights_cls": btag_weights_deepjet,
    "uses": _normalized_btag_weights.uses | {btag_weights_deepjet.PRODUCES},
})
normalized_btag_weights_pnet = _normalized_btag_weights.derive("normalized_btag_weights_pnet", cls_dict={
    "btag_weights_cls": btag_weights_pnet,
    "uses": _normalized_btag_weights.uses | {btag_weights_pnet.PRODUCES},
})
