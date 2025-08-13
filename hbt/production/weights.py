# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from __future__ import annotations

import re
import copy
import functools

import law

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
hist = maybe_import("hist")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


class stitched_normalization_weights_dy_tautau_drop(stitched_normalization_weights):
    """
    Same as the standard :py:class:`stitched_normalization_weights` producer, but it adjusts the dataset selection stats
    for DY datasets if needed to accommodate for the dropped tautau events in certain datasets.
    """

    def update_dataset_selection_stats_sum_weights(
        self,
        dataset_selection_stats: dict[str, dict[str, float | dict[str, float]]],
    ) -> dict[str, dict[str, float]]:
        # this only applies to dy datasets with the dy_lep_amcatnlo tag
        if not self.dataset_inst.has_tag("dy_lep_amcatnlo"):
            return dataset_selection_stats

        # start from a copy
        dataset_selection_stats = copy.deepcopy(dataset_selection_stats)

        # cached decisions on which tautau process ids to drop
        drop_ids = {}

        def drop_id(proc_id_str: str) -> bool:
            if proc_id_str not in drop_ids:
                proc = self.config_inst.get_process(int(proc_id_str))
                drop_ids[proc_id_str] = proc.x.lep_id == 15
            return drop_ids[proc_id_str]

        # start traversing the nested stats
        for dataset_name, stats in dataset_selection_stats.items():
            # the corresponding dataset needs to have the dy_drop_tautau tag
            if not self.config_inst.get_dataset(dataset_name).has_tag("dy_drop_tautau"):
                continue
            for entry_name, _stats in stats.items():
                # only consider dictionaries that map process ids (as strings) to other values
                if not isinstance(_stats, dict):
                    continue
                # only consider certain entries
                if not re.match(r"^(sum|num)_.+_per_process$", entry_name):
                    continue
                # loop over entries and potentially set to zero
                for proc_id_str in _stats:
                    if isinstance(_stats[proc_id_str], (int, float)) and drop_id(proc_id_str):
                        _stats[proc_id_str] = type(_stats[proc_id_str])()  # produces 0 with correct type

        return dataset_selection_stats


@producer(
    uses={pu_weight.PRODUCES, "process_id"},
    mc_only=True,
)
def normalized_pu_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self.pu_weight_names:
        # create a weight vector starting with ones
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]

        # store it
        norm_weight_per_pid = ak.values_astype(norm_weight_per_pid, np.float32)
        events = set_ak_column_f32(events, f"normalized_{weight_name}", norm_weight_per_pid)

    return events


@normalized_pu_weight.post_init
def normalized_pu_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    # remember pu columns to read and produce
    self.pu_weight_names = {
        weight_name
        for weight_name in map(str, self[pu_weight].produced_columns)
        if (
            weight_name.startswith("pu_weight") and
            (task.global_shift_inst.is_nominal or not weight_name.endswith(("_up", "_down")))
        )
    }
    # adjust columns
    self.uses -= {pu_weight.PRODUCES}
    self.uses |= self.pu_weight_names
    self.produces |= {f"normalized_{weight_name}" for weight_name in self.pu_weight_names}


@normalized_pu_weight.requires
def normalized_pu_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_pu_weight.setup
def normalized_pu_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:
    # load the selection stats
    hists = task.cached_value(
        key="selection_hists",
        func=lambda: inputs["selection_stats"]["hists"].load(formatter="pickle"),
    )

    # get the unique process ids in that dataset
    self.unique_process_ids = list(hists["sum_mc_weight_pu_weight"].axes["process"])

    # helper to get numerators and denominators
    def get_sum(pid, weight_name="", /):
        if weight_name:
            weight_name = "_" + weight_name
        key = f"sum_mc_weight{weight_name}"
        return hists[key][{"process": hist.loc(pid)}].sum().value

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(get_sum(pid), get_sum(pid, weight_name))
            for pid in self.unique_process_ids
        }
        for weight_name in (str(route) for route in self[pu_weight].produced_columns)
        if weight_name.startswith("pu_weight")
    }


@producer(
    uses={pdf_weights.PRODUCES},
    mc_only=True,
)
def normalized_pdf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self.pdf_weight_names:
        # create the normalized weight
        avg = self.average_pdf_weights[weight_name]
        normalized_weight = events[weight_name] / avg

        # store it
        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized_weight)

    return events


@normalized_pdf_weight.post_init
def normalized_pdf_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    # remember pdf columns to read and produce
    self.pdf_weight_names = {
        weight_name
        for weight_name in map(str, self[pdf_weights].produced_columns)
        if (
            weight_name.startswith("pdf_weight") and
            (task.global_shift_inst.is_nominal or not weight_name.endswith(("_up", "_down")))
        )
    }
    # adjust columns
    self.uses.clear()
    self.uses |= self.pdf_weight_names
    self.produces |= {f"normalized_{weight_name}" for weight_name in self.pdf_weight_names}


@normalized_pdf_weight.requires
def normalized_pdf_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_pdf_weight.setup
def normalized_pdf_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:
    # load the selection stats
    hists = task.cached_value(
        key="selection_hists",
        func=lambda: inputs["selection_stats"]["hists"].load(formatter="pickle"),
    )

    # save average weights
    self.average_pdf_weights = {
        weight_name: safe_div(hists[f"sum_{weight_name}"].sum().value, hists["num_events"].sum())
        for weight_name in self.pdf_weight_names
    }


# variation of the pdf weights producer that does not store up and down shifted weights
# but that stores all available pdf weights for the full treatment based on histograms
all_pdf_weights = pdf_weights.derive("all_pdf_weights", cls_dict={"store_all_weights": True})


@producer(
    uses={murmuf_weights.PRODUCES},
    mc_only=True,
)
def normalized_murmuf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self.mu_weight_names:
        # create the normalized weight
        avg = self.average_mu_weights[weight_name]
        normalized_weight = events[weight_name] / avg

        # store it
        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized_weight)

    return events


@normalized_murmuf_weight.post_init
def normalized_murmuf_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    # remember mur/muf columns to read and produce
    self.mu_weight_names = {
        weight_name
        for weight_name in map(str, self[murmuf_weights].produced_columns)
        if (
            weight_name.startswith("murmuf_weight") and
            (task.global_shift_inst.is_nominal or not weight_name.endswith(("_up", "_down")))
        )
    }
    # adjust columns
    self.uses.clear()
    self.uses |= self.mu_weight_names
    self.produces |= {f"normalized_{weight_name}" for weight_name in self.mu_weight_names}


@normalized_murmuf_weight.requires
def normalized_murmuf_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_murmuf_weight.setup
def normalized_murmuf_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:
    # load the selection stats
    hists = task.cached_value(
        key="selection_hists",
        func=lambda: inputs["selection_stats"]["hists"].load(formatter="pickle"),
    )

    # save average weights
    self.average_mu_weights = {
        weight_name: safe_div(hists[f"sum_{weight_name}"].sum().value, hists["num_events"].sum())
        for weight_name in self.mu_weight_names
    }


@producer(
    uses={ps_weights.PRODUCES},
    mc_only=True,
)
def normalized_ps_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self.ps_weight_names:
        # create the normalized weight
        avg = self.average_ps_weights[weight_name]
        normalized_weight = events[weight_name] / avg

        # store it
        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized_weight)

    return events


@normalized_ps_weights.post_init
def normalized_ps_weights_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    # remember ps weight columns to read and produce
    self.ps_weight_names = {
        weight_name
        for weight_name in map(str, self[ps_weights].produced_columns)
        if (
            "weight" in weight_name and
            (task.global_shift_inst.is_nominal or not weight_name.endswith(("_up", "_down")))
        )
    }
    # adjust columns
    self.uses.clear()
    self.uses |= self.ps_weight_names
    self.produces |= {f"normalized_{weight_name}" for weight_name in self.ps_weight_names}


@normalized_ps_weights.requires
def normalized_ps_weights_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_ps_weights.setup
def normalized_ps_weights_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:
    # load the selection stats
    hists = task.cached_value(
        key="selection_hists",
        func=lambda: inputs["selection_stats"]["hists"].load(formatter="pickle"),
    )

    # save average weights
    self.average_ps_weights = {
        weight_name: safe_div(hists[f"sum_{weight_name}"].sum().value, hists["num_events"].sum())
        for weight_name in self.ps_weight_names
    }
