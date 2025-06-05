# coding: utf-8

""" production methods regarding ml stats """

from __future__ import annotations

import functools
from functools import reduce
from operator import or_

import law

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.util import maybe_import
from columnflow.ml import MLModel
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from columnflow.config_util import get_events_from_categories


from hbt.util import IF_MC
from hbt.histogramming.default import ml_weights as default_hist_producer
from hbt.reduction.default import ml


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

logger = law.logger.get_logger(__name__)


def del_sub_proc_stats(
    stats: dict,
    proc: str,
) -> np.ndarray:
    """
    Function deletes dict keys which are not part of the requested process

    :param stats: Dictionaire containing ML stats for each process.
    :param proc: String of the process.
    :param sub_id: List of ids of sub processes that should be reatined (!).
    """
    item_list = list(stats.weight_map.keys())
    for item in item_list:
        stats[item].pop()


@producer(
    uses={
        "channel_id",
        "category_ids",
        increment_stats,
        "process_id",
        "fold_indices",
        stitched_normalization_weights.PRODUCES,
        default_hist_producer,
    },
    produces={IF_MC("event_weight")},
)
def ml_preparation(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst: MLModel | None = None,
    **kwargs,
) -> ak.Array:
    """
    Producer that is run as part of PrepareMLEvents to collect relevant stats
    """
    if task.task_family == "cf.PrepareMLEvents":
        # pass category mask to only use events that belong to the main "signal region"
        # NOTE: we could also just require the pre_ml_cats Producer here
        n_events = len(events)
        # only select signal-like channels
        channel_id = events.channel_id

        signal_ids = [
            self.config_inst.channels.n.mutau.id,
            self.config_inst.channels.n.etau.id,
            self.config_inst.channels.n.tautau.id,
        ]

        event_mask = reduce(or_, [channel_id == x for x in signal_ids])

        events = events[event_mask]
        events = get_events_from_categories(events=events, categories=["os"], config_inst=self.config_inst)
        events = get_events_from_categories(events=events, categories=["iso"], config_inst=self.config_inst)

        logger.info(f"Select {len(events)} from {n_events} events for MLTraining using")

    weight_map = {
        "num_events": Ellipsis,  # all events
    }

    if task.dataset_inst.is_mc:
        # full event weight
        events, weight = self[default_hist_producer](events, **kwargs)
        events = set_ak_column_f32(events, "event_weight", weight)
        stats["sum_weights"] += float(ak.sum(weight, axis=0))
        weight_map["sum_weights"] = weight
        weight_map["sum_pos_weights"] = (weight, weight > 0)
        weight_map["sum_abs_weights"] = np.abs(weight)
        weight_map["num_events_pos_weights"] = weight > 0

        # normalization weight only
        norm_weight = events["normalization_weight"]
        stats["sum_norm_weights"] += float(ak.sum(norm_weight, axis=0))
        weight_map["sum_norm_weights"] = norm_weight
        weight_map["sum_pos_norm_weights"] = (norm_weight, norm_weight > 0)
        weight_map["sum_abs_norm_weights"] = np.abs(norm_weight)

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": events.fold_indices,
            "mask_fn": (lambda v: events.fold_indices == v),
            "combinations_only": True,
        },
    }

    group_combinations = [("process", "fold")]

    self[increment_stats](
        events,
        None,  # SelectionResult that is not required
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )

    key_list = list(weight_map.keys())
    for key in key_list:
        stats.pop(key, None)
        # TODO: pop 'num_fold_events'

    return events


@ml_preparation.init
def ml_preparation_init(self):
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    # self.uses.add("stitched_normalization_weight")
    self.uses.add(default_hist_producer)
