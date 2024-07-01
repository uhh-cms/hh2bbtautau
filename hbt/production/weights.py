# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.util import maybe_import, safe_div, InsertableDict, dev_sandbox
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult


ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses={
        pu_weight.PRODUCES,
        # custom columns created upstream, probably by a producer
        "process_id",
    },
    # only run on mc
    mc_only=True,
)
def normalized_pu_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self[pu_weight].produces:
        if not weight_name.startswith("pu_weight"):
            continue

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
        events = set_ak_column(events, f"normalized_{weight_name}", norm_weight_per_pid, value_type=np.float32)

    return events


@normalized_pu_weight.init
def normalized_pu_weight_init(self: Producer) -> None:
    self.produces |= {
        f"normalized_{weight_name}"
        for weight_name in self[pu_weight].produces
        if weight_name.startswith("pu_weight")
    }


@normalized_pu_weight.requires
def normalized_pu_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_pu_weight.setup
def normalized_pu_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # load the selection stats
    selection_stats = self.task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json"),
    )

    # get the unique process ids in that dataset
    key = "sum_mc_weight_pu_weight_per_process"
    self.unique_process_ids = list(map(int, selection_stats[key].keys()))

    # helper to get numerators and denominators
    def numerator_per_pid(pid):
        key = "sum_mc_weight_per_process"
        return selection_stats[key].get(str(pid), 0.0)

    def denominator_per_pid(weight_name, pid):
        key = f"sum_mc_weight_{weight_name}_per_process"
        return selection_stats[key].get(str(pid), 0.0)

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
            for pid in self.unique_process_ids
        }
        for weight_name in self[pu_weight].produces
        if weight_name.startswith("pu_weight")
    }


@producer(
    uses={
        "pdf_weight", "pdf_weight_up", "pdf_weight_down",
    },
    produces={
        "normalized_pdf_weight", "normalized_pdf_weight_up", "normalized_pdf_weight_down",
    },
    # only run on mc
    mc_only=True,
)
def normalized_pdf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for postfix in ["", "_up", "_down"]:
        # create the normalized weight
        avg = self.average_pdf_weights[postfix]
        normalized_weight = events[f"pdf_weight{postfix}"] / avg

        # store it
        events = set_ak_column(events, f"normalized_pdf_weight{postfix}", normalized_weight, value_type=np.float32)

    return events


@normalized_pdf_weight.requires
def normalized_pdf_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_pdf_weight.setup
def normalized_pdf_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # load the selection stats
    selection_stats = self.task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json"),
    )

    # save average weights
    self.average_pdf_weights = {
        postfix: safe_div(selection_stats[f"sum_pdf_weight{postfix}"], selection_stats["num_events"])
        for postfix in ["", "_up", "_down"]
    }


@producer(
    uses={
        "murmuf_weight", "murmuf_weight_up", "murmuf_weight_down",
    },
    produces={
        "normalized_murmuf_weight", "normalized_murmuf_weight_up", "normalized_murmuf_weight_down",
    },
    # only run on mc
    mc_only=True,
)
def normalized_murmuf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for postfix in ["", "_up", "_down"]:
        # create the normalized weight
        avg = self.average_murmuf_weights[postfix]
        normalized_weight = events[f"murmuf_weight{postfix}"] / avg

        # store it
        events = set_ak_column(events, f"normalized_murmuf_weight{postfix}", normalized_weight, value_type=np.float32)

    return events


@normalized_murmuf_weight.requires
def normalized_murmuf_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_murmuf_weight.setup
def normalized_murmuf_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # load the selection stats
    selection_stats = self.task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json"),
    )

    # save average weights
    self.average_murmuf_weights = {
        postfix: safe_div(selection_stats[f"sum_murmuf_weight{postfix}"], selection_stats["num_events"])
        for postfix in ["", "_up", "_down"]
    }


@producer(
    uses={
        mc_weight, pdf_weights, murmuf_weights, pu_weight, btag_weights,
    },
    produces={
        mc_weight, pdf_weights, murmuf_weights, pu_weight, btag_weights,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=False,
)
def selection_weights_for_mc(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> ak.Array:

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

    return events
