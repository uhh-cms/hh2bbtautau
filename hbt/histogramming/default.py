# coding: utf-8

"""
Default histogram producers (mostly for event weight generation).
"""

from __future__ import annotations

import law
import order as od

from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.columnar_util import Route
from columnflow.util import maybe_import, pattern_matcher, safe_div
from columnflow.hist_util import create_hist_from_variables
from columnflow.types import TYPE_CHECKING, Any

ak = maybe_import("awkward")
np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")


@cf_default.hist_producer(
    # both produced columns and dependent shifts are defined in init below
    # options to keep or drop specific weights
    keep_weights=None,
    drop_weights={"normalization_weight_inclusive"},
)
def default(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    weight = ak.Array(np.ones(len(events), dtype=np.float32))

    # build the full event weight
    if self.dataset_inst.is_mc and len(events):
        for column in self.weight_columns:
            weight = weight * Route(column).apply(events)

    return events, weight


@default.create_hist
def default_create_hist(self: HistProducer, variables: list[od.Variable], task: law.Task, **kwargs) -> hist.Hist:
    # default axis setup
    categorical_axes = [
        ("category", "intcat"),
        ("process", "intcat"),
        ("shift", "intcat", [0]),
    ]

    # extend by index of pdf/alphas or murmuf weight if requested
    if self.pdf_via_hist:
        categorical_axes.append(
            ("pdf", "intcat", list(range(len(self.average_pdf_weights) + len(self.average_alphas_weights)))),
        )
    elif self.murmuf_via_hist:
        categorical_axes.append(
            ("murmuf", "intcat", list(range(len(self.murmuf_weights)))),
        )

    # create the histogram
    return create_hist_from_variables(*variables, categorical_axes=categorical_axes, weight=True)


@default.fill_hist
def default_fill_hist(self: HistProducer, h: hist.Hist, data: dict[str, Any], events: ak.Array, **kwargs) -> None:
    def fill(data, additional_weight=None):
        if additional_weight is not None:
            data = {**data, "weight": data["weight"] * additional_weight}
        super(default, self).fill_hist_func(h=h, data=data, events=events, **kwargs)

    if self.pdf_via_hist:
        # loop over weights and fill into each pdf bin separately with the corresponding weight
        # (cannot be vectorized since the actual fill weights change)
        pdf_weights_hessian = ak.fill_none(ak.pad_none(events.pdf_weights_hessian, 100, axis=1), 1)
        pdf_weights_alphas = ak.fill_none(ak.pad_none(events.pdf_weights_alphas, 2, axis=1), 1)
        for i in range(100):
            pdf_weight = pdf_weights_hessian[:, i] / self.average_pdf_weights[i]
            fill({**data, "pdf": i}, additional_weight=pdf_weight)
        for i in range(2):
            alphas_weight = pdf_weights_alphas[:, i] / self.average_alphas_weights[i]
            fill({**data, "pdf": 100 + i}, additional_weight=alphas_weight)

    elif self.murmuf_via_hist:
        for i, weight_name in enumerate(self.murmuf_weights):
            murmuf_weight = events[weight_name] / self.average_murmuf_weights[weight_name]
            fill({**data, "murmuf": i}, additional_weight=murmuf_weight)

    else:
        fill(data)


@default.post_process_merged_hist
def default_post_process_merged_hist(self: HistProducer, h: hist.Hist, **kwargs) -> hist.Hist:
    h = super(default, self).post_process_merged_hist_func(h=h, **kwargs)

    if self.pdf_via_hist:
        from IPython import embed; embed(header="implement pdf+alphas merge")
        raise NotImplementedError(
            f"post_process_merged_hist for 'pdf_via_hist' is not yet implemented by {self.cls_name}",
        )

    elif self.murmuf_via_hist:
        from IPython import embed; embed(header="implement murmuf merge")
        raise NotImplementedError(
            f"post_process_merged_hist for 'murmuf_via_hist' is not yet implemented by {self.cls_name}",
        )

    return h


@default.init
def default_init(self: HistProducer) -> None:
    # use the config's auxiliary event_weights, drop some of them based on drop_weights, and on this
    # weight producer instance, store weight_columns, used columns, and shifts
    self.weight_columns = set()

    if self.dataset_inst.is_data:
        return

    # update shifts
    if self.config_inst.x.pdf_via_hist and not self.dataset_inst.has_tag("no_lhe_weights"):
        self.shifts |= {"pdf_up", "pdf_down"}
    if self.config_inst.x.murmuf_via_hist and not self.dataset_inst.has_tag("no_lhe_weights"):
        self.murmuf_weights = [
            "mur_down_muf_down",
            "mur_down_muf_nom",
            "mur_nom_muf_down",
            "mur_nom_muf_nom",
            "mur_nom_muf_up",
            "mur_up_muf_nom",
            "mur_up_muf_up",
        ]
        self.shifts |= {"murmuf_up", "murmuf_down"}

    # helpers to match to kept or dropped weights
    do_keep = pattern_matcher(self.keep_weights) if self.keep_weights else (lambda _, /: True)
    do_drop = pattern_matcher(self.drop_weights) if self.drop_weights else (lambda _, /: False)

    # collect all possible weight columns and affected shifts
    all_weights = self.config_inst.x.event_weights.copy()
    all_weights.update(self.dataset_inst.x("event_weights", {}))
    for weight_name, shift_insts in all_weights.items():
        if not do_keep(weight_name) or do_drop(weight_name):
            continue

        # manually skip pdf and scale weights for samples that do not have lhe info
        is_lhe_weight = any(shift_inst.has_tag("lhe_weight") for shift_inst in shift_insts)
        if is_lhe_weight and self.dataset_inst.has_tag("no_lhe_weights"):
            continue

        self.weight_columns.add(weight_name)
        self.uses.add(weight_name)
        self.shifts |= {shift_inst.name for shift_inst in shift_insts}


@default.post_init
def default_post_init(self: HistProducer, task: law.Task, **kwargs) -> None:
    super(default, self).post_init_func(task=task, **kwargs)

    # store flags denoting whether pdf/murmuf handling is done via histogram methods
    self.pdf_via_hist = task.local_shift_inst.source == "pdf" and self.config_inst.x.pdf_via_hist
    self.murmuf_via_hist = task.local_shift_inst.source == "murmuf" and self.config_inst.x.murmuf_via_hist

    # update used columns for histogram production
    if task.task_family == "cf.CreateHistograms":
        if self.pdf_via_hist:
            self.uses.add("pdf_weights_{hessian,alphas}")
        if self.murmuf_via_hist:
            self.uses |= set(self.murmuf_weights)

    # when using any histogram method, shift validation from to post-merge
    if self.pdf_via_hist or self.murmuf_via_hist:
        self.post_process_compatibility_check = False
        self.post_process_merged_compatibility_check = True


@default.requires
def default_requires(self: HistProducer, task: law.Task, reqs: dict, **kwargs) -> None:
    super(default, self).requires_func(task=task, reqs=reqs, **kwargs)

    # TODO: this should only be done in post_init but this is not called yet at the right moment in cf
    self.pdf_via_hist = task.local_shift_inst.source == "pdf" and self.config_inst.x.pdf_via_hist
    self.murmuf_via_hist = task.local_shift_inst.source == "murmuf" and self.config_inst.x.murmuf_via_hist

    # require selection stats in case the requested shift
    if self.pdf_via_hist or self.murmuf_via_hist:
        from columnflow.tasks.selection import MergeSelectionStats
        reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
            task,
            branch=-1 if task.is_workflow() else 0,
        )


@default.setup
def default_setup(self: HistProducer, task: law.Task, inputs: dict, **kwargs) -> None:
    super(default, self).setup_func(task=task, inputs=inputs, **kwargs)

    if self.pdf_via_hist or self.murmuf_via_hist:
        # load the selection stats
        stats = task.cached_value(
            key="selection_stats",
            func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
        )

        # save average weights
        if self.pdf_via_hist:
            self.average_pdf_weights = [
                safe_div(stats[f"sum_pdf_weight_{i}"], stats["num_events"])
                for i in range(100)
            ]
            self.average_alphas_weights = [
                safe_div(stats[f"sum_alphas_weight_{i}"], stats["num_events"])
                for i in range(2)
            ]
        elif self.murmuf_via_hist:
            self.average_murmuf_weights = {
                key: safe_div(stats[f"sum_{key}"], stats["num_events"])
                for key in self.murmuf_weights
            }


no_weight = default.derive("no_weight", cls_dict={
    "drop_weights": {"*"},
})

# weight producer for cross checking histograms without stitching
normalization_inclusive = default.derive("normalization_inclusive", cls_dict={
    "drop_weights": {"normalization_weight"},
})

normalization_only = default.derive("normalization_only", cls_dict={
    "keep_weights": {"normalization_weight"},
})

normalization_inclusive_only = default.derive("normalization_inclusive_only", cls_dict={
    "keep_weights": {"normalization_weight_inclusive"},
    "drop_weights": None,
})

for weight_name in [
    "normalized_pdf_weight",
    "normalized_murmuf_weight",
    "normalized_pu_weight",
    "normalized_isr_weight",
    "normalized_fsr_weight",
    "normalized_njet_btag_weight_pnet",
    "btag_weight",
    "electron_id_weight",
    "electron_reco_weight",
    "muon_id_weight",
    "muon_iso_weight",
    "tau_weight",
    "trigger_weight",
    "dy_weight",
    "top_pt_weight",
]:
    default.derive(f"no_{weight_name}", cls_dict={"drop_weights": {"normalization_weight_inclusive", weight_name}})

default.derive("no_tau_and_trigger_weight", cls_dict={
    "drop_weights": {"normalization_weight_inclusive", "tau_weight", "trigger_weight"},
})
