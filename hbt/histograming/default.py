# coding: utf-8

"""
Default histogram producers (mostly for event weight generation).
"""

from columnflow.histograming import HistProducer
from columnflow.histograming.default import cf_default
from columnflow.columnar_util import Route
from columnflow.util import maybe_import, pattern_matcher
from columnflow.config_util import get_shifts_from_sources

from hbt.production.default import top_pt_weight
from hbt.util import IF_DATASET_HAS_TOP

ak = maybe_import("awkward")
np = maybe_import("numpy")


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


@default.init
def default_init(self: HistProducer) -> None:
    # use the config's auxiliary event_weights, drop some of them based on drop_weights, and on this
    # weight producer instance, store weight_columns, used columns, and shifts
    self.weight_columns = []

    if self.dataset_inst.is_data:
        return

    # helpers to match to kept or dropped weights
    do_keep = pattern_matcher(self.keep_weights) if self.keep_weights else (lambda _, /: True)
    do_drop = pattern_matcher(self.drop_weights) if self.drop_weights else (lambda _, /: False)

    # collect all possible weight columns and affected shifts
    all_weights = self.config_inst.x.event_weights
    all_weights.update(self.dataset_inst.x("event_weights", {}))
    self.uses |= set((IF_DATASET_HAS_TOP(top_pt_weight.produces),))
    for weight_name, shift_insts in all_weights.items():
        if not do_keep(weight_name) or do_drop(weight_name):
            continue

        # manually skip pdf and scale weights for samples that do not have lhe info
        is_lhe_weight = any(shift_inst.has_tag("lhe_weight") for shift_inst in shift_insts)
        if is_lhe_weight and self.dataset_inst.has_tag("no_lhe_weights"):
            continue

        self.weight_columns.append(weight_name)
        self.uses.add(weight_name)
        self.shifts |= {shift_inst.name for shift_inst in shift_insts}
    
    if self.has_dep(top_pt_weight):
        self.weight_columns.append(top_pt_weight.produces)
        self.shifts.add(get_shifts_from_sources(self.config_inst, "top_pt"))


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
