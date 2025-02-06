# coding: utf-8

"""
Default event weight definitions.
"""

from columnflow.weight import WeightProducer, weight_producer
from columnflow.columnar_util import Route
from columnflow.util import maybe_import, pattern_matcher

ak = maybe_import("awkward")
np = maybe_import("numpy")


@weight_producer(
    # both produced columns and dependent shifts are defined in init below
    # only run on mc
    mc_only=True,
    # options to keep or drop specific weights
    keep_weights=None,
    drop_weights={"normalization_weight_inclusive"},
)
def default(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float32))
    for column in self.weight_columns:
        weight = weight * Route(column).apply(events)

    return events, weight


@default.init
def default_init(self: WeightProducer) -> None:
    dataset_inst = getattr(self, "dataset_inst", None)

    # use the config's auxiliary event_weights, drop some of them based on drop_weights, and on this
    # weight producer instance, store weight_columns, used columns, and shifts
    self.weight_columns = []

    # helpers to match to kept or dropped weights
    do_keep = pattern_matcher(self.keep_weights) if self.keep_weights else (lambda _, /: True)
    do_drop = pattern_matcher(self.drop_weights) if self.drop_weights else (lambda _, /: False)

    # collect all possible weight columns and affected shifts
    all_weights = self.config_inst.x.event_weights
    if dataset_inst:
        all_weights.update(dataset_inst.x("event_weights", {}))
    for weight_name, shift_insts in all_weights.items():
        if not do_keep(weight_name) or do_drop(weight_name):
            continue

        # manually skip pdf and scale weights for samples that do not have lhe info
        if dataset_inst:
            is_lhe_weight = any(shift_inst.has_tag("lhe_weight") for shift_inst in shift_insts)
            if is_lhe_weight and self.dataset_inst.has_tag("no_lhe_weights"):
                continue

        self.weight_columns.append(weight_name)
        self.uses.add(weight_name)
        self.shifts |= {shift_inst.name for shift_inst in shift_insts}


normalization_inclusive = default.derive(
    "normalization_inclusive",
    cls_dict={"drop_weights": {"normalization_weight"}},
)


normalization_only = default.derive(
    "normalization_only",
    cls_dict={"keep_weights": {"normalization_weight"}},
)


normalization_inclusive_only = default.derive(
    "normalization_inclusive_only",
    cls_dict={"keep_weights": {"normalization_weight_inclusive"}, "drop_weights": None},
)
