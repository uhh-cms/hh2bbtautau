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
    drop_weights=None,
)
def default(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float32))
    for column in self.weight_columns:
        weight = weight * Route(column).apply(events)

    return events, weight


@default.init
def default_init(self: WeightProducer) -> None:
    # use the config's auxiliary event_weights, drop some of them based on drop_weights, and on this
    # weight producer instance, store weight_columns, used columns, and shifts
    self.weight_columns = []

    # helpers to match to kept or dropped weights
    do_keep = pattern_matcher(self.keep_weights) if self.keep_weights else (lambda _: True)
    do_drop = pattern_matcher(self.drop_weights) if self.drop_weights else (lambda _: False)

    for weight_name in self.config_inst.x.event_weights:
        if not do_keep(weight_name) or do_drop(weight_name):
            continue

        # manually skip pdf and scale weights for samples that do not have lhe info
        if getattr(self, "dataset_inst", None) is not None:
            is_lhe_weight = any(
                shift_inst.has_tag("lhe_weight")
                for shift_inst in self.config_inst.x.event_weights[weight_name]
            )
            if is_lhe_weight and self.dataset_inst.has_tag("no_lhe_weights"):
                continue

        self.weight_columns.append(weight_name)
        self.uses.add(weight_name)
        self.shifts |= {
            shift_inst.name
            for shift_inst in self.config_inst.x.event_weights[weight_name]
        }


normalization_only = default.derive(
    "normalization_only",
    cls_dict={"keep_weights": "normalization_weight"},
)
