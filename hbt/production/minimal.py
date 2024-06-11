# coding: utf-8

"""
Minimal producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, normalization_weights,
    },
    produces={
        category_ids, normalization_weights,
    },
)
def minimal(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    # category ids
    events = self[category_ids](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

    return events
