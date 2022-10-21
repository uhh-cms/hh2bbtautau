# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.electron import electron_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import event_weights
from hbt.production.btag import normalized_btag_weight

ak = maybe_import("awkward")


@producer(
    uses={
        features, category_ids, event_weights, normalized_btag_weight, electron_weights,
    },
    produces={
        features, category_ids, event_weights, normalized_btag_weight, electron_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # event weights
    events = self[event_weights](events, **kwargs)

    # btag weights
    events = self[normalized_btag_weight](events, **kwargs)

    # electron sf weights
    events = self[electron_weights](events, **kwargs)

    return events
