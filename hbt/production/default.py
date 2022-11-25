# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.electron import electron_weights
from columnflow.production.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight
from hbt.production.btag import normalized_btag_weights

ak = maybe_import("awkward")


@producer(
    uses={
        features, category_ids, normalization_weights, normalized_pu_weight,
        normalized_btag_weights, electron_weights, muon_weights,
    },
    produces={
        features, category_ids, normalization_weights, normalized_pu_weight,
        normalized_btag_weights, electron_weights, muon_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute normalized pu weights
    events = self[normalized_pu_weight](events, **kwargs)

    # btag weights
    events = self[normalized_btag_weights](events, **kwargs)

    # electron sf weights
    events = self[electron_weights](events, **kwargs)

    # muon sf weights
    events = self[muon_weights](events, **kwargs)

    return events
