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
from hbt.production.tau import tau_weights, trigger_weights

ak = maybe_import("awkward")


@producer(
    uses={features, category_ids},
    produces={features, category_ids},
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # compute normalization weights
    if self.dataset_inst.is_mc:
        events = self[normalization_weights](events, **kwargs)

    # compute normalized pu weights
    if self.dataset_inst.is_mc:
        events = self[normalized_pu_weight](events, **kwargs)

    # btag weights
    if self.dataset_inst.is_mc:
        events = self[normalized_btag_weights](events, **kwargs)

    # tau weights
    if self.dataset_inst.is_mc:
        events = self[tau_weights](events, **kwargs)

    # electron weights
    if self.dataset_inst.is_mc:
        events = self[electron_weights](events, **kwargs)

    # muon weights
    if self.dataset_inst.is_mc:
        events = self[muon_weights](events, **kwargs)

    # trigger weights
    if self.dataset_inst.is_mc:
        events = self[trigger_weights](events, **kwargs)

    return events


@default.init
def default_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    # my only producers
    producers = {
        normalization_weights, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights,
    }
    self.uses |= producers
    self.produces |= producers
