# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.weights import (
    normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight,
)
from hbt.production.btag import normalized_btag_weights_deepjet, normalized_btag_weights_pnet
from hbt.production.tau import tau_weights, trigger_weights
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3

ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, stitched_normalization_weights, normalized_pu_weight,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    produces={
        category_ids, stitched_normalization_weights, normalized_pu_weight,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    # whether weight producers should be added and called
    produce_weights=True,
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = self[category_ids](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[stitched_normalization_weights](events, **kwargs)

        # normalized pdf weight
        if self.has_dep(normalized_pdf_weight):
            events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        if self.has_dep(normalized_murmuf_weight):
            events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights_deepjet](events, **kwargs)
        if self.has_dep(normalized_btag_weights_pnet):
            events = self[normalized_btag_weights_pnet](events, **kwargs)

        # tau weights
        if self.has_dep(tau_weights):
            events = self[tau_weights](events, **kwargs)

        # electron weights
        if self.has_dep(electron_weights):
            events = self[electron_weights](events, **kwargs)

        # muon weights
        if self.has_dep(muon_weights):
            events = self[muon_weights](events, **kwargs)

        # trigger weights
        if self.has_dep(trigger_weights):
            events = self[trigger_weights](events, **kwargs)

    return events


@default.init
def default_init(self: Producer) -> None:
    if self.produce_weights:
        weight_producers = {tau_weights, electron_weights, muon_weights, trigger_weights}
        self.uses |= weight_producers
        self.produces |= weight_producers


empty = default.derive("empty", cls_dict={"produce_weights": False})
