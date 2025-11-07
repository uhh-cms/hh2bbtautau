# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.top_pt_weight import top_pt_weight as cf_top_pt_weight
from columnflow.production.cms.dy import dy_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import attach_coffea_behavior, default_coffea_collections, set_ak_column

from hbt.production.weights import (
    stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_pdf_weight,
    normalized_murmuf_weight, normalized_ps_weights, normalized_btag_weights_deepjet, normalized_btag_weights_pnet,
)
from hbt.production.tau import tau_weights
from hbt.production.trigger_sf import trigger_weight
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3

ak = maybe_import("awkward")


top_pt_weight = cf_top_pt_weight.derive("top_pt_weight", cls_dict={"require_dataset_tag": None})

muon_weights_lowpt = muon_weights.derive(
    "muon_weights_lowpt",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_sf_lowpt),
        "get_muon_config": (lambda self: self.config_inst.x.muon_sf_lowpt),
    },
)


@producer(
    uses={
        category_ids, stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_ps_weights,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    produces={
        category_ids, stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_ps_weights,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    shifts={
        "minbias_xs_{up,down}",  # PuppiMET used in categories, and depends on pu/minbias_xs through met phi correction
    },
    # whether weight producers should be added and called
    produce_weights=True,
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = attach_coffea_behavior(
        events,
        collections={"HHBJet": default_coffea_collections["Jet"]},
    )
    events = self[category_ids](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[stitched_normalization_weights_dy_tautau_drop](events, **kwargs)

        # normalized pdf weight
        if self.has_dep(normalized_pdf_weight):
            events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        if self.has_dep(normalized_murmuf_weight):
            events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # normalized parton shower weights
        events = self[normalized_ps_weights](events, **kwargs)

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
            # when the muon weight producer has a minimum pt configured, also run the low-pt weights for muons below
            # that pt threshold
            if self.has_dep(muon_weights_lowpt) and self[muon_weights].muon_config.min_pt > 0.0:
                lowpt_mask = events.Muon.pt < self[muon_weights].muon_config.min_pt
                if ak.any(lowpt_mask):
                    # evaluate low-pt weights and multiple to existing ones
                    events2 = self[muon_weights_lowpt](events, muon_mask=lowpt_mask, **kwargs)
                    for r in self[muon_weights].produced_columns:
                        events = set_ak_column(events, r, r.apply(events) * r.apply(events2))
                    del events2

        # trigger weight
        if self.has_dep(trigger_weight):
            events = self[trigger_weight](events, **kwargs)

        # top pt weight
        if self.has_dep(top_pt_weight):
            events = self[top_pt_weight](events, **kwargs)

        # dy weights
        if self.has_dep(dy_weights):
            events = self[dy_weights](events, **kwargs)

    return events


@default.init
def default_init(self: Producer, **kwargs) -> None:
    if self.produce_weights:
        weight_producers = {tau_weights, electron_weights, muon_weights, trigger_weight}
        if self.dataset_inst.has_tag("ttbar"):
            weight_producers.add(top_pt_weight)
        if self.dataset_inst.has_tag("dy"):
            weight_producers.add(dy_weights)

        self.uses |= weight_producers
        self.produces |= weight_producers

        # additional muon weight for low-pt
        if self.config_inst.has_aux("muon_sf_lowpt"):
            self.uses.add(muon_weights_lowpt)


empty = default.derive("empty", cls_dict={"produce_weights": False})
