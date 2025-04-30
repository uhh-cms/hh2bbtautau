# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import stitched_normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.top_pt_weight import top_pt_weight as cf_top_pt_weight
from columnflow.production.cms.dy import dy_weights, recoil_corrected_met
from columnflow.util import maybe_import
from columnflow.columnar_util import attach_coffea_behavior, default_coffea_collections
from columnflow.columnar_util import set_ak_column

from hbt.production.weights import (
    normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight, normalized_ps_weights,
)
from hbt.production.btag import normalized_btag_weights_deepjet, normalized_btag_weights_pnet
from hbt.production.tau import tau_weights
from hbt.production.trigger_sf import trigger_weight
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3, IF_DATASET_IS_RUN3_DY

ak = maybe_import("awkward")
np = maybe_import("numpy")

top_pt_weight = cf_top_pt_weight.derive("top_pt_weight", cls_dict={"require_dataset_tag": None})


@producer(
    uses={
        category_ids, stitched_normalization_weights, normalized_pu_weight, normalized_ps_weights,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_IS_RUN3_DY(recoil_corrected_met),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    produces={
        category_ids, stitched_normalization_weights, normalized_pu_weight, normalized_ps_weights,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        # IF_DATASET_IS_RUN3_DY(recoil_corrected_met),
        "RecoilCorrMET.{pt,phi}",
        "RecoilCorrMET.{pt,phi}_{recoilresp,recoilres}_{up,down}",
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
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

    # dy MET corrections
    if self.has_dep(recoil_corrected_met):
        events = self[recoil_corrected_met](events, **kwargs)
    else:
        events = set_ak_column(events, "RecoilCorrMET.pt", events.PuppiMET.pt, value_type=np.float32)
        events = set_ak_column(events, "RecoilCorrMET.phi", events.PuppiMET.phi, value_type=np.float32)
        for syst, postfix in [
            ("RespUp", "recoilresp_up"),
            ("RespDown", "recoilresp_down"),
            ("ResolUp", "recoilres_up"),
            ("ResolDown", "recoilres_down"),
        ]:
            events = set_ak_column(events, f"RecoilCorrMET.pt_{postfix}", events.PuppiMET.pt, value_type=np.float32)
            events = set_ak_column(events, f"RecoilCorrMET.phi_{postfix}", events.PuppiMET.phi, value_type=np.float32)

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


empty = default.derive("empty", cls_dict={"produce_weights": False})
