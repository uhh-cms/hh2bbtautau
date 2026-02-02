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
from columnflow.columnar_util import attach_coffea_behavior, set_ak_column, full_like

from hbt.production.weights import (
    stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_pdf_weight,
    normalized_murmuf_weight, normalized_ps_weights, normalized_btag_weights_deepjet, normalized_btag_weights_pnet,
)
from hbt.production.tau import tau_weights
from hbt.production.trigger_sf import trigger_weight
from hbt.util import IF_DATASET_HAS_LHE_WEIGHTS, IF_RUN_3

np = maybe_import("numpy")
ak = maybe_import("awkward")


hbt_category_ids = category_ids.derive("hbt_category_ids", cls_dict={"require_producers": ["vbf_dnn_moe"]})

electron_id_weights = electron_weights.derive(
    "electron_id_weights",
    cls_dict={
        "get_electron_config": (lambda self: self.config_inst.x.electron_id_sf),
        "weight_name": "electron_id_weight",
    },
)

electron_reco_weights = electron_weights.derive(
    "electron_reco_weights",
    cls_dict={
        "get_electron_config": (lambda self: self.config_inst.x.electron_reco_sf),
        "weight_name": "electron_reco_weight",
    },
)

muon_id_weights = muon_weights.derive(
    "muon_id_weights",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.muon_id_sf),
        "weight_name": "muon_id_weight",
    },
)

muon_id_weights_lowpt = muon_weights.derive(
    "muon_id_weights_lowpt",
    cls_dict={
        "get_muon_file": (lambda self, external_files: external_files.muon_sf_lowpt),
        "get_muon_config": (lambda self: self.config_inst.x.muon_id_sf_lowpt),
        "weight_name": "muon_id_weight",
    },
)

muon_iso_weights = muon_weights.derive(
    "muon_iso_weights",
    cls_dict={
        "get_muon_config": (lambda self: self.config_inst.x.muon_iso_sf),
        "weight_name": "muon_iso_weight",
    },
)

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
        hbt_category_ids, stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_ps_weights,
        normalized_btag_weights_deepjet, IF_RUN_3(normalized_btag_weights_pnet),
        IF_DATASET_HAS_LHE_WEIGHTS(normalized_pdf_weight, normalized_murmuf_weight),
        # weight producers added dynamically if produce_weights is set
    },
    produces={
        hbt_category_ids, stitched_normalization_weights_dy_tautau_drop, normalized_pu_weight, normalized_ps_weights,
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
    events = attach_coffea_behavior(events, collections={"HHBJet": "Jet"})
    events = self[hbt_category_ids](events, **kwargs)

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
        if self.has_dep(electron_id_weights):
            events = self[electron_id_weights](events, **kwargs)
        if self.has_dep(electron_reco_weights):
            events = self[electron_reco_weights](events, **kwargs)

        # muon weights
        if self.has_dep(muon_id_weights):
            events = self[muon_id_weights](events, **kwargs)
            # when the muon weight producer has a minimum pt configured, also run the low-pt weights for muons below
            # that pt threshold
            if self.has_dep(muon_id_weights_lowpt) and self[muon_id_weights].muon_config.min_pt > 0.0:
                low_pt_mask = events.Muon.pt < self[muon_id_weights].muon_config.min_pt
                if ak.any(low_pt_mask):
                    # evaluate low-pt weights and multiply to existing ones
                    events2 = self[muon_id_weights_lowpt](events, muon_mask=low_pt_mask, **kwargs)
                    for r in self[muon_id_weights].produced_columns:
                        events = set_ak_column(events, r, r.apply(events) * r.apply(events2))
                    del events2
        if self.has_dep(muon_iso_weights):
            events = self[muon_iso_weights](events, **kwargs)

        # trigger weight
        if self.has_dep(trigger_weight):
            events = self[trigger_weight](events, **kwargs)
        elif self.produce_weights:
            # TODO: 2024: fake trigger weights for 2024 for now
            assert self.config_inst.campaign.x.year == 2024
            ones = full_like(events.event, 1.0, dtype=np.float32)
            for trigger_weight_name in self.trigger_weight_names:
                events = set_ak_column(events, trigger_weight_name, ones)

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
        weight_producers = {
            tau_weights,
            electron_id_weights,
            electron_reco_weights,
            muon_id_weights,
            muon_iso_weights,
        }
        # TODO: 2024: use trigger_weight producer for 2024 when available
        if self.config_inst.campaign.x.year != 2024:
            weight_producers.add(trigger_weight)
        else:
            self.trigger_weight_names = [
                "trigger_weight",
                "trigger_weight_e_down",
                "trigger_weight_e_up",
                "trigger_weight_jet_down",
                "trigger_weight_jet_up",
                "trigger_weight_mu_down",
                "trigger_weight_mu_up",
                "trigger_weight_tau_dm0_down",
                "trigger_weight_tau_dm0_up",
                "trigger_weight_tau_dm10_down",
                "trigger_weight_tau_dm10_up",
                "trigger_weight_tau_dm11_down",
                "trigger_weight_tau_dm11_up",
                "trigger_weight_tau_dm1_down",
                "trigger_weight_tau_dm1_up",
            ]
            self.produces.update(self.trigger_weight_names)

        if self.dataset_inst.has_tag("ttbar"):
            weight_producers.add(top_pt_weight)
        if self.dataset_inst.has_tag("dy"):
            weight_producers.add(dy_weights)

        self.uses |= weight_producers
        self.produces |= weight_producers

        # additional muon weight for low-pt
        if self.config_inst.has_aux("muon_sf_lowpt"):
            self.uses.add(muon_id_weights_lowpt)


empty = default.derive("empty", cls_dict={"produce_weights": False})
