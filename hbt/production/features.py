# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools
import operator

import law

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.columnar_util import (
    EMPTY_FLOAT, Route, set_ak_column, attach_coffea_behavior, default_coffea_collections,
)
from columnflow.util import maybe_import

from hbt.util import IF_MC, IF_DATASET_HAS_LHE_WEIGHTS, IF_DATASET_IS_TT, IF_DATASET_IS_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@producer(
    uses={
        # nano columns
        "Electron.pt", "Muon.pt", "Jet.pt", "HHBJet.pt",
    },
    produces={
        # new columns
        "n_electron", "ht", "n_jet", "n_hhbtag", "n_electron", "n_muon",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_i32(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column_i32(events, "n_hhbtag", ak.num(events.HHBJet.pt, axis=1))
    events = set_ak_column_i32(events, "n_electron", ak.num(events.Electron.pt, axis=1))
    events = set_ak_column_i32(events, "n_muon", ak.num(events.Muon.pt, axis=1))
    return events


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi", "Electron.pt",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.n_jet", "cutflow.n_jet_selected", "cutflow.ht", "cutflow.jet1_pt",
        "cutflow.jet1_eta", "cutflow.jet1_phi", "cutflow.jet2_pt", "cutflow.n_ele",
        "cutflow.n_ele_selected",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    # columns required for cutflow plots
    events = self[category_ids](events, **kwargs)
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply per-object selections
    selected_jet = events.Jet[object_masks["Jet"]["Jet"]]
    selected_ele = events.Electron[object_masks["Electron"]["Electron"]]

    # add feature columns
    events = set_ak_column_i32(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column_i32(events, "cutflow.n_jet_selected", ak.num(selected_jet, axis=1))
    events = set_ak_column_f32(events, "cutflow.ht", ak.sum(selected_jet.pt, axis=1))
    events = set_ak_column_f32(events, "cutflow.jet1_pt", Route("pt[:,0]").apply(selected_jet, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "cutflow.jet1_eta", Route("eta[:,0]").apply(selected_jet, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "cutflow.jet1_phi", Route("phi[:,0]").apply(selected_jet, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "cutflow.jet2_pt", Route("pt[:,1]").apply(selected_jet, EMPTY_FLOAT))
    events = set_ak_column_i32(events, "cutflow.n_ele", ak.num(events.Electron, axis=1))
    events = set_ak_column_i32(events, "cutflow.n_ele_selected", ak.num(selected_ele, axis=1))

    return events


@producer(
    uses={
        "channel_id", "leptons_os",
        "{Electron,Muon,HHBJet}.{pt,eta,phi,mass}", "{Jet,HHBJet}.{pt,eta,phi,mass,btagPNetB}",
        IF_DATASET_IS_DY("gen_dilepton_{pt,pdgid}"),
    },
    produces={
        "keep_in_union", "n_jet", "n_btag_pnet", "n_btag_pnet_hhb",
        "{ll,bb,llbb}_{pt,eta,phi,mass}", "{jet,lep}1_{pt,eta,phi}",
        IF_MC("event_weight"),
        IF_DATASET_IS_DY("gen_ll_{pt,pdgid}"),
    },
)
def dy_dnn_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = attach_coffea_behavior(events, {"HHBJet": default_coffea_collections["Jet"]})

    # only keep ee and mumu events in the analysis region
    # (not need to check tau2 isolation in these channels)
    keep_in_union = (
        (
            (events.channel_id == self.config_inst.channels.n.ee.id) |
            (events.channel_id == self.config_inst.channels.n.mumu.id)
        ) &
        (events.leptons_os == 1)
    )
    events = set_ak_column(events, "keep_in_union", keep_in_union, value_type=bool)

    # construct the overall event weight
    if self.dataset_inst.is_mc:
        weight_vals = [events[weight_name] for weight_name in self.weight_names]
        event_weight = functools.reduce(operator.mul, weight_vals)
        events = set_ak_column_f32(events, "event_weight", event_weight)

    # rename some existing dy columns
    if IF_DATASET_IS_DY(True)(self):
        events = set_ak_column(events, "gen_ll_pt", events.gen_dilepton_pt)
        events = set_ak_column(events, "gen_ll_pdgid", events.gen_dilepton_pdgid)

    # number of jets
    events = set_ak_column_i32(events, "n_jet", ak.num(events.Jet, axis=1))

    # number of btags among all jets
    nb_all = ak.sum(events.Jet.btagPNetB > self.config_inst.x.btag_working_points.particleNet.medium, axis=1)
    events = set_ak_column_i32(events, "n_btag_pnet", nb_all)

    # number of btags among hhbtag jets
    nb_hbb = ak.sum(events.HHBJet.btagPNetB > self.config_inst.x.btag_working_points.particleNet.medium, axis=1)
    events = set_ak_column_i32(events, "n_btag_pnet_hhb", nb_hbb)

    # dilepton system variables
    dilep = ak.concatenate([events.Electron * 1, events.Muon * 1], axis=1)[:, :2].sum(axis=1)
    events = set_ak_column_f32(events, "ll_pt", dilep.pt)
    events = set_ak_column_f32(events, "ll_eta", dilep.eta)
    events = set_ak_column_f32(events, "ll_phi", dilep.phi)
    events = set_ak_column_f32(events, "ll_mass", dilep.mass)

    # di-bjet system variables
    dibjet = events.HHBJet.sum(axis=1)
    events = set_ak_column_f32(events, "bb_pt", dibjet.pt)
    events = set_ak_column_f32(events, "bb_eta", dibjet.eta)
    events = set_ak_column_f32(events, "bb_phi", dibjet.phi)
    events = set_ak_column_f32(events, "bb_mass", dibjet.mass)

    # ll+bb system
    llbb = ak.concatenate([dilep[:, None] * 1, dibjet[:, None] * 1], axis=1).sum(axis=1)
    events = set_ak_column_f32(events, "llbb_pt", llbb.pt)
    events = set_ak_column_f32(events, "llbb_eta", llbb.eta)
    events = set_ak_column_f32(events, "llbb_phi", llbb.phi)
    events = set_ak_column_f32(events, "llbb_mass", llbb.mass)

    # leading lepton
    lep = ak.concatenate([events.Electron * 1, events.Muon * 1], axis=1)[:, :1].sum(axis=1)
    events = set_ak_column_f32(events, "lep1_pt", lep.pt)
    events = set_ak_column_f32(events, "lep1_eta", lep.eta)
    events = set_ak_column_f32(events, "lep1_phi", lep.phi)

    # leading jet
    jet1 = (events.Jet[:, :1] * 1).sum(axis=1)
    events = set_ak_column_f32(events, "jet1_pt", jet1.pt)
    events = set_ak_column_f32(events, "jet1_eta", jet1.eta)
    events = set_ak_column_f32(events, "jet1_phi", jet1.phi)

    return events


@dy_dnn_features.init
def dy_dnn_features_init(self: Producer, **kwargs) -> None:
    # define mc weights that are to be multiplied
    if self.dataset_inst.is_mc:
        self.weight_names = [
            "normalization_weight",
            "normalized_pu_weight",
            "normalized_isr_weight",
            "normalized_fsr_weight",
            "normalized_njet_btag_weight_pnet",
            "electron_weight",
            "muon_weight",
            "tau_weight",
            "trigger_weight",
        ]

        if IF_DATASET_HAS_LHE_WEIGHTS(True)(self):
            self.weight_names.append("normalized_pdf_weight")
            self.weight_names.append("normalized_murmuf_weight")
        if IF_DATASET_IS_TT(True)(self):
            self.weight_names.append("top_pt_weight")
        self.uses.update(self.weight_names)


@dy_dnn_features.requires
def dy_dnn_features_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.production import ProduceColumns
    reqs["default_prod"] = ProduceColumns.req(task, producer="default", producer_inst=None, known_shifts=None)


@dy_dnn_features.setup
def dy_dnn_features_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    reader_targets["default_prod"] = inputs["default_prod"]["columns"]
