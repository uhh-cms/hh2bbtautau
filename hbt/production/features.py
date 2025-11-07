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
from columnflow.production.cms.gen_particles import transform_gen_part
from columnflow.production.util import lv_mass
from columnflow.columnar_util import (
    EMPTY_FLOAT, Route, set_ak_column, attach_coffea_behavior, default_coffea_collections,
)
from columnflow.util import maybe_import

from hbt.util import IF_MC, IF_DATASET_HAS_LHE_WEIGHTS, IF_DATASET_IS_TT, IF_DATASET_IS_DY, IF_DATASET_HAS_HIGGS

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@producer(
    uses={
        "Electron.pt", "Muon.pt", "Jet.pt", "HHBJet.pt",
    },
    produces={
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
        "Jet.{pt,eta,phi}", "Electron.pt",
    },
    produces={
        mc_weight, category_ids,
        "cutflow.n_jet", "cutflow.n_jet_selected", "cutflow.ht", "cutflow.jet1_pt", "cutflow.jet1_eta",
        "cutflow.jet1_phi", "cutflow.jet2_pt", "cutflow.n_ele", "cutflow.n_ele_selected",
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
        "{Electron,Muon,HHBJet}.{pt,eta,phi,mass}", "{Jet,HHBJet}.{pt,eta,phi,mass,btagPNetB}", "PuppiMET.{pt,phi}",
        IF_DATASET_IS_DY("gen_dilepton_{pt,pdgid}"),
    },
    produces={
        "keep_in_union", "n_jet", "n_btag_pnet", "n_btag_pnet_hhb",
        "{ll,bb,llbb}_{pt,eta,phi,mass}", "{jet,lep}1_{pt,eta,phi}", "met_{pt,phi}",
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
    events = set_ak_column(events, "met_pt", events.PuppiMET.pt)
    events = set_ak_column(events, "met_phi", events.PuppiMET.phi)

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
    reqs["default_prod"] = ProduceColumns.req_other_producer(task, producer="default")


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


@producer(
    uses={"gen_higgs.*"},
    produces={"nu_truth.nu.{pt,eta,phi,mass,pdgId}", "nu_truth.tau_vis.{pt,eta,phi,mass}"},
)
def nu_truth_htt(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # for single higgs -> tautau datasets, there is just one higgs decay in gen_higgs
    # for hh -> bb tautau datasets, there are two decay, but the tautau one is always last
    # so in any case, the higgs index is -1
    HTT = -1

    # get tau neutrino (always first in the list of tau children)
    nu_tau = events.gen_higgs.tau_children[:, HTT, :, 0]
    # check which tau decayed leptonically (charged lepton, if any, is always first)
    first_tau_w_child_id = abs(events.gen_higgs.tau_w_children[:, HTT, :, 0].pdgId)
    tau_lep_mask = (first_tau_w_child_id == 11) | (first_tau_w_child_id == 13) | (first_tau_w_child_id == 15)
    # get the neutrino on these cases
    nu_lep = events.gen_higgs.tau_w_children[:, HTT, :][ak.mask(tau_lep_mask, tau_lep_mask)][:, :, 1]
    # concatenate them to get one _or_ two neutrinos per tau decay
    nu = ak.drop_none(ak.concatenate([nu_tau[:, :, None], nu_lep[:, :, None]], axis=2))

    # also define the visible tau component from all non-neutrino w children
    w_children = events.gen_higgs.tau_w_children[:, HTT]
    w_children_id = abs(w_children.pdgId)
    w_nu_mask = (w_children_id == 12) | (w_children_id == 14) | (w_children_id == 16)
    tau_vis = lv_mass(w_children[~w_nu_mask]).sum(axis=-1)

    # combine to final structure
    nu_truth = ak.zip(
        {
            "nu": transform_gen_part(nu, depth_limit=3),
            "tau_vis": transform_gen_part(tau_vis, depth_limit=2, optional=True),
        },
        depth_limit=1,
    )

    # save the column
    events = set_ak_column(events, "nu_truth", nu_truth)

    return events


@producer(
    uses={"gen_top.*"},
    produces={"nu_truth.nu.{pt,eta,phi,mass}"},
)
def nu_truth_ttbar(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    raise NotImplementedError("nu_truth_ttbar is not implemented yet")
    return events


@producer(
    uses={"gen_dy.*"},
    produces={"nu_truth.{nu,tau_vis}.{pt,eta,phi,mass}"},
)
def nu_truth_dy(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    raise NotImplementedError("nu_truth_dy is not implemented yet")
    return events


@producer(
    uses={
        IF_DATASET_HAS_HIGGS(nu_truth_htt),
        # IF_DATASET_IS_TT(nu_truth_ttbar),
        # IF_DATASET_IS_DY(nu_truth_dy),
    },
    produces={
        IF_DATASET_HAS_HIGGS(nu_truth_htt),
        # IF_DATASET_IS_TT(nu_truth_ttbar),
        # IF_DATASET_IS_DY(nu_truth_dy),
    },
)
def nu_truth(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # defer to dataset/content-specific producers
    if self.has_dep(nu_truth_htt):
        events = self[nu_truth_htt](events, **kwargs)
    if self.has_dep(nu_truth_ttbar):
        events = self[nu_truth_ttbar](events, **kwargs)
    if self.has_dep(nu_truth_dy):
        events = self[nu_truth_dy](events, **kwargs)
    return events
