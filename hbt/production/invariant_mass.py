# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from columnflow.production.util import attach_coffea_behavior


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# Invariant Mass Producers
@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mjj",
    },
)
def invariant_mass_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    # invariant mass of two hardest jets
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "mjj", (events.Jet[:, 0] + events.Jet[:, 1]).mass)
    events = set_ak_column(events, "mjj", ak.fill_none(events.mjj, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mtautau",
    },
)
def invariant_mass_tau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of tau 1 and 2
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    ditau = events.Tau[:, :2].sum(axis=1)
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mtautau",
        ak.where(ditau_mask, ditau.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mbjetbjet",
    },
)
def invariant_mass_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    diBJet = events.BJet[:, :2].sum(axis=1)
    diBJet_mask = ak.num(events.BJet, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mbjetbjet",
        ak.where(diBJet_mask, diBJet.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mHH",
    },
)
def invariant_mass_HH(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}, "Tau": {"type_name": "Tau"}}, **kwargs)
    diHH = events.BJet[:, :2].sum(axis=1) + events.Tau[:, :2].sum(axis=1)
    dibjet_mask = ak.num(events.BJet, axis=1) >= 2
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    diHH_mask = np.logical_and(dibjet_mask, ditau_mask)
    events = set_ak_column_f32(
        events,
        "mHH",
        ak.where(diHH_mask, diHH.mass, EMPTY_FLOAT),
    )
    return events


# Producers for the columns of the kinetmatic variables (four vectors) of the jets, bjets and taus
@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.E", "Jet.area",
        "Jet.nConstituents", "Jet.jetID",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["pt", "eta", "phi", "mass"]
    },
)
def kinematic_vars_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Jet 1 and 2 kinematic variables
    # Jet 1
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column_f32(events, "jet1_mass", Route("Jet.mass[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_eta", Route("Jet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_phi", Route("Jet.phi[:,0]").apply(events, EMPTY_FLOAT))

    # Jet 2
    events = set_ak_column_f32(events, "jet2_mass", Route("Jet.mass[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_pt", Route("Jet.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_eta", Route("Jet.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_phi", Route("Jet.phi[:,1]").apply(events, EMPTY_FLOAT))
    # get energy seperately
    jets_e = ak.pad_none(events.Jet.E, 2, axis=1)
    events = set_ak_column_f32(events, "jet1_e", ak.fill_none(jets_e[:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_e", ak.fill_none(jets_e[:, 1], EMPTY_FLOAT))
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass", "BJet.E",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # BJet 1 and 2 kinematic variables
    # BJet 1
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_mass", Route("BJet.mass[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_pt", Route("BJet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_eta", Route("BJet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_phi", Route("BJet.phi[:,0]").apply(events, EMPTY_FLOAT))

    # BJet 2
    events = set_ak_column_f32(events, "bjet2_mass", Route("BJet.mass[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_pt", Route("BJet.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_eta", Route("BJet.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_phi", Route("BJet.phi[:,1]").apply(events, EMPTY_FLOAT))
    # get energy seperately
    bjets_e = ak.pad_none(events.BJet.E, 2, axis=1)
    events = set_ak_column_f32(events, "bjet1_e", ak.fill_none(bjets_e[:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_e", ak.fill_none(bjets_e[:, 1], EMPTY_FLOAT))
    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.E",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["tau1", "tau2"]
        for var in ["pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_taus(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Tau 1 and 2 kinematic variables
    # Tau 1
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column_f32(events, "tau1_mass", Route("Tau.mass[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_pt", Route("Tau.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_eta", Route("Tau.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_phi", Route("Tau.phi[:,0]").apply(events, EMPTY_FLOAT))

    # Tau 2
    events = set_ak_column_f32(events, "tau2_mass", Route("Tau.mass[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_pt", Route("Tau.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_eta", Route("Tau.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_phi", Route("Tau.phi[:,1]").apply(events, EMPTY_FLOAT))
    # get energy seperately
    taus_e = ak.pad_none(events.Tau.E, 2, axis=1)
    from IPython import embed; embed()
    events = set_ak_column_f32(events, "tau1_e", ak.fill_none(taus_e[:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_e", ak.fill_none(taus_e[:, 1], EMPTY_FLOAT))
    return events


# Producers for aditinal jet exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "Jet.hadronFlavour", "Jet.btagDeepFlavB", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["hadronFlavour", "btag", "DeepTau_e", "btag_dummy", "jet_oneHot", "bjet_oneHot",
        "tau_oneHot"]]
    },
)
def jet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce jet exclusive features
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column_f32(events, "jet1_hadronFlavour", Route("Jet.hadronFlavour[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_hadronFlavour", Route("Jet.hadronFlavour[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_btag", Route("Jet.btagDeepFlavB[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_btag", Route("Jet.btagDeepFlavB[:,1]").apply(events, EMPTY_FLOAT))
    # Produce dummy fills for non jet features
    padded_mass = ak.pad_none(events.Jet.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "jet1_DeepTau_e", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_DeepTau_e", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_btag_dummy", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_btag_dummy", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "jet1_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_jet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_jet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    return events


# Producers for aditinal bjet exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "BJet.btagDeepFlavB", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["btag", "DeepTau_e", "jet_oneHot", "bjet_oneHot", "tau_oneHot"]]
    },
)
def bjet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce bjet exclusive features
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_btag", Route("BJet.btagDeepFlavB[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_btag", Route("BJet.btagDeepFlavB[:,1]").apply(events, EMPTY_FLOAT))

    # Produce dummy fills for non bjet features
    padded_mass = ak.pad_none(events.BJet.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "bjet1_DeepTau_e", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_DeepTau_e", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "bjet1_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_bjet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_bjet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    return events


# Producers for aditinal tau exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "Tau.idDeepTau2017v2p1VSe", "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet",
        "Tau.decayMode", "Tau.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["tau1", "tau2"]
        for var in ["DeepTau_e", "DeepTau_mu", "DeepTau_jet", "decayMode", "btag", "jet_oneHot",
        "bjet_oneHot", "tau_oneHot"]]
    },
)
def tau_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce tau exclusive features
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column_f32(events, "tau1_DeepTau_e", Route("Tau.idDeepTau2017v2p1VSe[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_e", Route("Tau.idDeepTau2017v2p1VSe[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_DeepTau_mu", Route("Tau.idDeepTau2017v2p1VSmu[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_mu", Route("Tau.idDeepTau2017v2p1VSmu[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_DeepTau_jet", Route("Tau.idDeepTau2017v2p1VSjet[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_jet", Route("Tau.idDeepTau2017v2p1VSjet[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_decayMode", Route("Tau.idDeepTau2017v2p1VSjet[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_decayMode", Route("Tau.idDeepTau2017v2p1VSjet[:,1]").apply(events, EMPTY_FLOAT))

    # Produce dummy fills for non tau features
    padded_mass = ak.pad_none(events.Tau.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "tau1_btag", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_btag", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "tau1_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_tau_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_tau_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    return events
