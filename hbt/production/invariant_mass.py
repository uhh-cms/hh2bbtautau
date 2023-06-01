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
        "Jet.nConstituents", "Jet.jetID", "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        attach_coffea_behavior,
    },
    produces={
        # *[f"{obj}_{var}"
        # for obj in [f"jet{n}" for n in range(1, 7, 1)]
        # for var in ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]], "nJets", "nConstituents", "jets_pt",
        "nJets", "jets_pt", "jets_e", "jets_eta", "jets_phi", "jets_mass", "jets_btag", "jets_hadFlav",
    },
)
def kinematic_vars_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column_f32(events, "nJets", ak.fill_none(events.n_jet, EMPTY_FLOAT))
    # jets_mass = ak.pad_none(events.Jet.mass, 6, axis=1)
    # jets_e = ak.pad_none(events.Jet.E, 6, axis=1)
    # jets_pt = ak.pad_none(events.Jet.pt, 6, axis=1)
    # jets_eta = ak.pad_none(events.Jet.eta, 6, axis=1)
    # jets_phi = ak.pad_none(events.Jet.phi, 6, axis=1)
    # jets_btag = ak.pad_none(events.Jet.btagDeepFlavB, 6, axis=1)
    # jets_hadFlav = ak.pad_none(events.Jet.hadronFlavour, 6, axis=1)
    jets_pt = ak.pad_none(events.Jet.pt, max(events.n_jet))
    jets_pt = ak.to_regular(jets_pt, axis=1)
    jets_pt = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_pt", jets_pt)
    jets_eta = ak.pad_none(events.Jet.eta, max(events.n_jet))
    jets_eta = ak.to_regular(jets_eta, axis=1)
    jets_eta = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_eta", jets_eta)
    jets_phi = ak.pad_none(events.Jet.phi, max(events.n_jet))
    jets_phi = ak.to_regular(jets_phi, axis=1)
    jets_phi = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_phi", jets_phi)
    jets_mass = ak.pad_none(events.Jet.mass, max(events.n_jet))
    jets_mass = ak.to_regular(jets_mass, axis=1)
    jets_mass = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_mass", jets_mass)
    jets_e = ak.pad_none(events.Jet.E, max(events.n_jet))
    jets_e = ak.to_regular(jets_e, axis=1)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_e", jets_e)
    jets_btag = ak.pad_none(events.Jet.btagDeepFlavB, max(events.n_jet))
    jets_btag = ak.to_regular(jets_btag, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_btag", jets_btag)
    jets_hadFlav = ak.pad_none(events.Jet.hadronFlavour, max(events.n_jet))
    jets_hadFlav = ak.to_regular(jets_hadFlav, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_hadFlav", jets_hadFlav)
    # for i in range(0, 6, 1):
    #     events = set_ak_column_f32(events, f"jet{i+1}_mass", ak.fill_none(jets_mass[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_e", ak.fill_none(jets_e[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_pt", ak.fill_none(jets_pt[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_eta", ak.fill_none(jets_eta[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_phi", ak.fill_none(jets_phi[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_btag", ak.fill_none(jets_btag[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_hadronFlavour", ak.fill_none(jets_hadFlav[:, i], EMPTY_FLOAT))

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
        "tau_oneHot", "object_type"]]
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

    # Alternateively use integer to define object type (Jet:1, BJet:2, Tau:3) and apply embendding
    # layer in network
    events = set_ak_column_f32(events, "jet1_object_type", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_object_type", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

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
        for var in ["btag", "DeepTau_e", "jet_oneHot", "bjet_oneHot", "tau_oneHot", "object_type"]]
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

    # Create object type integer
    object_type_1 = ak.full_like(padded_mass[:, 0], 2)
    object_type_2 = ak.full_like(padded_mass[:, 1], 2)
    events = set_ak_column_f32(events, "bjet1_object_type", ak.fill_none(object_type_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_object_type", ak.fill_none(object_type_2, EMPTY_FLOAT))

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
        "bjet_oneHot", "tau_oneHot", "object_type"]]
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

    # Create object type integer
    object_type_1 = ak.full_like(padded_mass[:, 0], 3)
    object_type_2 = ak.full_like(padded_mass[:, 1], 3)
    events = set_ak_column_f32(events, "tau1_object_type", ak.fill_none(object_type_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_object_type", ak.fill_none(object_type_2, EMPTY_FLOAT))

    return events
