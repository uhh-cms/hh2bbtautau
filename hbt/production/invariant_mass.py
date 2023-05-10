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


# Producers for the columns of the kinetmatic variables of the jets, bjets and taus
@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.E", "Jet.area",
        "Jet.nConstituents", "Jet.jetID",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["pt", "eta", "phi", "mass", "e"]
    },
)
def kinetamic_vars_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Jet 1 and 2 kinematic variables
    # Jet 1
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    from IPython import embed; embed()
    events = set_ak_column_f32(events, "jet1_mass", Route("Jet.mass[:,0]").apply(events, EMPTY_FLOAT))
    # events = set_ak_column_f32(events, "jet1_e", Route("Jet.energy[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_eta", Route("Jet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_phi", Route("Jet.phi[:,0]").apply(events, EMPTY_FLOAT))

    # Jet 2
    events = set_ak_column_f32(events, "jet2_mass", Route("Jet.mass[:,1]").apply(events, EMPTY_FLOAT))
    # events = set_ak_column_f32(events, "jet2_e", Route("Jet.energy[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_pt", Route("Jet.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_eta", Route("Jet.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_phi", Route("Jet.phi[:,1]").apply(events, EMPTY_FLOAT))
    jet1_e = events.Jet.E[:, 0]
    jet1_mask = ak.num(events.Jet, axis=1) >= 1
    jet2_e = events.Jet.E[:, 1]
    jet2_mask = ak.num(events.Jet, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "jet1_e",
        ak.where(jet1_mask, jet1_e, EMPTY_FLOAT),
    )
    events = set_ak_column_f32(
        events,
        "jet2_e",
        ak.where(jet2_mask, jet2_e, EMPTY_FLOAT),
    )
    from IPython import embed; embed()
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
def kinetamic_vars_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # BJet 1 and 2 kinematic variables
    # BJet 1
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_mass", Route("BJet.mass[:,0]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "bjet1_e", Route("BJet.E[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_pt", Route("BJet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_eta", Route("BJet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_phi", Route("BJet.phi[:,0]").apply(events, EMPTY_FLOAT))

    # BJet 2
    events = set_ak_column_f32(events, "bjet2_mass", Route("BJet.mass[:,1]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "bjet2_e", Route("BJet.E[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_pt", Route("BJet.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_eta", Route("BJet.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_phi", Route("BJet.phi[:,1]").apply(events, EMPTY_FLOAT))
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
def kinetamic_vars_taus(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Tau 1 and 2 kinematic variables
    # Tau 1
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column_f32(events, "tau1_mass", Route("Tau.mass[:,0]").apply(events, EMPTY_FLOAT))
  #  events = set_ak_column_f32(events, "tau1_e", Route("Tau.E[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_pt", Route("Tau.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_eta", Route("Tau.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_phi", Route("Tau.phi[:,0]").apply(events, EMPTY_FLOAT))

    # Tau 2
    events = set_ak_column_f32(events, "tau2_mass", Route("Tau.mass[:,1]").apply(events, EMPTY_FLOAT))
  #  events = set_ak_column_f32(events, "tau2_e", Route("Tau.E[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_pt", Route("Tau.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_eta", Route("Tau.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_phi", Route("Tau.phi[:,1]").apply(events, EMPTY_FLOAT))
    return events


# Producers for additional event information on the Jets
@producer(
    uses={
        "Jet.area", "Jet.nConstituents", "nJet", "Jet.hadronFlavour",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["area", "nConstituents", "hadronFlavour"]], "jets_nJets"
    },
)
def jet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    from IPython import embed; embed()
    events = set_ak_column_f32(events, "jet1_area", Route("Jet.area[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_area", Route("Jet.area[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_nConstituents", Route("Jet.nConstituents[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_nConstituents", Route("Jet.nConstituents[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_hadronFlavour", Route("Jet.hadronFlavour[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_hadronFlavour", Route("Jet.hadronFlavour[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jets_nJets", Route("Jet.nJet").apply(events, EMPTY_FLOAT))
    return events


# Producers for additional event information on the BJets
@producer(
    uses={
        "BJet.area", "BJet.nConstituents", "BJet.btagDeepFlavB", "BJet.nJet",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["area", "nConstituents", "btag"]], "bjets_nJets"
    },
)
def bjet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_area", Route("BJet.area[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_area", Route("BJet.area[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_nConstituents", Route("BJet.nConstituents[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_nConstituents", Route("BJet.nConstituents[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_btag", Route("BJet.btagDeepFlavB[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_btag", Route("BJet.btagDeepFlavB[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjets_nJets", Route("BJet.nJet").apply(events, EMPTY_FLOAT))
    return events
