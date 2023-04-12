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


@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.e",
        attach_coffea_behavior,
    },
    produces={
        "mjj",
        # "HardestJetPair.mass",
    },
)
def invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    # njet_mask = ak.count(events.Jet.pt, axis=1) >= 2
    # hardest_jet_pair_mass = sum(events.Jet[:, :2]).mass  # sum of _up to_ two jets
    # hardest_jet_pair_mass = ak.where(ak.num(events.Jet, axis=1) < 2, EMPTY_FLOAT, hardest_jet_pair_mass)
    # hardest_jet_pair_mass = ak.where(
    #     ak.count(events.Jet.pt, axis=1) >= 2,
    #     (events.Jet[:, 0].add(events.Jet[:, 1])).mass,
    #     EMPTY_FLOAT
    # )
    # from IPython import embed; embed()
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))

    events = set_ak_column(events, "mjj", (events.Jet[:, 0] + events.Jet[:, 1]).mass)
    events = set_ak_column(events, "mjj", ak.fill_none(events.mjj, EMPTY_FLOAT))
    # events = set_ak_column_f32(events, "HardestJetPair.mass", hardest_jet_pair_mass)
    return events
