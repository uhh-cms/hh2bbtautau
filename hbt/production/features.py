# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@producer(
    uses={
        # nano columns
        "Electron.pt", "Muon.pt", "Jet.pt", "HHBJet.pt", "BJet.pt",
    },
    produces={
        # new columns
        "ht", "n_jet", "n_hhbtag", "n_electron", "n_muon",
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
        "Jet.pt", "Jet.eta", "Jet.phi",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.n_jet", "cutflow.ht", "cutflow.jet1_pt", "cutflow.jet1_eta", "cutflow.jet1_phi",
    },
)
def cutflow_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    events = set_ak_column_i32(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column_f32(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_f32(events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "cutflow.jet1_eta", Route("Jet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "cutflow.jet1_phi", Route("Jet.phi[:,0]").apply(events, EMPTY_FLOAT))

    return events
