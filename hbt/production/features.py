# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from hbt.production.weights import event_weights

ak = maybe_import("awkward")


@producer
def jet_energy_shifts(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Pseudo-producer that registers jet energy shifts.
    """
    return events


@jet_energy_shifts.init
def jet_energy_shifts_init(self: Producer) -> None:
    """
    Register shifts.
    """
    self.shifts |= {
        f"jec_{junc_name}_{junc_dir}"
        for junc_name in self.config_inst.x.jec.uncertainty_sources
        for junc_dir in ("up", "down")
    } | {"jer_up", "jer_down"}


@producer(
    uses={
        event_weights,
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta", "Jet.pt", "Jet.eta",
        "Jet.btagDeepFlavB",
    },
    produces={
        event_weights,
        "ht", "n_jet", "n_electron", "n_muon", "n_deepjet",
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=1))
    events = set_ak_column(events, "n_deepjet", ak.num(events.Jet.pt[events.Jet.btagDeepFlavB > 0.3], axis=1))

    # add event weights
    events = self[event_weights](events, **kwargs)

    return events


@producer(
    uses={"Jet.pt"},
    produces={"cutflow.n_jet", "cutflow.ht", "cutflow.jet1_pt"},
)
def cutflow_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))

    return events
