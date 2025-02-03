# coding: utf-8

"""
Exemplary selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import
from columnflow.columnar_util import mask_from_indices


ak = maybe_import("awkward")


#
# dummy selector
#

@categorizer(uses={"event"})
def cat_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # keep all events
    return events, ak.ones_like(events.event) == 1


#
# lepton channels
#

@categorizer(uses={"channel_id"})
def cat_etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.etau.id


@categorizer(uses={"channel_id"})
def cat_mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.mutau.id


@categorizer(uses={"channel_id"})
def cat_tautau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.tautau.id


@categorizer(uses={"channel_id"})
def cat_ee(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ee.id


@categorizer(uses={"channel_id"})
def cat_mumu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.mumu.id


@categorizer(uses={"channel_id"})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.emu.id


#
# QCD regions
#

@categorizer(uses={"leptons_os"})
def cat_os(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # oppositive sign leptons
    return events, events.leptons_os == 1


@categorizer(uses={"leptons_os"})
def cat_ss(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # same sign leptons
    return events, events.leptons_os == 0


@categorizer(uses={"tau2_isolated"})
def cat_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # isolated tau2
    return events, events.tau2_isolated == 1


@categorizer(uses={"tau2_isolated"})
def cat_noniso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # noon-isolated tau2
    return events, events.tau2_isolated == 0


#
# kinematic regions
#

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.pt"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2


@categorizer(uses={"Jet.{pt,btagPNetB}"})
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp = "medium"
    btag_scoring_indices = ak.argsort(events.Jet.btagPNetB, axis=1, ascending=False)
    btag_mask = mask_from_indices(btag_scoring_indices[:, :2], events.Jet.btagPNetB)

    atleast_2_passing_btag = ak.sum(
        (events.Jet.btagPNetB > self.config_inst.x.btag_working_points["particleNet"][wp]), axis=1,
    ) >= 2

    mask = (
        btag_mask &
        (events.Jet.pt > 20) &
        atleast_2_passing_btag
    )
    return events, mask


@categorizer(uses={"Jet.*"})
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp = "medium"
    btag_scoring_indices = ak.argsort(events.Jet.btagPNetB, axis=1, ascending=False)
    btag_mask = mask_from_indices(btag_scoring_indices[:, :2], events.Jet.btagPNetB)

    atleast_1_passing_btag = ak.sum(
        (events.Jet.btagPNetB > self.config_inst.x.btag_working_points["particleNet"][wp]), axis=1,
    ) >= 1

    mask = (
        btag_mask &
        (events.Jet.pt > 20) &
        atleast_1_passing_btag
    )
    return events, mask


@categorizer(uses={"FatJet.*"})
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        ak.num(events.FatJet.pt, axis=1) >= 1 &
        events.FatJet.pt > 350 &
        events.FatJet.mass > 30
    )
    return events, mask
