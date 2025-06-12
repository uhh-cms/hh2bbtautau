# coding: utf-8

"""
Exemplary selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.columnar_util import attach_coffea_behavior, default_coffea_collections
from columnflow.util import maybe_import

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


@categorizer(uses={"Jet.{pt,phi}"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2


@categorizer(uses={"HHBJet.{mass,pt,eta,phi}"})
def di_bjet_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events = attach_coffea_behavior(events, {"HHBJet": default_coffea_collections["Jet"]})
    di_bjet_mass = events.HHBJet.sum(axis=1).mass
    mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )
    return events, mask


@categorizer(uses={"Tau.{mass,pt,eta,phi}"})
def di_tau_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leptons = [events.Electron * 1, events.Muon * 1, events.Tau * 1]
    di_tau_mass = ak.concatenate(leptons, axis=1)[:, :2].sum(axis=1).mass
    mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )
    return events, mask


@categorizer(uses={
    di_bjet_mass_window, di_tau_mass_window, "Jet.{btagPNetB,mass,hhbtag,pt,eta,phi}",
})
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (ak.sum(tagged, axis=1) == 1) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(uses={
    di_bjet_mass_window, di_tau_mass_window, "Jet.{btagPNetB,mass,hhbtag,pt,eta,phi}",
})
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least two medium pnet b-tags
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (ak.sum(tagged, axis=1) == 2) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(uses={
    cat_res1b, cat_res2b, di_tau_mass_window, "FatJet.{pt,phi,msoftdrop,particleNet_XbbVsQCD,mass,eta}",
})
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exclude res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    # TODO: run3 wp are not released, falling back to run2
    wp = self.config_inst.x.btag_working_points["particleNetMD"]["lp"]
    tagged = events.FatJet.particleNet_XbbVsQCD > wp
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        (ak.sum(tagged, axis=1) >= 1) &
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1] &
        tau_mass_mask &
        ak.any(events.FatJet.msoftdrop >= 30, axis=1) &
        ak.any(events.FatJet.msoftdrop <= 450, axis=1)
    )
    return events, mask


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_dy(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # e/mu driven DY region: mll > 40 and met < 30 (to supress tau decays into e/mu)
    leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    mask = (
        (leps.sum(axis=1).mass > 40) &
        (events[self.config_inst.x.met_name].pt < 30)
    )
    return events, mask


@cat_dy.init
def cat_dy_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_tt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tt region: met > 30 (due to neutrino presence in leptonic w decays)
    mask = events[self.config_inst.x.met_name].pt > 30
    return events, mask


@cat_tt.init
def cat_tt_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")
