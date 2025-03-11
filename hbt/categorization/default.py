# coding: utf-8

"""
Exemplary selection methods.
"""

from typing import Optional

from columnflow.categorization import Categorizer, categorizer
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
    # non-isolated tau2
    return events, events.tau2_isolated == 0


# positive and negative SS regions
@categorizer(uses={"leptons_os", "Tau.charge"})
def cat_ss_pos(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # same sign leptons (both positive)
    ss_pos_mask = (events.leptons_os == 0 & events.Tau.charge > 0)

    return events, ss_pos_mask == 1


@categorizer(uses={"leptons_os", "Tau.charge"})
def cat_ss_neg(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # same sign leptons (both negative)
    ss_neg_mask = (events.leptons_os == 0 & events.Tau.charge < 0)

    return events, ss_neg_mask == 1


# alternative isolation categorization

# helper function
def get_iso_mask(
    self: Categorizer,
    events: ak.Array,
    passes_wp: str,
    fails_wp: Optional[str] = None,
) -> ak.Array:

    # get tagger
    tagger_name = self.config_inst.x.tau_tagger

    # define lowest working point
    wp_pass = getattr(self.config_inst.x.tau_id_working_points.tau_vs_jet, passes_wp)

    # define isolation mask
    iso_mask = (
        events.sel_tau_mask == 1 &
        # passes wp_down
        (events.Tau[tagger_name] >= wp_pass)
    )

    # define upper working point, if given
    if fails_wp is not None:
        wp_fail = getattr(self.config_inst.x.tau_id_working_points.tau_vs_jet, fails_wp)
        iso_mask = iso_mask & (events.Tau[tagger_name] < wp_fail)

    # channel dependent mask
    is_iso_etau = ak.sum(iso_mask, axis=1) >= 1  # same for mutau
    is_iso_tautau = ak.sum(iso_mask, axis=1) >= 2

    # define channels
    is_etau = events.channel_id == self.config_inst.channels.n.etau.id
    is_mutau = events.channel_id == self.config_inst.channels.n.mutau.id
    is_tautau = events.channel_id == self.config_inst.channels.n.tautau.id

    # get tau2 loose
    tau2_mask = (abs(events.event) < 0)
    tau2_mask = (
        ak.where(is_etau, is_iso_etau, tau2_mask) |
        ak.where(is_mutau, is_iso_etau, tau2_mask) |
        ak.where(is_tautau, is_iso_tautau, tau2_mask)
    )

    return tau2_mask


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_iso_l(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get tau2 passing Loose
    tau2_mask = get_iso_mask(self, events, "loose")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_iso_vl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get tau2 passing VLoose
    tau2_mask = get_iso_mask(self, events, "vloose")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_iso_vvl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get tau2 passing VVLoose
    tau2_mask = get_iso_mask(self, events, "vvloose")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_l_m(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get isolated tau2 (passing Loose but failing Medium)
    tau2_mask = get_iso_mask(self, events, "loose", "medium")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_vl_l(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get isolated tau2 (passing VLoose but failing Loose)
    tau2_mask = get_iso_mask(self, events, "vloose", "loose")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_vvl_vl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get isolated tau2 (passing VVLoose but failing VLoose)
    tau2_mask = get_iso_mask(self, events, "vvloose", "vloose")

    return events, tau2_mask == 1


@categorizer(uses={"Tau.*", "sel_tau_mask", "channel_id"})
def cat_vvvl_vvl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:

    # get isolated tau2 (passing VVVLoose but failing VVLoose)
    tau2_mask = get_iso_mask(self, events, "vvvloose", "vvloose")

    return events, tau2_mask == 1


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


@categorizer(uses={"Jet.btagPNetB"})
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exactly pnet b-tags
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    return events, ak.sum(tagged, axis=1) == 1


@categorizer(uses={"Jet.btagPNetB"})
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least two medium pnet b-tags
    wp = self.config_inst.x.btag_working_points["particleNet"]["medium"]
    tagged = events.Jet.btagPNetB > wp
    return events, ak.sum(tagged, axis=1) >= 2


@categorizer(uses={cat_res1b, cat_res2b, "FatJet.{pt,phi}"})
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # not res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    # note: this is just a draft
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1]
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
