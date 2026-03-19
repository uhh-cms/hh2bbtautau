# coding: utf-8

"""
Exemplary selection methods.
"""

from __future__ import annotations

import operator

from columnflow.categorization import Categorizer, categorizer
from columnflow.columnar_util import attach_coffea_behavior, full_like, ak_concatenate_safe
from columnflow.util import maybe_import

from hbt.production.jet import jet_multiplicity, bjet_multiplicity
from hbt.util import MET_COLUMN, stack_lvectors, create_lvector_xyz, rotate_px_py

ak = maybe_import("awkward")


# helpers
all_true = lambda events: full_like(events.event, True, dtype=bool)

hhbjet_multiplicity = bjet_multiplicity.derive("hhbjet_multiplicity", cls_dict={"jet_name": "HHBJet"})


#
# dummy selector
#

@categorizer(uses={"event"})
def cat_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # keep all events
    return events, all_true(events)


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
# additional special regions
#

@categorizer(uses={"single_triggered"})
def cat_single_triggered(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.single_triggered == 1


#
# kinematic regions
#

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, all_true(events)


@categorizer(uses={jet_multiplicity})
def cat_ge0j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 0


@categorizer(uses={jet_multiplicity})
def cat_eq0j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 0


@categorizer(uses={jet_multiplicity})
def cat_eq1j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 1


@categorizer(uses={jet_multiplicity})
def cat_ge2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 2


@categorizer(uses={jet_multiplicity})
def cat_eq2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 2


@categorizer(uses={jet_multiplicity})
def cat_eq3j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 3


@categorizer(uses={jet_multiplicity})
def cat_eq4j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 4


@categorizer(uses={jet_multiplicity})
def cat_eq5j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 5


@categorizer(uses={jet_multiplicity})
def cat_ge4j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 4


@categorizer(uses={jet_multiplicity})
def cat_ge5j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 5


@categorizer(uses={jet_multiplicity})
def cat_ge6j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 6


@categorizer(uses={bjet_multiplicity})
def cat_ge0b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 0


@categorizer(uses={bjet_multiplicity})
def cat_eq0b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 0


@categorizer(uses={bjet_multiplicity})
def cat_eq1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 1


@categorizer(uses={bjet_multiplicity})
def cat_eq2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 2


@categorizer(uses={bjet_multiplicity})
def cat_eq3b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 3


@categorizer(uses={bjet_multiplicity})
def cat_ge1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 1


@categorizer(uses={bjet_multiplicity})
def cat_ge2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 2


@categorizer(uses={"HHBJet.{mass,pt,eta,phi}"})
def di_bjet_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    di_bjet_mass = events.HHBJet.sum(axis=1).mass
    mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )
    return events, mask


@categorizer(uses={"Tau.{mass,pt,eta,phi}"})
def di_tau_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leptons = [events.Electron * 1, events.Muon * 1, events.Tau * 1]
    di_tau_mass = ak_concatenate_safe(leptons, axis=1)[:, :2].sum(axis=1).mass
    mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )
    return events, mask


@categorizer(
    uses={di_bjet_mass_window, di_tau_mass_window, hhbjet_multiplicity},
)
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (self[hhbjet_multiplicity](events, **kwargs) == 1) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(
    uses={di_bjet_mass_window, di_tau_mass_window, hhbjet_multiplicity},
)
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (self[hhbjet_multiplicity](events, **kwargs) >= 2) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(
    uses={
        cat_res1b, cat_res2b, di_tau_mass_window,
        "FatJet.{pt,phi,msoftdrop,particleNet_XbbVsQCD,mass,eta}",
    },
)
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exclude res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    mask = (
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1] &
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        (ak.sum(events.FatJet.particleNet_XbbVsQCD > 0.75, axis=1) >= 1) &  # wp defined by cclub
        self[di_tau_mass_window](events, **kwargs)[1] &
        ak.any(events.FatJet.msoftdrop >= 30, axis=1) &
        ak.any(events.FatJet.msoftdrop <= 450, axis=1)
    )
    return events, mask


@categorizer(uses={"vbf_dnn_moe_hh_vbf"})
def cat_vbf_0p5(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.vbf_dnn_moe_hh_vbf > 0.5)


@categorizer(uses={cat_res1b, cat_vbf_0p5})
def cat_res1b_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, res1b_mask = self[cat_res1b](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (res1b_mask & ~vbf_mask)


@categorizer(uses={cat_res2b, cat_vbf_0p5})
def cat_res2b_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, res2b_mask = self[cat_res2b](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (res2b_mask & ~vbf_mask)


@categorizer(uses={cat_boosted, cat_vbf_0p5})
def cat_boosted_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, boosted_mask = self[cat_boosted](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (boosted_mask & ~vbf_mask)


@categorizer(
    uses={"FatJet.{pt,phi,msoftdrop,particleNet_XbbVsQCD,mass,eta}"},
)
def cat_boosted_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        ak.any(events.FatJet.msoftdrop >= 80, axis=1) &
        ak.any(events.FatJet.msoftdrop <= 170, axis=1) &
        (ak.sum(events.FatJet.pt > 300, axis=1) == 1) &
        (ak.sum(events.FatJet.particleNet_XbbVsQCD > 0.75, axis=1) >= 1)  # wp defined by cclub
    )
    return events, mask


@categorizer(uses={cat_boosted_cc, "vbf_dnn_moe_hh_vbf"})
def cat_vbf_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        ~self[cat_boosted_cc](events, **kwargs)[1] &
        (events.vbf_dnn_moe_hh_vbf > 0.5)
    )
    return events, mask


@categorizer(
    uses={"{Electron,Muon,Tau,HHBJet}.{mass,pt,eta,phi}", "reg_dnn_moe_nu{1,2}_p{x,y,z}", "channel_id"},
    # channel dependent mean and std values for the regressed tautau system mass
    tautau_window={
        # regressed values
        "etau": (111.0, 38.0),
        "mutau": (110.0, 45.0),
        "tautau": (130.0, 36.0),
        # values actually used for svfit based tautau system
        "ee": (119.0, 57.0),
        "mumu": (116.0, 61.0),
        "emu": (116.0, 61.0),  # values from mumu
    },
    # channel dependent mean and std values for the bb (from HHBJet) system mass
    bb_window={
        "etau": (118.0, 225.0),
        "mutau": (118.0, 215.0),
        "tautau": (125.0, 222.0),
        "ee": (109.0, 232.0),
        "mumu": (114.0, 228.0),
        "emu": (114.0, 228.0),  # values from mumu
    },
)
def cat_hh_reg_mass_window_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # build bb system
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    bb = events.HHBJet[:, :2].sum(axis=1)
    # build visible tautau system
    tautau_vis = stack_lvectors([events.Electron, events.Muon, events.Tau])[..., :2].sum(axis=1)
    # get regressed neutrinos and rotate them
    ref_phi = tautau_vis.phi
    nu1 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu1_px, events.reg_dnn_moe_nu1_py, ref_phi),
        events.reg_dnn_moe_nu1_pz,
    )
    nu2 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu2_px, events.reg_dnn_moe_nu2_py, ref_phi),
        events.reg_dnn_moe_nu2_pz,
    )
    # build regressed tautau system
    tautau_reg = stack_lvectors([nu1, nu2, tautau_vis]).sum(axis=-1)
    # get masses
    m_bb = bb.mass
    m_tautau = tautau_reg.mass

    # apply channel dependent mass window cuts
    def mass_window(channel_name: str) -> ak.Array:
        channel_mask = events.channel_id == self.config_inst.get_channel(channel_name).id
        chi_tautau = (m_tautau - self.tautau_window[channel_name][0]) / self.tautau_window[channel_name][1]
        chi_bb = (m_bb - self.bb_window[channel_name][0]) / self.bb_window[channel_name][1]
        return channel_mask & ((chi_tautau**2 + chi_bb**2) <= 1.0)

    mass_mask = (
        full_like(events.event, False, dtype=bool) |
        mass_window("etau") |
        mass_window("mutau") |
        mass_window("tautau") |
        mass_window("ee") |
        mass_window("mumu") |
        mass_window("emu")
    )
    return events, mass_mask


@categorizer(
    uses={cat_boosted_cc, cat_vbf_cc, hhbjet_multiplicity, cat_hh_reg_mass_window_cc},
    n_btags=None,  # needs to be set
    n_btags_op=None,  # needs to be set
    reject_boosted=True,
    reject_vbf=True,
)
def _cat_res_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        self[cat_hh_reg_mass_window_cc](events, **kwargs)[1] &
        (~self[cat_boosted_cc](events, **kwargs)[1] if self.reject_boosted else True) &
        (~self[cat_vbf_cc](events, **kwargs)[1] if self.reject_vbf else True) &
        (self.n_btags_op(self[hhbjet_multiplicity](events, **kwargs), self.n_btags))
    )
    return events, mask


cat_res1b_cc = _cat_res_cc.derive("cat_res1b_cc", cls_dict={"n_btags": 1, "n_btags_op": operator.eq})
cat_res2b_cc = _cat_res_cc.derive("cat_res2b_cc", cls_dict={"n_btags": 2, "n_btags_op": operator.ge})
cat_res1b_inclvbf_cc = cat_res1b_cc.derive("cat_res1b_inclvbf_cc", cls_dict={"reject_vbf": False})
cat_res2b_inclvbf_cc = cat_res2b_cc.derive("cat_res2b_inclvbf_cc", cls_dict={"reject_vbf": False})


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_mll40(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    return events, leps.sum(axis=1).mass > 40.0


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_dy(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # e/mu driven DY region: mll > 40 and met < 30 (to supress tau decays into e/mu)
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    mask = (
        (leps.sum(axis=1).mass > 40) &
        (events[self.config_inst.x.met_name].pt < 30)
    )
    return events, mask


@cat_dy.init
def cat_dy_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@categorizer(
    uses={
        "{Electron,Muon,Tau}.{pt,eta,phi,mass}",
        MET_COLUMN("{pt,phi}"),
    },
)
def cat_dyc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]

    mask_cclub = (
        (leps.sum(axis=1).mass >= 70) &
        (leps.sum(axis=1).mass <= 110) &
        (events[self.config_inst.x.met_name].pt < 45)
    )

    return events, mask_cclub


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_tt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tt region: met > 30 (due to neutrino presence in leptonic w decays)
    mask = events[self.config_inst.x.met_name].pt > 30
    return events, mask


@cat_tt.init
def cat_tt_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")
