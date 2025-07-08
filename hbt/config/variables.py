# coding: utf-8

"""
Definition of variables.
"""

from __future__ import annotations

from functools import partial

import order as od

from columnflow.columnar_util import EMPTY_FLOAT, attach_coffea_behavior, default_coffea_collections
from columnflow.util import maybe_import

from hbt.util import create_lvector_xyz

ak = maybe_import("awkward")


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    add_variable(
        config,
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    add_variable(
        config,
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    add_variable(
        config,
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    add_variable(
        config,
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    def build_ht(events):
        objects = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1, events.Jet * 1], axis=1)[:, :]
        objects_sum = objects.sum(axis=1)
        return objects_sum.pt
    build_ht.inputs = ["{Electron,Muon,Tau,Jet}.{pt,eta,phi,mass}"]
    add_variable(
        config,
        name="ht",
        expression=partial(build_ht),
        aux={"inputs": build_ht.inputs},
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    add_variable(
        config,
        name="jet_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"all Jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        binning=(30, -3.0, 3.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        config,
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        config,
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        binning=(30, -3.0, 3.0),
        x_title=r"Subleading jet $\eta$",
    )
    add_variable(
        config,
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading jet $\phi$",
    )
    add_variable(
        config,
        name="met_pt",
        expression="PuppiMET.pt",
        binning=(40, 0, 200),
        x_title=r"MET $p_T$",
    )
    add_variable(
        config,
        name="met_phi",
        expression="PuppiMET.phi",
        binning=(66, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    add_variable(
        config,
        name="met_px",
        expression=lambda events: events.PuppiMET.px,
        aux={"inputs": ["PuppiMET.{pt,phi}"]},
        binning=(50, -250, 250),
        x_title=r"MET $p_x$",
    )
    add_variable(
        config,
        name="met_py",
        expression=lambda events: events.PuppiMET.py,
        aux={"inputs": ["PuppiMET.{pt,phi}"]},
        binning=(50, -250, 250),
        x_title=r"MET $p_y$",
    )
    for n in range(1, 2 + 1):
        for v in ["px", "py", "pz"]:
            add_variable(
                config,
                name=f"reg_dnn_nu{n}_{v}",
                binning=(40, -150, 150),
                x_title=rf"Regressed $\nu_{n} {v}$",
            )

    def build_reg_h(events, which=None):
        import numpy as np
        vis_leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
        ref_phi = vis_leps.sum(axis=1).phi
        def rotate_px_py(px, py):
            new_phi = np.arctan2(py, px) + ref_phi  # mind the "+"
            pt = (px**2 + py**2)**0.5
            return pt * np.cos(new_phi), pt * np.sin(new_phi)
        nu1 = create_lvector_xyz(*rotate_px_py(events.reg_dnn_nu1_px, events.reg_dnn_nu1_py), events.reg_dnn_nu1_pz)
        nu2 = create_lvector_xyz(*rotate_px_py(events.reg_dnn_nu2_px, events.reg_dnn_nu2_py), events.reg_dnn_nu2_pz)
        if which == "nus":
            return nu1, nu2
        # build the higgs
        h = ak.concatenate([nu1[:, None] * 1, nu2[:, None] * 1, vis_leps], axis=1).sum(axis=1)
        if which is None:
            return h
        if which == "mass":
            return h.mass
        raise ValueError(f"Unknown which: {which}")
    build_reg_h.inputs = ["{Electron,Muon,Tau}.{pt,eta,phi,mass}", "reg_dnn_nu{1,2}_p{x,y,z}"]
    add_variable(
        config,
        name="reg_h_mass",
        expression=partial(build_reg_h, which="mass"),
        aux={"inputs": build_reg_h.inputs},
        binning=(50, 0.0, 250.0),
        x_title=r"Regressed $m_{H}$",
    )

    def build_vis_h(events, which=None):
        vis_h = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2].sum(axis=1)
        if which is None:
            return vis_h
        if which == "mass":
            return vis_h.mass
        raise ValueError(f"Unknown which: {which}")
    build_vis_h.inputs = ["{Electron,Muon,Tau}.{pt,eta,phi,mass}"]
    add_variable(
        config,
        name="vis_h_mass",
        expression=partial(build_vis_h, which="mass"),
        aux={"inputs": build_vis_h.inputs},
        binning=(50, 0.0, 250.0),
        x_title=r"Visible $m_{H}$",
    )

    def build_reg_met(events, which=None):
        nu1, nu2 = build_reg_h(events, which="nus")
        if which == "px":
            return nu1.px + nu2.px
        if which == "py":
            return nu1.py + nu2.py
        raise ValueError(f"Unknown which: {which}")
    build_reg_met.inputs = build_reg_h.inputs

    add_variable(
        config,
        name="reg_met_px",
        expression=partial(build_reg_met, which="px"),
        aux={"inputs": build_reg_met.inputs},
        binning=(50, -250.0, 250.0),
        x_title=r"Regressed $\nu_1 p_x + \nu_2 p_x$",
    )
    add_variable(
        config,
        name="reg_met_py",
        expression=partial(build_reg_met, which="py"),
        aux={"inputs": build_reg_met.inputs},
        binning=(50, -250.0, 250.0),
        x_title=r"Regressed $\nu_1 p_y + \nu_2 p_y$",
    )

    # weights
    add_variable(
        config,
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    add_variable(
        config,
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    add_variable(
        config,
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    add_variable(
        config,
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    add_variable(
        config,
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    add_variable(
        config,
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    add_variable(
        config,
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    add_variable(
        config,
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    add_variable(
        config,
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        config,
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        config,
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        config,
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )

    # build variables for dilepton, dijet, and hh
    def delta_r12(vectors):
        # delta r between first two elements
        dr = ak.firsts(vectors[:, :1], axis=1).delta_r(ak.firsts(vectors[:, 1:2], axis=1))
        return ak.fill_none(dr, EMPTY_FLOAT)

    def build_dilep(events, which=None):
        leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
        if which == "dr":
            return delta_r12(leps)
        dilep = leps.sum(axis=1)
        if which is None:
            return dilep * 1
        if which == "mass":
            return dilep.mass
        if which == "pt":
            return dilep.pt
        if which == "eta":
            return dilep.eta
        if which == "abs_eta":
            return abs(dilep.eta)
        if which == "phi":
            return dilep.phi
        if which == "energy":
            return dilep.energy
        raise ValueError(f"Unknown which: {which}")

    build_dilep.inputs = ["{Electron,Muon,Tau}.{pt,eta,phi,mass}"]

    def build_dibjet(events, which=None):
        events = attach_coffea_behavior(events, {"HHBJet": default_coffea_collections["Jet"]})
        hhbjets = events.HHBJet[:, :2]
        if which == "dr":
            return delta_r12(hhbjets)
        dijet = hhbjets.sum(axis=1)
        if which is None:
            return dijet * 1
        if which == "mass":
            return dijet.mass
        if which == "pt":
            return dijet.pt
        if which == "eta":
            return dijet.eta
        if which == "abs_eta":
            return abs(dijet.eta)
        if which == "phi":
            return dijet.phi
        if which == "energy":
            return dijet.energy
        raise ValueError(f"Unknown which: {which}")

    build_dibjet.inputs = ["HHBJet.{pt,eta,phi,mass}"]

    def build_hh(events, which=None):
        dijet = build_dibjet(events)
        dilep = build_dilep(events)
        hs = ak.concatenate([dijet[..., None], dilep[..., None]], axis=1)
        if which == "dr":
            return delta_r12(hs)
        hh = hs.sum(axis=1)
        if which is None:
            return hh * 1
        if which == "mass":
            return hh.mass
        if which == "pt":
            return hh.pt
        if which == "eta":
            return hh.eta
        if which == "abs_eta":
            return abs(hh.eta)
        if which == "phi":
            return hh.phi
        if which == "energy":
            return hh.energy
        raise ValueError(f"Unknown which: {which}")

    build_hh.inputs = build_dibjet.inputs + build_dilep.inputs

    # dibjet variables
    add_variable(
        config,
        name="dibjet_energy",
        expression=partial(build_dibjet, which="energy"),
        aux={"inputs": build_dibjet.inputs},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{bb}$",
    )
    add_variable(
        config,
        name="dibjet_mass",
        expression=partial(build_dibjet, which="mass"),
        aux={"inputs": build_dibjet.inputs},
        binning=(30, 0, 300),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    add_variable(
        config,
        name="dibjet_pt",
        expression=partial(build_dibjet, which="pt"),
        aux={"inputs": build_dibjet.inputs},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,bb}$",
    )
    add_variable(
        config,
        name="dibjet_eta",
        expression=partial(build_dibjet, which="eta"),
        aux={"inputs": build_dibjet.inputs},
        binning=(50, -5, 5),
        x_title=r"$\eta_{bb}$",
    )
    add_variable(
        config,
        name="dibjet_phi",
        expression=partial(build_dibjet, which="phi"),
        aux={"inputs": build_dibjet.inputs},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{bb}$",
    )
    add_variable(
        config,
        name="dibjet_dr",
        expression=partial(build_dibjet, which="dr"),
        aux={"inputs": build_dibjet.inputs},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{bb}$",
    )

    def build_nbjets(events, which=None):
        wp = "medium"
        if which == "btagPNetB":
            wp_value = config.x.btag_working_points["particleNet"][wp]
        elif which == "btagDeepFlavB":
            wp_value = config.x.btag_working_points["deepjet"][wp]
        else:
            raise ValueError(f"Unknown which: {which}")
        bjet_mask = events.Jet[which] >= wp_value
        objects = events.Jet[bjet_mask]
        objects_num = ak.num(objects, axis=1)
        return objects_num

    build_nbjets.inputs = ["Jet.{btagPNetB,btagDeepFlavB}"]

    add_variable(
        config,
        name="nbjets_deepjet",
        expression=partial(build_nbjets, which="btagDeepFlavB"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (DeepJet medium)",
        discrete_x=True,
    )
    add_variable(
        config,
        name="nbjets_pnet",
        expression=partial(build_nbjets, which="btagPNetB"),
        aux={"inputs": build_nbjets.inputs},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (PNet medium)",
        discrete_x=True,
    )

    # dilepton variables
    add_variable(
        config,
        name="dilep_energy",
        expression=partial(build_dilep, which="energy"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{ll}$",
    )
    add_variable(
        config,
        name="dilep_mass",
        expression=partial(build_dilep, which="mass"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 40, 120),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    add_variable(
        config,
        name="dilep_pt",
        expression=partial(build_dilep, which="pt"),
        aux={"inputs": build_dilep.inputs},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,ll}$",
    )
    add_variable(
        config,
        name="dilep_eta",
        expression=partial(build_dilep, which="eta"),
        aux={"inputs": build_dilep.inputs},
        binning=(50, -5, 5),
        unit="GeV",
        x_title=r"$\eta_{ll}$",
    )
    add_variable(
        config,
        name="dilep_phi",
        expression=partial(build_dilep, which="phi"),
        aux={"inputs": build_dilep.inputs},
        binning=(66, -3.3, 3.3),
        unit="GeV",
        x_title=r"$\phi_{ll}$",
    )
    add_variable(
        config,
        name="dilep_dr",
        expression=partial(build_dilep, which="dr"),
        aux={"inputs": build_dilep.inputs},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll}$",
    )

    # hh variables
    add_variable(
        config,
        name="hh_energy",
        expression=partial(build_hh, which="energy"),
        aux={"inputs": build_hh.inputs},
        binning=(35, 100, 800),
        unit="GeV",
        x_title=r"$E_{ll+bb}$",
    )
    add_variable(
        config,
        name="hh_mass",
        expression=partial(build_hh, which="mass"),
        aux={"inputs": build_hh.inputs},
        binning=(50, 0, 1000),
        unit="GeV",
        x_title=r"$m_{ll+bb}$",
    )
    add_variable(
        config,
        name="hh_pt",
        expression=partial(build_hh, which="pt"),
        aux={"inputs": build_hh.inputs},
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T,ll+bb}$",
    )
    add_variable(
        config,
        name="hh_eta",
        expression=partial(build_hh, which="eta"),
        aux={"inputs": build_hh.inputs},
        binning=(50, -5, 5),
        unit="GeV",
        x_title=r"$\eta_{ll+bb}$",
    )
    add_variable(
        config,
        name="hh_phi",
        expression=partial(build_hh, which="phi"),
        aux={"inputs": build_hh.inputs},
        binning=(66, -3.3, 3.3),
        unit="GeV",
        x_title=r"$\phi_{ll+bb}$",
    )
    add_variable(
        config,
        name="hh_dr",
        expression=partial(build_hh, which="dr"),
        aux={"inputs": build_hh.inputs},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll,bb}$",
    )

    # single lepton variables
    # single electron
    add_variable(
        config,
        name="e1_pt",
        expression="Electron.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading electron $p_{T}$",
    )
    add_variable(
        config,
        name="e2_pt",
        expression="Electron.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading electron $p_{T}$",
    )
    add_variable(
        config,
        name="e1_eta",
        expression="Electron.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading electron $\eta$",
    )
    add_variable(
        config,
        name="e2_eta",
        expression="Electron.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading electron $\eta$",
    )
    add_variable(
        config,
        name="e1_phi",
        expression="Electron.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading electron $\phi$",
    )
    add_variable(
        config,
        name="e2_phi",
        expression="Electron.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading electron $\phi$",
    )

    # single tau
    add_variable(
        config,
        name="tau1_pt",
        expression="Tau.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading tau p$_{T}$",
    )
    add_variable(
        config,
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading tau $p_{T}$",
    )
    add_variable(
        config,
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading tau $\eta$",
    )
    add_variable(
        config,
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading tau $\eta$",
    )
    add_variable(
        config,
        name="tau1_phi",
        expression="Tau.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading tau $\phi$",
    )
    add_variable(
        config,
        name="tau2_phi",
        expression="Tau.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading tau $\phi$",
    )

    # single mu
    add_variable(
        config,
        name="mu1_pt",
        expression="Muon.pt[:,0]",
        binning=(30, 0, 150),
        x_title=r"Leading muon $p_{T}$",
    )
    add_variable(
        config,
        name="mu2_pt",
        expression="Muon.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading muon $p_{T}$",
    )
    add_variable(
        config,
        name="mu1_eta",
        expression="Muon.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading muon $\eta$",
    )
    add_variable(
        config,
        name="mu2_eta",
        expression="Muon.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading muon $\eta$",
    )
    add_variable(
        config,
        name="mu1_phi",
        expression="Muon.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading muon $\phi$",
    )
    add_variable(
        config,
        name="mu2_phi",
        expression="Muon.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading muon $\phi$",
    )

    add_variable(
        config,
        name="njets",
        expression=lambda events: ak.num(events.Jet["pt"], axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of jets",
    )

    for proc in ["hh", "tt", "dy"]:
        # outputs of the resonant pDNN at SM-like mass and spin values
        add_variable(
            config,
            name=f"res_pdnn_{proc}",
            expression=f"res_pdnn_s0_m500_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. pDNN$_{{m_{{HH}}=500\,GeV,s=0}}$",
        )

        # outputs of the resonant DNN trained over flat masses
        add_variable(
            config,
            name=f"res_dnn_{proc}",
            expression=f"res_dnn_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. DNN",
        )

        add_variable(
            config,
            name=f"res_dnn_{proc}_fine",
            expression=f"res_dnn_{proc}",
            binning=(5000, 0.0, 1.0),
            x_title=rf"{proc.upper()} output bin, res. DNN",
            aux={"x_transformations": "equal_distance_with_indices"},
        )


# helper to add a variable to the config with some defaults
def add_variable(config: od.Config, *args, **kwargs) -> od.Variable:
    kwargs.setdefault("null_value", EMPTY_FLOAT)

    # create the variable
    variable = config.add_variable(*args, **kwargs)

    # defaults
    if not variable.has_aux("underflow"):
        variable.x.underflow = True
    if not variable.has_aux("overflow"):
        variable.x.overflow = True

    return variable
