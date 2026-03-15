# coding: utf-8

"""
Definition of variables.
"""

from __future__ import annotations

import functools

import order as od

from columnflow.columnar_util import EMPTY_FLOAT, Route, attach_coffea_behavior
from columnflow.util import maybe_import
from columnflow.types import Sequence, Callable, Type, Any

from hbt.util import create_lvector_xyz, stack_lvectors

np = maybe_import("numpy")
ak = maybe_import("awkward")


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    # helper that automatically adds null_value and under/overflow flags if not specified
    def add_variable(*args, **kwargs) -> od.Variable:
        kwargs.setdefault("null_value", EMPTY_FLOAT)
        # create the variable
        variable = config.add_variable(*args, **kwargs)
        # defaults
        if not variable.has_aux("underflow"):
            variable.x.underflow = True
        if not variable.has_aux("overflow"):
            variable.x.overflow = True
        # return result
        return variable

    add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    add_variable(
        name="ht",
        expression=(var_ht := VarHt()),
        aux={"inputs": var_ht.uses},
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    add_variable(
        name="jet_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"all Jet $p_{T}$",
    )
    add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        binning=(30, -3.0, 3.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        name="jet1_chEmEF",
        expression="Jet.chEmEF[:,0]",
        binning=(40, 0.0, 0.2),
        x_title="Leading jet chEmEF",
    )
    add_variable(
        name="jet1_muEF",
        expression="Jet.muEF[:,0]",
        binning=(40, 0.6, 1.0),
        x_title="Leading jet muEF",
    )
    add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )
    add_variable(
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        binning=(30, -3.0, 3.0),
        x_title=r"Subleading jet $\eta$",
    )
    add_variable(
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading jet $\phi$",
    )

    add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    add_variable(
        name="njets",
        expression=lambda events: ak.num(events.Jet["pt"], axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of jets",
    )
    add_variable(
        name="nbjets_deepjet",
        expression=(var_nbjets := VarNBTags()).partial(config_inst=config, attr="btagDeepFlavB"),
        aux={"inputs": var_nbjets.uses},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (DeepJet medium)",
        discrete_x=True,
    )
    add_variable(
        name="nbjets_pnet",
        expression=(var_nbjets := VarNBTags()).partial(config_inst=config, attr="btagPNetB"),
        aux={"inputs": var_nbjets.uses},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (PNet medium)",
        discrete_x=True,
    )

    add_variable(
        name="nbjets_pnet_overflow",
        expression=config.variables.n.nbjets_pnet.expression,
        aux={**config.variables.n.nbjets_pnet.aux, "overflow": True},
        binning=(4, -0.5, 3.5),
        x_title=r"Number of b-jets (PNet medium)",
        discrete_x=True,
    )
    add_variable(
        name="nbjets_upart",
        expression=(var_nbjets := VarNBTags()).partial(config_inst=config, attr="btagUParTAK4B"),
        aux={"inputs": var_nbjets.uses},
        binning=(11, -0.5, 10.5),
        x_title=r"Number of b-jets (UParT medium)",
        discrete_x=True,
    )
    add_variable(
        name="nbjets_upart_overflow",
        expression=config.variables.n.nbjets_upart.expression,
        aux={**config.variables.n.nbjets_upart.aux, "overflow": True},
        binning=(4, -0.5, 3.5),
        x_title=r"Number of b-jets (UParT medium)",
        discrete_x=True,
    )
    add_variable(
        name="met_pt",
        expression="PuppiMET.pt",
        binning=(40, 0, 200),
        x_title=r"MET $p_T$",
    )
    add_variable(
        name="met_phi",
        expression="PuppiMET.phi",
        binning=(66, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    add_variable(
        name="met_px",
        expression=lambda events: events.PuppiMET.px,
        aux={"inputs": ["PuppiMET.{pt,phi}"]},
        binning=(50, -250, 250),
        x_title=r"MET $p_x$",
    )
    add_variable(
        name="met_py",
        expression=lambda events: events.PuppiMET.py,
        aux={"inputs": ["PuppiMET.{pt,phi}"]},
        binning=(50, -250, 250),
        x_title=r"MET $p_y$",
    )

    # regression variables
    for n in range(1, 2 + 1):
        for v in ["px", "py", "pz"]:
            add_variable(
                name=f"reg_dnn_nu{n}_{v}",
                expression=f"reg_dnn_moe_nu{n}_{v}",
                binning=(40, -150, 150),
                x_title=rf"Regressed $\nu_{n} {v}$",
            )

    add_variable(
        name="reg_met_px",
        expression=(var_met_reg := VarMETReg()).partial(attr="px"),
        aux={"inputs": var_met_reg.uses},
        binning=(50, -250.0, 250.0),
        x_title=r"$\nu_1 p_x + \nu_2 p_x$ (regressed)",
    )
    add_variable(
        name="reg_met_py",
        expression=var_met_reg.partial(attr="py"),
        aux={"inputs": var_met_reg.uses},
        binning=(50, -250.0, 250.0),
        x_title=r"$\nu_1 p_y + \nu_2 p_y$ (regressed)",
    )

    # weights
    add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Leading jet $p_{T}$",
    )
    add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Leading jet $\eta$",
    )
    add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading jet $\phi$",
    )
    add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Subleading jet $p_{T}$",
    )

    # dihhbjet variables
    add_variable(
        name="dihhbjet_energy",
        expression=(var_dihhbjet := VarDiHHBJet()).partial(attr="energy"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{bb}$",
    )
    add_variable(
        name="dihhbjet_mass",
        expression=var_dihhbjet.partial(attr="mass"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(30, 0, 300),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    add_variable(
        name="dihhbjet_pt",
        expression=var_dihhbjet.partial(attr="pt"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,bb}$",
    )
    add_variable(
        name="dihhbjet_eta",
        expression=var_dihhbjet.partial(attr="eta"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(50, -5, 5),
        x_title=r"$\eta_{bb}$",
    )
    add_variable(
        name="dihhbjet_phi",
        expression=var_dihhbjet.partial(attr="phi"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{bb}$",
    )
    add_variable(
        name="dihhbjet_dr",
        expression=var_dihhbjet.partial(attr="dr"),
        aux={"inputs": var_dihhbjet.uses},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{bb}$",
    )

    # visible dilepton variables
    add_variable(
        name="dilep_vis_energy",
        expression=(var_dilepvis := VarDiLepVis()).partial(attr="energy"),
        aux={"inputs": var_dilepvis.uses},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_mass",
        expression=var_dilepvis.partial(attr="mass"),
        aux={"inputs": var_dilepvis.uses},
        binning=(50, 0.0, 250.0),
        unit="GeV",
        x_title=r"$m_{ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_pt",
        expression=var_dilepvis.partial(attr="pt"),
        aux={"inputs": var_dilepvis.uses},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_pt_low",
        expression=var_dilepvis.partial(attr="pt"),
        aux={"inputs": var_dilepvis.uses, "overflow": False, "underflow": False},
        binning=(50, 0, 50),
        unit="GeV",
        x_title=r"$p_{T,ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_eta",
        expression=var_dilepvis.partial(attr="eta"),
        aux={"inputs": var_dilepvis.uses},
        binning=(50, -5, 5),
        x_title=r"$\eta_{ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_phi",
        expression=var_dilepvis.partial(attr="phi"),
        aux={"inputs": var_dilepvis.uses},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{ll}$ (visible)",
    )
    add_variable(
        name="dilep_vis_dr",
        expression=var_dilepvis.partial(attr="dr"),
        aux={"inputs": var_dilepvis.uses},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll}$ (visible)",
    )

    # regressed dilepton variables
    add_variable(
        name="dilep_reg_energy",
        expression=(var_dilepreg := VarDiLepReg()).partial(attr="energy"),
        aux={"inputs": var_dilepreg.uses},
        binning=(40, 40, 300),
        unit="GeV",
        x_title=r"$E_{ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_mass",
        expression=var_dilepreg.partial(attr="mass"),
        aux={"inputs": var_dilepreg.uses},
        binning=(50, 0.0, 250.0),
        unit="GeV",
        x_title=r"$m_{ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_pt",
        expression=var_dilepreg.partial(attr="pt"),
        aux={"inputs": var_dilepreg.uses},
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$p_{T,ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_pt_low",
        expression=var_dilepreg.partial(attr="pt"),
        aux={"inputs": var_dilepreg.uses, "overflow": False, "underflow": False},
        binning=(50, 0, 50),
        unit="GeV",
        x_title=r"$p_{T,ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_eta",
        expression=var_dilepreg.partial(attr="eta"),
        aux={"inputs": var_dilepreg.uses},
        binning=(50, -5, 5),
        x_title=r"$\eta_{ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_phi",
        expression=var_dilepreg.partial(attr="phi"),
        aux={"inputs": var_dilepreg.uses},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{ll}$ (regressed)",
    )
    add_variable(
        name="dilep_reg_dr",
        expression=var_dilepreg.partial(attr="dr"),
        aux={"inputs": var_dilepreg.uses},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll}$ (regressed)",
    )

    # visible hh variables
    add_variable(
        name="hh_vis_energy",
        expression=(var_hhvis := VarHHVis()).partial(attr="energy"),
        aux={"inputs": var_hhvis.uses},
        binning=(35, 100, 800),
        unit="GeV",
        x_title=r"$E_{ll+bb}$ (visible)",
    )
    add_variable(
        name="hh_vis_mass",
        expression=var_hhvis.partial(attr="mass"),
        aux={"inputs": var_hhvis.uses},
        binning=(50, 0, 1000),
        unit="GeV",
        x_title=r"$m_{ll+bb}$ (visible)",
    )
    add_variable(
        name="hh_vis_pt",
        expression=var_hhvis.partial(attr="pt"),
        aux={"inputs": var_hhvis.uses},
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T,ll+bb}$ (visible)",
    )
    add_variable(
        name="hh_vis_eta",
        expression=var_hhvis.partial(attr="eta"),
        aux={"inputs": var_hhvis.uses},
        binning=(50, -5, 5),
        x_title=r"$\eta_{ll+bb}$ (visible)",
    )
    add_variable(
        name="hh_vis_phi",
        expression=var_hhvis.partial(attr="phi"),
        aux={"inputs": var_hhvis.uses},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{ll+bb}$ (visible)",
    )
    add_variable(
        name="hh_vis_dr",
        expression=var_hhvis.partial(attr="dr"),
        aux={"inputs": var_hhvis.uses},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll,bb}$ (visible)",
    )

    # Regressed hh variables
    add_variable(
        name="hh_reg_energy",
        expression=(var_hhreg := VarHHReg()).partial(attr="energy"),
        aux={"inputs": var_hhreg.uses},
        binning=(35, 100, 800),
        unit="GeV",
        x_title=r"$E_{ll+bb}$ (regressed)",
    )
    add_variable(
        name="hh_reg_mass",
        expression=var_hhreg.partial(attr="mass"),
        aux={"inputs": var_hhreg.uses},
        binning=(50, 0, 1000),
        unit="GeV",
        x_title=r"$m_{ll+bb}$ (regressed)",
    )
    add_variable(
        name="hh_reg_pt",
        expression=var_hhreg.partial(attr="pt"),
        aux={"inputs": var_hhreg.uses},
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T,ll+bb}$ (regressed)",
    )
    add_variable(
        name="hh_reg_eta",
        expression=var_hhreg.partial(attr="eta"),
        aux={"inputs": var_hhreg.uses},
        binning=(50, -5, 5),
        x_title=r"$\eta_{ll+bb}$ (regressed)",
    )
    add_variable(
        name="hh_reg_phi",
        expression=var_hhreg.partial(attr="phi"),
        aux={"inputs": var_hhreg.uses},
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi_{ll+bb}$ (regressed)",
    )
    add_variable(
        name="hh_reg_dr",
        expression=var_hhreg.partial(attr="dr"),
        aux={"inputs": var_hhreg.uses},
        binning=(30, 0, 6),
        x_title=r"$\Delta R_{ll,bb}$ (regressed)",
    )

    # single lepton variables
    # single electron
    add_variable(
        name="e1_pt",
        expression="Electron.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading electron $p_{T}$",
    )
    add_variable(
        name="e2_pt",
        expression="Electron.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading electron $p_{T}$",
    )
    add_variable(
        name="e1_eta",
        expression="Electron.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading electron $\eta$",
    )
    add_variable(
        name="e2_eta",
        expression="Electron.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading electron $\eta$",
    )
    add_variable(
        name="e1_phi",
        expression="Electron.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading electron $\phi$",
    )
    add_variable(
        name="e2_phi",
        expression="Electron.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading electron $\phi$",
    )

    # single tau
    add_variable(
        name="tau1_pt",
        expression="Tau.pt[:, 0]",
        binning=(30, 0, 150),
        x_title=r"Leading tau p$_{T}$",
    )
    add_variable(
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading tau $p_{T}$",
    )
    add_variable(
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading tau $\eta$",
    )
    add_variable(
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading tau $\eta$",
    )
    add_variable(
        name="tau1_phi",
        expression="Tau.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading tau $\phi$",
    )
    add_variable(
        name="tau2_phi",
        expression="Tau.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading tau $\phi$",
    )

    # single mu
    add_variable(
        name="mu1_pt",
        expression="Muon.pt[:,0]",
        binning=(30, 0, 150),
        x_title=r"Leading muon $p_{T}$",
    )
    add_variable(
        name="mu2_pt",
        expression="Muon.pt[:,1]",
        binning=(30, 0, 150),
        x_title=r"Subleading muon $p_{T}$",
    )
    add_variable(
        name="mu1_eta",
        expression="Muon.eta[:,0]",
        binning=(50, -2.5, 2.5),
        x_title=r"Leading muon $\eta$",
    )
    add_variable(
        name="mu2_eta",
        expression="Muon.eta[:,1]",
        binning=(50, -2.5, 2.5),
        x_title=r"Subleading muon $\eta$",
    )
    add_variable(
        name="mu1_phi",
        expression="Muon.phi[:,0]",
        binning=(66, -3.3, 3.3),
        x_title=r"Leading muon $\phi$",
    )
    add_variable(
        name="mu2_phi",
        expression="Muon.phi[:,1]",
        binning=(66, -3.3, 3.3),
        x_title=r"Subleading muon $\phi$",
    )

    # DNN outputs
    for proc in ["hh", "tt", "dy"]:
        # outputs of the resonant pDNN at SM-like mass and spin values
        add_variable(
            name=f"res_pdnn_{proc}",
            expression=f"res_pdnn_s0_m500_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. pDNN$_{{m_{{HH}}=500\,GeV,s=0}}$",
        )

        # outputs of the resonant DNN trained over flat masses
        add_variable(
            name=f"res_dnn_{proc}",
            expression=f"res_dnn_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. DNN",
        )

        add_variable(
            name=f"res_dnn_{proc}_fine",
            expression=f"res_dnn_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"{proc.upper()} output bin, res. DNN",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_moe_{proc}",
            expression=f"run3_dnn_moe_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"DNN {proc.upper()} output",
        )

        add_variable(
            name=f"run3_dnn_moe_{proc}_10",
            expression=f"run3_dnn_moe_{proc}",
            binning=(10, 0.0, 1.0),
            x_title=rf"DNN {proc.upper()} output",
        )

        add_variable(
            name=f"run3_dnn_simple_{proc}",
            expression=f"run3_dnn_simple_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_moe_{proc}_fine",
            expression=f"run3_dnn_moe_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        def logit(events: ak.Array, col: str, eps: float = 1e-6) -> ak.Array | np.ndarray:
            # eps confines the range of the transformed values to approx. [-13.8, 13.8] for x in [0, 1]
            import numpy as np
            x = events[col]
            return np.log((x + eps) / (1 - x + eps))

        add_variable(
            name=f"run3_dnn_moe_{proc}_logit",
            expression=functools.partial(logit, col=f"run3_dnn_moe_{proc}"),
            binning=(30, -15, 15),
            x_title=rf"logit(DNN {proc.upper()} output)",
            aux={"inputs": [f"run3_dnn_moe_{proc}"]},
        )

        add_variable(
            name=f"run3_dnn_moe_{proc}_logit_fine",
            expression=functools.partial(logit, col=f"run3_dnn_moe_{proc}"),
            binning=(3000, -15, 15),
            x_title=rf"logit(DNN {proc.upper()} output)",
            aux={"inputs": [f"run3_dnn_moe_{proc}"],
                "x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_moe_{proc}_fine_5k",
            expression=f"run3_dnn_moe_{proc}",
            binning=(5000, 0.0, 1.0),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_simple_{proc}_fine",
            expression=f"run3_dnn_simple_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_simple_kl1_{proc}_fine",
            expression=f"run3_dnn_simple_kl1_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_simple_kl0_{proc}_fine",
            expression=f"run3_dnn_simple_kl0_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

        add_variable(
            name=f"run3_dnn_simple_allkl_{proc}_fine",
            expression=f"run3_dnn_simple_allkl_{proc}",
            binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
            x_title=rf"DNN {proc.upper()} output",
            aux={"x_transformations": "equal_distance_with_indices"},
        )

    # end-to-end DNN outputs
    add_variable(
        name="e2e_model1_hh_fine",
        expression="e2e_model1_hh",
        binning=np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist(),
        x_title=rf"E2E DNN {proc.upper()} output",
        aux={"x_transformations": "equal_distance_with_indices"},
    )

    # to be use with --hist-producer e2e
    add_variable(
        name="e2e_model1_bins",
        expression=(lambda events: [events[f"e2e_model1_bin{i}"] for i in range(50)]),
        binning=(50, -0.5, 49.5),
        x_title="E2E latent space bins",
        aux={"inputs": ["e2e_model1_bin*"]},
    )


#
# tools for defining variable functions
#

class VarExp:

    uses: set[str | Route] | None = None
    compose: dict[str, Type[VarExp]] | None = None

    def __init__(
        self,
        uses: Sequence[str | Route] | set[str | Route] | None = None,
        compose: dict[str, Type[VarExp]] | None = None,
    ) -> None:
        super().__init__()

        # store used columns
        self.uses = set(uses or self.__class__.uses or set())

        # create sub expressions for composing
        for attr, cls in (compose or self.__class__.compose or {}).items():
            assert attr not in {"uses", "compose"}
            inst = cls()
            setattr(self, attr, inst)
            self.uses.update(inst.uses)

    def partial(self, *args, **kwargs) -> Callable[[ak.Array], ak.Array | np.ndarray]:
        return functools.partial(self.__call__, *args, **kwargs) if args or kwargs else self.__call__

    def __call__(self, events) -> ak.Array | np.ndarray:
        raise NotImplementedError

    def raise_unknown_attr(self, attr) -> None:
        raise ValueError(f"unknown {self.__class__.__name__} attr: {attr}")

    def var_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        return {
            "expression": self.partial(*args, **kwargs),
            "aux": {"inputs": self.uses},
        }


#
# kinematic helpers
#

def delta_r12(vectors: ak.Array) -> ak.Array:
    # delta r between first two elements
    dr = ak.firsts(vectors[:, :1], axis=1).delta_r(ak.firsts(vectors[:, 1:2], axis=1))
    return ak.fill_none(dr, EMPTY_FLOAT)


def rotate_px_py(
    px: ak.Array | np.ndarray,
    py: ak.Array | np.ndarray,
    ref_phi: ak.Array | np.ndarray,
) -> ak.Array | np.ndarray:
    new_phi = np.arctan2(py, px) + ref_phi  # mind the "+"
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


#
# advanced variables
#

class VarHt(VarExp):

    uses = {"{Electron,Muon,Tau,Jet}.{pt,eta,phi,mass}"}

    def __call__(self, events: ak.Array) -> ak.Array:
        vectors = stack_lvectors([events.Electron, events.Muon, events.Tau, events.Jet])
        return vectors.sum(axis=-1).pt


class VarDiHHBJet(VarExp):

    uses = {"HHBJet.{pt,eta,phi,mass}"}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
        hhbjets = events.HHBJet[:, :2]

        if attr == "dr":
            return delta_r12(hhbjets)

        dijet = hhbjets.sum(axis=1)

        if attr is None:
            return dijet
        if attr == "mass":
            return dijet.mass
        if attr == "pt":
            return dijet.pt
        if attr == "eta":
            return dijet.eta
        if attr == "abs_eta":
            return abs(dijet.eta)
        if attr == "phi":
            return dijet.phi
        if attr == "energy":
            return dijet.energy

        self.raise_unknown_attr()


class VarDiLepVis(VarExp):

    uses = {"{Electron,Muon,Tau}.{pt,eta,phi,mass}"}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        leps = stack_lvectors([events.Electron, events.Muon, events.Tau])[..., :2]

        if attr == "dr":
            return delta_r12(leps)

        dilep = leps.sum(axis=1)

        if attr is None:
            return dilep
        if attr == "mass":
            return dilep.mass
        if attr == "pt":
            return dilep.pt
        if attr == "eta":
            return dilep.eta
        if attr == "abs_eta":
            return abs(dilep.eta)
        if attr == "phi":
            return dilep.phi
        if attr == "energy":
            return dilep.energy

        self.raise_unknown_attr(attr)


class VarDiLepReg(VarExp):

    uses = {"reg_dnn_moe_nu{1,2}_p{x,y,z}"}
    compose = {"dilepvis": VarDiLepVis}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        dilepvis = self.dilepvis(events)

        # regressed nu components are relative to the visible dilep system and need to be rotated back
        ref_phi = dilepvis.phi
        nu1 = create_lvector_xyz(
            *rotate_px_py(events.reg_dnn_moe_nu1_px, events.reg_dnn_moe_nu1_py, ref_phi),
            events.reg_dnn_moe_nu1_pz,
        )
        nu2 = create_lvector_xyz(
            *rotate_px_py(events.reg_dnn_moe_nu2_px, events.reg_dnn_moe_nu2_py, ref_phi),
            events.reg_dnn_moe_nu2_pz,
        )

        if attr == "nus":
            return nu1, nu2

        # build the full system
        dilepreg = stack_lvectors([nu1, nu2, dilepvis]).sum(axis=-1)

        if attr is None:
            return dilepreg
        if attr == "mass":
            return dilepreg.mass

        self.raise_unknown_attr(attr)


class VarMETReg(VarExp):

    compose = {"dilepreg": VarDiLepReg}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        # build met
        nu1, nu2 = self.dilepreg(events, attr="nus")
        met = stack_lvectors([nu1, nu2]).sum(axis=-11)

        if attr is None:
            return met
        if attr == "px":
            return met.px
        if attr == "py":
            return met.py + met.py
        if attr == "phi":
            return met.phi

        self.raise_unknown_attr(attr)


class VarHHVis(VarExp):

    compose = {"dihhbjet": VarDiHHBJet, "dilepvis": VarDiLepVis}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        dihhbjet = self.dihhbjet(events)
        dilepvis = self.dilepvis(events)
        hs = stack_lvectors([dihhbjet, dilepvis])

        if attr == "dr":
            return delta_r12(hs)

        hh = hs.sum(axis=1)

        if attr is None:
            return hh * 1
        if attr == "mass":
            return hh.mass
        if attr == "pt":
            return hh.pt
        if attr == "eta":
            return hh.eta
        if attr == "abs_eta":
            return abs(hh.eta)
        if attr == "phi":
            return hh.phi
        if attr == "energy":
            return hh.energy

        self.raise_unknown_attr(attr)


class VarHHReg(VarExp):

    compose = {"dihhbjet": VarDiHHBJet, "dilepreg": VarDiLepReg}

    def __call__(self, events: ak.Array, attr: str | None = None) -> ak.Array:
        dihhbjet = self.dihhbjet(events)
        dilepreg = self.dilepreg(events)
        hs = stack_lvectors([dihhbjet, dilepreg])

        if attr == "dr":
            return delta_r12(hs)

        hh = hs.sum(axis=1)

        if attr is None:
            return hh * 1
        if attr == "mass":
            return hh.mass
        if attr == "pt":
            return hh.pt
        if attr == "eta":
            return hh.eta
        if attr == "abs_eta":
            return abs(hh.eta)
        if attr == "phi":
            return hh.phi
        if attr == "energy":
            return hh.energy

        self.raise_unknown_attr(attr)


class VarNBTags(VarExp):

    uses = {"Jet.{btagPNetB,btagDeepFlavB,btagUParTAK4B}"}

    def __call__(self, events: ak.Array, config_inst: od.Config, attr: str | None = None) -> ak.Array:
        wp = "medium"
        if attr == "btagPNetB":
            wp_value = config_inst.x.btag_working_points["particleNet"][wp]
        elif attr == "btagDeepFlavB":
            wp_value = config_inst.x.btag_working_points["deepjet"][wp]
        elif attr == "btagUParTAK4B":
            wp_value = config_inst.x.btag_working_points["upart"][wp]
        else:
            self.raise_unknown_attr(attr)

        return ak.sum(events.Jet[attr] >= wp_value, axis=1)
