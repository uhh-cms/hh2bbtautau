# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


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
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
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
    add_variable(
        config,
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    add_variable(
        config,
        name="jet_pt",
        expression="Jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"all Jet $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        config,
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        config,
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        config,
        name="met_phi",
        expression="PuppiMET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    add_variable(
        config,
        name="e1_pt",
        expression="Electron.pt[:, 0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 400),
        x_title=r"Leading electron p$_{T}$",
    )
    add_variable(
        config,
        name="mu1_pt",
        expression="Muon.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 400),
        x_title=r"Leading muon p$_{T}$",
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
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        config,
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        config,
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        config,
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )

    # variables of interest
    add_variable(
        config,
        name="hh_mass",
        expression="hh.mass",
        binning=(20, 250, 750.0),
        unit="GeV",
        x_title=r"$m_{hh}$",
    )
    add_variable(
        config,
        name="hh_pt",
        expression="hh.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    add_variable(
        config,
        name="hh_eta",
        expression="hh.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    add_variable(
        config,
        name="ditau_mass",
        expression="diTau.mass",
        binning=(20, 50, 200.0),
        unit="GeV",
        x_title=r"$m_{\tau\tau}$",
    )
    add_variable(
        config,
        name="ditau_pt",
        expression="diTau.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    add_variable(
        config,
        name="ditau_eta",
        expression="diTau.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    add_variable(
        config,
        name="dibjet_mass",
        expression="diBJet.mass",
        binning=(20, 0, 500.0),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    add_variable(
        config,
        name="dibjet_pt",
        expression="diBJet.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    add_variable(
        config,
        name="dibjet_eta",
        expression="diBJet.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    def dilep_mass_test(events):
        import awkward as ak
        leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
        dilep = leps.sum(axis=1)
        return dilep.mass

    add_variable(
        config,
        name="dilep_mass",
        expression=dilep_mass_test,
        aux={"inputs": ["{Electron,Muon,Tau}.{pt,eta,phi,mass}"]},
        binning=(40, 40, 120),
        unit="GeV",
        x_title=r"$m_{ll}$",
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
            x_title=rf"{proc.upper()} output node, res. DNN",
        )


# helper to add a variable to the config with some defaults
def add_variable(config: od.Config, *args, **kwargs) -> od.Variable:
    # create the variable
    variable = config.add_variable(*args, **kwargs)

    # defaults
    if not variable.has_aux("underflow"):
        variable.x.underflow = True
    if not variable.has_aux("overflow"):
        variable.x.overflow = True

    return variable
