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
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )

    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    config.add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    config.add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    config.add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
