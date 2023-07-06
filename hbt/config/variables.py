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
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )

    # Jet Plots
    # jet 1
    config.add_variable(
        name="jet1_energy",
        expression="CollJet.E[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 energy",
    )
    config.add_variable(
        name="jet1_mass",
        expression="CollJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 mass",
    )
    config.add_variable(
        name="jet1_pt",
        expression="CollJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="CollJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 1 $\eta$",
    )
    config.add_variable(
        name="jet1_phi",
        expression="CollJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 1 $\phi$",
    )
    config.add_variable(
        name="jet1_btag",
        expression="CollJet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 1 b-tag",
    )

    # Jet 2
    config.add_variable(
        name="jet2_energy",
        expression="CollJet.E[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 energy",
    )
    config.add_variable(
        name="jet2_mass",
        expression="CollJet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 mass",
    )
    config.add_variable(
        name="jet2_pt",
        expression="CollJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 2 $p_{T}$",
    )
    config.add_variable(
        name="jet2_eta",
        expression="CollJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 2 $\eta$",
    )
    config.add_variable(
        name="jet2_phi",
        expression="CollJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 2 $\phi$",
    )
    config.add_variable(
        name="jet2_btag",
        expression="CollJet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 2 b-tag",
    )
    # Jet 3
    config.add_variable(
        name="jet3_energy",
        expression="CollJet.E[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 energy",
    )
    config.add_variable(
        name="jet3_mass",
        expression="CollJet.mass[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 mass",
    )
    config.add_variable(
        name="jet3_pt",
        expression="CollJet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 3 $p_{T}$",
    )
    config.add_variable(
        name="jet3_eta",
        expression="CollJet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 3 $\eta$",
    )
    config.add_variable(
        name="jet3_phi",
        expression="CollJet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 3 $\phi$",
    )
    config.add_variable(
        name="jet3_btag",
        expression="CollJet.btagDeepFlavB[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 3 b-tag",
    )
    # Jet 4
    config.add_variable(
        name="jet4_energy",
        expression="CollJet.E[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 energy",
    )
    config.add_variable(
        name="jet4_mass",
        expression="CollJet.mass[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 mass",
    )
    config.add_variable(
        name="jet4_pt",
        expression="CollJet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 4 $p_{T}$",
    )
    config.add_variable(
        name="jet4_eta",
        expression="CollJet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 4 $\eta$",
    )
    config.add_variable(
        name="jet4_phi",
        expression="CollJet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 4 $\phi$",
    )
    config.add_variable(
        name="jet4_btag",
        expression="CollJet.btagDeepFlavB[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 4 b-tag",
    )
    # Jet 5
    config.add_variable(
        name="jet5_energy",
        expression="CollJet.E[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 energy",
    )
    config.add_variable(
        name="jet5_mass",
        expression="CollJet.mass[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 mass",
    )
    config.add_variable(
        name="jet5_pt",
        expression="CollJet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 5 $p_{T}$",
    )
    config.add_variable(
        name="jet5_eta",
        expression="CollJet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 5 $\eta$",
    )
    config.add_variable(
        name="jet5_phi",
        expression="CollJet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 5 $\phi$",
    )
    config.add_variable(
        name="jet5_btag",
        expression="CollJet.btagDeepFlavB[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 5 b-tag",
    )
    # Jet 6
    config.add_variable(
        name="jet6_energy",
        expression="CollJet.E[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 energy",
    )
    config.add_variable(
        name="jet6_mass",
        expression="CollJet.mass[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 mass",
    )
    config.add_variable(
        name="jet6_pt",
        expression="CollJet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Colljet 6 $p_{T}$",
    )
    config.add_variable(
        name="jet6_eta",
        expression="CollJet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 6 $\eta$",
    )
    config.add_variable(
        name="jet6_phi",
        expression="CollJet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Colljet 6 $\phi$",
    )
    config.add_variable(
        name="jet6_btag",
        expression="CollJet.btagDeepFlavB[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(0.1, 0.0, 1.0),
        x_title=r"Colljet 6 b-tag",
    )

    # Tau Plots
    config.add_variable(
        name="tau1_mass",
        expression="Tau.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 1 mass",
    )
    config.add_variable(
        name="tau1_pt",
        expression="Tau.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $p_{T}$",
    )
    config.add_variable(
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\eta$",
    )
    config.add_variable(
        name="tau1_phi",
        expression="Tau.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\phi$",
    )
    config.add_variable(
        name="tau2_mass",
        expression="Tau.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 2 mass",
    )
    config.add_variable(
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $p_{T}$",
    )
    config.add_variable(
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 2 $\eta$",
    )
    config.add_variable(
        name="tau2_phi",
        expression="Tau.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 2 $\phi$",
    )

    # Invariant mass Plots
    config.add_variable(
        name="bjet_pair_mass",
        expression="mbjetbjet",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet Pair Mass",
    )
    config.add_variable(
        name="HH_pair_mass",
        expression="mHH",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"HH Pair Mass",
    )
    config.add_variable(
        name="tau_pair_mass",
        expression="mtautau",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau Pair Mass",
    )
    config.add_variable(
        name="inv_mass_d_eta",
        expression="jets_d_eta_inv_mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 5000.0),
        unit="GeV",
        x_title=r"Invariant Mass of Jets with maximum $\Delta \eta$",
    )
    config.add_variable(
        name="hardest_jet_pair_mass",
        expression="mjj",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 800.0),
        unit="GeV",
        x_title=r"Hardest Jet Pair Mass",
    )

    # Others
    config.add_variable(
        name="max_d_eta",
        expression="jets_max_d_eta",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 8.0),
        x_title=r"Maximum $\Delta \eta$ of Jets",
    )
    config.add_variable(
        name="hardest_jet_pair_pt",
        expression="hardest_jet_pair_pt",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 900.0),
        unit="GeV",
        x_title=r"$p_{T}$ of hardest Jets",
    )
    config.add_variable(
        name="ht",
        binning=(50, 0.0, 1200.0),
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="n_jet",
        expression="n_jets",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="energy_corr",
        expression="energy_corr",
        binning=(100, 700, 2000000),
        unit=r"$GeV^{2}$",
        x_title="Energy Correlation",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )

    # B Jets
    config.add_variable(
        name="bjet1_mass",
        expression="BJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 1 mass",
    )
    config.add_variable(
        name="bjet1_pt",
        expression="BJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    config.add_variable(
        name="bjet1_eta",
        expression="BJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 1 $\eta$",
    )
    config.add_variable(
        name="bjet1_phi",
        expression="BJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 1 $\phi$",
    )
    config.add_variable(
        name="bjet2_mass",
        expression="BJet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 2 mass",
    )
    config.add_variable(
        name="bjet2_pt",
        expression="BJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    config.add_variable(
        name="bjet2_eta",
        expression="BJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 2 $\eta$",
    )
    config.add_variable(
        name="bjet2_phi",
        expression="BJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 2 $\phi$",
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
        discrete_x=True,
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
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet2_eta",
        expression="cutflow.jet1_eta",
        null_value=EMPTY_FLOAT,
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 2 $\eta$",
    )
    config.add_variable(
        name="cf_jet3_pt",
        expression="cutflow.jet3_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet4_pt",
        expression="cutflow.jet4_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet5_pt",
        expression="cutflow.jet5_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet6_pt",
        expression="cutflow.jet6_pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
