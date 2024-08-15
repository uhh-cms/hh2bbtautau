# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

from __future__ import annotations

import os
import re
import itertools
import functools

import yaml
import law
import order as od
from scinum import Number

from columnflow.util import DotDict, dev_sandbox
from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases, get_shifts_from_sources,
    verify_config_processes,
)
from columnflow.columnar_util import ColumnCollection, skip_column


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    # gather campaign data
    run = campaign.x.run
    year = campaign.x.year
    year2 = year % 100

    # some validations
    assert run in {2, 3}
    assert year in {2016, 2017, 2018, 2022, 2023}

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = od.Config(name=config_name, id=config_id, campaign=campaign)

    ################################################################################################
    # helpers
    ################################################################################################

    # helper to enable processes / datasets only for a specific era
    def if_era(
        *,
        run: int | None = None,
        year: int | None = None,
        postfix: str | None = None,
        tag: str | None = None,
        values: list[str] | None = None,
    ) -> list[str]:
        match = (
            (run is None or campaign.x.run == run) and
            (year is None or campaign.x.year == year) and
            (postfix is None or campaign.x.postfix == postfix) and
            (tag is None or campaign.has_tag(tag))
        )
        return (values or []) if match else []

    ################################################################################################
    # processes
    ################################################################################################

    # add custom processes
    cfg.add_process(
        name="v",
        id=7997,
        label="W/Z",
        processes=[procs.n.w, procs.n.z],
    )
    cfg.add_process(
        name="multiboson",
        id=7998,
        label="Multiboson",
        processes=[procs.n.vv, procs.n.vvv],
    )
    cfg.add_process(
        name="tt_multiboson",
        id=7999,
        label=r"$t\bar{t}$ + Multiboson",
        processes=[procs.n.ttv, procs.n.ttvv],
    )

    # add processes we are interested in
    process_names = [
        "data",
        "tt",
        "st",
        "dy",
        "v",
        "multiboson",
        "tt_multiboson",
        "qcd",
        "h",
        "hh_ggf_hbb_htt_kl1_kt1",
        *if_era(run=3, year=2022, values=[
            "hh_ggf_hbb_htt_kl0_kt1",
            "hh_ggf_hbb_htt_kl2p45_kt1",
            "hh_ggf_hbb_htt_kl5_kt1",
            "hh_ggf_hbb_htt_kl0_kt1_c21",
            "hh_ggf_hbb_htt_kl1_kt1_c23",
            "hh_vbf_hbb_htt_kv1_k2v1_kl1",
            "hh_vbf_hbb_htt_kv1_k2v0_kl1",
            "hh_vbf_hbb_htt_kv1_k2v1_kl2",
            "hh_vbf_hbb_htt_kv1_k2v2_kl1",
            "hh_vbf_hbb_htt_kv1p74_k2v1p37_kl14p4",
            "hh_vbf_hbb_htt_kvm0p012_k2v0p03_kl10p2",
            "hh_vbf_hbb_htt_kvm0p758_k2v1p44_klm19p3",
            "hh_vbf_hbb_htt_kvm0p962_k2v0p959_klm1p43",
            "hh_vbf_hbb_htt_kvm1p21_k2v1p94_klm0p94",
            "hh_vbf_hbb_htt_kvm1p6_k2v2p72_klm1p36",
            "hh_vbf_hbb_htt_kvm1p83_k2v3p57_klm3p39",
            "hh_vbf_hbb_htt_kvm2p12_k2v3p87_klm5p96",
        ]),
        "graviton_hh_ggf_hbb_htt_m450",
        "graviton_hh_ggf_hbb_htt_m1200",
        "radion_hh_ggf_hbb_htt_m700",
    ]
    for process_name in process_names:
        if process_name in procs:
            proc = procs.get(process_name)
        elif process_name == "qcd":
            # qcd is not part of procs since there is no dataset registered for it
            from cmsdb.processes.qcd import qcd
            proc = qcd
        else:
            # development switch in case datasets are not _yet_ there
            continue

        # add tags to processes
        if process_name.startswith("hh_"):
            proc.add_tag("signal")
            proc.add_tag("nonresonant_signal")
        if process_name.startswith(("graviton_hh_", "radion_hh_")):
            proc.add_tag("signal")
            proc.add_tag("resonant_signal")

        # add the process
        cfg.add_process(proc)

    # configure colors, labels, etc
    from hbt.config.styles import stylize_processes
    stylize_processes(cfg)

    ################################################################################################
    # datasets
    ################################################################################################

    # add datasets we need to study
    dataset_names = [
        # signals
        *if_era(run=3, year=2022, values=[
            # ggf
            "hh_ggf_hbb_htt_kl1_kt1_powheg",
            "hh_ggf_hbb_htt_kl0_kt1_powheg",
            "hh_ggf_hbb_htt_kl2p45_kt1_powheg",
            "hh_ggf_hbb_htt_kl5_kt1_powheg",
            # vbf
            "hh_vbf_hbb_htt_kv1_k2v1_kl1_madgraph",
            "hh_vbf_hbb_htt_kv1_k2v1_kl2_madgraph",
            "hh_vbf_hbb_htt_kv1_k2v0_kl1_madgraph",
            "hh_vbf_hbb_htt_kv1_k2v2_kl1_madgraph",
            "hh_vbf_hbb_htt_kv1p74_k2v1p37_kl14p4_madgraph",
            "hh_vbf_hbb_htt_kvm0p012_k2v0p03_kl10p2_madgraph",
            "hh_vbf_hbb_htt_kvm0p758_k2v1p44_klm19p3_madgraph",
            "hh_vbf_hbb_htt_kvm0p962_k2v0p959_klm1p43_madgraph",
            "hh_vbf_hbb_htt_kvm1p21_k2v1p94_klm0p94_madgraph",
            "hh_vbf_hbb_htt_kvm1p6_k2v2p72_klm1p36_madgraph",
            "hh_vbf_hbb_htt_kvm1p83_k2v3p57_klm3p39_madgraph",
            "hh_vbf_hbb_htt_kvm2p12_k2v3p87_klm5p96_madgraph",
            # some resonances
            "graviton_hh_ggf_hbb_htt_m450_madgraph",
            "graviton_hh_ggf_hbb_htt_m1200_madgraph",
            "radion_hh_ggf_hbb_htt_m700_madgraph",
        ]),

        # backgrounds
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        *if_era(run=3, year=2022, values=[
            "ttw_wlnu_amcatnlo",
            "ttz_zqq_amcatnlo",
            "ttz_zll_m4to50_amcatnlo",
            "ttz_zll_m50toinf_amcatnlo",
            "ttzz_madgraph",
            "ttww_madgraph",
            # "ttwz_madgraph",  # not available yet
            "st_tchannel_t_4f_powheg",
            "st_tchannel_tbar_4f_powheg",
            "st_twchannel_t_sl_powheg",
            "st_twchannel_tbar_sl_powheg",
            "st_twchannel_t_dl_powheg",
            "st_twchannel_tbar_dl_powheg",
            "st_twchannel_t_fh_powheg",
            "st_twchannel_tbar_fh_powheg",
            # "st_schannel_t_lep_4f_amcatnlo",  # no cross section yet
            # "st_schannel_tbar_lep_4f_amcatnlo",  # no cross section yet
            "dy_m4to10_amcatnlo",
            "dy_m10to50_amcatnlo",
            "dy_m50toinf_amcatnlo",
            "dy_m50toinf_0j_amcatnlo",
            "dy_m50toinf_1j_amcatnlo",
            "dy_m50toinf_2j_amcatnlo",
            "dy_m50toinf_1j_pt40to100_amcatnlo",
            "dy_m50toinf_1j_pt100to200_amcatnlo",
            "dy_m50toinf_1j_pt200to400_amcatnlo",
            "dy_m50toinf_1j_pt400to600_amcatnlo",
            "dy_m50toinf_1j_pt600toinf_amcatnlo",
            "dy_m50toinf_2j_pt40to100_amcatnlo",
            "dy_m50toinf_2j_pt100to200_amcatnlo",
            "dy_m50toinf_2j_pt200to400_amcatnlo",
            "dy_m50toinf_2j_pt400to600_amcatnlo",
            "dy_m50toinf_2j_pt600toinf_amcatnlo",
            "w_lnu_amcatnlo",
            "z_qq_pt100to200_1j_amcatnlo",
            "z_qq_pt100to200_2j_amcatnlo",
            "z_qq_pt200to400_1j_amcatnlo",
            "z_qq_pt200to400_2j_amcatnlo",  # literally no events selected above 400 GeV
            "zz_pythia",
            "wz_pythia",
            "ww_pythia",
            "zzz_amcatnlo",
            "wzz_amcatnlo",
            "wwz_4f_amcatnlo",
            "www_4f_amcatnlo",
            "h_ggf_htt_powheg",
            "h_vbf_htt_powheg",
            "vh_hnonbb_amcatnlo",
            "zh_zll_hbb_powheg",
            "zh_zqq_hbb_powheg",
            "zh_htt_powheg",
            "wph_htt_powheg",
            "wmh_htt_powheg",
            "wph_wlnu_hbb_powheg",
            "wmh_wlnu_hbb_powheg",
            "zh_gg_zll_hbb_powheg",
            "zh_gg_znunu_hbb_powheg",
            "zh_gg_zqq_hbb_powheg",
            "tth_hbb_powheg",
            "tth_hnonbb_powheg",
        ]),

        # data
        *if_era(run=3, year=2022, tag="preEE", values=[
            f"data_{stream}_{period}" for stream in ["mu", "e", "tau", "met"] for period in "cd"
        ]),
    ]
    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # add tags to datasets
        if dataset.name.startswith("tt"):
            dataset.add_tag(("has_top", "is_ttbar"))
        elif dataset.name.startswith("st"):
            dataset.add_tag(("has_top", "is_single_top"))
        if dataset.name.startswith("dy"):
            dataset.add_tag("is_dy")
        if re.match(r"^(ww|wz|zz)_.*pythia$", dataset.name):
            dataset.add_tag("no_lhe_weights")
        if dataset_name.startswith("hh_"):
            dataset.add_tag("signal")
            dataset.add_tag("nonresonant_signal")
        if dataset_name.startswith(("graviton_hh_", "radion_hh_")):
            dataset.add_tag("signal")
            dataset.add_tag("resonant_signal")

        # apply an optional limit on the number of files
        if limit_dataset_files:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

    # verify that the root process of all datasets is part of any of the registered processes
    verify_config_processes(cfg, warn=True)

    ################################################################################################
    # task defaults and groups
    ################################################################################################

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "default_no_shifts"
    cfg.x.default_categories = ("incl__os__iso",)
    cfg.x.default_variables = ("n_jet", "n_btag", "res_pdnn_hh", "res_dnn_hh")
    cfg.x.default_weight_producer = "default"

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "signals": [
            "hh_ggf_hbb_htt_kl1_kt1",
        ],
        "signals_ggf": [
            "hh_ggf_hbb_htt_kl0_kt1",
            "hh_ggf_hbb_htt_kl1_kt1",
            "hh_ggf_hbb_htt_kl2p45_kt1",
            "hh_ggf_hbb_htt_kl5_kt1",
        ],
        "backgrounds": (backgrounds := [
            "h",
            "tt",
            "dy",
            "qcd",
            "st",
            "v",
            "multiboson",
            "tt_multiboson",
            "ewk",
        ]),
        "sm_ggf": (sm_ggf_group := ["hh_ggf_hbb_htt_kl1_kt1", *backgrounds]),
        "sm": (sm_group := ["hh_ggf_hbb_htt_kl1_kt1", "hh_vbf_hbb_htt_kv1_k2v1_kl1", *backgrounds]),
        "sm_ggf_data": ["data"] + sm_ggf_group,
        "sm_data": ["data"] + sm_group,
    }

    # define inclusive datasets for the dy process identification with corresponding leaf processes
    if run == 3:
        cfg.x.dy_stitching = {
            "m50toinf": {
                "inclusive_dataset": cfg.datasets.n.dy_m50toinf_amcatnlo,
                "leaf_processes": [
                    # the following processes cover the full njet and pt phasespace
                    procs.n.dy_m50toinf_0j,
                    *(
                        procs.get(f"dy_m50toinf_{nj}j_pt{pt}")
                        for nj in [1, 2]
                        for pt in ["0to40", "40to100", "100to200", "200to400", "400to600", "600toinf"]
                    ),
                    procs.n.dy_m50toinf_ge3j,
                ],
            },
        }

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {}

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["json", "trigger", "met_filter", "jet_veto_map", "lepton", "jet", "bjet"],
    }
    cfg.x.default_selector_steps = "default"

    cfg.x.custom_style_config_groups = {
        "small_legend": {
            "legend_cfg": {"ncols": 2, "fontsize": 16, "columnspacing": 0.6},
        },
    }
    cfg.x.default_custom_style_config = "small_legend"

    cfg.x.default_blinding_threshold = 3e-4

    ################################################################################################
    # luminosity and normalization
    ################################################################################################

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis
    # difference pre-post VFP: https://cds.cern.ch/record/2854610/files/DP2023_006.pdf
    if year == 2016 and campaign.has_tag("preVFP"):
        cfg.x.luminosity = Number(19_500, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2016 and campaign.has_tag("postVFP"):
        cfg.x.luminosity = Number(16_800, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2017:
        cfg.x.luminosity = Number(41_480, {
            "lumi_13TeV_2017": 0.02j,
            "lumi_13TeV_1718": 0.006j,
            "lumi_13TeV_correlated": 0.009j,
        })
    elif year == 2018:
        cfg.x.luminosity = Number(59_830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })
    elif year == 2022 and campaign.has_tag("preEE"):
        cfg.x.luminosity = Number(7_980.4, {
            "lumi_13p6TeV_correlated": 0.014j,
        })
    elif year == 2022 and campaign.has_tag("postEE"):
        cfg.x.luminosity = Number(26_671.7, {
            "lumi_13p6TeV_correlated": 0.014j,
        })
    elif year == 2023:
        cfg.x.luminosity = Number(27_208, {
            "lumi_13p6TeV_correlated": 0.0j,
        })
    elif year == 2024:
        cfg.x.luminosity = Number(0, {
            "lumi_13p6TeV_correlated": 0.0j,
        })
    else:
        assert False

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    ################################################################################################
    # jet settings
    # TODO: keep a single table somewhere that configures all settings: btag correlation, year
    #       dependence, usage in calibrator, etc
    ################################################################################################

    # common jec/jer settings configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    if run == 2:
        # TODO: check versions
        jec_campaign = f"Summer19UL{year2}{campaign.x.postfix}"
        jec_version = {2016: "V7", 2017: "V5", 2018: "V5"}[year]
        jer_campaign = f"Summer{'20' if year == 2016 else '19'}UL{year2}{campaign.x.postfix}"
        jer_version = "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year]
        jet_type = "AK4PFchs"
    elif run == 3 and year == 2022:
        # TODO: check versions
        jec_campaign = f"Summer{year2}{campaign.x.postfix}_22Sep2023"
        jec_version = {2022: "V2"}[year]
        jer_campaign = f"Summer{year2}{campaign.x.postfix}_22Sep2023"
        jer_version = "JR" + {2022: "V1"}[year]
        jet_type = "AK4PFPuppi"
    else:
        assert False

    cfg.x.jec = DotDict.wrap({
        "campaign": jec_campaign,
        "version": jec_version,
        "jet_type": jet_type,
        "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
        "levels_for_type1_met": ["L1FastJet"],
        "uncertainty_sources": [
            # "AbsoluteStat",
            # "AbsoluteScale",
            # "AbsoluteSample",
            # "AbsoluteFlavMap",
            # "AbsoluteMPFBias",
            # "Fragmentation",
            # "SinglePionECAL",
            # "SinglePionHCAL",
            # "FlavorQCD",
            # "TimePtEta",
            # "RelativeJEREC1",
            # "RelativeJEREC2",
            # "RelativeJERHF",
            # "RelativePtBB",
            # "RelativePtEC1",
            # "RelativePtEC2",
            # "RelativePtHF",
            # "RelativeBal",
            # "RelativeSample",
            # "RelativeFSR",
            # "RelativeStatFSR",
            # "RelativeStatEC",
            # "RelativeStatHF",
            # "PileUpDataMC",
            # "PileUpPtRef",
            # "PileUpPtBB",
            # "PileUpPtEC1",
            # "PileUpPtEC2",
            # "PileUpPtHF",
            # "PileUpMuZero",
            # "PileUpEnvelope",
            # "SubTotalPileUp",
            # "SubTotalRelative",
            # "SubTotalPt",
            # "SubTotalScale",
            # "SubTotalAbsolute",
            # "SubTotalMC",
            "Total",
            # "TotalNoFlavor",
            # "TotalNoTime",
            # "TotalNoFlavorNoTime",
            # "FlavorZJet",
            # "FlavorPhotonJet",
            # "FlavorPureGluon",
            # "FlavorPureQuark",
            # "FlavorPureCharm",
            # "FlavorPureBottom",
            "CorrelationGroupMPFInSitu",
            "CorrelationGroupIntercalibration",
            "CorrelationGroupbJES",
            "CorrelationGroupFlavor",
            "CorrelationGroupUncorrelated",
        ],
    })

    # JER
    cfg.x.jer = DotDict.wrap({
        "campaign": jer_campaign,
        "version": jer_version,
        "jet_type": jet_type,
    })

    ################################################################################################
    # tau settings
    ################################################################################################

    # tau tagger name
    # (needed by TECConfig below as well as tau selection)
    if run == 2:
        # TODO: still correct? what about 2p5?
        cfg.x.tau_tagger = "DeepTau2017v2p1"
    elif run == 3:
        # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3
        cfg.x.tau_tagger = "DeepTau2018v2p5"
    else:
        assert False

    # tec config
    from columnflow.calibration.cms.tau import TECConfig
    corrector_kwargs = {"wp": "Tight", "wp_VSe": "Tight"} if run == 3 else {}
    cfg.x.tec = TECConfig(tagger=cfg.x.tau_tagger, corrector_kwargs=corrector_kwargs)

    # tau ID working points
    if campaign.x.version < 10:
        cfg.x.tau_id_working_points = DotDict.wrap({
            "tau_vs_e": {"vvvloose": 1, "vvloose": 2, "vloose": 4, "loose": 8, "medium": 16, "tight": 32, "vtight": 64, "vvtight": 128},  # noqa
            "tau_vs_jet": {"vvvloose": 1, "vvloose": 2, "vloose": 4, "loose": 8, "medium": 16, "tight": 32, "vtight": 64, "vvtight": 128},  # noqa
            "tau_vs_mu": {"vloose": 1, "loose": 2, "medium": 4, "tight": 8},
        })
    else:
        cfg.x.tau_id_working_points = DotDict.wrap({
            "tau_vs_e": {"vvvloose": 1, "vvloose": 2, "vloose": 3, "loose": 4, "medium": 5, "tight": 6, "vtight": 7, "vvtight": 8},  # noqa
            "tau_vs_jet": {"vvvloose": 1, "vvloose": 2, "vloose": 3, "loose": 4, "medium": 5, "tight": 6, "vtight": 7, "vvtight": 8},  # noqa
            "tau_vs_mu": {"vloose": 1, "loose": 2, "medium": 3, "tight": 4},
        })

    # tau trigger working points
    cfg.x.tau_trigger_working_points = DotDict.wrap({
        "id_vs_jet_v0": "VVLoose",
        "id_vs_jet_gv0": ("Loose", "VVLoose"),
        "id_vs_mu_single": "Tight",
        "id_vs_mu_cross": "VLoose",
        "id_vs_e_single": "VVLoose",
        "id_vs_e_cross": "VVLoose",
        "trigger_corr": "VVLoose",
    })

    ################################################################################################
    # electron settings
    ################################################################################################

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    if run == 2:
        e_postfix = ""
        if year == 2016:
            e_postfix = "preVFP" if campaign.has_tag("preVFP") else "postVFP"
        cfg.x.electron_sf_names = (
            "UL-Electron-ID-SF",
            f"{year}{e_postfix}",
            "wp80iso",
        )
    elif run == 3 and year == 2022:
        cfg.x.electron_sf_names = (
            "Electron-ID-SF",
            "2022Re-recoBCD" if campaign.has_tag("preEE") else "2022Re-recoE+PromptFG",
            "wp80iso",
        )
    else:
        assert False

    ################################################################################################
    # muon settings
    ################################################################################################

    # names of muon correction sets and working points
    # (used in the muon producer)
    if run == 2:
        mu_postfix = ""
        if year == 2016:
            mu_postfix = "preVFP" if campaign.has_tag("preVFP") else "postVFP"
        cfg.x.muon_sf_names = (
            "NUM_TightRelIso_DEN_TightIDandIPCut",
            f"{year}{mu_postfix}_UL",
        )
    elif run == 3 and year == 2022:
        cfg.x.muon_sf_names = (
            "NUM_TightPFIso_DEN_TightID",
            "2022_preEE" if campaign.has_tag("preEE") else "2022_postEE",
        )
    else:
        assert False

    ################################################################################################
    # met settings
    ################################################################################################

    # name of the MET phi correction set
    # (used in the met_phi calibrator)
    cfg.x.met_phi_correction_set = r"{variable}_metphicorr_pfmet_{data_source}"

    ################################################################################################
    # b tagging
    ################################################################################################

    # b-tag working points
    btag_key = f"{year}{campaign.x.postfix}"
    if run == 2:
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18?rev=18
        cfg.x.btag_working_points = DotDict.wrap({
            "deepjet": {
                "loose": {"2016APV": 0.0508, "2016": 0.0480, "2017": 0.0532, "2018": 0.0490}[btag_key],
                "medium": {"2016APV": 0.2598, "2016": 0.2489, "2017": 0.3040, "2018": 0.2783}[btag_key],
                "tight": {"2016APV": 0.6502, "2016": 0.6377, "2017": 0.7476, "2018": 0.7100}[btag_key],
            },
            "deepcsv": {
                "loose": {"2016APV": 0.2027, "2016": 0.1918, "2017": 0.1355, "2018": 0.1208}[btag_key],
                "medium": {"2016APV": 0.6001, "2016": 0.5847, "2017": 0.4506, "2018": 0.4168}[btag_key],
                "tight": {"2016APV": 0.8819, "2016": 0.8767, "2017": 0.7738, "2018": 0.7665}[btag_key],
            },
        })
    elif run == 3:
        # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
        # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
        # TODO later: complete WP when data becomes available
        cfg.x.btag_working_points = DotDict.wrap({
            "deepjet": {
                "loose": {"2022": 0.0583, "2022EE": 0.0614, "2023": 0.0, "2024": 0.0}[btag_key],
                "medium": {"2022": 0.3086, "2022EE": 0.3196, "2023": 0.0, "2024": 0.0}[btag_key],
                "tight": {"2022": 0.7183, "2022EE": 0.73, "2023": 0.0, "2024": 0.0}[btag_key],
                "xtight": {"2022": 0.8111, "2022EE": 0.8184, "2023": 0.0, "2024": 0.0}[btag_key],
                "xxtight": {"2022": 0.9512, "2022EE": 0.9542, "2023": 0.0, "2024": 0.0}[btag_key],
            },
            "robustParticleTransformer": {
                "loose": {"2022": 0.0849, "2022EE": 0.0897, "2023": 0.0, "2024": 0.0}[btag_key],
                "medium": {"2022": 0.4319, "2022EE": 0.451, "2023": 0.0, "2024": 0.0}[btag_key],
                "tight": {"2022": 0.8482, "2022EE": 0.8604, "2023": 0.0, "2024": 0.0}[btag_key],
                "xtight": {"2022": 0.9151, "2022EE": 0.9234, "2023": 0.0, "2024": 0.0}[btag_key],
                "xxtight": {"2022": 0.9874, "2022EE": 0.9893, "2023": 0.0, "2024": 0.0}[btag_key],
            },
            "particleNet": {
                "loose": {"2022": 0.047, "2022EE": 0.0499, "2023": 0.0, "2024": 0.0}[btag_key],
                "medium": {"2022": 0.245, "2022EE": 0.2605, "2023": 0.0, "2024": 0.0}[btag_key],
                "tight": {"2022": 0.6734, "2022EE": 0.6915, "2023": 0.0, "2024": 0.0}[btag_key],
                "xtight": {"2022": 0.7862, "2022EE": 0.8033, "2023": 0.0, "2024": 0.0}[btag_key],
                "xxtight": {"2022": 0.961, "2022EE": 0.9664, "2023": 0.0, "2024": 0.0}[btag_key],
            },
        })
    else:
        assert False

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
        "",  # same as "Total"
        "Absolute",
        "AbsoluteMPFBias",
        "AbsoluteScale",
        "AbsoluteStat",
        f"Absolute_{year}",
        "BBEC1",
        f"BBEC1_{year}",
        "EC2",
        f"EC2_{year}",
        "FlavorQCD",
        "Fragmentation",
        "HF",
        f"HF_{year}",
        "PileUpDataMC",
        "PileUpPtBB",
        "PileUpPtEC1",
        "PileUpPtEC2",
        "PileUpPtHF",
        "PileUpPtRef",
        "RelativeBal",
        "RelativeFSR",
        "RelativeJEREC1",
        "RelativeJEREC2",
        "RelativeJERHF",
        "RelativePtBB",
        "RelativePtEC1",
        "RelativePtEC2",
        "RelativePtHF",
        "RelativeSample",
        f"RelativeSample_{year}",
        "RelativeStatEC",
        "RelativeStatFSR",
        "RelativeStatHF",
        "SinglePionECAL",
        "SinglePionHCAL",
        "TimePtEta",
    ]

    from columnflow.production.cms.btag import BTagSFConfig
    cfg.x.btag_sf = BTagSFConfig(
        correction_set="particleNet_shape",
        jec_sources=cfg.x.btag_sf_jec_sources,
        discriminator="btagPNetB",
    )

    ################################################################################################
    # shifts
    ################################################################################################

    # load jec sources
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]

    # register shifts
    cfg.add_shift(name="nominal", id=0)

    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

    cfg.add_shift(name="mtop_up", id=5, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="mtop_down", id=6, type="shape", tags={"disjoint_from_nominal"})

    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_shift_aliases(
        cfg,
        "minbias_xs",
        {
            "pu_weight": "pu_weight_{name}",
            "normalized_pu_weight": "normalized_pu_weight_{name}",
        },
    )

    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

    for jec_source in cfg.x.jec.uncertainty_sources:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(
            name=f"jec_{jec_source}_up",
            id=5000 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        cfg.add_shift(
            name=f"jec_{jec_source}_down",
            id=5001 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        add_shift_aliases(
            cfg,
            f"jec_{jec_source}",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                "MET.pt": "MET.pt_{name}",
                "MET.phi": "MET.phi_{name}",
            },
        )
        # TODO: check the JEC de/correlation across years and the interplay with btag weights
        if ("" if jec_source == "Total" else jec_source) in cfg.x.btag_sf_jec_sources:
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    "normalized_btag_weight": "normalized_btag_weight_{name}",
                    "normalized_njet_btag_weight": "normalized_njet_btag_weight_{name}",
                },
            )

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"jer"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"jer"})
    add_shift_aliases(
        cfg,
        "jer",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            "MET.pt": "MET.pt_{name}",
            "MET.phi": "MET.phi_{name}",
        },
    )

    for i, (match, dm) in enumerate(itertools.product(["jet", "e"], [0, 1, 10, 11])):
        cfg.add_shift(name=f"tec_{match}_dm{dm}_up", id=20 + 2 * i, type="shape", tags={"tec"})
        cfg.add_shift(name=f"tec_{match}_dm{dm}_down", id=21 + 2 * i, type="shape", tags={"tec"})
        add_shift_aliases(
            cfg,
            f"tec_{match}_dm{dm}",
            {
                "Tau.pt": "Tau.pt_{name}",
                "Tau.mass": "Tau.mass_{name}",
                "MET.pt": "MET.pt_{name}",
                "MET.phi": "MET.phi_{name}",
            },
        )

    # start at id=50
    cfg.x.tau_unc_names = [
        "jet_dm0", "jet_dm1", "jet_dm10",
        "e_barrel", "e_endcap",
        "mu_0p0To0p4", "mu_0p4To0p8", "mu_0p8To1p2", "mu_1p2To1p7", "mu_1p7To2p3",
    ]
    for i, unc in enumerate(cfg.x.tau_unc_names):
        cfg.add_shift(name=f"tau_{unc}_up", id=50 + 2 * i, type="shape")
        cfg.add_shift(name=f"tau_{unc}_down", id=51 + 2 * i, type="shape")
        add_shift_aliases(cfg, f"tau_{unc}", {"tau_weight": f"tau_weight_{unc}_" + "{direction}"})

    cfg.add_shift(name="tautau_trigger_up", id=80, type="shape")
    cfg.add_shift(name="tautau_trigger_down", id=81, type="shape")
    add_shift_aliases(cfg, "tautau_trigger", {"tau_trigger_weight": "tau_trigger_weight_tautau_{direction}"})
    cfg.add_shift(name="etau_trigger_up", id=82, type="shape")
    cfg.add_shift(name="etau_trigger_down", id=83, type="shape")
    add_shift_aliases(cfg, "etau_trigger", {"tau_trigger_weight": "tau_trigger_weight_etau_{direction}"})
    cfg.add_shift(name="mutau_trigger_up", id=84, type="shape")
    cfg.add_shift(name="mutau_trigger_down", id=85, type="shape")
    add_shift_aliases(cfg, "mutau_trigger", {"tau_trigger_weight": "tau_trigger_weight_mutau_{direction}"})
    # no uncertainty for di-tau VBF trigger existing yet
    # cfg.add_shift(name="mutau_trigger_up", id=86, type="shape")
    # cfg.add_shift(name="tautauvbf_trigger_down", id=86, type="shape")
    # add_shift_aliases(cfg, "tautauvbf_trigger", {"tau_trigger_weight": "tau_trigger_weight_tautauvbf_{direction}"})

    cfg.add_shift(name="e_up", id=90, type="shape")
    cfg.add_shift(name="e_down", id=91, type="shape")
    add_shift_aliases(cfg, "e", {"electron_weight": "electron_weight_{direction}"})

    cfg.add_shift(name="mu_up", id=100, type="shape")
    cfg.add_shift(name="mu_down", id=101, type="shape")
    add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

    cfg.x.btag_unc_names = [
        "hf", "lf",
        f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}",
        "cferr1", "cferr2",
    ]
    for i, unc in enumerate(cfg.x.btag_unc_names):
        cfg.add_shift(name=f"btag_{unc}_up", id=110 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=111 + 2 * i, type="shape")
        add_shift_aliases(
            cfg,
            f"btag_{unc}",
            {
                "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
            },
        )

    cfg.add_shift(name="pdf_up", id=130, type="shape", tags={"lhe_weight"})
    cfg.add_shift(name="pdf_down", id=131, type="shape", tags={"lhe_weight"})
    add_shift_aliases(
        cfg,
        "pdf",
        {
            "pdf_weight": "pdf_weight_{direction}",
            "normalized_pdf_weight": "normalized_pdf_weight_{direction}",
        },
    )

    cfg.add_shift(name="murmuf_up", id=140, type="shape", tags={"lhe_weight"})
    cfg.add_shift(name="murmuf_down", id=141, type="shape", tags={"lhe_weight"})
    add_shift_aliases(
        cfg,
        "murmuf",
        {
            "murmuf_weight": "murmuf_weight_{direction}",
            "normalized_murmuf_weight": "normalized_murmuf_weight_{direction}",
        },
    )

    ################################################################################################
    # external files
    ################################################################################################

    cfg.x.external_files = DotDict()

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    if run == 2:
        json_postfix = ""
        if year == 2016:
            json_postfix = f"{'pre' if campaign.has_tag('preVFP') else 'post'}VFP"
        json_pog_era = f"{year}{json_postfix}_UL"
        json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-6ce37404"
    elif run == 3:
        json_pog_era = f"{year}_Summer{year2}{campaign.x.postfix}"
        json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-6ce37404"
    else:
        assert False

    # common files
    # lumi files
    add_external("lumi", {
        "golden": {
            2016: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt", "v1"),  # noqa
            2017: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
            2018: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt", "v1"),  # noqa,
            # TODO: document source
            2022: ("/afs/cern.ch/user/a/anhaddad/public/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
            2023: ("/afs/cern.ch/user/a/anhaddad/public/Collisions23/Cert_Collisions2023_366442_370790_Golden.json", "v1"),  # noqa
        }[year],
        "normtag": {
            2016: ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            2017: ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            2018: ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            # TODO: check
            2022: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            2023: ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        }[year],
    })
    # pileup weight corrections
    add_external("pu_sf", (f"{json_mirror}/POG/LUM/{json_pog_era}/puWeights.json.gz", "v1"))
    # jet energy correction
    add_external("jet_jerc", (f"{json_mirror}/POG/JME/{json_pog_era}/jet_jerc.json.gz", "v1"))
    # jet veto map
    add_external("jet_veto_map", (f"{json_mirror}/POG/JME/{json_pog_era}/jetvetomaps.json.gz", "v1"))
    # btag scale factor
    add_external("btag_sf_corr", (f"{json_mirror}/POG/BTV/{json_pog_era}/btagging.json.gz", "v1"))
    # hh-btag repository (lightweight) with TF saved model directories
    add_external("hh_btag_repo", ("https://github.com/hh-italian-group/HHbtag/archive/df5220db5d4a32d05dc81d652083aece8c99ccab.tar.gz", "v2"))  # noqa
    # Tobias' tautauNN (https://github.com/uhh-cms/tautauNN)
    add_external("res_pdnn", ("/afs/cern.ch/work/m/mrieger/public/hbt/models/res_prod3/model_fold0.tgz", "v1"))
    # non-parametric (flat) training up to mX = 800 GeV
    add_external("res_dnn", ("/afs/cern.ch/work/m/mrieger/public/hbt/models/res_prod3_nonparam/model_fold0.tgz", "v1"))

    # run specific files
    if run == 2:
        # tau energy correction and scale factors
        add_external("tau_sf", (f"{json_mirror}/POG/TAU/{json_pog_era}/tau.json.gz", "v1"))
        # tau trigger scale factors
        add_external("tau_trigger_sf", (f"{json_mirror}/POG/TAU/{json_pog_era}/tau.json.gz", "v1"))
        # electron scale factors
        add_external("electron_sf", (f"{json_mirror}/POG/EGM/{json_pog_era}/electron.json.gz", "v1"))
        # muon scale factors
        add_external("muon_sf", (f"{json_mirror}/POG/MUO/{json_pog_era}/muon_Z.json.gz", "v1"))
        # met phi correction
        add_external("met_phi_corr", (f"{json_mirror}/POG/JME/{json_pog_era}/met.json.gz", "v1"))
    elif run == 3:
        if year == 2022 and campaign.has_tag("preEE"):
            # muon scale factors
            add_external("muon_sf", (f"{json_mirror}/POG/MUO/{json_pog_era}/muon_Z.json.gz", "v1"))
            # electron scale factors
            add_external("electron_sf", (f"{json_mirror}/POG/EGM/{json_pog_era}/electron.json.gz", "v1"))
            # tau energy correction and scale factors
            # TODO: remove tag pog mirror once integrated centrally
            json_mirror_tau_pog = "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-taupog"
            tau_pog_era = f"{year}_{'pre' if campaign.has_tag('preEE') else 'post'}EE"
            add_external("tau_sf", (f"{json_mirror_tau_pog}/POG/TAU/{tau_pog_era}/tau_DeepTau2018v2p5_{tau_pog_era}.json.gz", "v1"))  # noqa
    else:
        assert False

    ################################################################################################
    # reductions
    ################################################################################################

    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # mandatory
            ColumnCollection.MANDATORY_COFFEA,
            # object info
            "Jet.{pt,eta,phi,mass,hadronFlavour,puId,hhbtag,btagPNet*,btagDeep*}",
            "HHBJet.{pt,eta,phi,mass,hadronFlavour,puId,hhbtag,btagPNet*,btagDeep*}",
            "NonHHBJet.{pt,eta,phi,mass,hadronFlavour,puId,hhbtag,btagPNet*,btagDeep*}",
            "VBFJet.{pt,eta,phi,mass,hadronFlavour,puId,hhbtag,btagPNet*,btagDeep*}",
            "FatJet.*",
            "SubJet{1,2}.*",
            "Electron.*",
            "Muon.*",
            "Tau.*",
            "MET.{pt,phi,significance,covXX,covXY,covYY}",
            "PV.npvs",
            "FatJet.*",
            # keep all columns added during selection, but skip cutflow feature
            ColumnCollection.ALL_FROM_SELECTOR,
            skip_column("cutflow.*"),
        },
        "cf.MergeSelectionMasks": {
            "cutflow.*",
        },
        "cf.UniteColumns": {
            "*",
        },
    })

    ################################################################################################
    # weights
    ################################################################################################

    # configurations for all possible event weight columns as keys in an OrderedDict,
    # mapped to shift instances they depend on
    # (this info is used by weight producers)
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pdf_weight": get_shifts("pdf"),
        "murmuf_weight": get_shifts("murmuf"),
        "normalized_pu_weight": get_shifts("minbias_xs"),
        "normalized_njet_btag_weight": get_shifts(*(f"btag_{unc}" for unc in cfg.x.btag_unc_names)),
        "electron_weight": get_shifts("e"),
        "muon_weight": get_shifts("mu"),
        "tau_weight": get_shifts(*(f"tau_{unc}" for unc in cfg.x.tau_unc_names)),
        "tau_trigger_weight": get_shifts("etau_trigger", "mutau_trigger", "tautau_trigger"),
    })

    # define per-dataset event weights
    for dataset in cfg.datasets:
        if dataset.has_tag("is_ttbar"):
            dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    ################################################################################################
    # external configs: channels, categories, met filters, triggers, variables, hist hooks
    ################################################################################################

    # channels
    cfg.add_channel(name="mutau", id=1)
    cfg.add_channel(name="etau", id=2)
    cfg.add_channel(name="tautau", id=3)
    # cfg.add_channel(name="mumu", id=4)

    # add categories
    from hbt.config.categories import add_categories
    add_categories(cfg)

    # add variables
    from hbt.config.variables import add_variables
    add_variables(cfg)

    # add met filters
    from hbt.config.met_filters import add_met_filters
    add_met_filters(cfg)

    # add triggers
    if year == 2016:
        from hbt.config.triggers import add_triggers_2016
        add_triggers_2016(cfg)
    elif year == 2017:
        from hbt.config.triggers import add_triggers_2017
        add_triggers_2017(cfg)
    elif year == 2018:
        from hbt.config.triggers import add_triggers_2018
        add_triggers_2018(cfg)
    elif year == 2022:
        from hbt.config.triggers import add_triggers_2022
        add_triggers_2022(cfg)
    elif year == 2023:
        from hbt.config.triggers import add_triggers_2023
        add_triggers_2023(cfg)
    else:
        raise False

    # add hist hooks
    from hbt.config.hist_hooks import add_hist_hooks
    add_hist_hooks(cfg)

    ################################################################################################
    # LFN settings
    ################################################################################################

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # custom lfn retrieval method in case the underlying campaign is custom uhh
    if cfg.campaign.x("custom", {}).get("creator") == "uhh":
        def get_dataset_lfns(
            dataset_inst: od.Dataset,
            shift_inst: od.Shift,
            dataset_key: str,
        ) -> list[str]:
            # destructure dataset_key into parts and create the lfn base directory
            dataset_id, full_campaign, tier = dataset_key.split("/")[1:]
            main_campaign, sub_campaign = full_campaign.split("-", 1)
            lfn_base = law.wlcg.WLCGDirectoryTarget(
                f"/store/{dataset_inst.data_source}/{main_campaign}/{dataset_id}/{tier}/{sub_campaign}/0",
                fs=f"wlcg_fs_{cfg.campaign.x.custom['name']}",
            )

            # loop though files and interpret paths as lfns
            return [
                lfn_base.child(basename, type="f").path
                for basename in lfn_base.listdir(pattern="*.root")
            ]

        # define the lfn retrieval function
        cfg.x.get_dataset_lfns = get_dataset_lfns

        # define a custom sandbox
        cfg.x.get_dataset_lfns_sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cf.sh")

        # define custom remote fs's to look at
        cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: [
            f"local_fs_{cfg.campaign.x.custom['name']}",
            f"wlcg_fs_{cfg.campaign.x.custom['name']}",
        ]

    return cfg
