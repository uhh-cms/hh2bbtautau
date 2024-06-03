# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

from __future__ import annotations

import os
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
from columnflow.columnar_util import ColumnCollection


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    # some validations
    assert campaign.x.year in [2022, 2023, 2024]
    if campaign.x.year == 2024:
        raise NotImplementedError("It a bit too early for 2024 analysis :)")

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100

    # postfix for 2022 campaigns after ECAL Endcap water leak
    year_postfix = "EE" if campaign.x.postfix == "post" else ""
    postfixEE = f"{campaign.x.postfix}EE"

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)
    cfg.add_tag("run3")
    if not year_postfix:
        cfg.add_tag("pre")

    # add processes we are interested in
    process_names = [
        "data",
        "tt",
        # "st",
        # "ttv",
        # "ttvv",
        # "dy",
        # "w",
        # "ewk",
        # "vv",
        # "vvv",
        # "qcd",
        # "h",
        "hh_ggf_bbtautau",
        # "hh_vbf_bbtautau",
        # "graviton_hh_ggf_bbtautau_m400",
        # "graviton_hh_ggf_bbtautau_m1250",
    ]
    for process_name in process_names:
        # development switch in case datasets are not _yet_ there
        if process_name not in procs:
            continue

        # add the process
        cfg.add_process(procs.get(process_name))

        # # add the process (and set xsec to 0.1 if not available)
        # process_inst = procs.get(process_name)

        # cfg.add_process(process_inst)
        # for proc_inst in cfg.get_process(process_inst).get_leaf_processes():
        #     if campaign.ecm not in proc_inst.xsecs.keys():
        #         cfg.get_process(proc_inst.name).set_xsec(campaign.ecm, Number(0.1))

    # configure colors, labels, etc
    from hbt.config.styles import stylize_processes
    stylize_processes(cfg)

    # add datasets we need to study
    dataset_names = [
        # data
        # "data_e_b",
        # "data_e_c",
        # "data_e_d",
        # "data_e_e",
        # "data_e_f",
        # "data_mu_a",
        # "data_mu_b",
        # "data_mu_c",
        # "data_mu_d",
        # "data_mu_e",
        # "data_mu_f",
        # "data_mu_g",
        # "data_tau_b",
        # "data_tau_c",
        "data_tau_d",
        # "data_tau_e",
        # "data_tau_f",
        # "data_tau_g",
        # backgrounds
        "tt_sl_powheg",
        # "tt_dl_powheg",
        # "tt_fh_powheg",
        # "ttz_llnunu_amcatnlo",
        # "ttw_nlu_amcatnlo",
        # "ttw_qq_amcatnlo",
        # "ttzz_madgraph",
        # "ttwz_madgraph",
        # "ttww_madgraph",
        # "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg",
        # "st_twchannel_t_powheg",
        # "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo",
        # "st_schannel_had_amcatnlo",
        # "dy_lep_pt50To100_amcatnlo",
        # "dy_lep_pt100To250_amcatnlo",
        # "dy_lep_pt250To400_amcatnlo",
        # "dy_lep_pt400To650_amcatnlo",
        # "dy_lep_pt650_amcatnlo",
        # "w_lnu_madgraph",
        # "ewk_wm_lnu_madgraph",
        # "ewk_w_lnu_madgraph",
        # "ewk_z_ll_madgraph",
        # "zz_pythia",
        # "wz_pythia",
        # "ww_pythia",
        # "zzz_amcatnlo",
        # "wzz_amcatnlo",
        # "wwz_amcatnlo",
        # "www_amcatnlo",
        # "h_ggf_tautau_powheg",
        # "h_vbf_tautau_powheg",
        # "zh_tautau_powheg",
        # "zh_bb_powheg",
        # "wph_tautau_powheg",
        # "wmh_tautau_powheg",
        # "ggzh_llbb_powheg",
        # "tth_tautau_powheg",
        # "tth_bb_powheg",
        # "tth_nonbb_powheg",
        # # signals
        "hh_ggf_hbb_htt_kl1_kt1_c20_powheg",
    ]
    for dataset_name in dataset_names:
        # development switch in case datasets are not _yet_ there
        if dataset_name not in campaign.datasets:
            continue

        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # add tags to datasets
        if dataset.name.startswith("tt"):
            dataset.add_tag(("has_top", "is_ttbar"))
        elif dataset.name.startswith("st"):
            dataset.add_tag(("has_top", "is_single_top"))

        # apply an optional limit on the number of files
        if limit_dataset_files:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

    # verify that the root process of all datasets is part of any of the registered processes
    verify_config_processes(cfg, warn=True)

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "test_no_shifts"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = ("n_jet", "n_btag")
    cfg.x.default_weight_producer = "all_weights"

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {}

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
        "default": ["json", "met_filter", "trigger", "lepton", "jet", "bjet"],
    }

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # lumi values in inverse pb
    # TODO later: preliminary luminosity using norm tag. Must be corrected, when more data is available
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis
    if year == 2022:
        if campaign.x.postfix == "post":
            cfg.x.luminosity = Number(26671.7, {
                "total": 0.014j,
            })
        else:
            cfg.x.luminosity = Number(7980.4, {
                "total": 0.014j,
            })
    elif year == 2023:
        cfg.x.luminosity = Number(27208, {
            "lumi_13TeV_correlated": 0.0j,
        })
    else:  # 2024
        cfg.x.luminosity = Number(0, {
            "lumi_13TeV_correlated": 0.0j,
        })

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    # TODO later: Error for run three not available yet. Using error from run 2.
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # b-tag working points
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    # TODO later: complete WP when data becomes available
    btag_key = f"{year}{year_postfix}"
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

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    # TODO later: check this corrections summary correction_file (jet_jerc.json.gz) after setting sandbox_dev
    cfg.x.jec = DotDict.wrap({
        "campaign": f"Summer{year2}{year_postfix}_22Sep2023",
        "version": {2022: "V2"}[year],
        "jet_type": "AK4PFPuppi",
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
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107 # TODO later: check this
    cfg.x.jer = DotDict.wrap({
        "campaign": f"Summer{year2}{year_postfix}_22Sep2023",
        "version": {2022: "JRV1"}[year],
        "jet_type": "AK4PFPuppi",
    })

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

    # name of the btag_sf correction set and jec uncertainties to propagate through
    cfg.x.btag_sf = ("particleNet_shape", cfg.x.btag_sf_jec_sources, "btagPNetB")

    # name of the deep tau tagger
    # (used in the tec calibrator)
    # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3
    cfg.x.tau_tagger = "DeepTau2018v2p5"  # "DeepTauv2p5"

    # name of the MET phi correction set
    # (used in the met_phi calibrator)
    # cfg.x.met_phi_correction_set = "{variable}_metphicorr_pfmet_{data_source}"

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_sf_names = (
        "Electron-ID-SF",
        "2022Re-recoE+PromptFG" if year_postfix else "2022Re-recoBCD",
        "wp80iso",
    )

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{year}_{postfixEE}")

    # load jec sources
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]

    # register shifts
    cfg.add_shift(name="nominal", id=0)

    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

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
    tau_uncs = [
        "jet_dm0", "jet_dm1", "jet_dm10",
        "e_barrel", "e_endcap",
        "mu_0p0To0p4", "mu_0p4To0p8", "mu_0p8To1p2", "mu_1p2To1p7", "mu_1p7To2p3",
    ]
    for i, unc in enumerate(tau_uncs):
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

    btag_uncs = [
        "hf", "lf",
        f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}",
        "cferr1", "cferr2",
    ]
    for i, unc in enumerate(btag_uncs):
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

    cfg.add_shift(name="pdf_up", id=130, type="shape")
    cfg.add_shift(name="pdf_down", id=131, type="shape")
    add_shift_aliases(
        cfg,
        "pdf",
        {
            "pdf_weight": "pdf_weight_{direction}",
            "normalized_pdf_weight": "normalized_pdf_weight_{direction}",
        },
    )

    cfg.add_shift(name="murmuf_up", id=140, type="shape")
    cfg.add_shift(name="murmuf_down", id=141, type="shape")
    add_shift_aliases(
        cfg,
        "murmuf",
        {
            "murmuf_weight": "murmuf_weight_{direction}",
            "normalized_murmuf_weight": "normalized_murmuf_weight_{direction}",
        },
    )

    # external files
    json_mirror = "/afs/cern.ch/user/a/anhaddad/public/jsonpog-integration"
    json_mirror_alt = "/afs/cern.ch/user/a/anhaddad/public/jsonpog_alt"
    # TODO later: add factors for other POGs when available

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{json_mirror}/POG/LUM/{year}_Summer{year2}{year_postfix}/puWeights.json.gz", "v1"),

        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}_Summer{year2}{year_postfix}/jet_jerc.json.gz", "v1"),

        # tau energy correction and scale factors
        # "tau_sf": (f"{json_mirror}/POG/TAU/{year_folder}/tau.json.gz", "v1"),

        # electron scale factors
        # "electron_sf": (f"{json_mirror}/POG/EGM/{year_folder}/electron.json.gz", "v1"),

        # muon scale factors
        # "muon_sf": (f"{json_mirror}/POG/MUO/{year_folder}/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}_Summer{year2}{year_postfix}/btagging.json.gz", "v1"),

        # met phi corrector
        # "met_phi_corr": (f"{json_mirror}/POG/JME/2018_UL/met.json.gz", "v1"),

        # hh-btag repository (lightweight) with TF saved model directories
        "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/df5220db5d4a32d05dc81d652083aece8c99ccab.tar.gz", "v2"),  # noqa
    })

    if year == 2022:
        cfg.x.external_files.update(DotDict.wrap({
            # Add Muon POG scale factors
            "muon_sf": (f"{json_mirror}/POG/MUO/{year}_Summer{year2}{year_postfix}/muon_Z.json.gz", "v1"),

            # electron scale factors
            "electron_sf": (f"{json_mirror}/POG/EGM/{year}_Summer{year2}{year_postfix}/electron.json.gz", "v1"),

            # tau energy correction and scale factors
            "tau_sf": (f"{json_mirror_alt}/POG/TAU/{year}_{postfixEE}/tau_DeepTau2018v2p5_2022_{postfixEE}.json.gz", "v1"),  # noqa

            # tau trigger
            "tau_trigger_sf": (f"{json_mirror_alt}/POG/TAU/output/tau_trigger_DeepTau2018v2p5_{year}{postfixEE}.json", "v1"),  # noqa
        }))

    # external files with more complex year dependence # TODO: check this
    if year == 2022:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/user/a/anhaddad/public/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileup_JSON.txt", "v1"),  # noqa
                # Problem No file for 2022 --> using 2023 no matching shapes with root shape
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/203834e3ae301f2564423dd1cc84bebf660519b9/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))
    else:  # year 2023
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/user/a/anhaddad/public/Collisions23/Cert_Collisions2023_366442_370790_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/user/a/anhaddad/public/Collisions23/pileup_JSON.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/203834e3ae301f2564423dd1cc84bebf660519b9/SimGeneral/MixingModule/python/mix_2023_25ns_EraCD_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions23/pileupHistogram-Cert_Collisions2023_366442_370790_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions23/pileupHistogram-Cert_Collisions2023_366442_370790_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions23/pileupHistogram-Cert_Collisions2023_366442_370790_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))

    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # general event info
            "run", "luminosityBlock", "event",
            # object info
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.hadronFlavour",
            "Jet.hhbtag", "Jet.btagPNet*",
            "HHBJet.pt", "HHBJet.eta", "HHBJet.phi", "HHBJet.mass", "HHBJet.btagDeepFlavB",
            "HHBJet.hadronFlavour", "HHBJet.hhbtag", "Jet.puId",
            "NonHHBJet.pt", "NonHHBJet.eta", "NonHHBJet.phi", "NonHHBJet.mass",
            "NonHHBJet.btagDeepFlavB", "NonHHBJet.hadronFlavour", "NonHHBJet.hhbtag",
            "Electron.*",
            "Muon.*",
            "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.idDeepTau2017v2p1VSe", "Tau.charge",
            "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet", "Tau.genPartFlav",
            "Tau.decayMode",
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
            "PV.npvs",
            # columns added during selection
            "channel_id", "process_id", "category_ids", "mc_weight", "pdf_weight*", "murmuf_weight*",
            "leptons_os", "tau2_isolated", "single_triggered", "cross_triggered",
            "deterministic_seed", "pu_weight*", "btag_weight*", "cutflow.*",
            # columns added during selection
            ColumnCollection.ALL_FROM_SELECTOR,
        },
        "cf.MergeSelectionMasks": {
            "cutflow.*",
        },
        "cf.UniteColumns": {
            "*",
        },
    })

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pdf_weight": get_shifts("pdf"),
        "murmuf_weight": get_shifts("murmuf"),
        "normalized_pu_weight": get_shifts("minbias_xs"),
        "normalized_njet_btag_weight": get_shifts(*(f"btag_{unc}" for unc in btag_uncs)),
        "electron_weight": get_shifts("e"),
        "muon_weight": get_shifts("mu"),
        "tau_weight": get_shifts(*(f"tau_{unc}" for unc in tau_uncs)),
        "tau_trigger_weight": get_shifts("etau_trigger", "mutau_trigger", "tautau_trigger"),
    })

    # define per-dataset event weights
    for dataset in cfg.datasets:
        if dataset.x("is_ttbar", False):
            dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    # TODO: versioning is disabled for now and will be enabled once needed
    cfg.x.versions = {}
    # if cfg.name == "run2_2017_nano_v9":
    #     cfg.x.versions = {
    #         "cf.CalibrateEvents": "dev1",
    #         "cf.MergeSelectionStats": "dev1",
    #         "cf.MergeSelectionMasks": "dev1",
    #         "cf.SelectEvents": "dev1",
    #         "cf.ReduceEvents": "dev1",
    #         "cf.MergeReductionStats": "dev1",
    #         "cf.MergeReducedEvents": "dev1",
    #     }
    # else:
    #     raise NotImplementedError(f"config versions not implemented for {cfg.name}")

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

    # # add triggers
    if year == 2022:  # TODO later: check if still needed
        from hbt.config.triggers import add_triggers_2022
        add_triggers_2022(cfg)
    elif year == 2023:
        from hbt.config.triggers import add_triggers_2023
        add_triggers_2023(cfg)
    else:
        raise NotImplementedError(f"triggers not implemented for {year}")

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
        cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: f"wlcg_fs_{cfg.campaign.x.custom['name']}"
