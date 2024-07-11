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
    # some validations
    assert campaign.x.run == 3
    assert campaign.x.year in [2022, 2023, 2024]
    if campaign.x.year == 2022:
        assert campaign.x.postfix in ["pre", "post"]
    if campaign.x.year == 2024:
        raise NotImplementedError("it's a bit too early for the 2024 analysis :)")

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100

    # postfix for 2022 campaigns after ECAL Endcap water leak
    year_postfix = "EE" if campaign.x.postfix == "post" else ""
    postfix_ee = f"{campaign.x.postfix}EE"

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    if not year_postfix:
        cfg.add_tag("pre")

    # helper to enable processes / datasets only for a specific era
    def if_era(
        run: int | list[int],
        year: int | list[int],
        postfix: str | list[str],
        values: list[str],
    ) -> list[str]:
        match = (
            campaign.x.run == run and
            campaign.x.year == year and
            campaign.x("postfix", "") == postfix
        )
        return values if match else []

    # add custom processes
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
        # enable both processes once ttv exists
        # processes=[procs.n.ttv, procs.n.ttvv],
        processes=[procs.n.ttvv],
    )

    # add processes we are interested in
    process_names = [
        "data",
        "tt",
        "st",
        "ttv",
        "ttvv",
        "dy",
        "w",
        "ewk",
        "multiboson",
        "qcd",
        "h",
        "hh_ggf_hbb_htt_kl1_kt1",
        "hh_ggf_hbb_htt_kl0_kt1",
        "hh_ggf_hbb_htt_kl2p45_kt1",
        "hh_ggf_hbb_htt_kl5_kt1",
        "hh_ggf_hbb_htt_kl0_kt1_c21",
        "hh_ggf_hbb_htt_kl1_kt1_c23",
        "hh_vbf_hbb_htt_kv1_k2v1_kl1",
        "hh_vbf_hbb_htt_kv1_k2v0_kl1",
        "hh_vbf_hbb_htt_kv1_k2v1_kl2",
        "hh_vbf_hbb_htt_kv1_k2v2_kl1",
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

        # add the process
        cfg.add_process(proc)

    # configure colors, labels, etc
    from hbt.config.styles import stylize_processes
    stylize_processes(cfg)

    # add datasets we need to study
    dataset_names = [
        # signals
        "hh_ggf_hbb_htt_kl1_kt1_powheg",
        "hh_ggf_hbb_htt_kl0_kt1_powheg",
        "hh_ggf_hbb_htt_kl2p45_kt1_powheg",
        "hh_ggf_hbb_htt_kl5_kt1_powheg",
        "hh_ggf_hbb_htt_kl0_kt1_c21_powheg",
        "hh_ggf_hbb_htt_kl1_kt1_c23_powheg",

        "hh_vbf_hbb_htt_kv1_k2v1_kl1_madgraph",
        "hh_vbf_hbb_htt_kv1_k2v0_kl1_madgraph",
        "hh_vbf_hbb_htt_kv1_k2v1_kl2_madgraph",
        "hh_vbf_hbb_htt_kv1_k2v2_kl1_madgraph",

        "graviton_hh_ggf_hbb_htt_m450_madgraph",
        "graviton_hh_ggf_hbb_htt_m1200_madgraph",
        "radion_hh_ggf_hbb_htt_m700_madgraph",

        # backgrounds
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # TODO: add more
        # "ttz_llnunu_amcatnlo", not available
        # "ttw_nlu_amcatnlo", not available
        # "ttw_qq_amcatnlo", not available
        "ttz_zqq_amcatnlo",
        "ttzz_madgraph",
        # "ttwz_madgraph", not available
        "ttww_madgraph",
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # "st_schannel_lep_amcatnlo", not available
        # "st_schannel_had_amcatnlo", not available
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
        # "ewk_wm_lnu_m50toinf_madgraph", not available
        # "ewk_w_lnu_m50toinf_madgraph", not available
        # "ewk_z_ll_m50toinf_madgraph", not available
        "zz_pythia",
        "wz_pythia",
        "ww_pythia",
        "zzz_amcatnlo",
        "wzz_amcatnlo",
        "wwz_4f_amcatnlo",
        "www_4f_amcatnlo",
        "h_ggf_htt_powheg",
        "h_vbf_htt_powheg",
        # "zh_tautau_powheg", not available
        "vh_hnonbb_amcatnlo",
        "zh_zll_hbb_powheg",
        "zh_zqq_hbb_powheg",
        "wmh_wlnu_hbb_powheg",
        "wph_wlnu_hbb_powheg",
        "zh_gg_zll_hbb_powheg",
        "zh_gg_znunu_hbb_powheg",
        "zh_gg_zqq_hbb_powheg",
        # "wph_tautau_powheg", not available
        # "wmh_tautau_powheg", not available
        # "tth_tautau_powheg", not available
        "tth_hbb_powheg",
        "tth_hnonbb_powheg",

        # data
        *if_era(run=3, year=2022, postfix="pre", values=[
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
    cfg.x.default_weight_producer = "default"

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "backgrounds": (backgrounds := [
            "h",
            "tt",
            "dy",
            "qcd",
            "st",
            "w",
            "multiboson",
            "tt_multiboson",
            "ewk",
        ]),
        "sm_ggf": (sm_ggf_group := ["hh_ggf_hbb_htt_kl1_kt1", *backgrounds]),
        "sm": (sm_group := ["hh_ggf_hbb_htt_kl1_kt1", "hh_vbf_hbb_htt_kv1_k2v1_kl1", *backgrounds]),
        "sm_ggf_data": ["data"] + sm_ggf_group,
        "sm_data": ["data"] + sm_group,
    }

    # define inclusive datasets for the dy process identification
    cfg.x.dy_inclusive_datasets = {
        "m50toinf": cfg.datasets.n.dy_m50toinf_amcatnlo,
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
        "default": ["json", "met_filter", "trigger", "lepton", "jet", "bjet"],
    }

    cfg.x.custom_style_config_groups = {
        "small_legend": {
            "legend_cfg": {"ncols": 2, "fontsize": 16, "columnspacing": 0.6},
        },
    }
    cfg.x.default_custom_style_config = "small_legend"

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # lumi values in inverse pb
    # TODO later: preliminary luminosity using norm tag. Must be corrected, when more data is available
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis
    if year == 2022:
        if campaign.x.postfix == "pre":
            cfg.x.luminosity = Number(7980.4, {
                "total": 0.014j,
            })
        else:  # post
            cfg.x.luminosity = Number(26671.7, {
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

    # tau id working points for the selection for nanoAoD > 10
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

    cfg.x.tau_energy_calibration = ("Tight", "Tight")

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

    from columnflow.production.cms.btag import BTagSFConfig
    cfg.x.btag_sf = BTagSFConfig(
        correction_set="particleNet_shape",
        jec_sources=cfg.x.btag_sf_jec_sources,
        discriminator="btagPNetB",
    )

    # name of the deep tau tagger
    # (used in the tec calibrator)
    # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3
    cfg.x.tau_tagger = "DeepTau2018v2p5"

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_sf_names = (
        "Electron-ID-SF",
        "2022Re-recoE+PromptFG" if year_postfix else "2022Re-recoBCD",
        "wp80iso",
    )

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{year}_{postfix_ee}")

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

    # external files
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-6ce37404"
    # remove the taupog specific json files once they are integrated centrally
    json_mirror_taupog = "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-taupog"

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{json_mirror}/POG/LUM/{year}_Summer{year2}{year_postfix}/puWeights.json.gz", "v1"),

        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}_Summer{year2}{year_postfix}/jet_jerc.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}_Summer{year2}{year_postfix}/btagging.json.gz", "v1"),

        # hh-btag repository (lightweight) with TF saved model directories
        "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/df5220db5d4a32d05dc81d652083aece8c99ccab.tar.gz", "v2"),  # noqa

        # Tobias' tautauNN (https://github.com/uhh-cms/tautauNN)
        "res_pdnn": ("/afs/cern.ch/work/m/mrieger/public/hbt/models/res_prod3/model_fold0.tgz", "v1")  # noqa
    })

    if year == 2022:
        cfg.x.external_files.update(DotDict.wrap({
            # Add Muon POG scale factors
            "muon_sf": (f"{json_mirror}/POG/MUO/{year}_Summer{year2}{year_postfix}/muon_Z.json.gz", "v1"),

            # electron scale factors
            "electron_sf": (f"{json_mirror}/POG/EGM/{year}_Summer{year2}{year_postfix}/electron.json.gz", "v1"),

            # tau energy correction and scale factors
            "tau_sf": (f"{json_mirror_taupog}/POG/TAU/{year}_{postfix_ee}/tau_DeepTau2018v2p5_{year}_{postfix_ee}.json.gz", "v1"),  # noqa
        }))

    # external files with more complex year dependence # TODO: check this
    if year == 2022:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/user/a/anhaddad/public/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            },
        }))
    elif year == 2023:  # year 2023
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/user/a/anhaddad/public/Collisions23/Cert_Collisions2023_366442_370790_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },
        }))

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
        if dataset.x("is_ttbar", False):
            dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    # pinned versions
    # (empty since we use the lookup from the law.cfg instead)
    cfg.x.versions = {}

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
        cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: [
            f"local_fs_{cfg.campaign.x.custom['name']}",
            f"wlcg_fs_{cfg.campaign.x.custom['name']}",
        ]
