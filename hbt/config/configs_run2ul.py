# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import os
import re
import itertools

import yaml
from scinum import Number
import order as od

from columnflow.util import DotDict, get_root_processes_from_campaign


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(analysis: od.Analysis, campaign: od.Campaign) -> od.Config:
    # some validations
    assert campaign.x.year in [2016, 2017, 2018]
    if campaign.x.year == 2016:
        assert campaign.x.vfp in ["pre", "post"]

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = f"{campaign.x.vfp}VFP" if year == 2016 else ""

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis.add_config(campaign)

    # add processes we are interested in
    cfg.add_process(procs.n.data)
    cfg.add_process(procs.n.tt)
    cfg.add_process(procs.n.st)
    cfg.add_process(procs.n.ttv)
    cfg.add_process(procs.n.ttvv)
    cfg.add_process(procs.n.dy)
    cfg.add_process(procs.n.w)
    cfg.add_process(procs.n.ewk)
    cfg.add_process(procs.n.vv)
    cfg.add_process(procs.n.vvv)
    cfg.add_process(procs.n.qcd)
    cfg.add_process(procs.n.h)
    cfg.add_process(procs.n.hh_ggf_bbtautau)

    # configure colors, labels, etc
    from hbt.config.styles import stylize_processes
    stylize_processes(cfg)

    # add datasets we need to study
    dataset_names = [
        # data
        "data_e_b",
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
        "data_mu_b",
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
        "data_tau_b",
        "data_tau_c",
        "data_tau_d",
        "data_tau_e",
        "data_tau_f",
        # backgrounds
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        "ttz_llnunu_amcatnlo",
        "ttw_nlu_amcatnlo",
        "ttw_qq_amcatnlo",
        "ttzz_madgraph",
        "ttwz_madgraph",
        "ttww_madgraph",
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_amcatnlo",
        "st_schannel_had_amcatnlo",
        "dy_lep_pt50To100_amcatnlo",
        "dy_lep_pt100To250_amcatnlo",
        "dy_lep_pt250To400_amcatnlo",
        "dy_lep_pt400To650_amcatnlo",
        "dy_lep_pt650_amcatnlo",
        "w_lnu_madgraph",
        "ewk_wm_lnu_madgraph",
        "ewk_w_lnu_madgraph",
        "ewk_z_ll_madgraph",
        "zz_pythia",
        "wz_pythia",
        "ww_pythia",
        "zzz_amcatnlo",
        "wzz_amcatnlo",
        "wwz_amcatnlo",
        "www_amcatnlo",
        "h_ggf_tautau_powheg",
        "h_vbf_tautau_powheg",
        "zh_tautau_powheg",
        "zh_bb_powheg",
        "wph_tautau_powheg",
        "wmh_tautau_powheg",
        "ggzh_llbb_powheg",
        "tth_tautau_powheg",
        "tth_bb_powheg",
        "tth_nonbb_powheg",
        # signals
        "hh_ggf_bbtautau_madgraph",
    ]
    for dataset_name in dataset_names:
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # add aux info to datasets
        if dataset.name.startswith(("st", "tt")):
            dataset.x.has_top = True
        if dataset.name.startswith("tt"):
            dataset.x.is_ttbar = True

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "test"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = ("n_jet", "n_btag")

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
        "default": ["met_filter", "trigger_fired", "leptons", "jet", "bjet"],
    }

    # custom method for determining dataset lfns
    cfg.x.determine_dataset_lfns = None

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    if year == 2016:
        cfg.x.luminosity = Number(36310, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2017:
        cfg.x.luminosity = Number(41480, {
            "lumi_13TeV_2017": 0.02j,
            "lumi_13TeV_1718": 0.006j,
            "lumi_13TeV_correlated": 0.009j,
        })
    else:  # 2018
        cfg.x.luminosity = Number(59830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = True

    # b-tag working points
    if year == 2016:
        if campaign.x.vfp == "pre":
            # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
            cfg.x.btag_working_points = DotDict.wrap({
                "deepjet": {
                    "loose": 0.0508,
                    "medium": 0.2598,
                    "tight": 0.6502,
                },
                "deepcsv": {
                    "loose": 0.2027,
                    "medium": 0.6001,
                    "tight": 0.8819,
                },
            })
        else:  # post
            # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
            cfg.x.btag_working_points = DotDict.wrap({
                "deepjet": {
                    "loose": 0.0480,
                    "medium": 0.2489,
                    "tight": 0.6377,
                },
                "deepcsv": {
                    "loose": 0.1918,
                    "medium": 0.5847,
                    "tight": 0.8767,
                },
            })
    elif year == 2017:
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
        cfg.x.btag_working_points = DotDict.wrap({
            "deepjet": {
                "loose": 0.0532,
                "medium": 0.3040,
                "tight": 0.7476,
            },
            "deepcsv": {
                "loose": 0.1355,
                "medium": 0.4506,
                "tight": 0.7738,
            },
        })
    else:  # 2018
        # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=17
        cfg.x.btag_working_points = DotDict.wrap({
            "deepjet": {
                "loose": 0.0490,
                "medium": 0.2783,
                "tight": 0.7100,
            },
            "deepcsv": {
                "loose": 0.1208,
                "medium": 0.4168,
                "tight": 0.7665,
            },
        })

    # name of the btag_sf correction set
    cfg.x.btag_sf_correction_set = "deepJet_shape"

    # name of the deep tau tagger
    # (used in the tec calibrator)
    cfg.x.tau_tagger = "DeepTau2017v2p1"

    # name of the MET phi correction set
    # (used in the met_phi calibrator)
    cfg.x.met_phi_correction_set = "{variable}_metphicorr_pfmet_{data_source}"

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_sf_names = ("UL-Electron-ID-SF", f"{year}{corr_postfix}", "wp80iso")

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}{corr_postfix}_UL")

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = "APV" if year == 2016 and campaign.x.vfp == "post" else ""
    cfg.x.jec = DotDict.wrap({
        "campaign": f"Summer19UL{year2}{jerc_postfix}",
        "version": {2016: "V7", 2017: "V5", 2018: "V5"}[year],
        "jet_type": "AK4PFchs",
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
            # "TimeRunA",
            # "TimeRunB",
            # "TimeRunC",
            # "TimeRunD",
            "CorrelationGroupMPFInSitu",
            "CorrelationGroupIntercalibration",
            "CorrelationGroupbJES",
            "CorrelationGroupFlavor",
            "CorrelationGroupUncorrelated",
        ],
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "campaign": f"Summer19UL{year2}{jerc_postfix}",
        "version": "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year],
        "jet_type": "AK4PFchs",
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

    # helper to add column aliases for both shifts of a source
    def add_aliases(
        shift_source: str,
        aliases: dict,
        selection_dependent: bool = False,
    ):
        aux_key = "column_aliases" + ("_selection_dependent" if selection_dependent else "")
        for direction in ["up", "down"]:
            shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
            _aliases = shift.x(aux_key, {})
            # format keys and values
            inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
            _aliases.update({inject_shift(key): inject_shift(value) for key, value in aliases.items()})
            # extend existing or register new column aliases
            shift.set_aux(aux_key, _aliases)

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
    add_aliases(
        "minbias_xs",
        {
            "pu_weight": "pu_weight_{name}",
            "normalized_pu_weight": "normalized_pu_weight_{name}",
        },
    )

    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

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
        # selection dependent aliases
        add_aliases(
            f"jec_{jec_source}",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                "MET.pt": "MET.pt_{name}",
                "MET.phi": "MET.phi_{name}",
            },
            selection_dependent=True,
        )
        # selection independent aliases
        # TODO: check the JEC de/correlation across years and the interplay with btag weights
        if ("" if jec_source == "Total" else jec_source) in cfg.x.btag_sf_jec_sources:
            add_aliases(
                f"jec_{jec_source}",
                {
                    "normalized_btag_weight": "normalized_btag_weight_{name}",
                    "normalized_njet_btag_weight": "normalized_njet_btag_weight_{name}",
                },
            )

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"jer"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"jer"})
    add_aliases(
        "jer",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            "MET.pt": "MET.pt_{name}",
            "MET.phi": "MET.phi_{name}",
        },
        selection_dependent=True,
    )

    for i, (match, dm) in enumerate(itertools.product(["jet", "e"], [0, 1, 10, 11])):
        cfg.add_shift(name=f"tec_{match}_dm{dm}_up", id=20 + 2 * i, type="shape", tags={"tec"})
        cfg.add_shift(name=f"tec_{match}_dm{dm}_down", id=21 + 2 * i, type="shape", tags={"tec"})
        add_aliases(
            f"tec_{match}_dm{dm}",
            {
                "Tau.pt": "Tau.pt_{name}",
                "Tau.mass": "Tau.mass_{name}",
                "MET.pt": "MET.pt_{name}",
                "MET.phi": "MET.phi_{name}",
            },
            selection_dependent=True,
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
        add_aliases(f"tau_{unc}", {"tau_weight": f"tau_weight_{unc}_" + "{direction}"})

    cfg.add_shift(name="tautau_trigger_up", id=80, type="shape")
    cfg.add_shift(name="tautau_trigger_down", id=81, type="shape")
    add_aliases("tautau_trigger", {"tau_trigger_weight": "tau_trigger_weight_tautau_{direction}"})
    cfg.add_shift(name="etau_trigger_up", id=82, type="shape")
    cfg.add_shift(name="etau_trigger_down", id=83, type="shape")
    add_aliases("etau_trigger", {"tau_trigger_weight": "tau_trigger_weight_etau_{direction}"})
    cfg.add_shift(name="mutau_trigger_up", id=84, type="shape")
    cfg.add_shift(name="mutau_trigger_down", id=85, type="shape")
    add_aliases("mutau_trigger", {"tau_trigger_weight": "tau_trigger_weight_mutau_{direction}"})
    # no uncertainty for di-tau VBF trigger existing yet
    # cfg.add_shift(name="mutau_trigger_up", id=86, type="shape")
    # cfg.add_shift(name="tautauvbf_trigger_down", id=86, type="shape")
    # add_aliases("tautauvbf_trigger", {"tau_trigger_weight": "tau_trigger_weight_tautauvbf_{direction}"})

    cfg.add_shift(name="e_up", id=90, type="shape")
    cfg.add_shift(name="e_down", id=91, type="shape")
    add_aliases("e", {"electron_weight": "electron_weight_{direction}"})

    cfg.add_shift(name="mu_up", id=100, type="shape")
    cfg.add_shift(name="mu_down", id=101, type="shape")
    add_aliases("mu", {"muon_weight": "muon_weight_{direction}"})

    btag_uncs = [
        "hf", "lf",
        f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}",
        "cferr1", "cferr2",
    ]
    for i, unc in enumerate(btag_uncs):
        cfg.add_shift(name=f"btag_{unc}_up", id=110 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=111 + 2 * i, type="shape")
        add_aliases(
            f"btag_{unc}",
            {
                "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
            },
        )

    # external files
    corrlib_base = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-849c6a6e"
    cfg.x.external_files = DotDict.wrap({
        # jet energy correction
        "jet_jerc": (f"{corrlib_base}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

        # tau energy correction and scale factors
        "tau_sf": (f"{corrlib_base}/POG/TAU/{year}{corr_postfix}_UL/tau.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{corrlib_base}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{corrlib_base}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{corrlib_base}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{corrlib_base}/POG/JME/{year}{corr_postfix}_UL/met.json.gz", "v1"),

        # hh-btag repository (lightweight) with TF saved model directories
        "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/1dc426053418e1cab2aec021802faf31ddf3c5cd.tar.gz", "v1"),  # noqa
    })

    # external files with more complex year dependence
    if year == 2016:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/a65c2e1a23f2e7fe036237e2e34cda8af06b8182/SimGeneral/MixingModule/python/mix_2016_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-66000ub-99bins.root", "v1"),  # noqa
                },
            },
        }))
    elif year == 2017:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
                },
            },
        }))
    else:  # year 2018
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions1/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/a65c2e1a23f2e7fe036237e2e34cda8af06b8182/SimGeneral/MixingModule/python/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-66000ub-99bins.root", "v1"),  # noqa
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
            "Jet.hhbtag",
            "HHBJet.pt", "HHBJet.eta", "HHBJet.phi", "HHBJet.mass", "HHBJet.btagDeepFlavB",
            "HHBJet.hadronFlavour", "HHBJet.hhbtag",
            "NonHHBJet.pt", "NonHHBJet.eta", "NonHHBJet.phi", "NonHHBJet.mass",
            "NonHHBJet.btagDeepFlavB", "NonHHBJet.hadronFlavour", "NonHHBJet.hhbtag",
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.deltaEtaSC",
            "Electron.pfRelIso03_all",
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.pfRelIso04_all",
            "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.idDeepTau2017v2p1VSe",
            "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet", "Tau.genPartFlav",
            "Tau.decayMode",
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
            "PV.npvs",
            # columns added during selection
            "channel_id", "process_id", "category_ids", "mc_weight", "leptons_os", "tau2_isolated",
            "single_triggered", "cross_triggered", "deterministic_seed", "pu_weight*", "btag_weight*",
            "cutflow.*",
        },
        "cf.MergeSelectionMasks": {
            "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
        },
        "cf.CoalesceColumns": {
            "*",
        },
    })

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = lambda *keys: sum(([cfg.get_shift(f"{k}_up"), cfg.get_shift(f"{k}_down")] for k in keys), [])
    cfg.x.event_weights = DotDict()
    cfg.x.event_weights["normalization_weight"] = []
    cfg.x.event_weights["normalized_pu_weight"] = get_shifts("minbias_xs")
    cfg.x.event_weights["normalized_njet_btag_weight"] = get_shifts(*(f"btag_{unc}" for unc in btag_uncs))
    cfg.x.event_weights["electron_weight"] = get_shifts("e")
    cfg.x.event_weights["muon_weight"] = get_shifts("mu")
    cfg.x.event_weights["tau_weight"] = get_shifts(*(f"tau_{unc}" for unc in tau_uncs))
    cfg.x.event_weights["tau_trigger_weight"] = get_shifts("etau_trigger", "mutau_trigger", "tautau_trigger")

    # define per-dataset event weights
    for dataset in cfg.datasets:
        if dataset.x("is_ttbar", False):
            dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    if cfg.name == "run2_2017_nano_v9":
        cfg.x.versions = {
            # "cf.CalibrateEvents": "dev1",
            # "cf.MergeSelectionStats": "dev1",
            # "cf.MergeSelectionMasks": "dev1",
            # "cf.SelectEvents": "dev1",
            # "cf.ReduceEvents": "dev1",
            # "cf.MergeReductionStats": "dev1",
            # "cf.MergeReducedEvents": "dev1",
        }
    else:
        raise NotImplementedError(f"config versions not implemented for {cfg.name}")

    # cannels
    cfg.add_channel(name="mutau", id=1)
    cfg.add_channel(name="etau", id=2)
    cfg.add_channel(name="tautau", id=3)

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
    if year == 2017:
        from hbt.config.triggers import add_triggers_2017
        add_triggers_2017(cfg)
    else:
        raise NotImplementedError(f"triggers not implemented for {year}")
