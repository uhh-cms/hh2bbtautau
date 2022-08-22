# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import os
import re

import yaml
from scinum import Number
import order as od
import cmsdb
import cmsdb.campaigns.run2_2017

from columnflow.util import DotDict, get_root_processes_from_campaign
from hbt.config.categories import add_categories
from hbt.config.variables import add_variables


thisdir = os.path.dirname(os.path.abspath(__file__))


#
# the main analysis object
#

analysis_hbt = ana = od.Analysis(
    name="analysis_hbt",
    id=1,
)

# analysis-global versions
ana.set_aux("versions", {
})

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
ana.set_aux("cmssw_sandboxes", [
    # "cmssw_default.sh",
])

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.set_aux("config_groups", {})


#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017 = cmsdb.campaigns.run2_2017.campaign_run2_2017.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017)

# create a config by passing the campaign, so id and name will be identical
config_2017 = cfg = ana.add_config(campaign_run2_2017)

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
cfg.add_process(procs.n.h)
cfg.add_process(procs.n.hh_ggf_bbtautau)

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
    "st_tchannel_t",
    "st_tchannel_tbar",
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
    # signal
    "hh_ggf_bbtautau_madgraph",
]
for dataset_name in dataset_names:
    dataset = cfg.add_dataset(campaign_run2_2017.get_dataset(dataset_name))

    # add aux info to datasets
    if dataset.name.startswith(("st", "tt")):
        dataset.x.has_top = True
    if dataset.name.startswith("tt"):
        dataset.x.is_ttbar = True
        dataset.x.event_weights = ["top_pt_weight"]


# default calibrator, selector, producer, ml model and inference model
cfg.set_aux("default_calibrator", "test")
cfg.set_aux("default_selector", "test")
cfg.set_aux("default_producer", "features")
cfg.set_aux("default_ml_model", None)
cfg.set_aux("default_inference_model", "test")

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
cfg.set_aux("process_groups", {})

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
cfg.set_aux("dataset_groups", {})

# category groups for conveniently looping over certain categories
# (used during plotting)
cfg.set_aux("category_groups", {})

# variable groups for conveniently looping over certain variables
# (used during plotting)
cfg.set_aux("variable_groups", {})

# shift groups for conveniently looping over certain shifts
# (used during plotting)
cfg.set_aux("shift_groups", {})

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
cfg.set_aux("selector_step_groups", {
    "test": ["Jet"],
})

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
cfg.set_aux("luminosity", Number(41480, {
    "lumi_13TeV_2018": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
}))

# 2018 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
cfg.set_aux("minbiasxs", Number(69.2, 0.046j))

# location of JEC txt files
cfg.set_aux("jec", DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "V6",
    "jet_type": "AK4PFchs",
    "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
    "data_eras": ["RunB", "RunC", "RunD", "RunE", "RunF"],
    "uncertainty_sources": [
        # comment out most for now to prevent large file sizes
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
}))

cfg.set_aux("jer", DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JRDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "JRV3",
    "jet_type": "AK4PFchs",
}))


# helper to add column aliases for both shifts of a source
def add_aliases(shift_source, aliases):
    for direction in ["up", "down"]:
        shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
        # format keys and values
        inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
        _aliases = {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
        # extend existing or register new column aliases
        shift.set_aux("column_aliases", shift.get_aux("column_aliases", {})).update(_aliases)


# register shifts
cfg.add_shift(name="nominal", id=0)
cfg.add_shift(name="tune_up", id=1, type="shape", aux={"disjoint_from_nominal": True})
cfg.add_shift(name="tune_down", id=2, type="shape", aux={"disjoint_from_nominal": True})
cfg.add_shift(name="hdamp_up", id=3, type="shape", aux={"disjoint_from_nominal": True})
cfg.add_shift(name="hdamp_down", id=4, type="shape", aux={"disjoint_from_nominal": True})
cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"})
cfg.add_shift(name="top_pt_up", id=9, type="shape")
cfg.add_shift(name="top_pt_down", id=10, type="shape")
add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
    all_jec_sources = yaml.load(f, yaml.Loader)["names"]
for jec_source in cfg.x.jec["uncertainty_sources"]:
    idx = all_jec_sources.index(jec_source)
    cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
    cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
    add_aliases(f"jec_{jec_source}", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"})

cfg.add_shift(name="jer_up", id=6000, type="shape")
cfg.add_shift(name="jer_down", id=6001, type="shape")
add_aliases("jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"})


def make_jme_filenames(jme_aux, sample_type, names, era=None):
    """Convenience function to compute paths to JEC files."""

    # normalize and validate sample type
    sample_type = sample_type.upper()
    if sample_type not in ("DATA", "MC"):
        raise ValueError(f"Invalid sample type '{sample_type}'. Expected either 'DATA' or 'MC'.")

    jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

    return [
        f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"
        for name in names
    ]


# external files
cfg.set_aux("external_files", DotDict.wrap({
    # files from TODO
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
    "pu": {
        "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
        "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
        "data_profile": {
            "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
            "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
            "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
        },
    },

    # jet energy correction
    "jec": {
        "mc": [
            (fname, "v1")
            for fname in make_jme_filenames(cfg.x.jec, "mc", names=cfg.x.jec.levels)
        ],
        "data": {
            era: [
                (fname, "v1")
                for fname in make_jme_filenames(cfg.x.jec, "data", names=cfg.x.jec.levels, era=era)
            ]
            for era in cfg.x.jec.data_eras
        },
    },

    # jec energy correction uncertainties
    "junc": {
        "mc": [(make_jme_filenames(cfg.x.jec, "mc", names=["UncertaintySources"])[0], "v1")],
        "data": {
            era: [(make_jme_filenames(cfg.x.jec, "data", names=["UncertaintySources"], era=era)[0], "v1")]
            for era in cfg.x.jec.data_eras
        },
    },

    # jet energy resolution (pt resolution)
    "jer": {
        "mc": [(make_jme_filenames(cfg.x.jer, "mc", names=["PtResolution"])[0], "v1")],
    },

    # jet energy resolution (data/mc scale factors)
    "jersf": {
        "mc": [(make_jme_filenames(cfg.x.jer, "mc", names=["SF"])[0], "v1")],
    },

}))

# columns to keep after certain steps
cfg.set_aux("keep_columns", DotDict.wrap({
    "cf.ReduceEvents": {
        "run", "luminosityBlock", "event",
        "nJet", "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Deepjet.pt", "Deepjet.eta", "Deepjet.btagDeepFlavB",
        "nMuon", "Muon.pt", "Muon.eta",
        "nElectron", "Electron.pt", "Electron.eta",
        "LHEWeight.originalXWGTUP",
        "PV.npvs",
        "category_ids", "deterministic_seed",
    },
    "cf.MergeSelectionMasks": {
        "LHEWeight.originalXWGTUP", "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
}))

# event weight columns
cfg.set_aux("event_weights", ["normalization_weight", "pu_weight"])

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.set_aux("versions", {
})

# add categories
add_categories(cfg)

# add variables
add_variables(cfg)
