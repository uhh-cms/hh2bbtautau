# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import os
import re
from collections import OrderedDict

import yaml
from scinum import Number
import law
import order as od
import cmsdb
import cmsdb.campaigns.run2_2017_nano_v9

from columnflow.util import DotDict, get_root_processes_from_campaign
from hbt.config.styles import stylize_processes
from hbt.config.categories import add_categories
from hbt.config.variables import add_variables
from hbt.config.met_filters import add_met_filters
from hbt.config.triggers import add_triggers_2017


thisdir = os.path.dirname(os.path.abspath(__file__))


#
# the main analysis object
#

analysis_hbt = ana = od.Analysis(
    name="analysis_hbt",
    id=1,
)

# analysis-global versions
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf_prod.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    "$HBT_BASE/sandboxes/venv_columnar_tf.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("HBT_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017_nano_v9 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017_nano_v9)

# create a config by passing the campaign, so id and name will be identical
config_2017 = cfg = ana.add_config(campaign_run2_2017_nano_v9)

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
    dataset = cfg.add_dataset(campaign_run2_2017_nano_v9.get_dataset(dataset_name))

    # add aux info to datasets
    if dataset.name.startswith(("st", "tt")):
        dataset.x.has_top = True
    if dataset.name.startswith("tt"):
        dataset.x.is_ttbar = True
        dataset.x.event_weights = ["top_pt_weight"]


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
    "test": ["Jet"],
}

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
cfg.x.luminosity = Number(41480, {
    "lumi_13TeV_2017": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
})

# 2018 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
cfg.x.minbias_xs = Number(69.2, 0.046j)

# b-tag working points
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

# name of the btag_sf correction set
cfg.x.btag_sf_correction_set = "deepJet_shape"

# name of the deep tau tagger
# (used in the tec calibrator)
cfg.x.tau_tagger = "DeepTau2017v2p1"

# name of the MET phi correction set
# (used in the met_phi calibrator)
cfg.x.met_phi_correction_set = "metphicorr_{variable}_pfmet_{data_source}_2017"

# location of JEC txt files
cfg.x.jec = DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "V6",
    "jet_type": "AK4PFchs",
    "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
    "data_eras": sorted(filter(None, {d.x("jec_era", None) for d in cfg.datasets if d.is_data})),
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
})

# JEC uncertainty sources propagated to btag scale factors
# (names derived from contents in BTV correctionlib file)
cfg.x.btag_sf_jec_sources = [
    "",  # total
    "Absolute",
    "AbsoluteMPFBias",
    "AbsoluteScale",
    "AbsoluteStat",
    "Absolute_2017",
    "BBEC1",
    "BBEC1_2017",
    "EC2",
    "EC2_2017",
    "FlavorQCD",
    "Fragmentation",
    "HF",
    "HF_2017",
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
    "RelativeSample_2017",
    "RelativeStatEC",
    "RelativeStatFSR",
    "RelativeStatHF",
    "SinglePionECAL",
    "SinglePionHCAL",
    "TimePtEta",
]

cfg.x.jer = DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JRDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "JRV3",
    "jet_type": "AK4PFchs",
})


# helper to add column aliases for both shifts of a source
def add_aliases(
    shift_source: str,
    aliases: dict,
    selection_dependent: bool = False,
):
    aux_key = "column_aliases" + ("_selection_dependent" if selection_dependent else "")
    for direction in ["up", "down"]:
        shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
        # format keys and values
        inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
        _aliases = {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
        # extend existing or register new column aliases
        shift.set_aux(aux_key, shift.x(aux_key, {}).update(_aliases))


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
add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"})
cfg.add_shift(name="top_pt_up", id=9, type="shape")
cfg.add_shift(name="top_pt_down", id=10, type="shape")
add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})
for jec_source in cfg.x.jec["uncertainty_sources"]:
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
cfg.add_shift(name="jer_up", id=6000, type="shape")
cfg.add_shift(name="jer_down", id=6001, type="shape")
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
for i, dm in enumerate(["0", "1", "10", "11"]):
    cfg.add_shift(name=f"tec_dm{dm}_up", id=20 + 2 * i, type="shape")
    cfg.add_shift(name=f"tec_dm{dm}_down", id=21 + 2 * i, type="shape")
    add_aliases(
        f"tec_dm{dm}",
        {
            "Tau.pt": "Tau.pt_{name}",
            "Tau.mass": "Tau.mass_{name}",
            "MET.pt": "MET.pt_{name}",
            "MET.phi": "MET.phi_{name}",
        },
        selection_dependent=True,
    )

# tau weight shifts go here, ids 60 to 99

cfg.add_shift(name="hf_up", id=100, type="shape")
cfg.add_shift(name="hf_down", id=101, type="shape")
cfg.add_shift(name="lf_up", id=102, type="shape")
cfg.add_shift(name="lf_down", id=103, type="shape")
cfg.add_shift(name="hfstats1_2017_up", id=104, type="shape")
cfg.add_shift(name="hfstats1_2017_down", id=105, type="shape")
cfg.add_shift(name="hfstats2_2017_up", id=106, type="shape")
cfg.add_shift(name="hfstats2_2017_down", id=107, type="shape")
cfg.add_shift(name="lfstats1_2017_up", id=108, type="shape")
cfg.add_shift(name="lfstats1_2017_down", id=109, type="shape")
cfg.add_shift(name="lfstats2_2017_up", id=110, type="shape")
cfg.add_shift(name="lfstats2_2017_down", id=111, type="shape")
cfg.add_shift(name="cferr1_up", id=112, type="shape")
cfg.add_shift(name="cferr1_down", id=113, type="shape")
cfg.add_shift(name="cferr2_up", id=114, type="shape")
cfg.add_shift(name="cferr2_down", id=115, type="shape")


def make_jme_filename(jme_aux, sample_type, name, era=None):
    """
    Convenience function to compute paths to JEC files.
    """
    # normalize and validate sample type
    sample_type = sample_type.upper()
    if sample_type not in ("DATA", "MC"):
        raise ValueError(f"invalid sample type '{sample_type}', expected either 'DATA' or 'MC'")

    jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

    return f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"


# external files
cfg.x.external_files = DotDict.wrap({
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
        "mc": OrderedDict([
            (level, (make_jme_filename(cfg.x.jec, "mc", name=level), "v1"))
            for level in cfg.x.jec.levels
        ]),
        "data": {
            era: OrderedDict([
                (level, (make_jme_filename(cfg.x.jec, "data", name=level, era=era), "v1"))
                for level in cfg.x.jec.levels
            ])
            for era in cfg.x.jec.data_eras
        },
    },

    # jec energy correction uncertainties
    "junc": {
        "mc": [(make_jme_filename(cfg.x.jec, "mc", name="UncertaintySources"), "v1")],
        "data": {
            era: [(make_jme_filename(cfg.x.jec, "data", name="UncertaintySources", era=era), "v1")]
            for era in cfg.x.jec.data_eras
        },
    },

    # jet energy resolution (pt resolution)
    "jer": {
        "mc": [(make_jme_filename(cfg.x.jer, "mc", name="PtResolution"), "v1")],
    },

    # jet energy resolution (data/mc scale factors)
    "jersf": {
        "mc": [(make_jme_filename(cfg.x.jer, "mc", name="SF"), "v1")],
    },

    # tau energy correction and scale factors
    "tau": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-d0a522ea/POG/TAU/2017_UL/tau.json.gz", "v1"),  # noqa

    # met phi corrector
    "met_phi_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/cms-met/metphi_corrs_all.json.gz", "v1"),

    # hh-btag repository (lightweight) with TF saved model directories
    "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/1dc426053418e1cab2aec021802faf31ddf3c5cd.tar.gz", "v1"),  # noqa

    # btag scale factor
    "btag_sf_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-d0a522ea/POG/BTV/2017_UL/btagging.json.gz", "v1"),  # noqa

})

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
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass", "BJet.btagDeepFlavB", "BJet.hadronFlavour",
        "BJet.hhbtag",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.deltaEtaSC",
        "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.pfRelIso04_all",
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.idDeepTau2017v2p1VSe",
        "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet", "Tau.genPartFlav",
        "Tau.decayMode",
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
        # columns added during selection
        "channel", "leptons_os", "tau2_isolated", "single_triggered", "cross_triggered",
        "mc_weight", "PV.npvs", "category_ids", "deterministic_seed", "cutflow.*",
    },
    "cf.MergeSelectionMasks": {
        "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
})

# event weight columns
cfg.x.event_weights = ["normalization_weight", "pu_weight"]

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.x.versions = {}

# cannels
cfg.add_channel(name="mutau", id=1)
cfg.add_channel(name="etau", id=2)
cfg.add_channel(name="tautau", id=3)

# add categories
add_categories(cfg)

# add variables
add_variables(cfg)

# add met filters
add_met_filters(cfg)

# add triggers
add_triggers_2017(cfg)
