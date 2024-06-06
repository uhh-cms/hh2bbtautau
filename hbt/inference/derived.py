# coding: utf-8

"""
hbw inference model.
"""

import hbt.inference.constants as const  # noqa
from hbt.inference.base import HBTInferenceModelBase


#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "4classes_DeepSets_no_neg_weights"

# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
    "tt",
    "dy",
]

# All config categories to be included in the final datacard
config_categories = [
    f"incl__ml_{proc}" for proc in processes
]

rate_systematics = [
    # Lumi: should automatically choose viable uncertainties based on campaign
    "lumi_13TeV_2016",
    "lumi_13TeV_2017",
    "lumi_13TeV_1718",
    "lumi_13TeV_correlated",
    # Rate QCDScale uncertainties
    "QCDScale_ttbar",
    "QCDScale_V",
    "QCDScale_VV",
    "QCDScale_VVV",
    "QCDScale_ggH",
    "QCDScale_qqH",
    "QCDScale_VH",
    "QCDScale_ttH",
    "QCDScale_bbH",
    "QCDScale_ggHH",  # should be included in inference model (THU_HH)
    "QCDScale_qqHH",
    "QCDScale_VHH",
    "QCDScale_ttHH",
    # Rate PDF uncertainties
    "pdf_gg",
    "pdf_qqbar",
    "pdf_qg",
    "pdf_Higgs_gg",
    "pdf_Higgs_qqbar",
    "pdf_Higgs_qg",  # none so far
    "pdf_Higgs_ttH",
    "pdf_Higgs_bbH",  # removed
    "pdf_Higgs_ggHH",
    "pdf_Higgs_qqHH",
    "pdf_VHH",
    "pdf_ttHH",
]

# All systematics to be included in the final datacard
systematics = rate_systematics

default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "config_categories": config_categories,
    "systematics": systematics,
    "mc_stats": True,
    "skip_data": True,
}

default = HBTInferenceModelBase.derive(
    f"rates_only_{default_cls_dict['ml_model_name']}", cls_dict=default_cls_dict,
)

ggf_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "config_categories": ["incl"],
    "systematics": systematics,
    "mc_stats": True,
    "skip_data": True,
    "config_variable": lambda inference_model, config_cat_inst: "mlscore.graviton_hh_ggf_bbtautau_m400",
}

ml_score_ggf = HBTInferenceModelBase.derive(
    f"ggf_only_{default_cls_dict['ml_model_name']}", cls_dict=ggf_cls_dict,
)

#
# derive some additional Inference Models
#

cls_dict = default_cls_dict.copy()

cls_dict["systematics"] = rate_systematics
# inference model with only rate uncertainties
# sl_rates_only = default.derive("rates_only", cls_dict=cls_dict)

cls_dict["processes"] = [
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
    "tt",
    "dy",
]

cls_dict["config_categories"] = [
    "1e__ml_ggHH_kl_1_kt_1_sl_hbbhww",
    "1e__ml_st",
]

cls_dict["systematics"] = [
    "lumi_13TeV_2017",
]

cls_dict["ml_model_name"] = "dense_test"

# minimal model for quick test purposes
test = default.derive("test", cls_dict=cls_dict)

# model but with different fit variable
cls_dict["config_variable"] = lambda config_cat_inst: "jet1_pt"
jet1_pt = default.derive("jet1_pt", cls_dict=cls_dict)
