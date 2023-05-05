# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.first_nn_untouched import SimpleDNN


processes = [
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
]

custom_procweights = {
    "graviton_hh_ggf_bbtautau_m400": 1,
    "graviton_hh_vbf_bbtautau_m400": 1,
}

dataset_names = {
    "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
    # TTbar
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    # SingleTop
    "st_tchannel_t_powheg",
    "st_tchannel_tbar_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
    "st_schannel_lep_amcatnlo",
    # "st_schannel_had_amcatnlo",
    # WJets
    "w_lnu_ht70To100_madgraph",
    "w_lnu_ht100To200_madgraph",
    "w_lnu_ht200To400_madgraph",
    "w_lnu_ht400To600_madgraph",
    "w_lnu_ht600To800_madgraph",
    "w_lnu_ht800To1200_madgraph",
    "w_lnu_ht1200To2500_madgraph",
    "w_lnu_ht2500_madgraph",
    # DY
    "dy_lep_m50_ht70to100_madgraph",
    "dy_lep_m50_ht100to200_madgraph",
    "dy_lep_m50_ht200to400_madgraph",
    "dy_lep_m50_ht400to600_madgraph",
    "dy_lep_m50_ht600to800_madgraph",
    "dy_lep_m50_ht800to1200_madgraph",
    "dy_lep_m50_ht1200to2500_madgraph",
    "dy_lep_m50_ht2500_madgraph",
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
}

input_features = [
    f"{obj}_{var}"
    # for obj in ["bjet1", "bjet2", "jet1", "jet2", "tau1", "tau2"]
    for obj in ["jet1", "jet2", "tau1", "tau2", "bjet1", "bjet2"]
    for var in ["pt", "eta", "phi", "mass"]
] + ["mtautau", "mjj", "mbjetbjet", "mHH"]

default_cls_dict = {
    "folds": 5,
    # "max_events": 10**6,  # TODO
    "layers": [512, 512, 512],
    "activation": "relu",  # Options: elu, relu, prelu, selu, tanh, softmax
    "learningrate": 0.00050,
    "batchsize": 131072,
    "epochs": 200,
    "eqweight": True,
    "dropout": 0.50,
    "processes": processes,
    "custom_procweights": custom_procweights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "store_name": "inputs1",
}

# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)

# test model settings
cls_dict = default_cls_dict
cls_dict["epochs"] = 6
cls_dict["batchsize"] = 2048
cls_dict["processes"] = [
    "graviton_hh_ggf_bbtautau_m400",
    "graviton_hh_vbf_bbtautau_m400",
]
cls_dict["dataset_names"] = {
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
}

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)
