# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.first_nn import SimpleDNN


processes = [
    "graviton_hh_ggf_bbtautau_m400",
    "hh_ggf_bbtautau",
    "graviton_hh_vbf_bbtautau_m400",
]

ml_process_weights = {
    "graviton_hh_ggf_bbtautau_m400": 1,
    "hh_ggf_bbtautau": 1,
    "graviton_hh_vbf_bbtautau_m400": 1,
}

dataset_names = {
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "hh_ggf_bbtautau_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
}

# feature_list = ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]

# input_features = [
#     [f"{obj}_{var}"
#     for obj in [f"jet{i}" for i in range(1, 7, 1)]
#     for var in feature_list],
#     ["mjj", "mbjetbjet", "mtautau", "mHH"]]

input_features = [["jets_pt", "jets_e", "jets_mass", "jets_eta", "jets_phi", "jets_btag"],
                  ["mjj", "mbjetbjet", "mtautau", "mHH", "jets_max_d_eta", "jets_d_eta_inv_mass"]]

# Decide on dummy or proper btag of jets: If proper chosen coment out 4 lines below
# for i, name in enumerate(input_features[0]):
#     if name == 'jet1_btag' or name == 'jet2_btag':
#         name += "_dummy"
#         input_features[0][i] = name

default_cls_dict = {
    "folds": 3,
    # "max_events": 10**6,  # TODO
    "layers": [512, 512, 512],
    "activation": "relu",  # Options: elu, relu, prelu, selu, tanh, softmax
    "learningrate": 0.001,
    "batchsize": 131072,
    "epochs": 200,
    "eqweight": True,
    "dropout": 0.50,
    "processes": processes,
    "ml_process_weights": ml_process_weights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "store_name": "inputs1",
    "n_features": len(input_features[0]),
    "n_output_nodes": len(processes),
}

# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)

# test model settings
cls_dict = default_cls_dict
cls_dict["epochs"] = 150
cls_dict["batchsize"] = 2048
cls_dict["model_name"] = f"{len(processes)}classes_shap_plots_deep_sets"

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)

# 2classes_vbfjets_dr_inv_mass
# 2classes_vbfjets
# 3classes_vbfjets
# 3classes_vbfjets_dr_inv_mass
# 3classes_no_vbf
