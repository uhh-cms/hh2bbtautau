# coding: utf-8

"""
Default inference model.
"""

import functools

import law

from columnflow.inference import inference_model, ParameterType, FlowStrategy
from columnflow.config_util import get_datasets_from_process


logger = law.logger.get_logger(__name__)


@inference_model
def default(self):
    # gather config and campaign info
    year2 = self.config_inst.campaign.x.year % 100
    campaign_suffix = ""
    if self.config_inst.campaign.has_tag({"preEE", "preBPix"}):
        campaign_suffix = "pre"
    elif self.config_inst.campaign.has_tag({"postEE", "postBPix"}):
        campaign_suffix = "post"
    campaign_key = f"{year2}{campaign_suffix}"

    # helper
    find_datasets = functools.partial(get_datasets_from_process, self.config_inst, strategy="all")

    # mapping between names of processes in the config and how combine datacards should see them
    proc_map = dict([
        *[
            (f"hh_ggf_hbb_htt_kl{kl}_kt1", f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt")
            for kl in ["0", "1", "2p45", "5"]
        ],
        ("tt", "ttbar"),
        ("ttv", "ttbarV"),
        ("ttvv", "ttbarVV"),
        ("st", "singlet"),
        ("dy", "DY"),
        # ("z", "EWK"),  # currently not used
        ("w", "W"),
        ("vv", "VV"),
        ("vvv", "VVV"),
        ("wh", "WH_htt"),
        ("zh", "ZH_hbb"),
        ("h_ggf", "ggH_htt"),
        ("h_vbf", "qqH_htt"),
        ("tth", "ttH_hbb"),
        ("qcd", "QCD"),
    ])

    #
    # categories
    #

    for ch in ["etau", "mutau", "tautau"]:
        for cat in ["res1b", "res2b", "boosted"]:
            self.add_category(
                f"cat_{campaign_key}_{ch}_{cat}",
                config_category=f"{ch}__{cat}__os__iso",
                config_variable="res_dnn_hh_fine",
                config_data_datasets=["data_*"],
                data_from_processes=[
                    combine_name for proc_name, combine_name in proc_map.items()
                    if (
                        not self.config_inst.get_process(proc_name).has_tag("nonresonant_signal") and
                        proc_name != "qcd"
                    )
                ],
                mc_stats=10.0,
                flow_strategy=FlowStrategy.move,
            )

    #
    # processes
    #

    for proc_name, combine_name in proc_map.items():
        proc_inst = self.config_inst.get_process(proc_name)
        is_dynamic = proc_name == "qcd"
        dataset_names = []
        if not is_dynamic:
            dataset_names = [dataset.name for dataset in find_datasets(proc_name)]
            if not dataset_names:
                logger.debug(
                    f"skipping process {proc_name} in inference model {self.cls_name}, no matching "
                    f"datasets found in config {self.config_inst.name}",
                )
                continue
        self.add_process(
            name=combine_name,
            config_process=proc_name,
            config_mc_datasets=dataset_names,
            is_signal=proc_inst.has_tag("nonresonant_signal"),
            is_dynamic=is_dynamic,
        )

    #
    # parameters
    #

    # general groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # groups that contain parameters that solely affect the signal cross section and/or br
    self.add_parameter_group("signal_norm_xs")
    self.add_parameter_group("signal_norm_xsbr")

    # parameter that is added by the HH physics model, representing kl-dependent QCDscale + mtop
    # uncertainties on the ggHH cross section
    self.add_parameter_to_group("THU_HH", "theory")
    self.add_parameter_to_group("THU_HH", "signal_norm_xs")
    self.add_parameter_to_group("THU_HH", "signal_norm_xsbr")

    # theory uncertainties
    self.add_parameter(
        "BR_hbb",
        type=ParameterType.rate_gauss,
        process=["*_hbb", "*_hbbhtt"],
        effect=(0.9874, 1.0124),
        group=["theory", "signal_norm_xsbr"],
    )
    self.add_parameter(
        "BR_htt",
        type=ParameterType.rate_gauss,
        process=["*_htt", "*_hbbhtt"],
        effect=(0.9837, 1.0165),
        group=["theory", "signal_norm_xsbr"],
    )
    self.add_parameter(
        "BR_htt",
        type=ParameterType.rate_gauss,
        process=["tt*"],
        effect=(0.9, 1.1),
        group=["theory", "signal_norm_xsbr"],
    )
    self.add_parameter(
        "pdf_gg",  # contains alpha_s
        type=ParameterType.rate_gauss,
        process="TT",
        effect=1.042,
        group=["theory"],
    )
    self.add_parameter(
        "pdf_Higgs_ggHH",  # contains alpha_s
        type=ParameterType.rate_gauss,
        process="ggHH_*",
        effect=1.023,
        group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
    )
    # self.add_parameter(
    #     "pdf_Higgs_qqHH",  # contains alpha_s
    #     type=ParameterType.rate_gauss,
    #     process="qqHH_*",
    #     effect=1.027,
    #     group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
    # )
    self.add_parameter(
        "QCDscale_ttbar",
        type=ParameterType.rate_gauss,
        process="TT",
        effect=(0.965, 1.024),
        group=["theory"],
    )
    # self.add_parameter(
    #     "QCDscale_qqHH",
    #     type=ParameterType.rate_gauss,
    #     process="qqHH_*",
    #     effect=(0.9997, 1.0005),
    #     group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
    # )

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            group="experiment",
        )

    # btag
    # for name in self.config_inst.x.btag_unc_names:
    #     self.add_parameter(
    #         f"CMS_btag_{name}",
    #         type=ParameterType.shape,
    #         config_shift_source=f"btag_{name}",
    #         group="experiment",
    #     )

    # pileup
    # self.add_parameter(
    #     "CMS_pileup_2022",
    #     type=ParameterType.shape,
    #     config_shift_source="minbias_xs",
    #     group="experiment",
    # )

    #
    # cleanup
    #

    self.cleanup(keep_parameters="THU_HH")


@inference_model
def default_no_shifts(self):
    # same initialization as "default" above
    default.init_func.__get__(self, self.__class__)()

    #
    # remove all parameters that require a shift source other than nominal
    #

    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    #
    # cleanup
    #

    self.cleanup(keep_parameters="THU_HH")
