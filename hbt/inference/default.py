# coding: utf-8

"""
Default inference model.
"""

from columnflow.inference import inference_model, ParameterType


@inference_model
def default(self):

    #
    # categories
    #

    self.add_category(
        "incl",
        config_category="incl__os__iso",
        config_variable="res_dnn_hh",
        config_data_datasets=["data_*"],
        mc_stats=8.0,
    )

    #
    # processes
    #

    for kl in ["0", "1", "2p45", "5"]:
        self.add_process(
            f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt",
            is_signal=True,
            config_process=f"hh_ggf_hbb_htt_kl{kl}_kt1",
            config_mc_datasets=[f"hh_ggf_hbb_htt_kl{kl}_kt1_powheg"],
        )
    self.add_process(
        "TT",
        config_process="tt",
        config_mc_datasets=["^tt_(sl|dl|fh)_powheg$"],
    )
    self.add_process(
        "DY",
        config_process="dy",
        config_mc_datasets=["dy_*_amcatnlo"],
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
    self.add_parameter(
        "pdf_Higgs_qqHH",  # contains alpha_s
        type=ParameterType.rate_gauss,
        process="qqHH_*",
        effect=1.027,
        group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
    )
    self.add_parameter(
        "QCDscale_ttbar",
        type=ParameterType.rate_gauss,
        process="TT",
        effect=(0.965, 1.024),
        group=["theory"],
    )
    self.add_parameter(
        "QCDscale_qqHH",
        type=ParameterType.rate_gauss,
        process="qqHH_*",
        effect=(0.9997, 1.0005),
        group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
    )

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
    for name in self.config_inst.x.btag_unc_names:
        self.add_parameter(
            f"CMS_btag_{name}",
            type=ParameterType.shape,
            config_shift_source=f"btag_{name}",
            group="experiment",
        )

    # pileup
    self.add_parameter(
        "CMS_pileup_2022",
        type=ParameterType.shape,
        config_shift_source="minbias_xs",
        group="experiment",
    )

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
