from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def test_unc(self):
    self.add_category(
        "incl",
        config_category="incl",
        config_variable="hh_mass",
        # fake data
        data_from_processes=["TT", "dy"],
        mc_stats=True,
    )

    self.add_category(
        "2j",
        config_category="2j",
        config_variable="jet1_pt",
        # fake data
        data_from_processes=["TT", "dy"],
        mc_stats=True,
    )

    #
    # processes
    #
    self.add_process(
        "dy",
        config_process="dy",
    )

    self.add_process(
        "TT",
        config_process="tt_sl",
    )

    self.add_process(
        "hh_ggf",
        is_signal=True,
        config_process="hh_ggf_hbb_htt_kl1_kt1",
    )

    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
        )
        self.add_parameter_to_group(unc_name, "experiment")

    # electron uncertainty
    self.add_parameter(
        "CMS_eff_e",  # this is the name of the uncertainty as it will show in the datacard. Let's use some variant of the official naming # noqa
        process="*",
        type=ParameterType.shape,
        config_shift_source="e",  # this is the name of the shift (alias) in the config
    )
    self.add_parameter_to_group("CMS_eff_e", "experiment")

    # a custom asymmetric uncertainty
    self.add_parameter(
        "QCDscale_ttbar",
        process="TT",
        type=ParameterType.shape,
        transformations=[ParameterTransformation.effect_from_rate],
        effect=(0.85, 1.1),
    )
    self.add_parameter_to_group("QCDscale_ttbar", "experiment")

    """
    # tune uncertainty
    self.add_parameter(
        "tune",
        process="TT",
        type=ParameterType.shape,
        config_shift_source="tune",
    )
    self.add_parameter_to_group("tune", "experiment")
    """
