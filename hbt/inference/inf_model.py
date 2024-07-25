from columnflow.inference import inference_model, ParameterType


@inference_model
def test_unc(self):
    self.add_category(
        "channel",
        config_category="incl",
        config_variable="hh.mass",
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
        config_process="hh_ggf_bbtautau",
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

    # a custom asymmetric uncertainty
    """
    self.add_parameter(
        "QCDscale_ttbar",
        process="TT",
        type=ParameterType.shape,
        transformations=[ParameterTransformation.effect_from_rate],
        effect=(0.85, 1.1),
    )
    self.add_parameter_to_group("QCDscale_ttbar", "experiment")

    # tune uncertainty
    self.add_parameter(
        "tune",
        process="TT",
        type=ParameterType.shape,
        config_shift_source="tune",
    )
    self.add_parameter_to_group("tune", "experiment")
    """
