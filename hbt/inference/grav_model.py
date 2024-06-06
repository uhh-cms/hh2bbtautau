from columnflow.inference import inference_model, ParameterType


@inference_model
def grav_model(self):

    #
    # categories
    #

    for mass in ["400", "450", "500"]:
        self.add_category(
            f"cat{mass}",
            config_category="incl",
            config_variable="jet1_pt",
            # fake data from TT
            data_from_processes=["TT"],
            mc_stats=True,
        )

    #
    # processes
    #
    for mass in ["400", "450", "500"]:
        self.add_process(
            f"ggf_spin_2_mass_{mass}_hbbhtt",
            is_signal=True,
            config_process=f"graviton_hh_ggf_bbtautau_m{mass}",
            category=f"cat{mass}",
        )

    self.add_process(
        "TT",
        config_process="tt_sl",
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

    # a custom asymmetric uncertainty that is converted from rate to shape
    self.add_parameter(
        "QCDscale_ttbar",
        process="TT",
        type=ParameterType.rate_gauss,
        effect=(0.85, 1.1),
    )
    self.add_parameter_to_group("QCDscale_ttbar", "experiment")
