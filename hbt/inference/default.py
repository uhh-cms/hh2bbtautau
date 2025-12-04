# coding: utf-8

"""
Default inference model.
"""

from __future__ import annotations

import re
import functools
import itertools
import collections

import law
import order as od

from columnflow.inference import ParameterType  # , ParameterTransformation
from columnflow.config_util import get_datasets_from_process
from columnflow.util import DotDict

from hbt.inference.base import HBTInferenceModel


logger = law.logger.get_logger(__name__)

get_all_datasets_from_process = functools.partial(
    get_datasets_from_process,
    strategy="all",
    only_first=False,
)


class default(HBTInferenceModel):
    """
    Default statistical model for the HH -> bbtautau analysis.
    """

    # settings defined by HBTInferenceModel
    fake_data = True
    add_qcd = True

    # the default variable to use in all categories
    # (see get_category_variable for more details)
    variable = "run3_dnn_moe_hh_fine"

    # channels and phasespaces for category combinations
    channels = ["etau", "mutau", "tautau"]
    phasespaces = ["res1b", "res2b", "boosted"]

    def create_category_combinations(self) -> list[DotDict[str, str]]:
        return [
            DotDict(zip(["channel", "phasespace"], comb))
            for comb in itertools.product(self.channels, self.phasespaces)
        ]

    def create_category_info(self, *, channel: str, phasespace: str) -> HBTInferenceModel.CategoryInfo:
        return self.CategoryInfo(
            combine_category=f"cat_{self.campaign_key}_{channel}_{phasespace}",
            config_category=f"{channel}__{phasespace}__os__iso",
            config_variable=self.get_category_variable(channel=channel, phasespace=phasespace),
            config_data_datasets=["data_*"],
        )

    def get_category_variable(self, **kwargs) -> str:
        # return variable attribute if set
        if getattr(self, "variable", None):
            return self.variable

        raise NotImplementedError("get_category_variable not implemented in case of empty 'variable' attribute")

    def init_proc_map(self) -> None:
        name_map = self.create_proc_name_map()
        self.fill_proc_map(name_map)

    def create_proc_name_map(self) -> None:
        # mapping of process names in the datacard ("combine name") to configs and process names in a dict
        proc_name_map = {
            **{
                f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt": f"hh_ggf_hbb_htt_kl{kl}_kt1"
                for kl in ["1", "0", "2p45", "5"]
            },
            **{
                f"qqHH_CV_{kv}_C2V_{k2v}_kl_{kl}_13p6TeV_hbbhtt": f"hh_vbf_hbb_htt_kv{kv}_k2v{k2v}_kl{kl}"
                for kv, k2v, kl in [
                    ("1", "1", "1"),
                    ("1", "0", "1"),
                    ("1p74", "1p37", "14p4"),
                    ("2p12", "3p87", "m5p96"),
                    ("m0p012", "0p03", "10p2"),
                    ("m0p758", "1p44", "m19p3"),
                    ("m0p962", "0p959", "m1p43"),
                    ("m1p21", "1p94", "m0p94"),
                    ("m1p6", "2p72", "m1p36"),
                    ("m1p83", "3p57", "m3p39"),
                ]
            },
            "ttbar": "tt",
            "ttbarV": "ttv",
            "ttbarVV": "ttvv",
            "singlet": "st",
            "DY": "dy",
            # "EWK": "z",  # currently not used
            "W": "w",
            "VV": "vv",
            "VVV": "vvv",
            "WH_htt": "wh",
            "ZH_hbb": "zh",
            "ggH_htt": "h_ggf",
            "qqH_htt": "h_vbf",
            "ttH_hbb": "tth",
        }

        if self.add_qcd:
            proc_name_map["QCD"] = "qcd"

        return proc_name_map

    def init_parameters(self) -> None:
        # general groups
        self.add_parameter_group("experiment")
        self.add_parameter_group("theory")
        self.add_parameter_group("rate_nuisances")
        self.add_parameter_group("shape_nuisances")

        # groups that contain parameters that solely affect the signal cross section and/or br
        self.add_parameter_group("signal_norm_xs")
        self.add_parameter_group("signal_norm_xsbr")

        # parameter that is added by the HH physics model, representing kl-dependent QCDscale + mtop
        # uncertainties on the ggHH cross section
        self.add_parameter_to_group("THU_HH", "theory")
        self.add_parameter_to_group("THU_HH", "signal_norm_xs")
        self.add_parameter_to_group("THU_HH", "signal_norm_xsbr")

        #
        # simple rate parameters
        #

        # theory uncertainties
        self.add_parameter(
            "BR_hbb",
            type=ParameterType.rate_gauss,
            process=["*_hbb", "*_hbbhtt"],
            effect=(0.9874, 1.0124),
            group=["theory", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "BR_htt",
            type=ParameterType.rate_gauss,
            process=["*_htt", "*_hbbhtt"],
            effect=(0.9837, 1.0165),
            group=["theory", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_gg",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbar"),
            effect=1.042,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_ggHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="ggHH_*",
            effect=1.023,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=1.027,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbar"),
            effect=(0.965, 1.024),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_qqHH",
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=(0.9997, 1.0005),
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_ggH",
            type=ParameterType.rate_gauss,
            process="ggH_*",
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_qqH",
            type=ParameterType.rate_gauss,
            process="qqH_*",
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )
        # TODO: additional theory uncertainties, especially on background processes!

        # lumi
        for config_inst in self.config_insts:
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    process=self.process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "rate_nuisances"],
                )

        #
        # shape parameters from shifts acting on ProduceColumns or CreateHistograms (mostly weight variations)
        #

        # pileup
        for config_inst in self.config_insts:
            self.add_parameter(
                f"CMS_pileup_{self.campaign_keys[config_inst]}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source="minbias_xs"),
                },
                process=self.process_matches(configs=config_inst, skip_qcd=True),
                process_match_mode=all,
                group=["experiment", "shape_nuisances"],
            )

        # top pt weight
        self.add_parameter(
            "CMS_top_pT_reweighting",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="top_pt")
                for config_inst in self.config_insts
            },
            process=self.inject_all_eras("ttbar"),
            group=["experiment", "shape_nuisances"],
        )

        # pdf shape (could be decorrelated across some process groups if needed)
        self.add_parameter(
            "pdf_shape",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="pdf")
                for config_inst in self.config_insts
            },
            process=self.processes_with_lhe_weights,
            group=["theory", "shape_nuisances"],
        )

        # mur/muf shape (could be decorrelated across some process groups if needed)
        self.add_parameter(
            "scale_shape",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="murmuf")
                for config_inst in self.config_insts
            },
            process=self.processes_with_lhe_weights,
            group=["theory", "shape_nuisances"],
        )

        # isr and fsr (could be decorrelated across some process groups if needed)
        for source in ["isr", "fsr"]:
            self.add_parameter(
                f"ps_{source}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source=source)
                    for config_inst in self.config_insts
                },
                process=self.process_matches(skip_qcd=True),
                group=["theory", "shape_nuisances"],
            )

        # btag
        btag_map: collections.defaultdict[str, list[od.Config]] = collections.defaultdict(list)
        for config_inst in self.config_insts:
            for name in config_inst.x.btag_unc_names:
                btag_map[name].append(config_inst)
        for name, config_insts in btag_map.items():
            # decorrelate hf/lfstats across years, correlate others
            if re.match(r"^(l|h)fstats(1|2)$", name):
                for config_inst in config_insts:
                    self.add_parameter(
                        f"CMS_btag_{name}_{self.campaign_keys[config_inst]}",
                        type=ParameterType.shape,
                        config_data={
                            config_inst.name: self.parameter_config_spec(shift_source=f"btag_{name}"),
                        },
                        process=self.process_matches(configs=config_inst, skip_qcd=True),
                        process_match_mode=all,
                        group=["experiment", "shape_nuisances"],
                    )
            else:
                self.add_parameter(
                    f"CMS_btag_{name}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"btag_{name}")
                        for config_inst in config_insts
                    },
                    process=self.process_matches(configs=config_insts, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # electron weights
        # TODO: possibly correlate?
        for e_source in ["e_id", "e_reco"]:
            for config_inst in self.config_insts:
                self.add_parameter(
                    f"CMS_eff_{e_source}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=e_source),
                    },
                    category=["*_etau_*"],
                    process=self.process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # muon weights
        # TODO: possibly correlate?
        for mu_source in ["mu_id", "mu_iso"]:
            for config_inst in self.config_insts:
                self.add_parameter(
                    f"CMS_eff_{mu_source}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=mu_source),
                    },
                    category=["*_mutau_*"],
                    process=self.process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # tau weights
        for config_inst in self.config_insts:
            for name in config_inst.x.tau_unc_names:
                # each uncertainty only applies to specific channels
                unc_channels = config_inst.get_shift(f"tau_{name}_up").x.applies_to_channels
                self.add_parameter(
                    f"CMS_eff_t_{name}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"tau_{name}"),
                    },
                    category=[f"*_{ch}_*" for ch in unc_channels],
                    process=self.process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # trigger weights
        for config_inst in self.config_insts:
            for name in config_inst.x.trigger_legs:
                # each uncertainty only applies to specific channels
                unc_channels = config_inst.get_shift(f"trigger_{name}_up").x.applies_to_channels
                self.add_parameter(
                    f"CMS_bbtt_eff_trig_{name}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"trigger_{name}"),
                    },
                    category=[f"*_{ch}_*" for ch in unc_channels],
                    process=self.process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        #
        # shape parameters that alter the selection
        #

        # jec
        pass  # TODO

        # jer
        pass  # TODO

        # tec
        pass  # TODO

        # eec
        pass  # TODO

        # eer
        pass  # TODO

        # mec
        pass  # TODO

        # mer
        pass  # TODO

        # dy shifts
        for i, dy_name in enumerate(["syst", "stat"]):
            self.add_parameter(
                f"CMS_bbtt_dy_{dy_name}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source=f"dy_{dy_name}")
                    for config_inst in self.config_insts
                },
                process="DY*",
                group=["shape_nuisances"],
            )

        #
        # shape parameters based on entire dataset variations
        #

        # hdamp
        # self.add_parameter(
        #     "hdamp",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="hdamp")
        #         for config_inst in self.config_insts
        #     },
        #     process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )

        # tune
        # self.add_parameter(
        #     "underlying_event",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="tune")
        #         for config_inst in self.config_insts
        #     },
        #     process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )

        # mtop
        # self.add_parameter(
        #     "mtop",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="mtop")
        #         for config_inst in self.config_insts
        #     },
        #     process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )


@default.inference_model
def default_no_shifts(self):
    super(default_no_shifts, self).init_func()

    # remove all parameters that require a shift source other than nominal
    for category_name, process_name, parameter in self.iter_parameters():
        remove = (
            (parameter.type.is_shape and not parameter.transformations.any_from_rate) or
            (parameter.type.is_rate and parameter.transformations.any_from_shape)
        )
        if remove:
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    # repeat the cleanup
    self.init_cleanup()


default_no_shifts_no_vbf = default_no_shifts.derive(
    "default_no_shifts_no_vbf",
    cls_dict={"phasespaces": ["res1b_novbf", "res2b_novbf", "boosted_novbf"]},
)

default_no_shifts_simple = default_no_shifts.derive(
    "default_no_shifts_simple",
    cls_dict={"variable": "run3_dnn_simple_hh_fine"},
)

# for variables from networks trained with different kl variations
for kl in ["kl1", "kl0", "allkl"]:
    default_no_shifts.derive(
        f"default_no_shifts_simple_{kl}",
        cls_dict={"variable": f"run3_dnn_simple_{kl}_hh_fine"},
    )

# even 5k binning
default_no_shifts_simple_5k = default_no_shifts.derive(
    "default_no_shifts_simple_5k",
    cls_dict={"variable": "run3_dnn_moe_hh_fine_5k"},
)


@default.inference_model(variable="run3_dnn_moe_hh_fine_5k", empty_bin_value=0)
def default_bin_opt(self):
    # set everything up as in the default model
    super(default_bin_opt, self).init_func()

    # only keep certain parameters
    keep_parameters = {
        "BR_*",
        "QCDscale_*",
        "bbH_norm_*",
        "lumi_*",
        "CMS_bbtt_eff_trig_*",
        "CMS_btag_*",
        "CMS_eff_e_*",
        "CMS_eff_mu_*",
        "CMS_eff_t_*",
        "CMS_top_pT_reweighting",
        "pdf_*",
        "ps_*",
        "scale_*",
    }
    for category_name, process_name, parameter in self.iter_parameters():
        if not law.util.multi_match(parameter.name, keep_parameters):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    # repeat the cleanup
    self.init_cleanup()
